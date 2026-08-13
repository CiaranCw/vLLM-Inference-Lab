#!/usr/bin/env python3
"""M23 · L1 正式跨引擎对照：**2×2 execution-mode**（V-E / V-O / T-E / T-O）。

设计冻结 v4 §5.0。硬约束：
  - 正式结论**分别报告** `T-E vs V-E` 与 `T-O vs V-O`；**不得跨 lane 相减**；
    **任一差值都不得称为「纯 engine intrinsic speedup」**。
  - **Primary metrics 才承担正式结论**：端到端请求延迟 / 请求吞吐 / 输出 token 吞吐 /
    **总 GPU 显存** / 冷启动时间。engine 内部指标只用于解释。
  - **显存只报 total（host 侧 nvidia-smi 采样）**：Gate 2 已实测 TRT 侧主进程
    `torch.cuda.memory_allocated()` 读到 0（executor 在独立 runtime 内），
    分项不可得 → 标 `unavailable`，**禁止残差推导**。
  - 四组都**回读并保存实际 execution / graph 配置**（不信入参，只信回读）。
  - 两侧用**同一份 token ids**（`outputs/m23/bench_inputs.json`），逐字节一致。

用法：
    python scripts/m23_bench.py --build-inputs      # 只生成输入（host，无 GPU）
    python scripts/m23_bench.py --group V-E         # 跑单组
    python scripts/m23_bench.py --all               # 四组串行
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import subprocess
import threading
import time

OUT, LOGD = "outputs/m23", "logs/m23"
RAW = os.path.join(OUT, "raw")
DOCKER = os.path.expanduser("~/bin/docker")
HF_CACHE = os.path.expanduser("~/.cache/huggingface")
VLLM_IMAGE = "nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3"
TRT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24"
INPUTS = os.path.join(OUT, "bench_inputs.json")

# ---- 冻结的 workload 与网格 ----
IN_TOKENS, OUT_TOKENS = 256, 128          # Shape A（与 M12/M18/M20 同）
N_PROMPTS = 64                            # batch 吞吐用
SINGLE_WARMUP, SINGLE_MEASURE = 3, 10     # 单请求延迟
BATCH_WARMUP, BATCH_MEASURE = 2, 5        # batch-64 吞吐

# ---- 冻结的对齐配置（A2–A6）----
MAX_MODEL_LEN, MAX_BATCH, MAX_NUM_TOKENS, MEM_FRACTION = 2048, 64, 4096, 0.6

GROUPS = {
    "V-E": {"engine": "vllm", "lane": "E"},
    "V-O": {"engine": "vllm", "lane": "O"},
    "T-E": {"engine": "trt", "lane": "E"},
    "T-O": {"engine": "trt", "lane": "O"},
}

VLLM_PAYLOAD = r'''
import json, time, traceback
LANE="__LANE__"; IDS=json.loads(r"""__IDS__"""); MODEL="__MODELPATH__"
OUT_TOKENS=__OUTTOK__; SW,SM=__SW__,__SM__; BW,BM=__BW__,__BM__
rec={"group":"V-"+LANE,"engine":"vllm","lane":LANE}
try:
    import vllm
    from vllm import LLM, SamplingParams
    rec["engine_version"]=vllm.__version__
    t0=time.perf_counter()
    llm=LLM(model=MODEL, enforce_eager=(LANE=="E"), dtype="bfloat16",
            max_model_len=__MAXLEN__, gpu_memory_utilization=__MEMFRAC__,
            max_num_seqs=__MAXBATCH__, max_num_batched_tokens=__MAXTOK__,
            enable_prefix_caching=False, enable_chunked_prefill=False, seed=0)
    rec["cold_start_s"]=round(time.perf_counter()-t0,3)
    vc=llm.llm_engine.vllm_config; cc=vc.compilation_config
    rec["effective"]={"compilation_mode":str(getattr(cc,"mode",None)),
        "cudagraph_mode":str(getattr(cc,"cudagraph_mode",None)),
        "dtype":str(vc.model_config.dtype),"kv_cache_dtype":str(vc.cache_config.cache_dtype),
        "kv_cache_size_tokens":getattr(vc.cache_config,"kv_cache_size_tokens",None),
        "max_num_seqs":vc.scheduler_config.max_num_seqs,
        "max_num_batched_tokens":vc.scheduler_config.max_num_batched_tokens,
        "chunked_prefill_enabled":getattr(vc.scheduler_config,"chunked_prefill_enabled",None),
        "gpu_memory_utilization":vc.cache_config.gpu_memory_utilization,
        "enable_prefix_caching":vc.cache_config.enable_prefix_caching}
    sp=SamplingParams(temperature=0,max_tokens=OUT_TOKENS,ignore_eos=True,seed=0)
    def run(batch):
        t=time.perf_counter()
        o=llm.generate([{"prompt_token_ids":x} for x in batch],sp,use_tqdm=False)
        return (time.perf_counter()-t)*1e3,[len(y.outputs[0].token_ids) for y in o]
    for _ in range(SW): run([IDS[0]])
    single=[];lens=[]
    for i in range(SM):
        ms,L=run([IDS[i%len(IDS)]]); single.append(ms); lens+=L
    for _ in range(BW): run(IDS)
    batch=[]
    for _ in range(BM):
        ms,L=run(IDS); batch.append(ms); lens+=L
    rec["single_latency_ms"]=single; rec["batch_wall_ms"]=batch
    rec["out_len_all_equal"]=all(n==OUT_TOKENS for n in lens); rec["n_out_checked"]=len(lens)
except Exception as e:
    rec["error"]=f"{type(e).__name__}: {e}"
    rec["traceback_tail"]=traceback.format_exc().strip().splitlines()[-8:]
print("M23_BENCH "+json.dumps(rec,ensure_ascii=False))
'''

TRT_PAYLOAD = r'''
import json, time, traceback
LANE="__LANE__"; IDS=json.loads(r"""__IDS__"""); MODEL="__MODELPATH__"
OUT_TOKENS=__OUTTOK__; SW,SM=__SW__,__SM__; BW,BM=__BW__,__BM__
rec={"group":"T-"+LANE,"engine":"tensorrt_llm","lane":LANE}
try:
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig
    rec["engine_version"]=getattr(tensorrt_llm,"__version__","unknown")
    kv=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=__MEMFRAC__)
    # E lane：显式关掉 CUDA Graph；O lane：用默认 CudaGraphConfig（optimized ON）
    cg=None if LANE=="E" else CudaGraphConfig()
    t0=time.perf_counter()
    llm=LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=1,
            max_batch_size=__MAXBATCH__, max_num_tokens=__MAXTOK__,
            max_seq_len=__MAXLEN__, enable_chunked_prefill=False,
            kv_cache_config=kv, cuda_graph_config=cg)
    rec["cold_start_s"]=round(time.perf_counter()-t0,3)
    a=getattr(llm,"args",None)
    rec["effective"]={k:(str(getattr(a,k)) if not isinstance(getattr(a,k,None),(int,float,str,bool,type(None))) else getattr(a,k))
        for k in ("dtype","max_batch_size","max_num_tokens","max_seq_len","backend",
                  "enable_chunked_prefill","kv_cache_config","cuda_graph_config") if hasattr(a,k)}
    # TRT 的 SamplingParams 没有 ignore_eos：用 min_tokens=max_tokens 强制定长
    sp=SamplingParams(temperature=0,max_tokens=OUT_TOKENS,min_tokens=OUT_TOKENS,seed=0)
    def run(batch):
        t=time.perf_counter()
        o=llm.generate(batch,sp,use_tqdm=False)
        return (time.perf_counter()-t)*1e3,[len(y.outputs[0].token_ids) for y in o]
    for _ in range(SW): run([IDS[0]])
    single=[];lens=[]
    for i in range(SM):
        ms,L=run([IDS[i%len(IDS)]]); single.append(ms); lens+=L
    for _ in range(BW): run(IDS)
    batch=[]
    for _ in range(BM):
        ms,L=run(IDS); batch.append(ms); lens+=L
    rec["single_latency_ms"]=single; rec["batch_wall_ms"]=batch
    rec["out_len_all_equal"]=all(n==OUT_TOKENS for n in lens); rec["n_out_checked"]=len(lens)
except Exception as e:
    rec["error"]=f"{type(e).__name__}: {e}"
    rec["traceback_tail"]=traceback.format_exc().strip().splitlines()[-8:]
print("M23_BENCH "+json.dumps(rec,ensure_ascii=False))
'''


def model_path_in_ctr() -> str:
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/"))[0].rstrip("/")
    return ("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
            + os.path.basename(snap))


def build_inputs():
    """host 侧生成一次：N_PROMPTS 条、每条精确 IN_TOKENS 个 token，互不相同。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    base = tok.encode(
        "Large language model inference proceeds in two phases: prefill and decode. "
        "During prefill the entire prompt is processed at once, computing keys and values "
        "for every input token and writing them into the cache. During decode only a single "
        "new token is produced per step, yet the whole history of cached keys and values "
        "must be read back from memory. ", add_special_tokens=False)
    # 旋转基序列构造互不相同的等长 prompt（确定性；避免任何前缀复用带来的不公平）
    ids = [[base[(i * 7 + k) % len(base)] for k in range(IN_TOKENS)] for i in range(N_PROMPTS)]
    assert all(len(x) == IN_TOKENS for x in ids)
    assert len({tuple(x) for x in ids}) == N_PROMPTS, "prompt 必须互不相同"
    os.makedirs(OUT, exist_ok=True)
    json.dump({"n_prompts": N_PROMPTS, "in_tokens": IN_TOKENS, "out_tokens": OUT_TOKENS,
               "note": "两侧共用同一份 token ids，逐字节一致", "token_ids": ids},
              open(INPUTS, "w"))
    print(f"[m23] -> {INPUTS}  ({N_PROMPTS} × {IN_TOKENS} tokens)")


class GpuSampler(threading.Thread):
    """host 侧 nvidia-smi 采样：显存只报 total（分项在 TRT 侧不可得）。"""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        # 注意：**不能**叫 self._stop —— 会遮蔽 threading.Thread._stop()，
        # join() 时报 "Event object is not callable"。M12 当年踩过同一个坑
        # （notes/m12/serving.md §8 第①条），这里重蹈覆辙了一次。
        self.interval, self.samples, self._stopev = interval, [], threading.Event()

    def run(self):
        while not self._stopev.is_set():
            try:
                q = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
                used, util = [int(x.strip()) for x in q.split(",")]
                self.samples.append({"t": time.time(), "mem_used_mib": used, "util": util})
            except Exception:
                pass
            self._stopev.wait(self.interval)

    def stop(self):
        self._stopev.set()


def run_group(group: str) -> dict:
    cfg = GROUPS[group]
    ids = json.load(open(INPUTS))["token_ids"]
    tmpl = VLLM_PAYLOAD if cfg["engine"] == "vllm" else TRT_PAYLOAD
    payload = (tmpl.replace("__LANE__", cfg["lane"])
               .replace("__IDS__", json.dumps(ids))
               .replace("__MODELPATH__", model_path_in_ctr())
               .replace("__OUTTOK__", str(OUT_TOKENS))
               .replace("__SW__", str(SINGLE_WARMUP)).replace("__SM__", str(SINGLE_MEASURE))
               .replace("__BW__", str(BATCH_WARMUP)).replace("__BM__", str(BATCH_MEASURE))
               .replace("__MAXLEN__", str(MAX_MODEL_LEN)).replace("__MAXBATCH__", str(MAX_BATCH))
               .replace("__MAXTOK__", str(MAX_NUM_TOKENS)).replace("__MEMFRAC__", str(MEM_FRACTION)))
    image = VLLM_IMAGE if cfg["engine"] == "vllm" else TRT_IMAGE
    env = ["-e", "TQDM_DISABLE=1", "-e", "HF_HUB_OFFLINE=1",
           "-e", "HF_HOME=/root/.cache/huggingface"]
    if cfg["engine"] == "vllm":
        env += ["-e", "VLLM_USE_FLASHINFER_SAMPLER=0", "-e", "VLLM_ENABLE_V1_MULTIPROCESSING=0"]
    cmd = [DOCKER, "run", "--rm", "--gpus", "all", "--ipc", "host",
           "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", *env,
           "-v", f"{HF_CACHE}:/root/.cache/huggingface", image, "python3", "-c", payload]

    # 基线：容器启动前的整卡占用
    base = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True).strip()
    sampler = GpuSampler()
    sampler.start()
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    sampler.stop()
    sampler.join(timeout=5)
    elapsed = round(time.time() - t0, 1)

    log = os.path.join(LOGD, f"bench_{group}.log")
    open(log, "w").write(proc.stdout or "")
    rec, dec = None, json.JSONDecoder()
    for line in (proc.stdout or "").splitlines():
        i = line.find("M23_BENCH ")
        if i >= 0:
            try:
                rec, _ = dec.raw_decode(line[i + len("M23_BENCH "):])
            except ValueError:
                pass

    mem = [s["mem_used_mib"] for s in sampler.samples]
    return {"group": group, "image": image, "returncode": proc.returncode,
            "wall_elapsed_s": elapsed, "record": rec, "log": log,
            "gpu_total_memory_mib": {
                "baseline_before_container": int(base),
                "peak_during_run": max(mem) if mem else None,
                "median_during_run": statistics.median(mem) if mem else None,
                "n_samples": len(mem),
                "note": ("整卡口径（host nvidia-smi）。**分项 unavailable**：TRT 侧 executor 在独立 "
                         "runtime 内，主进程 torch 计数为 0（Gate 2 实测）。**禁止残差推导**。")},
            "gpu_samples": sampler.samples}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-inputs", action="store_true")
    ap.add_argument("--group", choices=list(GROUPS))
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()

    if a.build_inputs:
        build_inputs()
        return 0
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)
    if not os.path.exists(INPUTS):
        build_inputs()

    groups = list(GROUPS) if a.all else [a.group]
    for g in groups:
        print(f"[m23] === {g} ===", flush=True)
        r = run_group(g)
        json.dump(r, open(os.path.join(RAW, f"bench_{g}.json"), "w"), ensure_ascii=False, indent=2)
        rec = r["record"]
        ok = rec is not None and "error" not in (rec or {})
        print(f"[m23] {g}: rc={r['returncode']} wall={r['wall_elapsed_s']}s ok={ok} "
              f"peak_mem={r['gpu_total_memory_mib']['peak_during_run']}MiB", flush=True)
        if ok:
            s, b = rec["single_latency_ms"], rec["batch_wall_ms"]
            print(f"       cold_start={rec['cold_start_s']}s "
                  f"single_med={statistics.median(s):.1f}ms "
                  f"batch64_med={statistics.median(b):.1f}ms "
                  f"out_len_ok={rec['out_len_all_equal']}", flush=True)
        else:
            print(f"       ⚠️ 失败：{(rec or {}).get('error', 'no record')}", flush=True)
            print(f"       按冻结稿：保存证据并停止该 lane，回报所有者。log={r['log']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
