#!/usr/bin/env python3
"""M23 · 跨引擎 generation consistency / divergence check（L1 收口补项）。

复用 `outputs/m23/preflight_inputs.json` 的 **6 条固定 token-id**，
分别做 **V-E vs T-E**、**V-O vs T-O**。

记录：exact token match、common-prefix length、first-divergence position、
输出长度、functional sanity。

**判据纪律（跑前冻结）**：
  - **硬失败** = 进程崩溃 / 空输出 / 超长（n_tokens > max_tokens）。
    这才是 functional sanity 失败。
  - **TRT 侧 n_tokens < max_tokens 不自动判失败**：TRT-LLM SamplingParams
    **没有 ignore_eos**（bench 用 `min_tokens=max_tokens` 对齐，但遇 EOS 仍可能提前停）。
    记为 `eos_early_stop`。vLLM 用 `ignore_eos=True` 会在 EOS 后继续垫到定长。
  - **跨引擎 exact-match 失败只记为 numerical divergence**，
    **不得**直接判为 correctness failure。
    理由：同一 vLLM 的 V-E/V-O 已经只有 1/6 exact match
    （`outputs/m23/vctr_preflight.json`），跨引擎 mismatch 是预期的数值分歧。
  - **不扩成大规模质量 benchmark**：固定 6 条、greedy、max_tokens=64。

同引擎 execution-mode ablation（V-O vs V-E、T-O vs T-E）作为**诊断附表**落盘
（零额外 GPU 成本，因为四组都要跑），不承担跨引擎 correctness 结论。

用法：
    python scripts/m23_cross_engine_consistency.py                 # 四组串行
    python scripts/m23_cross_engine_consistency.py --group V-E
    python scripts/m23_cross_engine_consistency.py --summarize-only  # 不占 GPU，从 raw 重汇总
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

OUT, LOGD = "outputs/m23", "logs/m23"
RAW = os.path.join(OUT, "raw")
DOCKER = os.path.expanduser("~/bin/docker")
HF_CACHE = os.path.expanduser("~/.cache/huggingface")
VLLM_IMAGE = "nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3"
TRT_IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24"
INPUTS = os.path.join(OUT, "preflight_inputs.json")
MAX_TOKENS = 64
GPU_BUSY_MIB = 4000
MAX_MODEL_LEN, MAX_BATCH, MAX_NUM_TOKENS, MEM_FRACTION = 2048, 64, 4096, 0.6

GROUPS = {
    "V-E": {"engine": "vllm", "lane": "E"},
    "V-O": {"engine": "vllm", "lane": "O"},
    "T-E": {"engine": "trt", "lane": "E"},
    "T-O": {"engine": "trt", "lane": "O"},
}

# 正式跨引擎对照只允许同 lane；交叉 lane 不做 engine speedup / 正确性对照
CROSS_ENGINE_PAIRS = (("V-E", "T-E"), ("V-O", "T-O"))
# 同引擎 execution-mode ablation（允许；诊断附表）
SAME_ENGINE_PAIRS = (("V-E", "V-O"), ("T-E", "T-O"))

VLLM_PAYLOAD = r'''
import json, time, traceback
LANE="__LANE__"; IDS=json.loads(r"""__IDS__"""); MODEL="__MODELPATH__"
MAX_TOKENS=__MAXTOK__
rec={"group":"V-"+LANE,"engine":"vllm","lane":LANE}
try:
    import vllm
    from vllm import LLM, SamplingParams
    rec["engine_version"]=vllm.__version__
    t0=time.perf_counter()
    llm=LLM(model=MODEL, enforce_eager=(LANE=="E"), dtype="bfloat16",
            max_model_len=__MAXLEN__, gpu_memory_utilization=__MEMFRAC__,
            max_num_seqs=__MAXBATCH__, max_num_batched_tokens=__MAXNUMTOK__,
            enable_prefix_caching=False, enable_chunked_prefill=False, seed=0)
    rec["build_s"]=round(time.perf_counter()-t0,3)
    vc=llm.llm_engine.vllm_config; cc=vc.compilation_config
    rec["effective"]={"compilation_mode":str(getattr(cc,"mode",None)),
        "cudagraph_mode":str(getattr(cc,"cudagraph_mode",None)),
        "dtype":str(vc.model_config.dtype)}
    sp=SamplingParams(temperature=0,max_tokens=MAX_TOKENS,ignore_eos=True,seed=0)
    outs=llm.generate([{"prompt_token_ids":x} for x in IDS],sp,use_tqdm=False)
    rec["generations"]=[{
        "text":o.outputs[0].text,
        "token_ids":[int(t) for t in o.outputs[0].token_ids],
        "n_tokens":len(o.outputs[0].token_ids),
    } for o in outs]
except Exception as e:
    rec["error"]=f"{type(e).__name__}: {e}"
    rec["traceback_tail"]=traceback.format_exc().strip().splitlines()[-8:]
print("M23_CONS "+json.dumps(rec,ensure_ascii=False))
'''

TRT_PAYLOAD = r'''
import json, time, traceback
LANE="__LANE__"; IDS=json.loads(r"""__IDS__"""); MODEL="__MODELPATH__"
MAX_TOKENS=__MAXTOK__
rec={"group":"T-"+LANE,"engine":"tensorrt_llm","lane":LANE}
try:
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig, CudaGraphConfig
    rec["engine_version"]=getattr(tensorrt_llm,"__version__","unknown")
    kv=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=__MEMFRAC__)
    cg=None if LANE=="E" else CudaGraphConfig()
    t0=time.perf_counter()
    llm=LLM(model=MODEL, dtype="bfloat16", tensor_parallel_size=1,
            max_batch_size=__MAXBATCH__, max_num_tokens=__MAXNUMTOK__,
            max_seq_len=__MAXLEN__, enable_chunked_prefill=False,
            kv_cache_config=kv, cuda_graph_config=cg)
    rec["build_s"]=round(time.perf_counter()-t0,3)
    a=getattr(llm,"args",None)
    rec["effective"]={k:(str(getattr(a,k)) if not isinstance(getattr(a,k,None),(int,float,str,bool,type(None))) else getattr(a,k))
        for k in ("dtype","backend","cuda_graph_config") if hasattr(a,k)}
    sp=SamplingParams(temperature=0,max_tokens=MAX_TOKENS,min_tokens=MAX_TOKENS,seed=0)
    outs=llm.generate(IDS,sp,use_tqdm=False)
    rec["generations"]=[{
        "text":o.outputs[0].text,
        "token_ids":[int(t) for t in o.outputs[0].token_ids],
        "n_tokens":len(o.outputs[0].token_ids),
    } for o in outs]
except Exception as e:
    rec["error"]=f"{type(e).__name__}: {e}"
    rec["traceback_tail"]=traceback.format_exc().strip().splitlines()[-8:]
print("M23_CONS "+json.dumps(rec,ensure_ascii=False))
'''


def model_path_in_ctr() -> str:
    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/"))[0].rstrip("/")
    return ("/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/"
            + os.path.basename(snap))


def gpu_snapshot() -> dict:
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        used, total, util = [x.strip() for x in q.split(",")]
        return {"mem_used_mib": int(used), "mem_total_mib": int(total), "gpu_util_pct": int(util)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def common_prefix_len(a, b) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def first_divergence_pos(a, b):
    """0-based index of first mismatch; None if exact match (incl. equal length)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    if len(a) != len(b):
        return min(len(a), len(b))
    return None


def run_group(group: str, ids) -> dict:
    cfg = GROUPS[group]
    tmpl = VLLM_PAYLOAD if cfg["engine"] == "vllm" else TRT_PAYLOAD
    payload = (tmpl.replace("__LANE__", cfg["lane"])
               .replace("__IDS__", json.dumps(ids))
               .replace("__MODELPATH__", model_path_in_ctr())
               .replace("__MAXTOK__", str(MAX_TOKENS))
               .replace("__MAXLEN__", str(MAX_MODEL_LEN))
               .replace("__MAXBATCH__", str(MAX_BATCH))
               .replace("__MAXNUMTOK__", str(MAX_NUM_TOKENS))
               .replace("__MEMFRAC__", str(MEM_FRACTION)))
    image = VLLM_IMAGE if cfg["engine"] == "vllm" else TRT_IMAGE
    env = ["-e", "TQDM_DISABLE=1", "-e", "HF_HUB_OFFLINE=1",
           "-e", "HF_HOME=/root/.cache/huggingface"]
    if cfg["engine"] == "vllm":
        env += ["-e", "VLLM_USE_FLASHINFER_SAMPLER=0", "-e", "VLLM_ENABLE_V1_MULTIPROCESSING=0"]
    cmd = [DOCKER, "run", "--rm", "--gpus", "all", "--ipc", "host",
           "--ulimit", "memlock=-1", "--ulimit", "stack=67108864", *env,
           "-v", f"{HF_CACHE}:/root/.cache/huggingface", image, "python3", "-c", payload]
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = round(time.time() - t0, 1)
    log = os.path.join(LOGD, f"consistency_{group}.log")
    open(log, "w").write(proc.stdout or "")
    rec, dec = None, json.JSONDecoder()
    for line in (proc.stdout or "").splitlines():
        i = line.find("M23_CONS ")
        if i >= 0:
            try:
                rec, _ = dec.raw_decode(line[i + len("M23_CONS "):])
            except ValueError:
                pass
    return {"group": group, "image": image, "returncode": proc.returncode,
            "wall_elapsed_s": elapsed, "record": rec, "log": log}


def compare_pair(left_g, right_g, left_rec, right_rec, prompts) -> dict:
    lg = left_rec["generations"]
    rg = right_rec["generations"]
    rows = []
    exact = 0
    for i, (a, b) in enumerate(zip(lg, rg)):
        match = a["token_ids"] == b["token_ids"]
        if match:
            exact += 1
        cpl = common_prefix_len(a["token_ids"], b["token_ids"])
        fdp = first_divergence_pos(a["token_ids"], b["token_ids"])
        shorter_is_prefix = (not match) and cpl == min(len(a["token_ids"]), len(b["token_ids"]))
        eos_early = []
        if a["n_tokens"] < MAX_TOKENS:
            eos_early.append(left_g)
        if b["n_tokens"] < MAX_TOKENS:
            eos_early.append(right_g)
        rows.append({
            "prompt_index": i,
            "prompt": prompts[i],
            "exact_token_match": match,
            "common_prefix_len": cpl,
            "first_divergence_pos": fdp,
            "shorter_is_prefix_of_longer": shorter_is_prefix,
            "eos_early_stop_groups": eos_early,
            f"{left_g}_n_tokens": a["n_tokens"],
            f"{right_g}_n_tokens": b["n_tokens"],
            f"{left_g}_text": a["text"],
            f"{right_g}_text": b["text"],
            f"{left_g}_first_div_token": None if fdp is None else (
                a["token_ids"][fdp] if fdp < len(a["token_ids"]) else None),
            f"{right_g}_first_div_token": None if fdp is None else (
                b["token_ids"][fdp] if fdp < len(b["token_ids"]) else None),
        })
    n = len(rows)
    return {
        "left": left_g, "right": right_g,
        "exact_match": {"matched": exact, "total": n,
                        "rate": round(exact / n, 4) if n else None},
        "common_prefix_len": {
            "per_prompt": [r["common_prefix_len"] for r in rows],
            "median": sorted(r["common_prefix_len"] for r in rows)[n // 2] if n else None,
            "min": min((r["common_prefix_len"] for r in rows), default=None),
        },
        "first_divergence_pos": [r["first_divergence_pos"] for r in rows],
        "output_len": {
            left_g: [r[f"{left_g}_n_tokens"] for r in rows],
            right_g: [r[f"{right_g}_n_tokens"] for r in rows],
            "all_equal_requested": all(
                r[f"{left_g}_n_tokens"] == MAX_TOKENS and r[f"{right_g}_n_tokens"] == MAX_TOKENS
                for r in rows),
        },
        "per_prompt": rows,
        "n_shorter_is_prefix_of_longer": sum(1 for r in rows if r["shorter_is_prefix_of_longer"]),
        "note": ("跨引擎 mismatch 只作为 numerical divergence 记录，"
                 "不得直接判为 correctness failure。"
                 if {left_g[0], right_g[0]} == {"V", "T"} else
                 "同引擎 execution-mode ablation（诊断附表，不承担跨引擎结论）。"),
    }


def functional_sanity(group: str, rec) -> tuple[list[str], list[dict]]:
    """硬失败 vs EOS early-stop 分列。后者不是 correctness failure。"""
    hard, early = [], []
    if rec is None:
        hard.append(f"{group}: 无结构化记录")
        return hard, early
    if "error" in rec:
        hard.append(f"{group}: {rec['error']}")
        return hard, early
    gens = rec.get("generations") or []
    if len(gens) != 6:
        hard.append(f"{group}: 生成条数 {len(gens)} != 6")
    for i, g in enumerate(gens):
        n = g.get("n_tokens")
        if not (g.get("text") or "").strip():
            hard.append(f"{group} prompt{i}: 空输出")
        if n is None or n <= 0:
            hard.append(f"{group} prompt{i}: n_tokens={n}")
        elif n > MAX_TOKENS:
            hard.append(f"{group} prompt{i}: 超长 {n} > {MAX_TOKENS}")
        elif n < MAX_TOKENS:
            early.append({"group": group, "prompt_index": i, "n_tokens": n,
                          "note": "短于 max_tokens：TRT 无 ignore_eos，遇 EOS 可提前停；"
                                  "不作为 functional sanity 硬失败"})
    return hard, early


def summarize(runs, prompts, gpu) -> dict:
    hard, early = [], []
    for g, r in runs.items():
        if r.get("returncode") not in (0, None) and r.get("returncode") != 0:
            hard.append(f"{g}: 容器 rc={r['returncode']}")
        h, e = functional_sanity(g, r.get("record"))
        hard.extend(h)
        early.extend(e)

    recs = {g: r["record"] for g, r in runs.items()
            if r.get("record") and "error" not in r["record"] and r["record"].get("generations")}

    cross, same = {}, {}
    for left_g, right_g in CROSS_ENGINE_PAIRS:
        if left_g in recs and right_g in recs:
            cross[f"{right_g} vs {left_g}"] = compare_pair(
                left_g, right_g, recs[left_g], recs[right_g], prompts)
    for left_g, right_g in SAME_ENGINE_PAIRS:
        if left_g in recs and right_g in recs:
            same[f"{right_g} vs {left_g}"] = compare_pair(
                left_g, right_g, recs[left_g], recs[right_g], prompts)

    return {
        "generated_at": time.strftime("%F %T%z"),
        "inputs_file": INPUTS,
        "n_prompts": 6,
        "max_tokens": MAX_TOKENS,
        "gpu_before": gpu,
        "scope": ("Qwen2.5-0.5B-Instruct / RTX 5080 sm_120 / WSL2 / greedy / "
                  "max_tokens=64 / 6 fixed token-id prompts / containers"),
        "discipline": {
            "cross_engine_allowed": ["T-E vs V-E", "T-O vs V-O"],
            "same_engine_ablation_allowed": ["V-O vs V-E", "T-O vs T-E"],
            "forbidden_for_engine_speedup": ["T-E vs V-O", "T-O vs V-E"],
            "mismatch_is": "numerical divergence, NOT correctness failure",
            "not_a_quality_benchmark": True,
        },
        "groups": {g: {
            "returncode": r.get("returncode"), "wall_elapsed_s": r.get("wall_elapsed_s"),
            "engine_version": (r.get("record") or {}).get("engine_version"),
            "effective": (r.get("record") or {}).get("effective"),
            "build_s": (r.get("record") or {}).get("build_s"),
            "log": r.get("log"),
        } for g, r in runs.items()},
        "functional_sanity": {
            "hard_fail_reasons": hard,
            "eos_early_stop": early,
            "PASS": not hard,
            "note": ("硬失败 = 崩溃/空输出/超长。"
                     "TRT 遇 EOS 提前停记入 eos_early_stop，不是 correctness failure。"
                     "exact-match 不在硬失败列。"),
        },
        "cross_engine": cross,
        "same_engine_ablation": same,
        "verdict": (
            "functional sanity PASS；跨引擎 mismatch 记为 numerical divergence，"
            "不升级为 correctness failure。"
            if not hard else
            "**硬失败 → functional sanity 未过**；保存证据并回报，不把 mismatch 当根因。"
        ),
    }


def load_raw_runs():
    runs = {}
    for g in GROUPS:
        p = os.path.join(RAW, f"consistency_{g}.json")
        if os.path.exists(p):
            runs[g] = json.load(open(p))
    return runs


def emit_report(report) -> int:
    path = os.path.join(OUT, "cross_engine_consistency.json")
    json.dump(report, open(path, "w"), ensure_ascii=False, indent=2)
    hard = report["functional_sanity"]["hard_fail_reasons"]
    print(f"\n[m23] functional sanity PASS={report['functional_sanity']['PASS']}")
    for name, cmp_ in {**report["cross_engine"], **report["same_engine_ablation"]}.items():
        em = cmp_["exact_match"]
        print(f"[m23] {name}: exact {em['matched']}/{em['total']} = {em['rate']}  "
              f"prefix median={cmp_['common_prefix_len']['median']} "
              f"min={cmp_['common_prefix_len']['min']}  "
              f"div={cmp_['first_divergence_pos']}", flush=True)
    for e in report["functional_sanity"].get("eos_early_stop") or []:
        print(f"[m23] eos_early_stop: {e['group']} prompt{e['prompt_index']} "
              f"n_tokens={e['n_tokens']}  ({e['note']})")
    for h in hard:
        print("[m23] ⚠️ 硬失败:", h)
    print("[m23] ->", path)
    return 0 if not hard else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", choices=list(GROUPS))
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--summarize-only", action="store_true",
                    help="不占 GPU：从 outputs/m23/raw/consistency_*.json 重汇总")
    args = ap.parse_args()

    blob = json.load(open(INPUTS))
    ids, prompts = blob["token_ids"], blob["prompts"]
    assert len(ids) == 6 and len(prompts) == 6

    if args.summarize_only:
        runs = load_raw_runs()
        missing = [g for g in GROUPS if g not in runs]
        if missing:
            raise SystemExit(f"[m23] --summarize-only 缺 raw：{missing}")
        return emit_report(summarize(runs, prompts, gpu=None))

    gpu = gpu_snapshot()
    print(f"[m23] GPU: {gpu}", flush=True)
    if "error" in gpu:
        raise SystemExit("[m23] 读不到 nvidia-smi，拒绝启动")
    if gpu["mem_used_mib"] > GPU_BUSY_MIB and not args.allow_busy_gpu:
        raise SystemExit(
            f"[m23] GPU 已占用 {gpu['mem_used_mib']} MiB（阈值 {GPU_BUSY_MIB}）——"
            "可能有他人任务在跑，拒绝启动。")

    os.makedirs(RAW, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)

    groups = [args.group] if args.group else list(GROUPS)
    runs = {}
    for g in groups:
        print(f"[m23] === {g} ===", flush=True)
        r = run_group(g, ids)
        json.dump(r, open(os.path.join(RAW, f"consistency_{g}.json"), "w"),
                  ensure_ascii=False, indent=2)
        rec = r["record"]
        ok = rec is not None and "error" not in (rec or {})
        print(f"[m23] {g}: rc={r['returncode']} wall={r['wall_elapsed_s']}s ok={ok}",
              flush=True)
        if not ok:
            print(f"       ⚠️ {(rec or {}).get('error', 'no record')}  log={r['log']}",
                  flush=True)
        runs[g] = r

    if args.group:
        print("[m23] 单组模式：不写汇总。四组齐后再无 --group 跑一次做比对。")
        rec = runs[args.group]["record"]
        return 0 if rec and "error" not in rec else 1

    for g in GROUPS:
        if g not in runs:
            p = os.path.join(RAW, f"consistency_{g}.json")
            if os.path.exists(p):
                runs[g] = json.load(open(p))

    return emit_report(summarize(runs, prompts, gpu))


if __name__ == "__main__":
    raise SystemExit(main())
