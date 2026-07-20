"""M6 · 离线/闭环引擎 Benchmark —— 单配置 driver（一次只建一种 (mode, prefix) 引擎）。

用法（每种配置独立进程，见 scripts/m6_run_all.py）：
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python scripts/m6_benchmark.py --mode eager --prefix off \
    --experiments mem,prefill,decode --out-dir outputs/m6 [--quick]

设计要点（详见 notes/m6/benchmark.md 与批准的方案）：
- 进程内 LLMEngine（enable_multiprocessing=False）+ 手动 step() 循环；async_scheduling=False。
- 计时：torch.cuda.synchronize() 夹 perf_counter()，覆盖整个 engine step（进程内闭环指标，非在线 TTFT）。
- prompt 用精确 token 数（绕 chat template，直接 prompt_token_ids），断言 actual==target。
- 只读包裹 scheduler.schedule 存最近 SchedulerOutput 引用（不序列化/不打印/不写文件）；
  校验信息在计时区间之外读取。
- 正式计时区间：无 trace、无文件写、无打印。
- runtime_mode 在“诊断相”（warmup 期临时开 exec trace 到临时 jsonl）确认一次，不进入计时区。
- 每个样本落原始 JSON（含 warmup/measure 标记与校验字段）；统计交给 m6_summarize.py。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

import vllm
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext

# ---- 稳定长文本，用于构造精确 token 序列（tokenizer 编码后重复到足够长） ----
_LONG = (
    "大模型推理会把已经算过的 Key 和 Value 缓存起来，避免每一步都重算历史，"
    "从而让自回归解码更快。Prefill 阶段一次性处理整段提示词，Decode 阶段每步只处理一个新 token。"
)


def env_fingerprint() -> dict:
    return {
        "sys_executable": sys.executable,
        "vllm_file": os.path.realpath(vllm.__file__),
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "env": {
            k: os.environ.get(k)
            for k in (
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_USE_FLASHINFER_SAMPLER",
                "VLLM_LAB_TRACE",
            )
        },
    }


def mem_snapshot() -> dict:
    """设备级 + 进程级显存（MiB）。实测来源。"""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "device_total_MiB": round(total / 2**20, 1),
        "device_free_MiB": round(free / 2**20, 1),
        "device_used_MiB": round((total - free) / 2**20, 1),
        "torch_reserved_MiB": round(torch.cuda.memory_reserved() / 2**20, 1),
        "torch_allocated_MiB": round(torch.cuda.memory_allocated() / 2**20, 1),
    }


# ============================ 引擎构建 ============================
def build_engine(mode: str, prefix: str, args) -> LLMEngine:
    enforce_eager = mode == "eager"
    enable_prefix_caching = prefix == "on"
    engine_args = EngineArgs(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=enforce_eager,
        seed=args.seed,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        async_scheduling=False,
        disable_log_stats=True,
    )
    engine = LLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.ENGINE_CONTEXT,
        enable_multiprocessing=False,
    )
    return engine


def get_scheduler(engine: LLMEngine):
    return engine.engine_core.engine_core.scheduler


class ScheduleSpy:
    """只读包裹 scheduler.schedule：仅存最近一次 SchedulerOutput 引用。

    不序列化、不打印、不写文件；读取在计时区间之外。可 restore() 还原。
    """

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self._orig = scheduler.schedule
        self.last = None

        def wrapped(*a, **k):
            out = self._orig(*a, **k)
            self.last = out
            return out

        scheduler.schedule = wrapped

    def restore(self):
        self.scheduler.schedule = self._orig


# ============================ 精确 token 构造 ============================
class TokenFactory:
    def __init__(self, engine: LLMEngine):
        # 优先用引擎内的 tokenizer；退回 transformers。
        tok = getattr(engine, "tokenizer", None)
        if tok is not None and hasattr(tok, "encode"):
            self.tok = tok
        else:
            from transformers import AutoTokenizer

            self.tok = AutoTokenizer.from_pretrained(engine.model_config.model)
        base = self.tok.encode(_LONG, add_special_tokens=False)
        # 重复到 ~4000 token 备用
        self.pool = (base * (4000 // max(1, len(base)) + 2))[:4200]

    def ids(self, n: int, uniq: str = "") -> list[int]:
        """返回恰好 n 个 token id；uniq 非空则前置一段独立前缀（打破 prefix 共享）。"""
        if uniq:
            pre = self.tok.encode(uniq, add_special_tokens=False)
            body = (pre + self.pool)[:n]
        else:
            body = self.pool[:n]
        assert len(body) == n, f"token 构造长度不符: {len(body)} != {n}"
        return list(body)


# ============================ 计时原语 ============================
def timed_step(engine: LLMEngine):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = engine.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0), outs


def drain(engine: LLMEngine):
    while engine.has_unfinished_requests():
        engine.step()


# ============================ 诊断相：确认 runtime_mode ============================
def diagnose_runtime_mode(engine: LLMEngine, tf: "TokenFactory", tmp_jsonl: str) -> dict:
    """临时开 exec trace 跑若干 throwaway step，读回 prefill/decode 的 cudagraph_mode。

    只在 warmup/诊断阶段用，不进入任何计时样本。
    """
    result = {"prefill": None, "decode": None}
    if os.path.exists(tmp_jsonl):
        os.remove(tmp_jsonl)
    saved = {k: os.environ.get(k) for k in
             ("VLLM_LAB_TRACE", "VLLM_LAB_TRACE_COMPONENTS", "VLLM_LAB_TRACE_JSONL", "VLLM_LAB_TRACE_MAX_STEPS")}
    os.environ["VLLM_LAB_TRACE"] = "1"
    os.environ["VLLM_LAB_TRACE_COMPONENTS"] = "exec"
    os.environ["VLLM_LAB_TRACE_JSONL"] = tmp_jsonl
    os.environ["VLLM_LAB_TRACE_MAX_STEPS"] = "50"
    try:
        from vllm import _lab_trace as _lt
        _lt.bump_step()  # 让 step 计数进入被追踪窗口
        rid = "diag"
        engine.add_request(rid, {"prompt_token_ids": tf.ids(32)},
                           SamplingParams(temperature=0.0, max_tokens=6, ignore_eos=True, seed=0))
        for _ in range(8):
            _lt.bump_step()
            if not engine.has_unfinished_requests():
                break
            engine.step()
        engine.abort_request([rid])
        drain(engine)
    except Exception as e:  # noqa
        result["error"] = repr(e)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    # 解析
    try:
        rows = [json.loads(l) for l in open(tmp_jsonl, encoding="utf-8")] if os.path.exists(tmp_jsonl) else []
        for r in rows:
            if r.get("event") != "dispatch":
                continue
            f = r["fields"]
            if f.get("num_tokens_unpadded", 0) > 1 and result["prefill"] is None:
                result["prefill"] = f.get("cudagraph_mode")
            # decode：优先 uniform_decode（compiled 下为 True），否则用单 token 步兜底（eager）
            is_decode = f.get("uniform_decode") or f.get("num_tokens_unpadded", 0) == 1
            if is_decode and result["decode"] is None:
                result["decode"] = f.get("cudagraph_mode")
    except Exception as e:  # noqa
        result["parse_error"] = repr(e)
    return result


# ============================ E-mem/cap ============================
def dump_mem_cap(engine: LLMEngine, mode: str, prefix: str) -> dict:
    cc = engine.vllm_config.cache_config
    comp = engine.vllm_config.compilation_config
    sched = get_scheduler(engine)
    kvcfg = getattr(sched, "kv_cache_config", None)

    # 配置直读
    conf = {
        "num_gpu_blocks": cc.num_gpu_blocks,
        "block_size": cc.block_size,
        "kv_cache_size_tokens": cc.kv_cache_size_tokens,
        "kv_cache_max_concurrency": cc.kv_cache_max_concurrency,
        "max_model_len": engine.model_config.max_model_len,
        "max_num_seqs": engine.vllm_config.scheduler_config.max_num_seqs,
        "gpu_memory_utilization": cc.gpu_memory_utilization,
        "_source": "config-direct",
    }

    # KV 每 token 字节（配置推导）
    kv_derived = {"_source": "config-derived"}
    try:
        page_bytes = sum(g.kv_cache_spec.page_size_bytes for g in kvcfg.kv_cache_groups)
        kv_total = cc.num_gpu_blocks * page_bytes
        kv_derived.update({
            "page_size_bytes_sum_over_groups": page_bytes,
            "kv_total_bytes": kv_total,
            "kv_total_GiB": round(kv_total / 2**30, 3),
            "bytes_per_cached_token": round(kv_total / cc.kv_cache_size_tokens, 1)
            if cc.kv_cache_size_tokens else None,
        })
    except Exception as e:  # noqa
        kv_derived["error"] = repr(e)

    # KV 每 token 字节（理论交叉校验）
    kv_theory = {"_source": "theory"}
    try:
        hf = getattr(engine.model_config, "hf_text_config", None) or engine.model_config.hf_config
        L = getattr(hf, "num_hidden_layers", None)
        kvh = getattr(hf, "num_key_value_heads", None) or getattr(hf, "num_attention_heads", None)
        hd = getattr(hf, "head_dim", None)
        if hd is None and getattr(hf, "hidden_size", None) and getattr(hf, "num_attention_heads", None):
            hd = hf.hidden_size // hf.num_attention_heads
        dtype_bytes = 2  # bf16
        kv_theory.update({
            "num_hidden_layers": L, "num_key_value_heads": kvh, "head_dim": hd,
            "dtype_bytes": dtype_bytes,
            "bytes_per_token_theory": (2 * L * kvh * hd * dtype_bytes) if all(x for x in (L, kvh, hd)) else None,
        })
    except Exception as e:  # noqa
        kv_theory["error"] = repr(e)

    # KV tensor 实测（若可达）
    kv_measured = {"_source": "kv-tensor-sum"}
    try:
        def _kv_bytes(worker):
            mr = worker.model_runner
            return int(sum(t.numel() * t.element_size() for t in mr.kv_caches if t is not None))
        vals = engine.model_executor.collective_rpc(_kv_bytes)
        kv_measured["kv_tensor_bytes"] = vals[0] if isinstance(vals, list) else vals
        if cc.kv_cache_size_tokens:
            kv_measured["bytes_per_cached_token"] = round(kv_measured["kv_tensor_bytes"] / cc.kv_cache_size_tokens, 1)
    except Exception as e:  # noqa
        kv_measured["error"] = repr(e)

    # 编译配置（M5 口径）
    def nm(v):
        return getattr(v, "name", None) or (str(v) if v is not None else None)
    compilation = {
        "mode": nm(getattr(comp, "mode", None)),
        "cudagraph_mode": nm(getattr(comp, "cudagraph_mode", None)),
        "backend": getattr(comp, "backend", None),
        "cudagraph_capture_sizes_count": len(getattr(comp, "cudagraph_capture_sizes", []) or []),
        "max_cudagraph_capture_size": getattr(comp, "max_cudagraph_capture_size", None),
    }

    return {
        "mode": mode, "prefix": prefix,
        "config": conf,
        "kv_per_token_config_derived": kv_derived,
        "kv_per_token_theory": kv_theory,
        "kv_tensor_measured": kv_measured,
        "compilation": compilation,
        "mem_snapshot_current": mem_snapshot(),
        "_note": "weights/cudagraph 显存拆分见运行日志(Model loading took / Graph capturing finished)；MemorySnapshot 只分 torch/non-torch",
    }


# ============================ E-prefill ============================
def measure_prefill(engine, spy, tf, mode, prefix, lengths, warmup, repeats, quick):
    samples = []
    for n in lengths:
        ids = tf.ids(n)
        total = warmup + repeats
        for r in range(total):
            phase = "warmup" if r < warmup else "measure"
            rid = f"pf-{n}-{r}"
            engine.add_request(rid, {"prompt_token_ids": ids},
                               SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=False, seed=0))
            dt, outs = timed_step(engine)
            so = spy.last
            scheduled = int(so.total_num_scheduled_tokens) if so is not None else -1
            actual = len(outs[0].prompt_token_ids) if outs else n
            finished = sum(1 for o in outs if getattr(o, "finished", False))
            drain(engine)  # max_tokens=1 通常本步即完成；兜底清理
            chunked = scheduled != n
            valid = (phase == "measure") and (not chunked) and (actual == n)
            samples.append({
                "experiment": "prefill", "mode": mode, "prefix_caching": prefix,
                "repeat": r, "phase": phase, "batch_size": 1,
                "target_prompt_tokens": n, "actual_prompt_tokens": actual, "seq_len": actual,
                "scheduled_tokens": scheduled, "running_requests": 1,
                "latency_ms": round(dt * 1000, 4),
                "throughput_tokens_per_s": None,
                "finished_requests": finished, "preempted": False, "chunked": chunked,
                "valid_sample": valid,
                "invalid_reason": "" if valid else (
                    "warmup" if phase == "warmup" else ("chunked" if chunked else "token_mismatch")),
            })
        if quick:
            break
    return samples


# ============================ E-decode（方案 A：独立重复） ============================
def measure_decode(engine, spy, tf, mode, prefix, batches, prompt_len, warmup_steps,
                   repeats, window, quick):
    samples = []
    ids = tf.ids(prompt_len)
    sched = get_scheduler(engine)
    for B in batches:
        for r in range(repeats):
            rids = [f"dec-{B}-{r}-{b}" for b in range(B)]
            for rid in rids:
                engine.add_request(rid, {"prompt_token_ids": ids},
                                   SamplingParams(temperature=0.0, max_tokens=100000,
                                                  ignore_eos=True, seed=0))
            # 先把所有请求推进到全部处于 decode（每步 scheduled 应等于 running 数且各 1）
            guard = 0
            while guard < 20:
                so = spy.last
                running = len(getattr(sched, "running", []))
                # 判断是否已进入“全 decode”：running==B 且上一步 total==B
                if so is not None and running == B and int(so.total_num_scheduled_tokens) == B:
                    break
                engine.step()
                guard += 1
            # decode warmup
            for _ in range(warmup_steps):
                engine.step()
            # 测量窗口（固定位置，seq_len 每个 repeat 一致）
            for w in range(window):
                dt, outs = timed_step(engine)
                so = spy.last
                total_sched = int(so.total_num_scheduled_tokens) if so is not None else -1
                running = len(getattr(sched, "running", []))
                # seq_len：取任一 running 请求的已计算 token 数（近似 = prompt_len + 已产出）
                seq_len = None
                try:
                    for req in sched.running:
                        seq_len = int(getattr(req, "num_computed_tokens", 0))
                        break
                except Exception:  # noqa
                    pass
                preempted = False
                try:
                    preempted = any(getattr(sched.requests.get(x), "status", None) is not None
                                    and str(getattr(sched.requests.get(x), "status")).endswith("PREEMPTED")
                                    for x in rids)
                except Exception:  # noqa
                    pass
                valid = (total_sched == B) and (running == B) and (not preempted)
                thru = round(B / dt, 2) if dt > 0 else None
                samples.append({
                    "experiment": "decode", "mode": mode, "prefix_caching": prefix,
                    "repeat": r, "phase": "measure", "batch_size": B,
                    "target_prompt_tokens": prompt_len, "actual_prompt_tokens": prompt_len,
                    "seq_len": seq_len, "scheduled_tokens": total_sched, "running_requests": running,
                    "latency_ms": round(dt * 1000, 4),
                    "throughput_tokens_per_s": thru,
                    "finished_requests": 0, "preempted": preempted, "chunked": False,
                    "valid_sample": valid,
                    "invalid_reason": "" if valid else "batch_or_sched_mismatch",
                })
            engine.abort_request(rids)
            drain(engine)
        if quick:
            break
    return samples


# ============================ E-prefix（独立前缀族） ============================
def measure_prefix(engine, spy, tf, mode, prefix, body_len, repeats, quick):
    """在 prefix ON 引擎里：每 repeat 独立前缀族 Cold_i(全新,miss) -> Hot_i(同ids,命中)。

    prefix OFF 引擎里跑同 ids 作对照(Off)。命中证据靠 scheduled_tokens 减少(计时区外读)。
    """
    samples = []
    n = repeats if not quick else 1
    for i in range(n):
        ids = tf.ids(body_len, uniq=f"实验前缀编号{i}：")
        tag = "off" if prefix == "off" else None
        seq = [("cold", ids), ("hot", ids)] if prefix == "on" else [("off", ids)]
        for kind, kids in seq:
            rid = f"px-{kind}-{i}"
            engine.add_request(rid, {"prompt_token_ids": kids},
                               SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=False, seed=0))
            dt, outs = timed_step(engine)
            so = spy.last
            scheduled = int(so.total_num_scheduled_tokens) if so is not None else -1
            actual = len(outs[0].prompt_token_ids) if outs else body_len
            hit_tokens = actual - scheduled if scheduled >= 0 else None  # 命中则 scheduled < actual
            drain(engine)
            samples.append({
                "experiment": "prefix", "mode": mode, "prefix_caching": prefix,
                "repeat": i, "phase": "measure", "batch_size": 1, "prefix_kind": kind,
                "target_prompt_tokens": body_len, "actual_prompt_tokens": actual, "seq_len": actual,
                "scheduled_tokens": scheduled, "prefix_hit_tokens": hit_tokens,
                "running_requests": 1,
                "latency_ms": round(dt * 1000, 4), "throughput_tokens_per_s": None,
                "finished_requests": 1, "preempted": False, "chunked": scheduled > actual,
                "valid_sample": True, "invalid_reason": "",
            })
    return samples


def parse_args():
    p = argparse.ArgumentParser(description="vLLM M6 closed-loop engine benchmark (single config)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", choices=["eager", "compiled"], required=True)
    p.add_argument("--prefix", choices=["on", "off"], required=True)
    p.add_argument("--experiments", default="mem,prefill,decode")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--repeats", type=int, default=15)
    p.add_argument("--decode-warmup-steps", type=int, default=4)
    p.add_argument("--decode-window", type=int, default=2)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-dir", default="outputs/m6")
    return p.parse_args()


def main():
    args = parse_args()
    exps = [e.strip() for e in args.experiments.split(",") if e.strip()]
    tag = f"{args.mode}_{args.prefix}"

    prefill_lengths = [32, 128, 256, 512, 1024, 1536, 2000]
    decode_batches = [1, 2, 4, 8, 16, 32, 64]
    if args.quick:
        prefill_lengths = [32, 512]
        decode_batches = [1, 8]
        args.warmup, args.repeats = 2, 3
        args.decode_window = 1

    print("=" * 80)
    print(f"[m6] config={tag} vllm={os.path.realpath(vllm.__file__)}")
    print(f"[m6] experiments={exps} quick={args.quick}")
    print("=" * 80)

    t0 = time.perf_counter()
    engine = build_engine(args.mode, args.prefix, args)
    t_init = time.perf_counter() - t0
    spy = ScheduleSpy(get_scheduler(engine))

    tf = TokenFactory(engine)

    meta = {
        "config_tag": tag, "mode": args.mode, "prefix": args.prefix,
        "engine_init_s": round(t_init, 3),
        "env": env_fingerprint(),
        "args": vars(args),
    }

    # 诊断相：确认 runtime_mode（不进入计时样本）
    tmp_jsonl = os.path.join(args.out_dir, "config", f"_diag_{tag}.jsonl")
    meta["runtime_mode_diag"] = diagnose_runtime_mode(engine, tf, tmp_jsonl)
    print(f"[m6] runtime_mode diag: {meta['runtime_mode_diag']}")

    all_samples = []
    mem_cap = None
    if "mem" in exps:
        mem_cap = dump_mem_cap(engine, args.mode, args.prefix)
        print(f"[m6] mem/cap: num_gpu_blocks={mem_cap['config']['num_gpu_blocks']} "
              f"kv_size_tokens={mem_cap['config']['kv_cache_size_tokens']} "
              f"max_conc={mem_cap['config']['kv_cache_max_concurrency']}")
    if "prefill" in exps:
        all_samples += measure_prefill(engine, spy, tf, args.mode, args.prefix,
                                       prefill_lengths, args.warmup, args.repeats, args.quick)
        print(f"[m6] prefill done: {sum(1 for s in all_samples if s['experiment']=='prefill')} samples")
    if "decode" in exps:
        all_samples += measure_decode(engine, spy, tf, args.mode, args.prefix,
                                      decode_batches, 32, args.decode_warmup_steps,
                                      args.repeats, args.decode_window, args.quick)
        print(f"[m6] decode done: {sum(1 for s in all_samples if s['experiment']=='decode')} samples")
    if "prefix" in exps:
        all_samples += measure_prefix(engine, spy, tf, args.mode, args.prefix,
                                      256, args.repeats, args.quick)
        print(f"[m6] prefix done: {sum(1 for s in all_samples if s['experiment']=='prefix')} samples")

    spy.restore()

    # 落盘
    os.makedirs(os.path.join(args.out_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, "config"), exist_ok=True)
    raw_path = os.path.join(args.out_dir, "raw", f"{tag}{'_quick' if args.quick else ''}.jsonl")
    with open(raw_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    meta_path = os.path.join(args.out_dir, "config", f"{tag}{'_quick' if args.quick else ''}_meta.json")
    if mem_cap is not None:
        meta["mem_cap"] = mem_cap
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print(f"[m6] wrote {len(all_samples)} samples -> {raw_path}")
    print(f"[m6] wrote meta -> {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
