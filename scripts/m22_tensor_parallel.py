"""M22 · Tensor Parallel 对比实验（TP=1/2/4）。

============================================================================
**每个 TP 档一个独立进程**（沿用 M6/M17/M19 的纪律）。一个进程内只建一个引擎，
并在这一个引擎上依次跑 D1 / D2 / D3 —— 不为 D1 单独重复建引擎。

    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
    python scripts/m22_tensor_parallel.py --tp 1 --tag h20_formal
    python scripts/m22_tensor_parallel.py --tp 2 --tag h20_formal
    python scripts/m22_tensor_parallel.py --tp 4 --tag h20_formal
    python scripts/m22_tensor_parallel.py --tp 2 --tag h20_formal --shard profile
    python scripts/m22_tensor_parallel.py --shard summarize --tag h20_formal   # 不建引擎

设计 / 源码依据：notes/m22/tensor_parallel.md（R1–R7 逐条 file:line 核实）
============================================================================

以下每条设计都由 v0.24.0 的真实源码决定，不是习惯照抄：

1. **`VLLM_ENABLE_V1_MULTIPROCESSING=0` 是硬前置**
   `v1/engine/llm_engine.py:224-226` 会在该环境变量为真时**强制覆盖**传入的
   `enable_multiprocessing=False`。一旦被覆盖，EngineCore 会跑到独立进程里，
   `engine.engine_core.engine_core.scheduler` 就不存在了，逐 step 内省全部失效。
   本脚本启动时**硬断言**该变量。

2. **TP>1 时 scheduler 仍在本进程**（`core_client.py:304-315` InprocClient +
   `v1/engine/core.py:269-306`），所以 M6 的 `ScheduleSpy` 猴补丁依然可用 ——
   这是精确隔离 steady-state decode 的基础。

3. **driver 进程的显存读数无意义**：模型与 KV 全在 worker 进程
   （`multiproc_executor.py:625-634` 才 init_device）。故 per-rank 显存一律走
   `collective_rpc`（`v1/engine/llm_engine.py:492-499`），**不用 driver 的
   `torch.cuda.memory_allocated()` 冒充模型显存**。

4. **不在 driver 调 `torch.cuda.synchronize()`**：TP>1 时 driver 本来没有 CUDA context，
   调它反而会凭空建一个；而 `engine.step()` 本身是同步的（要把采样结果作为 Python 对象返回）。
   两档 TP 用同一套计时口径，避免 TP=1/TP>1 计时方式不对称。

5. **`async_scheduling` 三档一律显式 False**：它默认 `None`，而
   `config/vllm.py:938` 的自动决策依赖 `executor_class.supports_async_scheduling()`，
   executor 类在 TP=1(uni) 与 TP>1(mp) 下不同 → 不写死就会变成隐藏变量。

6. **`max_model_len` 必须 > 最长 prompt**：`v1/engine/input_processor.py:423-432`
   规定 `prompt_len == max_model_len` 时直接 ValueError（因为还要至少产出 1 个 token）。
   所以扫 4096 长 prompt 时 `max_model_len` 取 8192。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "m22")
RAW = os.path.join(OUT, "raw")

# ------------------------------ 冻结配置（跑前写死） ------------------------------
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEED = 0
# > 最长 prompt(4096)，否则 input_processor.py:423-432 直接 ValueError
MAX_MODEL_LEN = 8192
GPU_MEM_UTIL = 0.85
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 8192

# D2 · steady-state decode
D2_BATCHES = [1, 4, 16, 64]
D2_PROMPT_LEN = 128  # 短 prompt：让 prefill 尽快结束，把测量重心放在 decode
D2_OUTPUT_LEN = 64
D2_WARMUP = 5
D2_REPEATS = 15
# 进入稳态前丢弃的纯 decode step 数（排除 prefill 刚结束时的瞬态）
D2_STEADY_SKIP = 3

# D3 · prefill scaling
D3_PROMPT_LENS = [256, 1024, 4096]
D3_WARMUP = 5
D3_REPEATS = 15

# quick 模式（本地预演用，只为验证链路，不产出正式结论）
QUICK_BATCHES = [1, 4]
QUICK_PROMPT_LENS = [256, 1024]
QUICK_WARMUP = 1
QUICK_REPEATS = 3
QUICK_OUTPUT_LEN = 16

_LONG = (
    "推理引擎的连续批处理与分页注意力机制通过按块管理键值缓存来提升吞吐与显存利用率。"
    "The quick brown fox jumps over the lazy dog. "
)


def _h(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


# ============================ per-rank 观测（R2/R6） ============================
def _rank_report(worker):
    """在**每个 worker 进程内**执行，返回该 rank 的显存 / KV / 设备身份。

    经 `LLMEngine.collective_rpc(callable)` 下发（`v1/engine/llm_engine.py:492-499`），
    由 cloudpickle 序列化（`multiproc_executor.py:340-402`），worker 作为第一个参数传入
    （`v1/executor/abstract.py:152-184`）。**必须是模块级函数**，不能捕获不可序列化的状态。

    七类量刻意分开返回，语义**互不等同**（M6 曾把「KV 容量等价值」与实际并发混谈）：
      weight_bytes            模型权重
      torch_allocated/reserved torch allocator 视角
      non_torch_bytes         非 torch 占用（NCCL buffer / cuda context 等）
      available_kv_bytes      该 rank profile 出的可用 KV 字节
      num_gpu_blocks          实际分配的 KV block 数
    """
    import torch

    out = {"ok": True, "err": ""}
    try:
        out["rank"] = int(getattr(worker, "rank", -1))
        out["local_rank"] = int(getattr(worker, "local_rank", -1))
        dev = getattr(worker, "device", None)
        out["device"] = str(dev) if dev is not None else None
        out["device_name"] = (
            torch.cuda.get_device_name(dev) if dev is not None else None
        )
        out["torch_allocated_bytes"] = int(torch.cuda.memory_allocated(dev))
        out["torch_reserved_bytes"] = int(torch.cuda.memory_reserved(dev))
        free, total = torch.cuda.mem_get_info(dev)
        out["device_free_bytes"] = int(free)
        out["device_total_bytes"] = int(total)

        mr = getattr(worker, "model_runner", None)
        out["weight_bytes"] = (
            int(getattr(mr, "model_memory_usage", 0)) if mr is not None else None
        )
        out["available_kv_bytes"] = getattr(
            worker, "available_kv_cache_memory_bytes", None
        )
        if out["available_kv_bytes"] is not None:
            out["available_kv_bytes"] = int(out["available_kv_bytes"])
        for name in ("non_torch_memory", "peak_activation_memory"):
            v = getattr(worker, name, None)
            out[name + "_bytes"] = int(v) if v is not None else None
        cc = getattr(worker, "cache_config", None)
        out["num_gpu_blocks"] = (
            int(getattr(cc, "num_gpu_blocks", 0) or 0) if cc is not None else None
        )
        out["block_size"] = (
            int(getattr(cc, "block_size", 0) or 0) if cc is not None else None
        )
        try:
            from vllm.platforms import current_platform

            out["device_uuid"] = current_platform.get_device_uuid(out["local_rank"])
        except Exception as e:  # noqa: BLE001 —— UUID 拿不到不影响主结论
            out["device_uuid"] = None
            out["uuid_err"] = repr(e)
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["err"] = repr(e)
    return out


# ============================ 引擎 ============================
def assert_env():
    """硬前置：不满足就**立刻失败**，不要跑完才发现内省全是空的。"""
    v = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if v not in ("0", "false", "False"):
        raise SystemExit(
            "[m22][FATAL] 必须 export VLLM_ENABLE_V1_MULTIPROCESSING=0。\n"
            "  依据 vllm/v1/engine/llm_engine.py:224-226：该变量为真时会**强制覆盖**\n"
            "  enable_multiprocessing=False，导致 EngineCore 跑到独立进程，\n"
            "  engine.engine_core.engine_core.scheduler 不复存在，逐 step 内省全部失效。\n"
            f"  当前值: {v!r}"
        )


def build_engine(tp: int, model: str, profile_dir: str | None = None):
    from vllm import LLMEngine
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    kwargs = {
        "model": model,
        "dtype": "auto",
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "tensor_parallel_size": tp,
        "enforce_eager": True,
        "seed": SEED,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "enable_prefix_caching": False,
        # 三档 TP 必须一致：默认 None 会因 executor 类不同而自动取不同值
        # （config/vllm.py:938 executor_class.supports_async_scheduling()）
        "async_scheduling": False,
        # 本轮不依赖任何 stats 观测（与 M19 不同），关掉以免引入无关开销；三档一致
        "disable_log_stats": True,
    }
    if profile_dir:
        # v0.24.0 没有 VLLM_TORCH_PROFILER_DIR，必须走 ProfilerConfig
        # （config/profiler.py:34-67，经 engine/arg_utils.py:662 传入）
        from vllm.config.profiler import ProfilerConfig

        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=os.path.abspath(profile_dir),
            torch_profiler_with_stack=False,  # trace 体积可控
            torch_profiler_record_shapes=True,  # 需要 shape 才能解释 all-reduce payload
        )
    args = EngineArgs(**kwargs)
    return LLMEngine.from_engine_args(
        args, usage_context=UsageContext.ENGINE_CONTEXT, enable_multiprocessing=False
    )


def effective_cfg(engine, tp_requested: int) -> dict:
    """从**引擎**读回真正生效的配置——三层证据里的第二层。"""
    vc = engine.vllm_config
    pc, sc, cc, mc = (
        vc.parallel_config,
        vc.scheduler_config,
        vc.cache_config,
        vc.model_config,
    )
    return {
        "tp_requested": tp_requested,
        "tensor_parallel_size": int(pc.tensor_parallel_size),
        "pipeline_parallel_size": int(pc.pipeline_parallel_size),
        "world_size": int(getattr(pc, "world_size", pc.tensor_parallel_size)),
        # TP=1→uni / TP>1→mp，属 TP 的固有伴随变化，无法消除，如实记录
        "distributed_executor_backend": str(pc.distributed_executor_backend),
        "disable_custom_all_reduce": bool(pc.disable_custom_all_reduce),
        "async_scheduling": bool(sc.async_scheduling),
        "enable_chunked_prefill": bool(sc.enable_chunked_prefill),
        "max_num_batched_tokens": int(sc.max_num_batched_tokens),
        "max_num_seqs": int(sc.max_num_seqs),
        "enable_prefix_caching": bool(cc.enable_prefix_caching),
        "block_size": int(cc.block_size),
        "enforce_eager": bool(mc.enforce_eager),
        "max_model_len": int(mc.max_model_len),
        "dtype": str(mc.dtype),
        "model": str(mc.model),
        # ↓ 调度器视角的统一容量（min over ranks 之后），**不是**各 rank 之和，
        #   也**不是**运行时并发（kv_cache_utils.py:2135-2163 / :1800-1810）
        "sched_num_gpu_blocks": int(getattr(cc, "num_gpu_blocks", 0) or 0),
        "sched_kv_cache_size_tokens": getattr(cc, "kv_cache_size_tokens", None),
        "sched_kv_capacity_equiv_concurrency": getattr(
            cc, "kv_cache_max_concurrency", None
        ),
        # 模型结构（R7：整除性与 KV 复制的解释依据）
        "total_num_attention_heads": mc.model_arch_config.total_num_attention_heads,
        "total_num_kv_heads": mc.get_total_num_kv_heads(),
        "num_attention_heads_per_rank": mc.get_num_attention_heads(pc),
        "num_kv_heads_per_rank": mc.get_num_kv_heads(pc),
        "num_hidden_layers": getattr(mc.model_arch_config, "num_hidden_layers", None),
    }


def get_scheduler(engine):
    """R1：InprocClient 下 scheduler 仍在本进程；拿不到就退化（不中断实验）。"""
    try:
        return engine.engine_core.engine_core.scheduler
    except AttributeError:
        return None


class ScheduleSpy:
    """只读包裹 scheduler.schedule，仅存最近一次 SchedulerOutput。TP>1 下同样有效（R1）。"""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.last = None
        self.ok = scheduler is not None
        if self.ok:
            self._orig = scheduler.schedule

            def wrapped(*a, **k):
                out = self._orig(*a, **k)
                self.last = out
                return out

            scheduler.schedule = wrapped

    def restore(self):
        if self.ok:
            self.scheduler.schedule = self._orig


class TokenFactory:
    def __init__(self, engine, model: str):
        tok = getattr(engine, "tokenizer", None)
        if tok is None or not hasattr(tok, "encode"):
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(model)
        self.tok = tok
        base = self.tok.encode(_LONG, add_special_tokens=False)
        need = MAX_MODEL_LEN + 256
        self.pool = (base * (need // max(1, len(base)) + 2))[:need]

    def ids(self, n: int, uniq: str = "") -> list[int]:
        if uniq:
            pre = self.tok.encode(uniq, add_special_tokens=False)
            body = (pre + self.pool)[:n]
        else:
            body = self.pool[:n]
        assert len(body) == n, f"prompt 长度不符: {len(body)} != {n}"
        return list(body)


def _same_req(observed: str, rid: str) -> bool:
    return observed == rid or observed.startswith(rid + "-")


def _sched_map(so):
    """该 step 每个请求被调度的 token 数；拿不到返回 None（观测缺口，如实标记）。"""
    if so is None:
        return None
    d = getattr(so, "num_scheduled_tokens", None)
    return dict(d) if d else {}


# ============================ D1 · 显存 / KV 快照 ============================
def run_d1(engine, tp: int, eff: dict) -> dict:
    """引擎建好后直接采一次，不单独建引擎。"""
    ranks, method, err = None, None, ""
    try:
        ranks = engine.collective_rpc(_rank_report)
        method = "collective_rpc(callable)"
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        try:
            # fallback：只拿 device 身份，精度更低，明确标注
            ranks = engine.collective_rpc("get_kv_cache_spec")
            method = "collective_rpc('get_kv_cache_spec') [degraded]"
        except Exception as e2:  # noqa: BLE001
            err += " | " + repr(e2)
            method = "unavailable"

    def _tot(key):
        if not ranks or not isinstance(ranks[0], dict):
            return None
        vals = [r.get(key) for r in ranks if isinstance(r, dict)]
        return sum(v for v in vals if isinstance(v, int)) if vals else None

    return {
        "experiment": "D1_memory_kv",
        "tp": tp,
        "observation_method": method,
        "observation_error": err,
        "per_rank": ranks,
        "n_ranks_reported": len(ranks) if ranks else 0,
        "sum_weight_bytes": _tot("weight_bytes"),
        "sum_available_kv_bytes": _tot("available_kv_bytes"),
        "sum_torch_allocated_bytes": _tot("torch_allocated_bytes"),
        "effective_cfg": eff,
        "success": bool(ranks) and method.startswith("collective_rpc(callable)"),
        "invalid_reason": "" if ranks else f"per-rank 观测失败: {err}",
    }


# ============================ D2 · steady-state decode ============================
def run_d2_sample(engine, spy, tf, tp, batch, out_len, prompt_len):
    """一个样本 = batch 个**完全同构**的请求跑到全部结束。

    同构（相同 prompt 长度 + 相同 max_tokens + ignore_eos）是刻意的：
    这样 batch 内所有请求**同步推进、同一步结束**，从构造上消除
    「请求陆续完成导致 batch 缩水」的 tail effect —— 否则测到的
    「TP decode scaling」里会混进 batch 尾部收缩的效应。
    构造是否真的成立，由下面的 `steady_batch_stable` 事后核对，不靠假设。
    """
    from vllm import SamplingParams

    rids = [f"m22-tp{tp}-b{batch}-r{i}" for i in range(batch)]
    prompts = {r: tf.ids(prompt_len, uniq=f"d2_{i}：") for i, r in enumerate(rids)}
    for rid in rids:
        engine.add_request(
            rid,
            {"prompt_token_ids": prompts[rid]},
            SamplingParams(
                temperature=0.0, max_tokens=out_len, ignore_eos=True, seed=SEED
            ),
        )

    steps = []  # (wall_ms, n_reqs_scheduled, total_tokens, is_pure_decode)
    outputs: dict[str, dict] = {}
    first_token_s = None
    # 不在 driver 调 torch.cuda.synchronize()：TP>1 时 driver 无 CUDA context，
    # 且 engine.step() 本身同步（要把采样结果作为 Python 对象返回）
    t0 = time.perf_counter()
    n_step = 0
    while engine.has_unfinished_requests() and n_step < 100000:
        ts = time.perf_counter()
        outs = engine.step()
        te = time.perf_counter()
        sm = _sched_map(spy.last)
        if sm is None:
            n_req, tot, pure = None, None, None
        else:
            per = [v for k, v in sm.items() if any(_same_req(k, r) for r in rids)]
            n_req, tot = len(per), sum(per)
            # 纯 decode step：本批 batch 个请求全部在场，且每个恰好 1 个 token
            pure = (n_req == batch) and all(v == 1 for v in per)
        steps.append(
            {
                "wall_ms": round((te - ts) * 1000, 4),
                "n_reqs": n_req,
                "sched_tokens": tot,
                "pure_decode": pure,
            }
        )
        for o in outs:
            for rid in rids:
                if _same_req(o.request_id, rid) and o.outputs:
                    outputs[rid] = {
                        "n": len(o.outputs[0].token_ids),
                        "hash": _h(list(o.outputs[0].token_ids)),
                        "finished": bool(o.finished),
                    }
        if first_token_s is None and outputs:
            first_token_s = te - t0
        n_step += 1
    completion_s = time.perf_counter() - t0

    pure_idx = [i for i, s in enumerate(steps) if s["pure_decode"]]
    steady = [steps[i]["wall_ms"] for i in pure_idx[D2_STEADY_SKIP:]]
    out_lens = [outputs.get(r, {}).get("n", 0) for r in rids]
    total_out = sum(out_lens)
    ft_ms = None if first_token_s is None else round(first_token_s * 1000, 4)
    comp_ms = round(completion_s * 1000, 4)
    ok = (
        all(n == out_len for n in out_lens)
        and all(outputs.get(r, {}).get("finished") for r in rids)
        and len(steady) >= 5
    )
    reasons = []
    if any(n != out_len for n in out_lens):
        reasons.append(f"输出长度 {out_lens} != {out_len}")
    if not all(outputs.get(r, {}).get("finished") for r in rids):
        reasons.append("存在未正常结束的请求")
    if len(steady) < 5:
        reasons.append(f"稳态 decode step 仅 {len(steady)} 个（<5），无法给稳定中位数")
    return {
        "experiment": "D2_decode",
        "tp": tp,
        "batch_size": batch,
        "prompt_len": prompt_len,
        "max_tokens_requested": out_len,
        "engine_steps": n_step,
        "n_pure_decode_steps": len(pure_idx),
        "n_steady_steps": len(steady),
        # ★ 主指标：稳态单步 decode wall latency
        "steady_step_ms_median": round(statistics.median(steady), 4)
        if steady
        else None,
        "steady_step_ms_mean": round(statistics.fmean(steady), 4) if steady else None,
        "steady_step_ms_stdev": (
            round(statistics.stdev(steady), 4) if len(steady) > 1 else 0.0
        ),
        "steady_step_ms_p95": (
            round(sorted(steady)[int(0.95 * (len(steady) - 1))], 4) if steady else None
        ),
        # batch 在稳态窗口内是否恒定 —— tail effect 的独立核对，不靠"我构造得对"
        "steady_batch_stable": (
            all(steps[i]["n_reqs"] == batch for i in pure_idx[D2_STEADY_SKIP:])
            if steady and steps[pure_idx[0]]["n_reqs"] is not None
            else None
        ),
        "scheduler_introspection_ok": bool(spy.ok and _sched_map(spy.last) is not None),
        "in_process_first_token_latency_ms": ft_ms,
        "request_completion_latency_ms": comp_ms,
        "actual_output_tokens": out_lens,
        "total_output_tokens": total_out,
        "output_hashes": [outputs.get(r, {}).get("hash") for r in rids],
        "batch_output_hash": _h([outputs.get(r, {}).get("hash", "") for r in rids]),
        "batch_prompt_hash": _h([t for r in rids for t in prompts[r]]),
        "steps_detail": steps if batch == 1 else None,  # 只对 batch=1 留全量，控体积
        "success": ok,
        "invalid_reason": "; ".join(reasons),
    }


# ============================ D3 · prefill scaling ============================
def run_d3_sample(engine, spy, tf, tp, prompt_len):
    """batch=1，只变 prompt 长度；max_tokens=1 → 首 token 延迟即 prefill 成本。

    注意 prompt_len 必须 < MAX_MODEL_LEN（input_processor.py:423-432：
    prompt_len == max_model_len 会直接 ValueError，因为还要至少产出 1 个 token）。
    """
    from vllm import SamplingParams

    assert prompt_len < MAX_MODEL_LEN, (
        f"prompt_len={prompt_len} 必须 < max_model_len={MAX_MODEL_LEN}"
        "（input_processor.py:423-432）"
    )
    rid = f"m22-tp{tp}-p{prompt_len}"
    ids = tf.ids(prompt_len, uniq="d3：")
    engine.add_request(
        rid,
        {"prompt_token_ids": ids},
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=False, seed=SEED),
    )
    chunks, n_step, first_s, got = [], 0, None, None
    t0 = time.perf_counter()
    while engine.has_unfinished_requests() and n_step < 100000:
        outs = engine.step()
        sm = _sched_map(spy.last)
        if sm:
            n = sum(v for k, v in sm.items() if _same_req(k, rid))
            if n > 0:
                chunks.append(n)
        for o in outs:
            if _same_req(o.request_id, rid) and o.outputs:
                got = {"n": len(o.outputs[0].token_ids), "finished": bool(o.finished)}
                if first_s is None:
                    first_s = time.perf_counter() - t0
        n_step += 1
    total_s = time.perf_counter() - t0
    ok = got is not None and got["n"] == 1 and got["finished"]
    return {
        "experiment": "D3_prefill",
        "tp": tp,
        "batch_size": 1,
        "prompt_len": prompt_len,
        "prefill_step_count": len(chunks),
        "chunk_sizes": chunks,
        "sum_prefill_scheduled": sum(chunks),
        "engine_steps": n_step,
        "in_process_first_token_latency_ms": (
            round(first_s * 1000, 4) if first_s is not None else None
        ),
        "request_completion_latency_ms": round(total_s * 1000, 4),
        "prompt_hash": _h(ids),
        "actual_output_tokens": got["n"] if got else 0,
        "success": ok,
        "invalid_reason": "" if ok else f"got={got}",
    }


# ============================ shard 驱动 ============================
def run_side(tp, model, tag, quick=False, profile_dir=None):
    assert_env()
    batches = QUICK_BATCHES if quick else D2_BATCHES
    plens = QUICK_PROMPT_LENS if quick else D3_PROMPT_LENS
    n_warm = QUICK_WARMUP if quick else D2_WARMUP
    n_meas = QUICK_REPEATS if quick else D2_REPEATS
    out_len = QUICK_OUTPUT_LEN if quick else D2_OUTPUT_LEN

    import torch  # noqa: F401  （仅用于 worker 端；driver 不建 CUDA context）

    engine = build_engine(tp, model, profile_dir=profile_dir)
    samples = []
    try:
        eff = effective_cfg(engine, tp)
        print(
            f"[m22] effective config: {json.dumps(eff, ensure_ascii=False)}", flush=True
        )
        spy = ScheduleSpy(get_scheduler(engine))
        if not spy.ok:
            print(
                "[m22][WARN] 拿不到 scheduler，逐 step 内省将缺失（如实标记）",
                flush=True,
            )
        tf = TokenFactory(engine, model)

        # --- D1：引擎建好后直接采一次，不另建引擎 ---
        d1 = run_d1(engine, tp, eff)
        d1.update(
            {"tag": tag, "phase": "measure", "repeat": 0, "metric_scope": "in-process"}
        )
        samples.append(d1)
        print(
            f"[m22] D1 ranks={d1['n_ranks_reported']} method={d1['observation_method']}",
            flush=True,
        )

        # --- D2：steady-state decode ---
        for b in batches:
            for r in range(n_warm + n_meas):
                s = run_d2_sample(engine, spy, tf, tp, b, out_len, D2_PROMPT_LEN)
                s.update(
                    {
                        "tag": tag,
                        "repeat": r,
                        "phase": "warmup" if r < n_warm else "measure",
                        "effective_cfg": eff,
                        "metric_scope": "in-process",
                    }
                )
                samples.append(s)
            print(f"[m22] D2 tp={tp} batch={b} done ({n_warm}w+{n_meas}m)", flush=True)

        # --- D3：prefill scaling ---
        for p in plens:
            for r in range(n_warm + n_meas):
                s = run_d3_sample(engine, spy, tf, tp, p)
                s.update(
                    {
                        "tag": tag,
                        "repeat": r,
                        "phase": "warmup" if r < n_warm else "measure",
                        "effective_cfg": eff,
                        "metric_scope": "in-process",
                    }
                )
                samples.append(s)
            print(f"[m22] D3 tp={tp} prompt_len={p} done", flush=True)
    finally:
        spy_obj = locals().get("spy")
        if spy_obj is not None:
            spy_obj.restore()
        del engine
    return samples


def run_profile(tp, model, tag, profile_dir):
    """P1 · collective communication evidence（独立 shard，避免给主实验加 profiler 开销）。

    ⚠ v0.24.0 **没有** VLLM_TORCH_PROFILER_DIR，必须用 ProfilerConfig（见 build_engine）。
    每个 rank 各写一份 trace，文件名含 `tp{N}_rank{N}`
    （distributed/utils.py:664-696 + torch tensorboard_trace_handler）。
    """
    assert_env()
    os.makedirs(profile_dir, exist_ok=True)
    engine = build_engine(tp, model, profile_dir=profile_dir)
    recs = []
    try:
        eff = effective_cfg(engine, tp)
        spy = ScheduleSpy(get_scheduler(engine))
        tf = TokenFactory(engine, model)
        # 三个代表性 workload，而不是把整个网格都 profile
        cases = [
            ("decode_small_batch", "d2", 1),
            ("decode_large_batch", "d2", 16),
            ("prefill_long", "d3", 4096),
        ]
        for name, kind, param in cases:
            # 先跑几次 warmup（不采集），避免把首次 JIT 计进 trace
            for _ in range(2):
                if kind == "d2":
                    run_d2_sample(engine, spy, tf, tp, param, 16, D2_PROMPT_LEN)
                else:
                    run_d3_sample(engine, spy, tf, tp, param)
            engine.start_profile(profile_prefix=f"{tag}_tp{tp}_{name}")
            if kind == "d2":
                s = run_d2_sample(engine, spy, tf, tp, param, 16, D2_PROMPT_LEN)
            else:
                s = run_d3_sample(engine, spy, tf, tp, param)
            engine.stop_profile()
            s.update(
                {
                    "experiment": "P1_profile",
                    "profile_case": name,
                    "tag": tag,
                    "phase": "measure",
                    "profile_prefix": f"{tag}_tp{tp}_{name}",
                    "profile_dir": os.path.abspath(profile_dir),
                    "effective_cfg": eff,
                }
            )
            recs.append(s)
            print(f"[m22] profiled {name} (tp={tp})", flush=True)
    finally:
        spy_obj = locals().get("spy")
        if spy_obj is not None:
            spy_obj.restore()
        del engine
    return recs


# ============================ 落盘 / 汇总 ============================
def _dump(samples, name):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{name}.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
    print(f"[m22] shard written: {path} ({len(samples)} samples)")


def _load(tag):
    rows = []
    if not os.path.isdir(RAW):
        return rows
    for n in sorted(os.listdir(RAW)):
        if n.startswith(f"{tag}_tp") and n.endswith(".jsonl"):
            with open(os.path.join(RAW, n)) as f:
                rows.extend(json.loads(x) for x in f if x.strip())
    return rows


def _measure(rows):
    return [r for r in rows if r.get("phase") == "measure" and r.get("success") is True]


def _stats(xs):
    if not xs:
        return {}
    s = sorted(xs)
    return {
        "n": len(xs),
        "median": round(statistics.median(xs), 4),
        "mean": round(statistics.fmean(xs), 4),
        "stdev": round(statistics.stdev(xs), 4) if len(xs) > 1 else 0.0,
        "p95": round(s[int(0.95 * (len(s) - 1))], 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
    }


def _boot_ci_median_diff(a, b, iters=10000, seed=0):
    """median(a) - median(b) 的 Bootstrap 95% CI（与 M17/M19 同算法同种子）。"""
    import random

    rng = random.Random(seed)
    d = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        d.append(statistics.median(ra) - statistics.median(rb))
    d.sort()
    return (
        round(statistics.median(a) - statistics.median(b), 4),
        round(d[int(0.025 * (iters - 1))], 4),
        round(d[int(0.975 * (iters - 1))], 4),
    )


def summarize(tag):
    """离线汇总（不建引擎、不需 GPU）。speedup / efficiency 全部由 D2 派生，不另跑实验。"""
    os.makedirs(OUT, exist_ok=True)
    rows = _load(tag)
    if not rows:
        print(f"[m22] 未找到 tag={tag} 的样本 —— 实验尚未执行。")
        return False
    meas = _measure(rows)

    # ---------------- D1 ----------------
    d1 = [r for r in rows if r["experiment"] == "D1_memory_kv"]
    if d1:
        p = os.path.join(OUT, f"{tag}_memory_kv.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tag",
                    "tp",
                    "rank",
                    "device",
                    "device_uuid",
                    "weight_MiB",
                    "torch_allocated_MiB",
                    "torch_reserved_MiB",
                    "non_torch_MiB",
                    "available_kv_MiB",
                    "num_gpu_blocks",
                    "block_size",
                    "device_total_MiB",
                    # ↓ 引擎级（min over ranks 后的统一容量），非各 rank 之和、非运行时并发
                    "sched_num_gpu_blocks",
                    "sched_kv_cache_size_tokens",
                    "sched_kv_capacity_equiv_concurrency",
                    "runtime_max_num_seqs",
                    "num_attention_heads_per_rank",
                    "num_kv_heads_per_rank",
                    "total_num_kv_heads",
                    "kv_heads_replicated",
                    "observation_method",
                ]
            )
            for r in sorted(d1, key=lambda x: x["tp"]):
                eff = r["effective_cfg"]
                kv_rep = (
                    eff["num_kv_heads_per_rank"] * eff["tensor_parallel_size"]
                    > eff["total_num_kv_heads"]
                )
                for pr in r.get("per_rank") or []:
                    if not isinstance(pr, dict):
                        continue

                    def mib(k, _pr=pr):
                        v = _pr.get(k)
                        return round(v / 2**20, 2) if isinstance(v, int) else ""

                    w.writerow(
                        [
                            r["tag"],
                            r["tp"],
                            pr.get("rank"),
                            pr.get("device"),
                            pr.get("device_uuid"),
                            mib("weight_bytes"),
                            mib("torch_allocated_bytes"),
                            mib("torch_reserved_bytes"),
                            mib("non_torch_memory_bytes"),
                            mib("available_kv_bytes"),
                            pr.get("num_gpu_blocks"),
                            pr.get("block_size"),
                            mib("device_total_bytes"),
                            eff["sched_num_gpu_blocks"],
                            eff["sched_kv_cache_size_tokens"],
                            eff["sched_kv_capacity_equiv_concurrency"],
                            eff["max_num_seqs"],
                            eff["num_attention_heads_per_rank"],
                            eff["num_kv_heads_per_rank"],
                            eff["total_num_kv_heads"],
                            kv_rep,
                            r["observation_method"],
                        ]
                    )
        print(f"[m22] wrote {p}")

    # ---------------- D2 + 派生的 speedup / efficiency ----------------
    d2 = [r for r in meas if r["experiment"] == "D2_decode"]
    if d2:
        g = {}
        for r in d2:
            g.setdefault((r["tp"], r["batch_size"]), []).append(r)

        def thr(v):
            return [
                x["total_output_tokens"] / (x["request_completion_latency_ms"] / 1000)
                for x in v
            ]

        base = {b: g.get((1, b)) for b in {k[1] for k in g}}
        p = os.path.join(OUT, f"{tag}_decode.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tag",
                    "tp",
                    "batch_size",
                    "n",
                    "steady_step_ms_median",
                    "steady_step_ms_stdev",
                    "steady_step_ms_p95",
                    "n_steady_steps_median",
                    "steady_batch_stable_all",
                    "completion_ms_median",
                    "completion_ms_stdev",
                    "completion_ms_p95",
                    "first_token_ms_median",
                    "tpot_ms_median",
                    "output_tps_median",
                    "speedup_vs_tp1",
                    "efficiency_vs_tp1",
                    "scheduler_introspection_ok",
                    "metric_scope",
                ]
            )
            for k in sorted(g):
                v = g[k]
                st = _stats(
                    [
                        x["steady_step_ms_median"]
                        for x in v
                        if x["steady_step_ms_median"]
                    ]
                )
                cp = _stats([x["request_completion_latency_ms"] for x in v])
                ftl = _stats(
                    [
                        x["in_process_first_token_latency_ms"]
                        for x in v
                        if x["in_process_first_token_latency_ms"]
                    ]
                )
                tps = _stats(thr(v))
                # TPOT 由完成/首 token 派生，**不用它代表单步 decode**（那是 steady_step）
                tpot = _stats(
                    [
                        (
                            x["request_completion_latency_ms"]
                            - x["in_process_first_token_latency_ms"]
                        )
                        / max(1, x["max_tokens_requested"] - 1)
                        for x in v
                        if x["in_process_first_token_latency_ms"]
                    ]
                )
                b1 = base.get(k[1])
                sp = eff_ = ""
                if b1:
                    t1 = statistics.median(thr(b1))
                    tk = tps["median"]
                    sp = round(tk / t1, 4)
                    eff_ = round(tk / (k[0] * t1), 4)
                w.writerow(
                    [
                        v[0]["tag"],
                        k[0],
                        k[1],
                        len(v),
                        st.get("median"),
                        st.get("stdev"),
                        st.get("p95"),
                        _stats([x["n_steady_steps"] for x in v]).get("median"),
                        all(x.get("steady_batch_stable") for x in v),
                        cp.get("median"),
                        cp.get("stdev"),
                        cp.get("p95"),
                        ftl.get("median"),
                        tpot.get("median"),
                        tps.get("median"),
                        sp,
                        eff_,
                        all(x.get("scheduler_introspection_ok") for x in v),
                        "in-process",
                    ]
                )
        print(f"[m22] wrote {p}")

        # TP 之间的差值 + Bootstrap CI（主指标：稳态单步 decode）
        p2 = os.path.join(OUT, f"{tag}_decode_effect.csv")
        with open(p2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tag",
                    "batch_size",
                    "tp",
                    "n_per_group",
                    "tp1_steady_step_ms",
                    "tpk_steady_step_ms",
                    "median_diff_ms",
                    "ci95_lo_ms",
                    "ci95_hi_ms",
                    "ci_excludes_zero",
                    "rel_change_pct",
                    "metric_scope",
                ]
            )
            for b in sorted({k[1] for k in g}):
                a = g.get((1, b))
                if not a:
                    continue
                av = [
                    x["steady_step_ms_median"] for x in a if x["steady_step_ms_median"]
                ]
                for tp in sorted({k[0] for k in g} - {1}):
                    c = g.get((tp, b))
                    if not c:
                        continue
                    cv = [
                        x["steady_step_ms_median"]
                        for x in c
                        if x["steady_step_ms_median"]
                    ]
                    if not av or not cv:
                        continue
                    d, lo, hi = _boot_ci_median_diff(av, cv)
                    am, cm = statistics.median(av), statistics.median(cv)
                    w.writerow(
                        [
                            a[0]["tag"],
                            b,
                            tp,
                            min(len(av), len(cv)),
                            round(am, 4),
                            round(cm, 4),
                            d,
                            lo,
                            hi,
                            "yes" if (lo > 0 or hi < 0) else "no",
                            round((am - cm) / am * 100, 2),
                            "in-process",
                        ]
                    )
        print(f"[m22] wrote {p2}")

    # ---------------- D3 ----------------
    d3 = [r for r in meas if r["experiment"] == "D3_prefill"]
    if d3:
        g3 = {}
        for r in d3:
            g3.setdefault((r["tp"], r["prompt_len"]), []).append(r)
        p = os.path.join(OUT, f"{tag}_prefill.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "tag",
                    "tp",
                    "prompt_len",
                    "n",
                    "prefill_step_count",
                    "chunk_sizes",
                    "sum_prefill_scheduled",
                    "first_token_ms_median",
                    "first_token_ms_stdev",
                    "first_token_ms_p95",
                    "speedup_vs_tp1",
                    "metric_scope",
                ]
            )
            b1 = {k[1]: g3.get((1, k[1])) for k in g3}
            for k in sorted(g3):
                v = g3[k]
                st = _stats([x["in_process_first_token_latency_ms"] for x in v])
                ref = b1.get(k[1])
                sp = ""
                if ref:
                    r1 = statistics.median(
                        [x["in_process_first_token_latency_ms"] for x in ref]
                    )
                    sp = round(r1 / st["median"], 4)
                w.writerow(
                    [
                        v[0]["tag"],
                        k[0],
                        k[1],
                        len(v),
                        v[0]["prefill_step_count"],
                        v[0]["chunk_sizes"],
                        v[0]["sum_prefill_scheduled"],
                        st.get("median"),
                        st.get("stdev"),
                        st.get("p95"),
                        sp,
                        "in-process",
                    ]
                )
        print(f"[m22] wrote {p}")

    # ---------------- 输出一致性（跨 TP，如实记录） ----------------
    p4 = os.path.join(OUT, f"{tag}_output_consistency.csv")
    with open(p4, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "tag",
                "batch_size",
                "repeat",
                "tp_a",
                "tp_b",
                "prompt_hash_match",
                "output_hash_match",
                "out_lens_a",
                "out_lens_b",
            ]
        )
        idx = {}
        for r in d2:
            idx[(r["tp"], r["batch_size"], r["repeat"])] = r
        tps = sorted({k[0] for k in idx})
        for b in sorted({k[1] for k in idx}):
            for rep in sorted({k[2] for k in idx}):
                a = idx.get((1, b, rep))
                if not a:
                    continue
                for tp in tps:
                    if tp == 1:
                        continue
                    c = idx.get((tp, b, rep))
                    if not c:
                        continue
                    w.writerow(
                        [
                            a["tag"],
                            b,
                            rep,
                            1,
                            tp,
                            a["batch_prompt_hash"] == c["batch_prompt_hash"],
                            a["batch_output_hash"] == c["batch_output_hash"],
                            a["actual_output_tokens"],
                            c["actual_output_tokens"],
                        ]
                    )
    print(f"[m22] wrote {p4}")
    print(f"[m22] summarize done. tag={tag} measure samples: {len(meas)}")
    return True


def main():
    p = argparse.ArgumentParser(
        description="M22 tensor parallel（需多卡；一个进程一个引擎）"
    )
    p.add_argument("--tp", type=int, help="tensor_parallel_size")
    p.add_argument("--shard", choices=["run", "profile", "summarize"], default="run")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument(
        "--tag",
        required=True,
        help="数据标签。本地预演必须用 local_preflight，H20 正式用 h20_formal —— "
        "两者写入不同文件，绝不混入同一份 CSV",
    )
    p.add_argument("--quick", action="store_true", help="本地预演用的极小网格")
    p.add_argument(
        "--profile-dir", default=os.path.join(ROOT, "outputs", "m22", "traces")
    )
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.shard == "summarize":
        if not summarize(a.tag):
            raise SystemExit(1)
        return
    if a.tp is None:
        p.error("--shard run/profile 必须提供 --tp")
    if a.shard == "run":
        _dump(run_side(a.tp, a.model, a.tag, a.quick), f"{a.tag}_tp{a.tp}_run")
    else:
        _dump(
            run_profile(a.tp, a.model, a.tag, a.profile_dir),
            f"{a.tag}_tp{a.tp}_profile",
        )


if __name__ == "__main__":
    main()
