#!/usr/bin/env python3
"""M24 · Decode Context Parallel（TP=4 固定，唯一变量 dcp∈{1,2}）。

每个 DCP 档一个独立进程。一个引擎上依次 D1 + D2。

    export VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0
    python scripts/m24_dcp.py --dcp 1 --tag h20_formal --model /path/to/Qwen2.5-1.5B-Instruct
    python scripts/m24_dcp.py --dcp 2 --tag h20_formal --model /path/to/Qwen2.5-1.5B-Instruct
    python scripts/m24_dcp.py --shard summarize --tag h20_formal

矩阵：notes/m24/M24_H20_PLAN.md
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "m24")
RAW = os.path.join(OUT, "raw")

SEED = 0
TP = 4
MAX_MODEL_LEN = 16384
GPU_MEM_UTIL = 0.85
MAX_NUM_SEQS = 256
MAX_NUM_BATCHED_TOKENS = 8192

D2_CTX = [1024, 4096, 8192]
D2_BATCHES = [1, 8]
D2_OUTPUT_LEN = 32
D2_WARMUP = 5
D2_REPEATS = 15
D2_STEADY_SKIP = 3

QUICK_CTX = [1024]
QUICK_BATCHES = [1]
QUICK_WARMUP = 1
QUICK_REPEATS = 3
QUICK_OUTPUT_LEN = 8

_LONG = (
    "推理引擎的连续批处理与分页注意力机制通过按块管理键值缓存来提升吞吐与显存利用率。"
    "The quick brown fox jumps over the lazy dog. "
)


def _h(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


def assert_env():
    v = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
    if v not in ("0", "false", "False"):
        raise SystemExit(
            "[m24][FATAL] 必须 export VLLM_ENABLE_V1_MULTIPROCESSING=0。\n"
            "  依据 vllm/v1/engine/llm_engine.py：该变量为真时会强制覆盖 "
            "enable_multiprocessing=False，D2 scheduler 内省失效。\n"
            f"  当前值: {v!r}"
        )


def load_hf_config(model: str) -> dict:
    path = model
    cfg_path = os.path.join(path, "config.json")
    if not os.path.isfile(cfg_path):
        raise SystemExit(f"[m24][FATAL] 找不到 {cfg_path}。必须用本地模型目录，禁止 HF id。")
    with open(cfg_path, encoding="utf-8") as f:
        return json.load(f)


def assert_dcp_legal(model: str, tp: int, dcp: int) -> dict:
    """建引擎前用 config.json 做 GQA 算术。不合法则不碰 GPU。"""
    cfg = load_hf_config(model)
    hq = int(cfg["num_attention_heads"])
    hkv = int(cfg["num_key_value_heads"])
    info = {
        "model_path": os.path.abspath(model),
        "num_attention_heads": hq,
        "num_key_value_heads": hkv,
        "q_per_kv": hq // hkv if hkv else None,
        "tp": tp,
        "dcp": dcp,
        "max_dcp": (tp // hkv) if hkv else None,
    }
    if dcp <= 1:
        return info
    reasons = []
    if tp <= hkv:
        reasons.append(f"需要 tp > H_kv，得到 tp={tp} H_kv={hkv}")
    if hkv and dcp > tp // hkv:
        reasons.append(f"需要 dcp <= tp/H_kv={tp // hkv}，得到 dcp={dcp}")
    if hkv and (hq // hkv) % dcp != 0:
        reasons.append(f"需要 (H_q/H_kv) % dcp == 0，得到 {hq}/{hkv}={hq // hkv} dcp={dcp}")
    if tp % dcp != 0:
        reasons.append(f"需要 tp % dcp == 0，得到 tp={tp} dcp={dcp}")
    if reasons:
        raise SystemExit(
            "[m24][FATAL] 该模型在当前 TP 上不能开 DCP。不要换 7B/0.5B，不要改 TP。\n"
            + "\n".join(f"  - {r}" for r in reasons)
            + f"\n  config: {info}"
        )
    return info


def _rank_report(worker):
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
        cc = getattr(worker, "cache_config", None)
        out["num_gpu_blocks"] = (
            int(getattr(cc, "num_gpu_blocks", 0) or 0) if cc is not None else None
        )
        out["block_size"] = (
            int(getattr(cc, "block_size", 0) or 0) if cc is not None else None
        )
        try:
            from vllm.distributed.parallel_state import get_dcp_group

            g = get_dcp_group()
            out["dcp_world_size"] = int(g.world_size)
            out["dcp_rank"] = int(g.rank_in_group)
        except Exception as e:  # noqa: BLE001
            out["dcp_world_size"] = None
            out["dcp_rank"] = None
            out["dcp_group_err"] = repr(e)
    except Exception as e:  # noqa: BLE001
        out["ok"] = False
        out["err"] = repr(e)
    return out


def build_engine(dcp: int, model: str, profile_dir: str | None = None):
    from vllm import LLMEngine
    from vllm.engine.arg_utils import EngineArgs
    from vllm.usage.usage_lib import UsageContext

    kwargs = {
        "model": model,
        "dtype": "auto",
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEM_UTIL,
        "tensor_parallel_size": TP,
        "decode_context_parallel_size": dcp,
        "prefill_context_parallel_size": 1,
        "enforce_eager": True,
        "seed": SEED,
        "max_num_seqs": MAX_NUM_SEQS,
        "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "disable_log_stats": True,
    }
    if profile_dir:
        from vllm.config.profiler import ProfilerConfig

        kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch",
            torch_profiler_dir=os.path.abspath(profile_dir),
            torch_profiler_with_stack=False,
            torch_profiler_record_shapes=True,
        )
    args = EngineArgs(**kwargs)
    return LLMEngine.from_engine_args(
        args, usage_context=UsageContext.ENGINE_CONTEXT, enable_multiprocessing=False
    )


def effective_cfg(engine, dcp_requested: int) -> dict:
    vc = engine.vllm_config
    pc, sc, cc, mc = (
        vc.parallel_config,
        vc.scheduler_config,
        vc.cache_config,
        vc.model_config,
    )
    return {
        "dcp_requested": dcp_requested,
        "decode_context_parallel_size": int(pc.decode_context_parallel_size),
        "prefill_context_parallel_size": int(pc.prefill_context_parallel_size),
        "tensor_parallel_size": int(pc.tensor_parallel_size),
        "world_size": int(getattr(pc, "world_size", pc.tensor_parallel_size)),
        "cp_kv_cache_interleave_size": int(
            getattr(pc, "cp_kv_cache_interleave_size", 1)
        ),
        "dcp_comm_backend": str(getattr(pc, "dcp_comm_backend", "")),
        "distributed_executor_backend": str(pc.distributed_executor_backend),
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
        "sched_num_gpu_blocks": int(getattr(cc, "num_gpu_blocks", 0) or 0),
        "sched_kv_cache_size_tokens": getattr(cc, "kv_cache_size_tokens", None),
        "total_num_attention_heads": mc.model_arch_config.total_num_attention_heads,
        "total_num_kv_heads": mc.get_total_num_kv_heads(),
        "num_attention_heads_per_rank": mc.get_num_attention_heads(pc),
        "num_kv_heads_per_rank": mc.get_num_kv_heads(pc),
    }


def get_scheduler(engine):
    try:
        return engine.engine_core.engine_core.scheduler
    except AttributeError:
        return None


class ScheduleSpy:
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

            tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
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
    if so is None:
        return None
    d = getattr(so, "num_scheduled_tokens", None)
    return dict(d) if d else {}


def run_d1(engine, dcp: int, eff: dict) -> dict:
    ranks, method, err = None, None, ""
    try:
        ranks = engine.collective_rpc(_rank_report)
        method = "collective_rpc(callable)"
    except Exception as e:  # noqa: BLE001
        err = repr(e)
        method = "unavailable"
    def _tot(key):
        if not ranks or not isinstance(ranks[0], dict):
            return None
        vals = [r.get(key) for r in ranks if isinstance(r, dict)]
        return sum(v for v in vals if isinstance(v, int)) if vals else None

    return {
        "experiment": "D1_memory_kv",
        "tp": TP,
        "dcp": dcp,
        "observation_method": method,
        "observation_error": err,
        "per_rank": ranks,
        "n_ranks_reported": len(ranks) if ranks else 0,
        "sum_weight_bytes": _tot("weight_bytes"),
        "sum_available_kv_bytes": _tot("available_kv_bytes"),
        "sum_torch_allocated_bytes": _tot("torch_allocated_bytes"),
        "effective_cfg": eff,
        "success": bool(ranks) and str(method).startswith("collective_rpc(callable)"),
        "invalid_reason": "" if ranks else f"per-rank 观测失败: {err}",
    }


def run_d2_sample(engine, spy, tf, dcp, batch, out_len, prompt_len):
    from vllm import SamplingParams

    rids = [f"m24-dcp{dcp}-p{prompt_len}-b{batch}-r{i}" for i in range(batch)]
    prompts = {r: tf.ids(prompt_len, uniq=f"d2_{i}：") for i, r in enumerate(rids)}
    for rid in rids:
        engine.add_request(
            rid,
            {"prompt_token_ids": prompts[rid]},
            SamplingParams(
                temperature=0.0, max_tokens=out_len, ignore_eos=True, seed=SEED
            ),
        )

    steps = []
    outputs: dict[str, dict] = {}
    first_token_s = None
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
    ft_ms = None if first_token_s is None else round(first_token_s * 1000, 4)
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
        reasons.append(f"稳态 decode step 仅 {len(steady)} 个（<5）")
    return {
        "experiment": "D2_decode",
        "tp": TP,
        "dcp": dcp,
        "batch_size": batch,
        "prompt_len": prompt_len,
        "max_tokens_requested": out_len,
        "engine_steps": n_step,
        "n_pure_decode_steps": len(pure_idx),
        "n_steady_steps": len(steady),
        "steady_step_ms_median": round(statistics.median(steady), 4) if steady else None,
        "steady_step_ms_mean": round(statistics.fmean(steady), 4) if steady else None,
        "steady_step_ms_stdev": (
            round(statistics.stdev(steady), 4) if len(steady) > 1 else 0.0
        ),
        "steady_step_ms_p95": (
            round(sorted(steady)[int(0.95 * (len(steady) - 1))], 4) if steady else None
        ),
        "steady_batch_stable": (
            all(steps[i]["n_reqs"] == batch for i in pure_idx[D2_STEADY_SKIP:])
            if steady and steps[pure_idx[0]]["n_reqs"] is not None
            else None
        ),
        "scheduler_introspection_ok": bool(spy.ok and _sched_map(spy.last) is not None),
        "in_process_first_token_latency_ms": ft_ms,
        "request_completion_latency_ms": round(completion_s * 1000, 4),
        "actual_output_tokens": out_lens,
        "output_hashes": [outputs.get(r, {}).get("hash") for r in rids],
        "success": ok,
        "invalid_reason": "; ".join(reasons),
        "metric_scope": "in_process_steady_decode_step",
    }


def append_jsonl(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = dict(rec)
    rec["unix_ts"] = time.time()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_shard(dcp: int, tag: str, model: str, quick: bool, profile: bool) -> int:
    assert_env()
    gate = assert_dcp_legal(model, TP, dcp)
    os.makedirs(RAW, exist_ok=True)
    jsonl = os.path.join(RAW, f"{tag}_dcp{dcp}_{'profile' if profile else 'run'}.jsonl")
    env_path = os.path.join(ROOT, "logs", "m24", "environment.txt")
    os.makedirs(os.path.dirname(env_path), exist_ok=True)

    ctxs = QUICK_CTX if quick else D2_CTX
    batches = QUICK_BATCHES if quick else D2_BATCHES
    warmup = QUICK_WARMUP if quick else D2_WARMUP
    repeats = QUICK_REPEATS if quick else D2_REPEATS
    out_len = QUICK_OUTPUT_LEN if quick else D2_OUTPUT_LEN

    profile_dir = None
    if profile:
        profile_dir = os.path.join(ROOT, "outputs", "m24", "traces", f"{tag}_dcp{dcp}")
        os.makedirs(profile_dir, exist_ok=True)

    t_build = time.perf_counter()
    engine = build_engine(dcp, model, profile_dir=profile_dir)
    build_s = round(time.perf_counter() - t_build, 3)
    eff = effective_cfg(engine, dcp)
    if eff["decode_context_parallel_size"] != dcp:
        raise SystemExit(
            f"[m24][FATAL] 请求 dcp={dcp} 但引擎回读 "
            f"{eff['decode_context_parallel_size']}"
        )
    if eff["tensor_parallel_size"] != TP:
        raise SystemExit(f"[m24][FATAL] TP 回读 {eff['tensor_parallel_size']} != {TP}")

    with open(env_path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "tag": tag,
                    "dcp": dcp,
                    "build_s": build_s,
                    "gqa_gate": gate,
                    "effective_cfg": eff,
                    "vllm": __import__("vllm").__version__,
                    "vllm_file": __import__("vllm").__file__,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    meta = {
        "tag": tag,
        "phase": "meta",
        "dcp": dcp,
        "gqa_gate": gate,
        "effective_cfg": eff,
        "engine_build_s": build_s,
        "quick": quick,
        "profile": profile,
        "success": True,
    }
    append_jsonl(jsonl, meta)

    d1 = run_d1(engine, dcp, eff)
    d1.update({"tag": tag, "phase": "d1", "quick": quick})
    append_jsonl(jsonl, d1)

    spy = ScheduleSpy(get_scheduler(engine))
    tf = TokenFactory(engine, model)
    try:
        if profile:
            # 0.24.0：只传 ProfilerConfig 不会落盘 chrome trace，
            # 必须显式 start_profile()/stop_profile()（H20 首跑已实证）。
            # 本 shard 用 8 token 短跑，稳态 step 必然 <5 → D2 success=False
            # 是阈值，不是缺 trace；profile 成败看是否写出 trace。
            if hasattr(engine, "start_profile"):
                engine.start_profile()
            try:
                rec = run_d2_sample(engine, spy, tf, dcp, 1, 8, 1024)
            finally:
                if hasattr(engine, "stop_profile"):
                    engine.stop_profile()
            rec.update(
                {
                    "tag": tag,
                    "phase": "profile",
                    "metric_scope": "profile_only_not_formal",
                    "effective_cfg": eff,
                }
            )
            append_jsonl(jsonl, rec)
            if not rec.get("success"):
                print(
                    "[m24] profile 样本 success=False（8 token → 稳态 step<5，预期）；"
                    "正式 D2 以 run shard 为准",
                    flush=True,
                )
            return 0

        for prompt_len in ctxs:
            for batch in batches:
                for i in range(warmup):
                    rec = run_d2_sample(
                        engine, spy, tf, dcp, batch, out_len, prompt_len
                    )
                    rec.update(
                        {
                            "tag": tag,
                            "phase": "warmup",
                            "repeat": i,
                            "effective_cfg": eff,
                            "quick": quick,
                        }
                    )
                    append_jsonl(jsonl, rec)
                for i in range(repeats):
                    rec = run_d2_sample(
                        engine, spy, tf, dcp, batch, out_len, prompt_len
                    )
                    rec.update(
                        {
                            "tag": tag,
                            "phase": "measure",
                            "repeat": i,
                            "effective_cfg": eff,
                            "quick": quick,
                        }
                    )
                    append_jsonl(jsonl, rec)
                    print(
                        f"[m24] dcp={dcp} p={prompt_len} b={batch} "
                        f"i={i} ok={rec['success']} "
                        f"step_med={rec['steady_step_ms_median']}",
                        flush=True,
                    )
    finally:
        spy.restore()
    return 0


def _load_rows(tag: str) -> list[dict]:
    rows = []
    if not os.path.isdir(RAW):
        return rows
    for n in sorted(os.listdir(RAW)):
        if n.startswith(f"{tag}_dcp") and n.endswith("_run.jsonl"):
            with open(os.path.join(RAW, n), encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def bootstrap_median_diff(a, b, n=10000, seed=0):
    if not a or not b:
        return None
    rng = random.Random(seed)
    d = []
    for _ in range(n):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        d.append(statistics.median(sa) - statistics.median(sb))
    d.sort()
    lo, hi = d[int(0.025 * n)], d[int(0.975 * n) - 1]
    return {
        "median_diff": statistics.median(a) - statistics.median(b),
        "ci95_lo": lo,
        "ci95_hi": hi,
        "excludes_zero": not (lo <= 0 <= hi),
        "n_a": len(a),
        "n_b": len(b),
    }


def summarize(tag: str) -> int:
    rows = _load_rows(tag)
    if not rows:
        print(f"[m24] summarize: 没有 {tag} 的 run jsonl")
        return 1
    os.makedirs(OUT, exist_ok=True)
    meas = [r for r in rows if r.get("phase") == "measure" and r.get("success") is True]
    d1s = [r for r in rows if r.get("experiment") == "D1_memory_kv"]

    mem_path = os.path.join(OUT, f"{tag}_memory_kv.csv")
    with open(mem_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "dcp",
                "n_ranks",
                "sum_weight_bytes",
                "sum_available_kv_bytes",
                "sched_kv_cache_size_tokens",
                "num_kv_heads_per_rank",
                "dcp_effective",
            ],
        )
        w.writeheader()
        for r in d1s:
            eff = r.get("effective_cfg") or {}
            w.writerow(
                {
                    "tag": tag,
                    "dcp": r.get("dcp"),
                    "n_ranks": r.get("n_ranks_reported"),
                    "sum_weight_bytes": r.get("sum_weight_bytes"),
                    "sum_available_kv_bytes": r.get("sum_available_kv_bytes"),
                    "sched_kv_cache_size_tokens": eff.get("sched_kv_cache_size_tokens"),
                    "num_kv_heads_per_rank": eff.get("num_kv_heads_per_rank"),
                    "dcp_effective": eff.get("decode_context_parallel_size"),
                }
            )

    dec_path = os.path.join(OUT, f"{tag}_decode.csv")
    keys = [
        "tag",
        "dcp",
        "prompt_len",
        "batch_size",
        "n",
        "steady_step_ms_median",
        "steady_step_ms_stdev",
        "steady_step_ms_p95",
        "in_process_first_token_latency_ms_median",
    ]
    groups: dict[tuple, list] = {}
    for r in meas:
        if r.get("experiment") != "D2_decode":
            continue
        k = (r["dcp"], r["prompt_len"], r["batch_size"])
        groups.setdefault(k, []).append(r)
    with open(dec_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for (dcp, plen, bs), rs in sorted(groups.items()):
            steps = [x["steady_step_ms_median"] for x in rs if x.get("steady_step_ms_median") is not None]
            fts = [
                x["in_process_first_token_latency_ms"]
                for x in rs
                if x.get("in_process_first_token_latency_ms") is not None
            ]
            w.writerow(
                {
                    "tag": tag,
                    "dcp": dcp,
                    "prompt_len": plen,
                    "batch_size": bs,
                    "n": len(rs),
                    "steady_step_ms_median": round(statistics.median(steps), 4) if steps else "",
                    "steady_step_ms_stdev": round(statistics.stdev(steps), 4) if len(steps) > 1 else 0,
                    "steady_step_ms_p95": round(sorted(steps)[int(0.95 * (len(steps) - 1))], 4) if steps else "",
                    "in_process_first_token_latency_ms_median": round(statistics.median(fts), 4) if fts else "",
                }
            )

    eff_path = os.path.join(OUT, f"{tag}_decode_effect.csv")
    with open(eff_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tag",
                "prompt_len",
                "batch_size",
                "median_dcp2_minus_dcp1_ms",
                "ci95_lo",
                "ci95_hi",
                "excludes_zero",
                "n_dcp1",
                "n_dcp2",
                "note",
            ],
        )
        w.writeheader()
        pairs = {}
        for (dcp, plen, bs), rs in groups.items():
            pairs.setdefault((plen, bs), {})[dcp] = [
                x["steady_step_ms_median"]
                for x in rs
                if x.get("steady_step_ms_median") is not None
            ]
        for (plen, bs), dcps in sorted(pairs.items()):
            a = dcps.get(2) or []
            b = dcps.get(1) or []
            boot = bootstrap_median_diff(a, b)
            note = (
                "正值 = DCP=2 更慢。CI 含 0 → 未观察到超出噪声的差异。"
                "这不是 Ring Attention。"
            )
            if not boot:
                w.writerow(
                    {
                        "tag": tag,
                        "prompt_len": plen,
                        "batch_size": bs,
                        "median_dcp2_minus_dcp1_ms": "",
                        "ci95_lo": "",
                        "ci95_hi": "",
                        "excludes_zero": "",
                        "n_dcp1": len(b),
                        "n_dcp2": len(a),
                        "note": "缺一档，无法对照。" + note,
                    }
                )
                continue
            w.writerow(
                {
                    "tag": tag,
                    "prompt_len": plen,
                    "batch_size": bs,
                    "median_dcp2_minus_dcp1_ms": round(boot["median_diff"], 4),
                    "ci95_lo": round(boot["ci95_lo"], 4),
                    "ci95_hi": round(boot["ci95_hi"], 4),
                    "excludes_zero": boot["excludes_zero"],
                    "n_dcp1": boot["n_b"],
                    "n_dcp2": boot["n_a"],
                    "note": note,
                }
            )
    print(f"[m24] wrote {mem_path}\n[m24] wrote {dec_path}\n[m24] wrote {eff_path}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dcp", type=int, default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--model", default=os.environ.get("M24_MODEL", ""))
    p.add_argument("--shard", choices=["run", "profile", "summarize"], default="run")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.shard == "summarize":
        return summarize(a.tag)
    if not a.model or not os.path.isdir(a.model):
        raise SystemExit(
            "[m24][FATAL] --model 必须是本地目录（拷贝包里的 "
            "models/Qwen2.5-1.5B-Instruct）。禁止 HF id。"
        )
    if a.dcp is None:
        raise SystemExit("[m24][FATAL] run/profile 需要 --dcp")
    return run_shard(a.dcp, a.tag, a.model, a.quick, profile=(a.shard == "profile"))


if __name__ == "__main__":
    raise SystemExit(main())
