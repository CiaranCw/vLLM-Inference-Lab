"""M17 · Prefix Caching & Chunked Prefill 深化 —— 实验脚本（E1 + E3）。

============================================================================
GPU-FREE STATUS：本脚本为【设计定稿、尚未运行】。运行需 GPU（会建真实引擎）。
  GPU 空闲后一键跑：
    VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      python scripts/m17_prefix_cache.py
  设计/源码定位/不变量见 notes/m17/prefix_cache_and_chunked_prefill.md。
============================================================================

复用 M6 已验证的闭环内省法（见 scripts/m6_benchmark.py）：
  - LLMEngine + ScheduleSpy 抓 SchedulerOutput；
  - 命中证据 = actual_prompt_tokens - total_num_scheduled_tokens（命中→scheduled<actual）；
  - chunked = scheduled > actual（当一次 step 未算完 prompt）；
  - 计时 torch.cuda.synchronize() 夹 perf_counter()，覆盖整个 engine.step()。
固定 Qwen2.5-0.5B / enforce_eager / seed=0 / temperature=0；warmup 与正式分离、重复≥15。

E1：长共享前缀 on/off——随共享前缀长度 P 增大，看 APC 省出的 prefill 延迟能否超过噪声
    （M6 短前缀时落在噪声内，见 outputs/m6/baseline.csv 的 prefix 行）。
E3：chunked prefill——长 prompt > max_num_batched_tokens 时按预算分块跨 step，记录分块序列。
E2（block 生命周期 ref_cnt/驱逐）：标为 TODO（需先验证 block_pool 内省 API），不阻塞 E1/E3。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import time

import torch
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext

OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "m17"
)
RAW = os.path.join(OUT, "raw")

# ------------------------------ 冻结配置（跑前写死） ------------------------------
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.6
BLOCK_SIZE_HINT = 16  # 仅用于 I1 对齐核对的说明；实际 block_size 以引擎为准
# E1：共享前缀长度（block 对齐），各带独立短后缀打破完全命中
E1_PREFIX_LENS = [256, 512, 1024, 2048]
E1_SUFFIX_LEN = 16
E1_WARMUP = 5
E1_REPEATS = 15
# E3：chunked prefill
E3_PROMPT_LEN = 2048
E3_BATCHED_TOKENS = [512, 1024, 2048]
E3_CHUNKED = [True, False]

_LONG = (
    "推理引擎的连续批处理与分页注意力机制通过按块管理键值缓存来提升吞吐与显存利用率。"
    "The quick brown fox jumps over the lazy dog. "
)


# ============================ 引擎与内省（复用 M6 模式） ============================
def build_engine(
    enable_prefix_caching: bool,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool,
) -> LLMEngine:
    args = EngineArgs(
        model=MODEL,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        tensor_parallel_size=1,
        enforce_eager=True,
        seed=SEED,
        max_num_seqs=64,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=enable_prefix_caching,
        enable_chunked_prefill=enable_chunked_prefill,
        async_scheduling=False,
        disable_log_stats=True,
    )
    return LLMEngine.from_engine_args(
        args, usage_context=UsageContext.ENGINE_CONTEXT, enable_multiprocessing=False
    )


def get_scheduler(engine: LLMEngine):
    return engine.engine_core.engine_core.scheduler


class ScheduleSpy:
    """只读包裹 scheduler.schedule：仅存最近一次 SchedulerOutput。计时区外读取。"""

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


class TokenFactory:
    def __init__(self, engine: LLMEngine):
        tok = getattr(engine, "tokenizer", None)
        if tok is None or not hasattr(tok, "encode"):
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok = tok
        base = self.tok.encode(_LONG, add_special_tokens=False)
        self.pool = (base * (MAX_MODEL_LEN // max(1, len(base)) + 2))[
            : MAX_MODEL_LEN + 200
        ]

    def ids(self, n: int, uniq: str = "") -> list[int]:
        """恰好 n 个 token id；uniq 非空则前置独立前缀（打破 prefix 共享）。"""
        if uniq:
            pre = self.tok.encode(uniq, add_special_tokens=False)
            body = (pre + self.pool)[:n]
        else:
            body = self.pool[:n]
        assert len(body) == n, f"token 构造长度不符: {len(body)} != {n}"
        return list(body)


def timed_step(engine: LLMEngine):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = engine.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0), outs


def drain(engine: LLMEngine):
    while engine.has_unfinished_requests():
        engine.step()


def _prefill_once(engine, spy, rid, ids):
    """加一个 max_tokens=1 的请求跑一个 prefill step，返回 (latency_ms, scheduled, actual, chunked)。"""
    engine.add_request(
        rid,
        {"prompt_token_ids": ids},
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=False, seed=SEED),
    )
    dt, outs = timed_step(engine)
    so = spy.last
    scheduled = int(so.total_num_scheduled_tokens) if so is not None else -1
    actual = len(outs[0].prompt_token_ids) if outs else len(ids)
    drain(engine)
    return round(dt * 1000, 4), scheduled, actual


# ============================ E1：长共享前缀 on/off ============================
def run_e1():
    samples = []
    for prefix_on in (True, False):
        engine = build_engine(
            prefix_on, max_num_batched_tokens=8192, enable_chunked_prefill=True
        )
        spy = ScheduleSpy(get_scheduler(engine))
        tf = TokenFactory(engine)
        try:
            for P in E1_PREFIX_LENS:
                total = E1_WARMUP + E1_REPEATS
                for r in range(total):
                    phase = "warmup" if r < E1_WARMUP else "measure"
                    fam = f"族{P}_{r}："  # 每 repeat 独立前缀族，避免跨 repeat 串扰
                    prefix_ids = tf.ids(P, uniq=fam)
                    # cold：前缀 + 查询A（填缓存）；hot：同前缀 + 查询B（命中前缀）
                    cold = (prefix_ids + tf.ids(P + E1_SUFFIX_LEN, uniq=fam)[P:])[
                        : P + E1_SUFFIX_LEN
                    ]
                    hot = prefix_ids + tf.ids(P + E1_SUFFIX_LEN, uniq=fam + "B")[P:]
                    hot = hot[: P + E1_SUFFIX_LEN]
                    seq = (
                        [("cold", cold), ("hot", hot)] if prefix_on else [("off", hot)]
                    )
                    for kind, ids in seq:
                        lat, sched, actual = _prefill_once(
                            engine, spy, f"e1-{P}-{kind}-{r}", ids
                        )
                        hit = actual - sched if sched >= 0 else None
                        samples.append(
                            {
                                "experiment": "E1_prefix",
                                "prefix_caching": "on" if prefix_on else "off",
                                "prefix_len": P,
                                "kind": kind,
                                "repeat": r,
                                "phase": phase,
                                "actual_prompt_tokens": actual,
                                "scheduled_tokens": sched,
                                "hit_tokens": hit,
                                "chunked": sched > actual,
                                "latency_ms": lat,
                            }
                        )
        finally:
            spy.restore()
            del engine
            torch.cuda.empty_cache()
    return samples


# ============================ E3：chunked prefill ============================
def run_e3():
    samples = []
    for chunked in E3_CHUNKED:
        for mbt in E3_BATCHED_TOKENS:
            # 关闭 APC 以隔离 chunked 行为（前缀命中会改变待算 token 数）
            engine = build_engine(
                enable_prefix_caching=False,
                max_num_batched_tokens=mbt,
                enable_chunked_prefill=chunked,
            )
            spy = ScheduleSpy(get_scheduler(engine))
            tf = TokenFactory(engine)
            try:
                ids = tf.ids(E3_PROMPT_LEN, uniq=f"chunk{mbt}{chunked}：")
                engine.add_request(
                    "e3",
                    {"prompt_token_ids": ids},
                    SamplingParams(
                        temperature=0.0, max_tokens=1, ignore_eos=False, seed=SEED
                    ),
                )
                step_sched = []  # 每 step 调度的 token 数（分块序列）
                t0 = time.perf_counter()
                torch.cuda.synchronize()
                steps = 0
                while engine.has_unfinished_requests() and steps < 10000:
                    engine.step()
                    so = spy.last
                    if so is not None and int(so.total_num_scheduled_tokens) > 0:
                        step_sched.append(int(so.total_num_scheduled_tokens))
                    steps += 1
                torch.cuda.synchronize()
                ttft_ms = round((time.perf_counter() - t0) * 1000, 4)
                samples.append(
                    {
                        "experiment": "E3_chunked",
                        "enable_chunked_prefill": chunked,
                        "max_num_batched_tokens": mbt,
                        "prompt_len": E3_PROMPT_LEN,
                        "num_chunks": len(step_sched),
                        "chunk_sizes": step_sched,
                        "sum_scheduled": sum(step_sched),
                        "ttft_proxy_ms": ttft_ms,
                    }
                )
            finally:
                spy.restore()
                del engine
                torch.cuda.empty_cache()
    return samples


# ============================ E2：block 生命周期（TODO，需验证内省 API） ==========
def run_e2():
    """TODO(GPU 空闲后)：通过 get_scheduler(engine).kv_cache_manager / block_pool 读
    KVCacheBlock.ref_cnt、free_block_queue、cached_block_hash_to_block，跨"分配→命中→
    释放→cached-free→驱逐"记录 block 状态表，验证 §2 的尾部优先淘汰/前缀优先保留。
    先在真实引擎上确认这些属性的可读路径（block_pool.py:82/118/182），再实现。"""
    return []


def summarize(e1, e3):
    os.makedirs(RAW, exist_ok=True)
    with open(os.path.join(RAW, "e1.jsonl"), "w") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in e1)
    with open(os.path.join(RAW, "e3.jsonl"), "w") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in e3)

    # E1 汇总：按 (prefix_caching, prefix_len, kind) 聚合 measure 相的 median latency + hit
    agg = {}
    for s in e1:
        if s["phase"] != "measure":
            continue
        k = (s["prefix_caching"], s["prefix_len"], s["kind"])
        agg.setdefault(
            k,
            {
                "lat": [],
                "hit": s["hit_tokens"],
                "sched": s["scheduled_tokens"],
                "actual": s["actual_prompt_tokens"],
            },
        )
        agg[k]["lat"].append(s["latency_ms"])
    with open(os.path.join(OUT, "prefix_cache_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prefix_caching",
                "prefix_len",
                "kind",
                "n",
                "latency_ms_median",
                "latency_ms_stdev",
                "hit_tokens",
                "scheduled_tokens",
                "actual_prompt_tokens",
            ]
        )
        for (pc, P, kind), v in sorted(agg.items()):
            lat = v["lat"]
            w.writerow(
                [
                    pc,
                    P,
                    kind,
                    len(lat),
                    round(statistics.median(lat), 4),
                    round(statistics.stdev(lat), 4) if len(lat) > 1 else 0.0,
                    v["hit"],
                    v["sched"],
                    v["actual"],
                ]
            )

    with open(os.path.join(OUT, "chunked_prefill.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "enable_chunked_prefill",
                "max_num_batched_tokens",
                "prompt_len",
                "num_chunks",
                "sum_scheduled",
                "ttft_proxy_ms",
                "chunk_sizes",
            ]
        )
        for s in e3:
            w.writerow(
                [
                    s["enable_chunked_prefill"],
                    s["max_num_batched_tokens"],
                    s["prompt_len"],
                    s["num_chunks"],
                    s["sum_scheduled"],
                    s["ttft_proxy_ms"],
                    s["chunk_sizes"],
                ]
            )
    print(
        "[m17] wrote:",
        os.path.join(OUT, "prefix_cache_sweep.csv"),
        "and",
        os.path.join(OUT, "chunked_prefill.csv"),
    )


def main():
    p = argparse.ArgumentParser(
        description="M17 prefix caching & chunked prefill (needs GPU)"
    )
    p.add_argument("--only", choices=["e1", "e3", "all"], default="all")
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    e1 = run_e1() if a.only in ("e1", "all") else []
    e3 = run_e3() if a.only in ("e3", "all") else []
    summarize(e1, e3)
    print(
        "[m17] done. 逐条核对不变量 I1-I5，见 notes/m17/prefix_cache_and_chunked_prefill.md §4"
    )


if __name__ == "__main__":
    main()
