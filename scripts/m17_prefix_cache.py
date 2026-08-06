"""M17 · Prefix Caching & Chunked Prefill —— 实验脚本（E1 / E3-single / E3-concurrency）。

============================================================================
状态：**本脚本已实测运行过**（结论见 notes/m17/results.md）。
运行需 GPU（会建真实 vLLM 引擎）。

**必须用 --shard 分片跑：一个进程只建一个引擎。**
原因：vLLM 引擎的 `del engine + torch.cuda.empty_cache()` **不足以归还 KV cache
显存**，同一进程内建第二个引擎会因显存不足失败（nvidia-smi 显示空闲、但 vLLM 只看到
一小部分）。这与 M6 的做法一致（scripts/m6_run_all.py 每配置起一个子进程）。

真实可运行命令（逐条执行；shard 之间留几秒让显存归还）：
    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
    python scripts/m17_prefix_cache.py --shard e1-on
    python scripts/m17_prefix_cache.py --shard e1-off
    python scripts/m17_prefix_cache.py --shard e3 --e3-chunked on  --e3-mbt 512
    python scripts/m17_prefix_cache.py --shard e3 --e3-chunked on  --e3-mbt 1024
    python scripts/m17_prefix_cache.py --shard e3 --e3-chunked on  --e3-mbt 2048
    python scripts/m17_prefix_cache.py --shard e3 --e3-chunked on  --e3-mbt 4096
    python scripts/m17_prefix_cache.py --shard e3 --e3-chunked off --e3-mbt 4096
    python scripts/m17_prefix_cache.py --shard e3-conc --conc-mode chunked
    python scripts/m17_prefix_cache.py --shard e3-conc --conc-mode nochunk
    python scripts/m17_prefix_cache.py --shard summarize    # 不建引擎、不需 GPU

设计 / 源码定位 / 不变量：notes/m17/prefix_cache_and_chunked_prefill.md
============================================================================

复用 M6 已验证的闭环内省法（见 scripts/m6_benchmark.py）：
  - LLMEngine + ScheduleSpy 抓 SchedulerOutput；
  - 命中证据 = actual_prompt_tokens - scheduled_prompt_tokens（命中 → scheduled < actual）；
  - **注意区分两种 scheduled < actual 的原因**：
      · E1（APC）里是 **prefix 命中**导致的（每请求仍只跑 1 个 prefill step）
        → 记 `prefix_cache_hit` + `prefill_step_count`，**不写 chunked 字段**；
      · E3 里才是**未命中部分被切成多个 chunk** → 以 **prefill step 数 > 1** 为
        「请求真的跨多个 prefill step」的最终证据，并按每请求 num_scheduled_tokens
        累计到 prompt 算完为止（避免把 decode step 误计为 chunk）。
  - 计时：先 torch.cuda.synchronize() 再取 perf_counter()，收尾同样先 synchronize()。

**指标口径（重要，勿与 M12 的在线 TTFT 混淆）**：本脚本所有延迟都是
**进程内 engine 路径**测量（in-process），**不含** HTTP / 网络 / tokenization /
异步服务队列 / 跨请求排队。在线 TTFT 口径见 M12。

E1：长共享前缀 on/off —— 随共享前缀长度 P 增大，看 APC 省出的 prefill 延迟能否超过
    运行噪声（M6 短前缀时落在噪声内）。
E3-single：长 prompt 按预算分块，验证分块守恒 + 单请求下分块的固定成本。
E3-concurrency：D 稳定 decode 后插入长 prefill，测 D 的 ITL 抖动 —— chunked prefill
    的核心卖点（公平单变量：两侧同 max_num_batched_tokens，仅 chunked 开关 +
    long_prefill_token_threshold 不同）。
E2（block 生命周期 ref_cnt/驱逐）：可选未做（需先验证 block_pool 内省 API）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
# E1：共享前缀长度（16 对齐），各带独立短后缀打破完全命中
E1_PREFIX_LENS = [256, 512, 1024, 2048]
E1_SUFFIX_LEN = 16
E1_WARMUP = 5
E1_REPEATS = 15
# E3-single：长 prompt 分块
E3_PROMPT_LEN = 2048
E3_WARMUP = 5
E3_REPEATS = 15
# E3-concurrency：D 稳定 decode 后插入长 prefill
CONC_DECODE_MAX_TOKENS = 96
CONC_DECODE_PROMPT_LEN = 64
CONC_LONG_PROMPT_LEN = 2048
CONC_MBT = 4096  # 两侧相同的全局 token 预算（保证单变量）
CONC_LONG_THRESHOLD = 512  # 仅 chunked 侧：强制长 prefill 每步最多 512 token
CONC_WARMUP = 5
CONC_REPEATS = 15

_LONG = (
    "推理引擎的连续批处理与分页注意力机制通过按块管理键值缓存来提升吞吐与显存利用率。"
    "The quick brown fox jumps over the lazy dog. "
)


def _h(ids: list[int]) -> str:
    """token 序列的短哈希（用于记录前缀/后缀身份，不把全量 ids 写进 CSV）。"""
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


# ============================ 引擎与内省（复用 M6 模式） ============================
def build_engine(
    enable_prefix_caching: bool,
    max_num_batched_tokens: int,
    enable_chunked_prefill: bool,
    long_prefill_token_threshold: int = 0,
    max_num_partial_prefills: int = 1,
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
        long_prefill_token_threshold=long_prefill_token_threshold,
        max_num_partial_prefills=max_num_partial_prefills,
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
    """跑一个 engine.step()，返回 (秒, outputs)。sync 在计时开始前，避免把先前排队工作计入。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = engine.step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0), outs


def drain(engine: LLMEngine):
    while engine.has_unfinished_requests():
        engine.step()


def _same_req(observed: str, rid: str) -> bool:
    """输出里的 request_id 可能带 vLLM 追加的后缀，故容错匹配。"""
    return observed == rid or observed.startswith(rid + "-")


def _sched_for(so, rid: str) -> int:
    """该 step 为 rid 调度的 token 数（0 表示这步没调度它）。

    注意：vLLM 会给 add_request 的 id 追加后缀（"p" -> "p-8ef2061f"），
    因此按「精确匹配或以 rid + '-' 开头」查找，不能直接 dict.get(rid)——
    否则所有 per-request 计数会静默变成 0。
    """
    if so is None:
        return 0
    d = so.num_scheduled_tokens
    if rid in d:
        return int(d[rid])
    return int(sum(v for k, v in d.items() if _same_req(k, rid)))


def _prefill_once(engine, spy, rid, ids):
    """加一个 max_tokens=1 的请求跑一个 step，返回 (latency_ms, scheduled_for_rid, actual)。"""
    engine.add_request(
        rid,
        {"prompt_token_ids": ids},
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=False, seed=SEED),
    )
    dt, outs = timed_step(engine)
    scheduled = _sched_for(spy.last, rid)
    actual = len(outs[0].prompt_token_ids) if outs else len(ids)
    drain(engine)
    return round(dt * 1000, 4), scheduled, actual


# ============================ E1：长共享前缀 on/off ============================
def run_e1(prefix_on: bool):
    """跑一侧（APC on 或 off）。一个进程只建一个引擎。"""
    samples = []
    engine = build_engine(
        enable_prefix_caching=prefix_on,
        max_num_batched_tokens=8192,
        enable_chunked_prefill=True,
    )
    spy = ScheduleSpy(get_scheduler(engine))
    tf = TokenFactory(engine)
    try:
        for P in E1_PREFIX_LENS:
            for r in range(E1_WARMUP + E1_REPEATS):
                phase = "warmup" if r < E1_WARMUP else "measure"
                # 每 repeat 独立前缀族：避免跨 repeat 意外互相命中
                fam = f"族{P}_{r}："
                prefix_ids = tf.ids(P, uniq=fam)
                sfx_a = tf.ids(P + E1_SUFFIX_LEN, uniq=fam)[P:]
                sfx_b = tf.ids(P + E1_SUFFIX_LEN, uniq=fam + "B")[P:]
                cold = (prefix_ids + sfx_a)[: P + E1_SUFFIX_LEN]
                hot = (prefix_ids + sfx_b)[: P + E1_SUFFIX_LEN]
                # token 构造正确性断言（cold/hot 共享前缀、后缀必须不同）
                assert len(cold) == P + E1_SUFFIX_LEN, f"cold len {len(cold)}"
                assert len(hot) == P + E1_SUFFIX_LEN, f"hot len {len(hot)}"
                assert cold[:P] == hot[:P], "cold/hot 前缀必须完全相同"
                assert cold[P:] != hot[P:], "cold/hot 后缀必须不同（否则会完全命中）"

                seq = [("cold", cold), ("hot", hot)] if prefix_on else [("off", hot)]
                for kind, ids in seq:
                    lat, sched, actual = _prefill_once(
                        engine, spy, f"e1-{P}-{kind}-{r}", ids
                    )
                    hit = actual - sched if sched >= 0 else None
                    samples.append(
                        {
                            "experiment": "E1_prefix",
                            "metric_scope": "in-process",
                            "prefix_caching": "on" if prefix_on else "off",
                            "prefix_len": P,
                            "kind": kind,
                            "repeat": r,
                            "phase": phase,
                            "actual_prompt_tokens": actual,
                            "scheduled_prompt_tokens": sched,
                            "hit_tokens": hit,
                            # E1 里 scheduled < actual 的原因是 **prefix 命中**，
                            # 不是「未命中部分被切成多个 chunk」。E1 每个请求只跑
                            # 一个 prefill step（_prefill_once 只调一次 timed_step），
                            # 所以这里不写 chunked 字段，改用两个语义明确的字段。
                            "prefix_cache_hit": bool(hit) and hit > 0,
                            "prefill_step_count": 1,
                            "in_process_first_token_latency_ms": lat,
                            "prefix_hash": _h(prefix_ids),
                            "suffix_hash": _h(list(ids[P:])),
                            "success": True,
                            "invalid_reason": "",
                        }
                    )
    finally:
        spy.restore()
        del engine
        torch.cuda.empty_cache()
    return samples


# ======================= E3-single：长 prompt 分块（重复测量） =======================
def run_e3_single(chunked: bool, mbt: int):
    """单一配置（一个进程一个引擎）。含 warmup，measure 阶段不含首次 JIT。"""
    samples = []
    # 关闭 APC 以隔离 chunked 行为（前缀命中会改变待算 token 数）
    engine = build_engine(
        enable_prefix_caching=False,
        max_num_batched_tokens=mbt,
        enable_chunked_prefill=chunked,
    )
    spy = ScheduleSpy(get_scheduler(engine))
    tf = TokenFactory(engine)
    try:
        for r in range(E3_WARMUP + E3_REPEATS):
            phase = "warmup" if r < E3_WARMUP else "measure"
            rid = f"e3-{int(chunked)}-{mbt}-{r}"
            # 每 repeat 换前缀，避免（即便 APC 关着）任何跨轮复用
            ids = tf.ids(E3_PROMPT_LEN, uniq=f"chunk{mbt}{chunked}_{r}：")
            engine.add_request(
                rid,
                {"prompt_token_ids": ids},
                SamplingParams(
                    temperature=0.0, max_tokens=1, ignore_eos=False, seed=SEED
                ),
            )
            prefill_chunks: list[int] = []  # 仅 prefill step 的调度量
            decode_steps = 0
            done_prompt = 0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            first_token_s = None
            steps = 0
            while engine.has_unfinished_requests() and steps < 10000:
                outs = engine.step()
                n = _sched_for(spy.last, rid)
                if n > 0:
                    # prefill 阶段：累计到 prompt 全部算完为止；之后每步 1 token = decode
                    if done_prompt < E3_PROMPT_LEN:
                        prefill_chunks.append(n)
                        done_prompt += n
                    else:
                        decode_steps += 1
                if first_token_s is None and outs:
                    torch.cuda.synchronize()
                    first_token_s = time.perf_counter() - t0
                steps += 1
            torch.cuda.synchronize()
            completion_s = time.perf_counter() - t0

            ok = sum(prefill_chunks) == E3_PROMPT_LEN and all(
                c <= mbt for c in prefill_chunks
            )
            samples.append(
                {
                    "experiment": "E3_single",
                    "metric_scope": "in-process",
                    "enable_chunked_prefill": chunked,
                    "max_num_batched_tokens": mbt,
                    "prompt_len": E3_PROMPT_LEN,
                    "repeat": r,
                    "phase": phase,
                    "prefill_step_count": len(prefill_chunks),
                    "chunk_sizes": prefill_chunks,
                    "sum_prefill_scheduled": sum(prefill_chunks),
                    "decode_steps": decode_steps,
                    "in_process_first_token_latency_ms": (
                        round(first_token_s * 1000, 4)
                        if first_token_s is not None
                        else None
                    ),
                    "request_completion_latency_ms": round(completion_s * 1000, 4),
                    "success": ok,
                    "invalid_reason": (
                        ""
                        if ok
                        else f"sum={sum(prefill_chunks)} != {E3_PROMPT_LEN} or chunk>mbt"
                    ),
                }
            )
    finally:
        spy.restore()
        del engine
        torch.cuda.empty_cache()
    return samples


# =============== E3-concurrency：D 稳定 decode 后插入长 prefill（ITL 抖动） ===============
def run_e3_concurrency(mode: str):
    """mode='chunked'：chunked on + long_prefill_token_threshold 强制分块；
    mode='nochunk'：chunked off，长 prefill 单步算完。
    两侧 max_num_batched_tokens 相同（CONC_MBT），保证单变量。"""
    chunked = mode == "chunked"
    samples = []
    engine = build_engine(
        enable_prefix_caching=False,
        max_num_batched_tokens=CONC_MBT,
        enable_chunked_prefill=chunked,
        long_prefill_token_threshold=CONC_LONG_THRESHOLD if chunked else 0,
        # 必须保持默认 1：本版本 arg_utils.py:2350 对非默认值直接
        # raise NotImplementedError("Concurrent Partial Prefill")。已实测
        # long_prefill_token_threshold 单独即可强制长 prefill 分块（[512]*4）。
        max_num_partial_prefills=1,
    )
    spy = ScheduleSpy(get_scheduler(engine))
    tf = TokenFactory(engine)
    sched_cfg = get_scheduler(engine).scheduler_config
    eff = {
        "enable_chunked_prefill": bool(sched_cfg.enable_chunked_prefill),
        "long_prefill_token_threshold": int(sched_cfg.long_prefill_token_threshold),
        "max_num_partial_prefills": int(sched_cfg.max_num_partial_prefills),
        "max_num_batched_tokens": int(sched_cfg.max_num_batched_tokens),
    }
    print(f"[m17] effective scheduler cfg: {eff}")
    try:
        for r in range(CONC_WARMUP + CONC_REPEATS):
            phase = "warmup" if r < CONC_WARMUP else "measure"
            drid, prid = f"D-{mode}-{r}", f"P-{mode}-{r}"
            # 1) D 进入稳定 decode
            engine.add_request(
                drid,
                {"prompt_token_ids": tf.ids(CONC_DECODE_PROMPT_LEN, uniq=f"D{r}：")},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=CONC_DECODE_MAX_TOKENS,
                    ignore_eos=True,
                    seed=SEED,
                ),
            )
            for _ in range(4):  # 跑掉 D 的 prefill + 几步 decode，进入稳态
                engine.step()

            # 2) 插入长 prefill 请求 P，记录 D 每个输出 token 的间隔
            engine.add_request(
                prid,
                {"prompt_token_ids": tf.ids(CONC_LONG_PROMPT_LEN, uniq=f"P{r}：")},
                SamplingParams(
                    temperature=0.0, max_tokens=1, ignore_eos=False, seed=SEED
                ),
            )
            itls_ms: list[float] = []
            p_prefill_chunks: list[int] = []
            p_done = 0
            p_first_token_s = None
            steps_detail = []
            torch.cuda.synchronize()
            t_prev = time.perf_counter()
            t_start = t_prev
            steps = 0
            while engine.has_unfinished_requests() and steps < 10000:
                outs = engine.step()
                torch.cuda.synchronize()
                now = time.perf_counter()
                so = spy.last
                d_n, p_n = _sched_for(so, drid), _sched_for(so, prid)
                if p_n > 0 and p_done < CONC_LONG_PROMPT_LEN:
                    p_prefill_chunks.append(p_n)
                    p_done += p_n
                steps_detail.append(
                    {
                        "d": d_n,
                        "p": p_n,
                        "total": int(so.total_num_scheduled_tokens) if so else 0,
                    }
                )
                # D 产出一个 token → 记一个 ITL
                if any(_same_req(o.request_id, drid) for o in outs):
                    itls_ms.append((now - t_prev) * 1000)
                    t_prev = now
                if p_first_token_s is None and any(
                    _same_req(o.request_id, prid) for o in outs
                ):
                    p_first_token_s = now - t_start
                steps += 1
            torch.cuda.synchronize()
            total_s = time.perf_counter() - t_start

            ok = sum(p_prefill_chunks) == CONC_LONG_PROMPT_LEN and len(itls_ms) >= 2
            samples.append(
                {
                    "experiment": "E3_concurrency",
                    "metric_scope": "in-process",
                    "mode": mode,
                    "effective_cfg": eff,
                    "repeat": r,
                    "phase": phase,
                    "decode_itl_samples": len(itls_ms),
                    "decode_itl_median_ms": (
                        round(statistics.median(itls_ms), 4) if itls_ms else None
                    ),
                    "decode_itl_p95_ms": (
                        round(sorted(itls_ms)[int(0.95 * (len(itls_ms) - 1))], 4)
                        if itls_ms
                        else None
                    ),
                    "decode_itl_max_ms": (round(max(itls_ms), 4) if itls_ms else None),
                    "long_prefill_step_count": len(p_prefill_chunks),
                    "long_prefill_chunk_sizes": p_prefill_chunks,
                    "long_prefill_sum_scheduled": sum(p_prefill_chunks),
                    "long_prefill_first_token_latency_ms": (
                        round(p_first_token_s * 1000, 4)
                        if p_first_token_s is not None
                        else None
                    ),
                    "total_latency_ms": round(total_s * 1000, 4),
                    "steps_detail": steps_detail
                    if phase == "measure" and r == CONC_WARMUP
                    else None,  # 只留一份逐步明细，避免 JSONL 膨胀
                    "success": ok,
                    "invalid_reason": (
                        ""
                        if ok
                        else f"p_sum={sum(p_prefill_chunks)} or itl_n={len(itls_ms)}"
                    ),
                }
            )
    finally:
        spy.restore()
        del engine
        torch.cuda.empty_cache()
    return samples


# ============================ E2：block 生命周期（可选未做） ==========================
def run_e2():
    """可选 TODO：通过 block_pool 读 ref_cnt / free 队列 / cached_block_hash_to_block，
    跨「分配→命中→释放→cached-free→驱逐」记录 block 状态表。需先验证内省 API 可读。"""
    return []


# ============================ shard 落盘 / 汇总 ============================
def _dump(samples, tag):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{tag}.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
    print(f"[m17] shard written: {path} ({len(samples)} samples)")


def _load_shards(prefix):
    out = []
    if not os.path.isdir(RAW):
        return out
    for name in sorted(os.listdir(RAW)):
        if name.startswith(prefix) and name.endswith(".jsonl"):
            with open(os.path.join(RAW, name)) as f:
                out.extend(json.loads(line) for line in f if line.strip())
    return out


def _stats(xs):
    """median/mean/stdev/p95/min/max。"""
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
    """median(a) - median(b) 的 Bootstrap 95% CI（固定种子，仅用 stdlib random）。"""
    import random

    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.median(ra) - statistics.median(rb))
    diffs.sort()
    lo = diffs[int(0.025 * (iters - 1))]
    hi = diffs[int(0.975 * (iters - 1))]
    return (
        round(statistics.median(a) - statistics.median(b), 4),
        round(lo, 4),
        round(hi, 4),
    )


def _measure(rows):
    """只取有效 measure 样本（排除 warmup 与失败样本）。"""
    return [r for r in rows if r.get("phase") == "measure" and r.get("success", True)]


def summarize():
    """由已落盘 shard 汇总成 CSV（不建引擎、不需 GPU）。"""
    os.makedirs(OUT, exist_ok=True)
    e1 = _measure(_load_shards("e1_"))
    e3 = _measure(_load_shards("e3_single_"))
    conc = _measure(_load_shards("e3_conc_"))

    # ---------------- E1 ----------------
    if e1:
        groups: dict[tuple, list] = {}
        for r in e1:
            groups.setdefault(
                (r["prefix_caching"], r["prefix_len"], r["kind"]), []
            ).append(r)
        lat_by_key = {
            k: [x["in_process_first_token_latency_ms"] for x in v]
            for k, v in groups.items()
        }
        path = os.path.join(OUT, "prefix_cache_sweep.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "prefix_caching",
                    "prefix_len",
                    "kind",
                    "n",
                    "lat_median_ms",
                    "lat_mean_ms",
                    "lat_stdev_ms",
                    "lat_p95_ms",
                    "lat_min_ms",
                    "lat_max_ms",
                    "hit_tokens",
                    "scheduled_prompt_tokens",
                    "actual_prompt_tokens",
                    "metric_scope",
                ]
            )
            for k in sorted(groups):
                v = groups[k]
                st = _stats(lat_by_key[k])
                w.writerow(
                    [
                        k[0],
                        k[1],
                        k[2],
                        st["n"],
                        st["median"],
                        st["mean"],
                        st["stdev"],
                        st["p95"],
                        st["min"],
                        st["max"],
                        v[0]["hit_tokens"],
                        v[0]["scheduled_prompt_tokens"],
                        v[0]["actual_prompt_tokens"],
                        "in-process",
                    ]
                )
        print(f"[m17] wrote {path}")

        # APC 收益：hot vs off，Bootstrap 95% CI（不用 σ）
        path2 = os.path.join(OUT, "prefix_cache_effect.csv")
        with open(path2, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "prefix_len",
                    "off_median_ms",
                    "hot_median_ms",
                    "median_diff_ms",
                    "ci95_lo_ms",
                    "ci95_hi_ms",
                    "ci_excludes_zero",
                    "rel_reduction_pct",
                    "off_stdev_ms",
                    "hot_stdev_ms",
                    "n_per_group",
                ]
            )
            for P in E1_PREFIX_LENS:
                ka, kb = ("off", P, "off"), ("on", P, "hot")
                if ka not in lat_by_key or kb not in lat_by_key:
                    continue
                off_l, hot_l = lat_by_key[ka], lat_by_key[kb]
                d, lo, hi = _boot_ci_median_diff(off_l, hot_l)
                om, hm = statistics.median(off_l), statistics.median(hot_l)
                w.writerow(
                    [
                        P,
                        round(om, 4),
                        round(hm, 4),
                        d,
                        lo,
                        hi,
                        "yes" if (lo > 0 or hi < 0) else "no",
                        round((om - hm) / om * 100, 2),
                        round(statistics.stdev(off_l), 4),
                        round(statistics.stdev(hot_l), 4),
                        len(off_l),
                    ]
                )
        print(f"[m17] wrote {path2}")

    # ---------------- E3-single ----------------
    if e3:
        groups3: dict[tuple, list] = {}
        for r in e3:
            groups3.setdefault(
                (r["enable_chunked_prefill"], r["max_num_batched_tokens"]), []
            ).append(r)
        path = os.path.join(OUT, "chunked_prefill.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "enable_chunked_prefill",
                    "max_num_batched_tokens",
                    "prompt_len",
                    "prefill_step_count",
                    "chunk_sizes",
                    "sum_prefill_scheduled",
                    "n",
                    "ttft_median_ms",
                    "ttft_mean_ms",
                    "ttft_stdev_ms",
                    "ttft_p95_ms",
                    "ttft_min_ms",
                    "ttft_max_ms",
                    "metric_scope",
                ]
            )
            for k in sorted(groups3, key=lambda x: (x[0], x[1])):
                v = groups3[k]
                st = _stats([x["in_process_first_token_latency_ms"] for x in v])
                w.writerow(
                    [
                        k[0],
                        k[1],
                        v[0]["prompt_len"],
                        v[0]["prefill_step_count"],
                        v[0]["chunk_sizes"],
                        v[0]["sum_prefill_scheduled"],
                        st["n"],
                        st["median"],
                        st["mean"],
                        st["stdev"],
                        st["p95"],
                        st["min"],
                        st["max"],
                        "in-process",
                    ]
                )
        print(f"[m17] wrote {path}")

    # ---------------- E3-concurrency ----------------
    if conc:
        groupsc: dict[str, list] = {}
        for r in conc:
            groupsc.setdefault(r["mode"], []).append(r)
        path = os.path.join(OUT, "chunked_prefill_concurrency.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "mode",
                    "n",
                    "long_prefill_step_count",
                    "long_prefill_chunk_sizes",
                    "itl_median_of_medians_ms",
                    "itl_p95_median_ms",
                    "itl_max_median_ms",
                    "itl_max_worst_ms",
                    "long_prefill_ttft_median_ms",
                    "total_latency_median_ms",
                    "effective_cfg",
                    "metric_scope",
                ]
            )
            for m in sorted(groupsc):
                v = groupsc[m]
                med = [
                    x["decode_itl_median_ms"] for x in v if x["decode_itl_median_ms"]
                ]
                p95 = [x["decode_itl_p95_ms"] for x in v if x["decode_itl_p95_ms"]]
                mx = [x["decode_itl_max_ms"] for x in v if x["decode_itl_max_ms"]]
                ttft = [
                    x["long_prefill_first_token_latency_ms"]
                    for x in v
                    if x["long_prefill_first_token_latency_ms"]
                ]
                tot = [x["total_latency_ms"] for x in v]
                w.writerow(
                    [
                        m,
                        len(v),
                        v[0]["long_prefill_step_count"],
                        v[0]["long_prefill_chunk_sizes"],
                        round(statistics.median(med), 4) if med else None,
                        round(statistics.median(p95), 4) if p95 else None,
                        round(statistics.median(mx), 4) if mx else None,
                        round(max(mx), 4) if mx else None,
                        round(statistics.median(ttft), 4) if ttft else None,
                        round(statistics.median(tot), 4) if tot else None,
                        json.dumps(v[0]["effective_cfg"], ensure_ascii=False),
                        "in-process",
                    ]
                )
        print(f"[m17] wrote {path}")

    print(
        f"[m17] summarize done. measure samples: E1={len(e1)} "
        f"E3-single={len(e3)} E3-conc={len(conc)}"
    )


def main():
    p = argparse.ArgumentParser(
        description="M17 prefix caching & chunked prefill (需 GPU；一个进程一个引擎)"
    )
    p.add_argument(
        "--shard",
        choices=["e1-on", "e1-off", "e3", "e3-conc", "summarize"],
        required=True,
    )
    p.add_argument("--e3-chunked", choices=["on", "off"], help="--shard e3 必填")
    p.add_argument("--e3-mbt", type=int, help="--shard e3 必填")
    p.add_argument(
        "--conc-mode", choices=["chunked", "nochunk"], help="--shard e3-conc 必填"
    )
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.shard == "e1-on":
        _dump(run_e1(prefix_on=True), "e1_on")
    elif a.shard == "e1-off":
        _dump(run_e1(prefix_on=False), "e1_off")
    elif a.shard == "e3":
        # 强制单配置：一个进程只建一个引擎（否则第二个引擎会因显存无法归还而失败）
        if a.e3_chunked is None or a.e3_mbt is None:
            p.error(
                "--shard e3 必须同时提供 --e3-chunked {on,off} 与 --e3-mbt <int>："
                "一个进程只能建一个引擎"
            )
        ch = a.e3_chunked == "on"
        _dump(run_e3_single(ch, a.e3_mbt), f"e3_single_{a.e3_chunked}_{a.e3_mbt}")
    elif a.shard == "e3-conc":
        if a.conc_mode is None:
            p.error("--shard e3-conc 必须提供 --conc-mode {chunked,nochunk}")
        _dump(run_e3_concurrency(a.conc_mode), f"e3_conc_{a.conc_mode}")
    else:
        summarize()


if __name__ == "__main__":
    main()
