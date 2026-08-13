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

后续追加的两组实验（**实现完成，待 GPU 执行**；同样一个进程一个引擎）：
    # P0 · E3 no-chunk 生成正确性对照（补 M17 未验证的一环）
    python scripts/m17_prefix_cache.py --shard e3-correctness --conc-mode chunked
    python scripts/m17_prefix_cache.py --shard e3-correctness --conc-mode nochunk
    python scripts/m17_prefix_cache.py --shard e3-correctness-summarize   # 不建引擎

    # P2 · I5-b 输出长度定量实验
    python scripts/m17_prefix_cache.py --shard e1-output-len-on
    python scripts/m17_prefix_cache.py --shard e1-output-len-off
    python scripts/m17_prefix_cache.py --shard e1-output-len-summarize    # 不建引擎

**新增 shard 一律写独立文件**（e3_correctness.csv / apc_output_length*.csv），
不并入、不覆盖 M17 已核验的 prefix_cache_*.csv 与 chunked_prefill*.csv。

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

# ---- P0 · E3 no-chunk 生成正确性对照（chunked vs nochunk 的输出 token 是否逐一相同）----
# 动机：上游在 runner_type=="generate" 且手动 enable_chunked_prefill=False 时告警
# （vllm/engine/arg_utils.py:2476-2480：'This model does not officially support
# disabling chunked prefill. Disabling this manually may cause the engine to crash
# or produce incorrect outputs.'）。M17 E3 只验证了调度守恒与延迟，**没有比较两侧
# 生成的 token**——本 shard 专门补这一环。
CORR_MBT = 4096  # 两侧相同
CORR_LONG_THRESHOLD = 512  # 仅 chunked 侧
CORR_A_PROMPT_LEN = 2048  # 场景 A：单个长请求
CORR_A_MAX_TOKENS = 64
CORR_B_D_PROMPT_LEN = 64  # 场景 B：复现 E3-concurrency 的 D + P
CORR_B_D_MAX_TOKENS = 96
CORR_B_D_PRIMING_STEPS = 4  # 与 run_e3_concurrency 一致：先让 D 进入稳定 decode
CORR_B_P_PROMPT_LEN = 2048
CORR_B_P_MAX_TOKENS = 64
CORR_WARMUP = 1  # 吸收首次 Triton JIT；warmup 样本不参与比对
CORR_REPEATS = 3

# ---- P2 · I5-b 输出长度定量实验（APC 只省 prefill 不省 decode 的定量边界）----
# 冻结网格：只有当 128 档仍看不到相对收益下降趋势时，才可**再次征求用户确认**后追加 256；
# 严禁跑完再挑有利档位。
I5B_PREFIX_LEN = 2048
I5B_SUFFIX_LEN = 16
I5B_MAX_TOKENS = [1, 8, 32, 128]
I5B_MBT = 8192  # 与 E1 同侧配置（2064-token prompt 单步算完，prefill_step_count=1）
I5B_WARMUP = 5
I5B_REPEATS = 15

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


class OutputCollector:
    """累积每个请求的完整输出 token 序列。

    vLLM 的 `SamplingParams.output_kind` 默认是 **CUMULATIVE**
    （`vllm/sampling_params.py:301`），因此每次 `RequestOutput` 携带的
    `outputs[0].token_ids` 是**到目前为止的全量序列**而非增量
    （`vllm/v1/engine/output_processor.py:389-390`：非 DELTA 时直接取
    `detokenizer.output_token_ids`）。

    两个必须防的坑：
      1. `token_ids` 是引擎内部会继续 append 的**同一个 list 对象** → 必须 `list(...)` 拷贝，
         否则记录下来的"历史快照"会被后续 step 就地改写。
      2. 若上游哪天改成 DELTA 语义，长度会突然变短——此时直接把样本标为失败并记录原因，
         **不能让它静默产生一个"看起来相等"的错误结论**。
    """

    def __init__(self):
        self.by_req: dict[str, dict] = {}
        self.semantics_violation: str = ""

    def observe(self, outs, rids):
        for o in outs:
            for rid in rids:
                if not _same_req(o.request_id, rid):
                    continue
                if not o.outputs:
                    continue
                co = o.outputs[0]
                ids = list(co.token_ids)  # 必须拷贝：CUMULATIVE 复用同一 list 对象
                prev = self.by_req.get(rid)
                if prev is not None and len(ids) < len(prev["token_ids"]):
                    # 长度回退 → 不是 CUMULATIVE 语义，本次比对不可信
                    self.semantics_violation = (
                        f"{rid}: token_ids 长度回退 "
                        f"{len(prev['token_ids'])} -> {len(ids)}"
                        "（疑似 DELTA 语义，累积逻辑失效）"
                    )
                self.by_req[rid] = {
                    "token_ids": ids,
                    "finished": bool(o.finished),
                    "finish_reason": co.finish_reason,
                    "prompt_token_ids": list(o.prompt_token_ids or []),
                }

    def get(self, rid) -> dict:
        return self.by_req.get(
            rid,
            {
                "token_ids": [],
                "finished": False,
                "finish_reason": None,
                "prompt_token_ids": [],
            },
        )


def _run_request_to_completion(engine, spy, rid, ids, max_tokens, collector=None):
    """跑单个多 token 请求直到结束，返回逐样本计时与调度事实。

    prefill / decode 的分界：**该请求产出第一个 token 之前**被调度的 token 计入
    prefill chunk，之后每步 1 token 计入 decode——与 E3-single 同口径，避免把
    decode step 误计为 prompt chunk。

    计时：先 `torch.cuda.synchronize()` 再取 t0；首 token 与收尾各再 sync 一次。
    只在这两个点 sync（不逐 step sync），以免把测量开销灌进 decode 路径。
    """
    coll = collector if collector is not None else OutputCollector()
    engine.add_request(
        rid,
        {"prompt_token_ids": ids},
        SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True, seed=SEED
        ),
    )
    prefill_chunks: list[int] = []
    decode_steps = 0
    seen_first = False
    first_s = None
    steps = 0
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.has_unfinished_requests() and steps < 100000:
        outs = engine.step()
        n = _sched_for(spy.last, rid)
        if not seen_first:
            if n > 0:
                prefill_chunks.append(n)
        elif n > 0:
            decode_steps += 1
        coll.observe(outs, [rid])
        if not seen_first and any(_same_req(o.request_id, rid) for o in outs):
            torch.cuda.synchronize()
            first_s = time.perf_counter() - t0
            seen_first = True
        steps += 1
    torch.cuda.synchronize()
    completion_s = time.perf_counter() - t0

    got = coll.get(rid)
    return {
        "prefill_chunks": prefill_chunks,
        "prefill_step_count": len(prefill_chunks),
        "scheduled_prompt_tokens": sum(prefill_chunks),
        "decode_steps": decode_steps,
        "first_token_s": first_s,
        "completion_s": completion_s,
        "output_token_ids": got["token_ids"],
        "finished": got["finished"],
        "finish_reason": got["finish_reason"],
        "actual_prompt_tokens": len(got["prompt_token_ids"]) or len(ids),
        "collector": coll,
        "steps": steps,
    }


def _latency_fields(first_s, completion_s, n_out):
    """首 token / 完成 / 首 token 之后 / TPOT 四个口径（全为 in-process，非在线 E2E）。"""
    ft = None if first_s is None else round(first_s * 1000, 4)
    comp = round(completion_s * 1000, 4)
    post = None if ft is None else round(comp - ft, 4)
    if post is not None and n_out is not None and n_out > 1:
        tpot = round(post / (n_out - 1), 4)
    else:
        tpot = None  # 只有 1 个输出 token 时 TPOT 无定义
    return ft, comp, post, tpot


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


# ============ P0 · E3-correctness：chunked vs nochunk 的生成一致性（新增） ============
def _corr_sample(scenario, case, r, phase, mode, eff, ids, res, expected_out):
    """把一个请求的观测组装成逐样本记录（两侧共用同一份 schema，便于离线逐字段比对）。"""
    out_ids = res["output_token_ids"]
    n_out = len(out_ids)
    ft, comp, post, tpot = _latency_fields(
        res["first_token_s"], res["completion_s"], n_out
    )
    coll = res["collector"]
    ok = (
        n_out == expected_out
        and res["finished"]
        and n_out > 0
        and not coll.semantics_violation
    )
    reasons = []
    if n_out != expected_out:
        reasons.append(f"output_tokens={n_out} != expected {expected_out}")
    if not res["finished"]:
        reasons.append("request 未正常结束")
    if coll.semantics_violation:
        reasons.append(coll.semantics_violation)
    return {
        "experiment": "E3_correctness",
        "metric_scope": "in-process",
        "mode": mode,
        "scenario": scenario,
        "case": case,
        "repeat": r,
        "phase": phase,
        "effective_cfg": eff,
        "prompt_len": len(ids),
        "prompt_hash": _h(ids),
        "actual_prompt_tokens": res["actual_prompt_tokens"],
        "scheduled_prompt_tokens": res["scheduled_prompt_tokens"],
        "prefill_step_count": res["prefill_step_count"],
        "chunk_sizes": res["prefill_chunks"],
        "sum_prefill_scheduled": sum(res["prefill_chunks"]),
        "decode_steps": res["decode_steps"],
        "max_tokens_requested": expected_out,
        "actual_output_tokens": n_out,
        "output_token_ids": out_ids,
        "output_token_ids_hash": _h(out_ids) if out_ids else "",
        "finish_reason": res["finish_reason"],
        "in_process_first_token_latency_ms": ft,
        "request_completion_latency_ms": comp,
        "post_first_token_latency_ms": post,
        "tpot_ms": tpot,
        "success": ok,
        "invalid_reason": "; ".join(reasons),
    }


def run_e3_correctness(mode: str):
    """P0：验证「关闭 chunked prefill」这一侧生成的 token 与 chunked 侧是否逐一相同。

    背景：`enable_chunked_prefill=False` 在本模型上会触发上游告警
    （arg_utils.py:2476-2480，"may cause the engine to crash or produce incorrect
    outputs"）。M17 E3 已验证调度守恒与延迟，但**从未比较过两侧的输出 token**——
    调度守恒不等于生成正确。本 shard 补这一环。

    两个场景：
      A：单个长请求（prompt 2048 / max_tokens 64）；
      B：复现 E3-concurrency（D 稳定 decode 后插入长 prefill P），比较 D 与 P 的完整输出。

    prompt 由 TokenFactory 确定性构造（uniq 串**不含 mode**），故两个独立进程
    构造出的 prompt token ids 必然逐一相同——由落盘的 prompt_hash 事后核验，
    不依赖任何进程内随机状态。
    """
    chunked = mode == "chunked"
    samples = []
    engine = build_engine(
        enable_prefix_caching=False,  # 关 APC，隔离 chunked 行为
        max_num_batched_tokens=CORR_MBT,
        enable_chunked_prefill=chunked,
        long_prefill_token_threshold=CORR_LONG_THRESHOLD if chunked else 0,
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
    print(f"[m17] e3-correctness effective scheduler cfg: {eff}")
    try:
        for r in range(CORR_WARMUP + CORR_REPEATS):
            phase = "warmup" if r < CORR_WARMUP else "measure"

            # ---------------- 场景 A：单个长请求 ----------------
            a_ids = tf.ids(CORR_A_PROMPT_LEN, uniq=f"corrA_{r}：")
            res_a = _run_request_to_completion(
                engine, spy, f"corrA-{mode}-{r}", a_ids, CORR_A_MAX_TOKENS
            )
            samples.append(
                _corr_sample(
                    "A",
                    "single_long",
                    r,
                    phase,
                    mode,
                    eff,
                    a_ids,
                    res_a,
                    CORR_A_MAX_TOKENS,
                )
            )
            drain(engine)

            # ---------------- 场景 B：复现 E3-concurrency ----------------
            drid, prid = f"corrB-D-{mode}-{r}", f"corrB-P-{mode}-{r}"
            d_ids = tf.ids(CORR_B_D_PROMPT_LEN, uniq=f"corrBD_{r}：")
            p_ids = tf.ids(CORR_B_P_PROMPT_LEN, uniq=f"corrBP_{r}：")
            coll = OutputCollector()

            engine.add_request(
                drid,
                {"prompt_token_ids": d_ids},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=CORR_B_D_MAX_TOKENS,
                    ignore_eos=True,
                    seed=SEED,
                ),
            )
            d_chunks: list[int] = []
            d_first = False
            d_decode = 0
            torch.cuda.synchronize()
            t_d0 = time.perf_counter()
            d_first_s = None
            for _ in range(CORR_B_D_PRIMING_STEPS):  # D 进入稳定 decode
                outs = engine.step()
                n = _sched_for(spy.last, drid)
                if not d_first:
                    if n > 0:
                        d_chunks.append(n)
                elif n > 0:
                    d_decode += 1
                coll.observe(outs, [drid])
                if not d_first and any(_same_req(o.request_id, drid) for o in outs):
                    torch.cuda.synchronize()
                    d_first_s = time.perf_counter() - t_d0
                    d_first = True

            engine.add_request(  # 插入长 prefill P
                prid,
                {"prompt_token_ids": p_ids},
                SamplingParams(
                    temperature=0.0,
                    max_tokens=CORR_B_P_MAX_TOKENS,
                    ignore_eos=True,
                    seed=SEED,
                ),
            )
            p_chunks: list[int] = []
            p_first = False
            p_decode = 0
            p_first_s = None
            t_p0 = time.perf_counter()
            steps = 0
            while engine.has_unfinished_requests() and steps < 100000:
                outs = engine.step()
                dn, pn = _sched_for(spy.last, drid), _sched_for(spy.last, prid)
                if not d_first:
                    if dn > 0:
                        d_chunks.append(dn)
                elif dn > 0:
                    d_decode += 1
                if not p_first:
                    if pn > 0:
                        p_chunks.append(pn)
                elif pn > 0:
                    p_decode += 1
                coll.observe(outs, [drid, prid])
                if not d_first and any(_same_req(o.request_id, drid) for o in outs):
                    torch.cuda.synchronize()
                    d_first_s = time.perf_counter() - t_d0
                    d_first = True
                if not p_first and any(_same_req(o.request_id, prid) for o in outs):
                    torch.cuda.synchronize()
                    p_first_s = time.perf_counter() - t_p0
                    p_first = True
                steps += 1
            torch.cuda.synchronize()
            t_end = time.perf_counter()

            for case, rid, ids, chunks, first_s, t_base, decode_n, want in (
                (
                    "conc_decode_D",
                    drid,
                    d_ids,
                    d_chunks,
                    d_first_s,
                    t_d0,
                    d_decode,
                    CORR_B_D_MAX_TOKENS,
                ),
                (
                    "conc_long_prefill_P",
                    prid,
                    p_ids,
                    p_chunks,
                    p_first_s,
                    t_p0,
                    p_decode,
                    CORR_B_P_MAX_TOKENS,
                ),
            ):
                got = coll.get(rid)
                res = {
                    "prefill_chunks": chunks,
                    "prefill_step_count": len(chunks),
                    "scheduled_prompt_tokens": sum(chunks),
                    "decode_steps": decode_n,
                    "first_token_s": first_s,
                    "completion_s": t_end - t_base,
                    "output_token_ids": got["token_ids"],
                    "finished": got["finished"],
                    "finish_reason": got["finish_reason"],
                    "actual_prompt_tokens": len(got["prompt_token_ids"]) or len(ids),
                    "collector": coll,
                    "steps": steps,
                }
                samples.append(
                    _corr_sample("B", case, r, phase, mode, eff, ids, res, want)
                )
            drain(engine)
    finally:
        spy.restore()
        del engine
        torch.cuda.empty_cache()
    return samples


# ============ P2 · I5-b：APC 收益随输出长度变化（新增） ============
def run_e1_output_len(prefix_on: bool):
    """I5-b：固定 2048 共享前缀，扫 max_tokens ∈ {1,8,32,128}，看 APC 的**相对**收益
    是否随输出变长而下降（APC 只省 prefill、不省 decode 的定量边界）。

    与 E1 同一引擎配置（mbt=8192、chunked on），prompt 2064 token 单步算完
    → prefill_step_count 恒为 1，不与 chunked 行为混淆。
    每个 repeat 使用独立前缀族，避免跨 repeat 意外命中。
    """
    samples = []
    engine = build_engine(
        enable_prefix_caching=prefix_on,
        max_num_batched_tokens=I5B_MBT,
        enable_chunked_prefill=True,
    )
    spy = ScheduleSpy(get_scheduler(engine))
    tf = TokenFactory(engine)
    P, S = I5B_PREFIX_LEN, I5B_SUFFIX_LEN
    try:
        for mt in I5B_MAX_TOKENS:
            for r in range(I5B_WARMUP + I5B_REPEATS):
                phase = "warmup" if r < I5B_WARMUP else "measure"
                fam = f"i5b族{mt}_{r}："
                prefix_ids = tf.ids(P, uniq=fam)
                sfx_a = tf.ids(P + S, uniq=fam)[P:]
                sfx_b = tf.ids(P + S, uniq=fam + "B")[P:]
                cold = (prefix_ids + sfx_a)[: P + S]
                hot = (prefix_ids + sfx_b)[: P + S]
                assert len(cold) == P + S, f"cold len {len(cold)}"
                assert len(hot) == P + S, f"hot len {len(hot)}"
                assert cold[:P] == hot[:P], "cold/hot 前缀必须完全相同"
                assert cold[P:] != hot[P:], "cold/hot 后缀必须不同（否则会完全命中）"

                # APC on：cold 填缓存 → hot 命中；APC off：只跑与 hot 完全相同的 prompt
                seq = [("cold", cold), ("hot", hot)] if prefix_on else [("off", hot)]
                for kind, ids in seq:
                    rid = f"i5b-{mt}-{kind}-{r}"
                    res = _run_request_to_completion(engine, spy, rid, ids, mt)
                    out_ids = res["output_token_ids"]
                    n_out = len(out_ids)
                    ft, comp, post, tpot = _latency_fields(
                        res["first_token_s"], res["completion_s"], n_out
                    )
                    hit = res["actual_prompt_tokens"] - res["scheduled_prompt_tokens"]
                    ok = (
                        n_out == mt
                        and res["finished"]
                        and ft is not None
                        and ft >= 0
                        and post is not None
                        and post >= 0
                        and not res["collector"].semantics_violation
                    )
                    reasons = []
                    if n_out != mt:
                        reasons.append(f"output_tokens={n_out} != max_tokens {mt}")
                    if not res["finished"]:
                        reasons.append("request 未正常结束")
                    if ft is None or post is None or ft < 0 or post < 0:
                        reasons.append(f"latency 异常 ft={ft} post={post}")
                    if res["collector"].semantics_violation:
                        reasons.append(res["collector"].semantics_violation)
                    samples.append(
                        {
                            "experiment": "E1_output_len",
                            "metric_scope": "in-process",
                            "prefix_caching": "on" if prefix_on else "off",
                            "kind": kind,
                            "prefix_len": P,
                            "suffix_len": S,
                            "max_tokens_requested": mt,
                            "repeat": r,
                            "phase": phase,
                            "actual_prompt_tokens": res["actual_prompt_tokens"],
                            "scheduled_prompt_tokens": res["scheduled_prompt_tokens"],
                            "hit_tokens": hit,
                            "prefix_cache_hit": hit > 0,
                            "prefill_step_count": res["prefill_step_count"],
                            "chunk_sizes": res["prefill_chunks"],
                            "decode_steps": res["decode_steps"],
                            "actual_output_tokens": n_out,
                            "in_process_first_token_latency_ms": ft,
                            "request_completion_latency_ms": comp,
                            "post_first_token_latency_ms": post,
                            "tpot_ms": tpot,
                            "finish_reason": res["finish_reason"],
                            "output_token_ids_hash": _h(out_ids) if out_ids else "",
                            "prompt_hash": _h(ids),
                            "prefix_hash": _h(prefix_ids),
                            "suffix_hash": _h(list(ids[P:])),
                            "success": ok,
                            "invalid_reason": "; ".join(reasons),
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


def _first_divergence(a: list, b: list):
    """返回 (首个不同的下标, 公共前缀长度)；完全相等时下标为 None。"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, i
    if len(a) == len(b):
        return None, n
    return n, n  # 一方是另一方的前缀（长度不同）


def summarize_e3_correctness():
    """P0 汇总：逐 case 比对 chunked / nochunk 两侧的完整输出 token 序列。

    只读已落盘 JSONL，不建引擎、不需 GPU。任何一个 measure case 不一致 → 非零退出，
    并把「首个分歧下标 / 公共前缀长度」一并落盘，便于判断是哪一类失败
    （首 token 就分歧、长度不对、还是长尾分歧）。
    """
    os.makedirs(OUT, exist_ok=True)
    rows = {}
    for mode in ("chunked", "nochunk"):
        path = os.path.join(RAW, f"e3_correctness_{mode}.jsonl")
        if not os.path.exists(path):
            print(f"[m17] 缺少 {path} —— E3-correctness 尚未执行，无法汇总。")
            return False
        with open(path) as f:
            rows[mode] = [json.loads(line) for line in f if line.strip()]

    idx = {
        mode: {(r["scenario"], r["case"], r["repeat"]): r for r in _measure(rs)}
        for mode, rs in rows.items()
    }
    keys = sorted(set(idx["chunked"]) & set(idx["nochunk"]))
    only_c = sorted(set(idx["chunked"]) - set(idx["nochunk"]))
    only_n = sorted(set(idx["nochunk"]) - set(idx["chunked"]))
    if only_c or only_n:
        print(f"[m17][WARN] 两侧 case 不成对：仅 chunked={only_c} 仅 nochunk={only_n}")

    out_rows, failures = [], []
    for k in keys:
        c, n = idx["chunked"][k], idx["nochunk"][k]
        ci, common = _first_divergence(c["output_token_ids"], n["output_token_ids"])
        exact = ci is None
        prompt_match = c["prompt_hash"] == n["prompt_hash"]
        hash_match = c["output_token_ids_hash"] == n["output_token_ids_hash"]
        len_ok = (
            c["actual_output_tokens"]
            == n["actual_output_tokens"]
            == c["max_tokens_requested"]
        )
        conserved = (
            c["sum_prefill_scheduled"] == n["sum_prefill_scheduled"] == c["prompt_len"]
        )
        ok = bool(
            prompt_match
            and exact
            and hash_match
            and len_ok
            and conserved
            and c["success"]
            and n["success"]
        )
        why = []
        if not prompt_match:
            why.append("prompt_hash 不一致")
        if not len_ok:
            why.append(
                f"输出长度 chunked={c['actual_output_tokens']} "
                f"nochunk={n['actual_output_tokens']} "
                f"expected={c['max_tokens_requested']}"
            )
        if not exact:
            why.append(f"token 序列首个分歧下标={ci}（公共前缀 {common}）")
        if not hash_match:
            why.append("output hash 不一致")
        if not conserved:
            why.append("prefill 调度守恒不成立")
        if not (c["success"] and n["success"]):
            why.append(f"样本自检失败: {c['invalid_reason']} | {n['invalid_reason']}")
        if not ok:
            failures.append((k, "; ".join(why)))
        out_rows.append(
            {
                "scenario": k[0],
                "case": k[1],
                "repeat": k[2],
                "prompt_len": c["prompt_len"],
                "prompt_hash_chunked": c["prompt_hash"],
                "prompt_hash_nochunk": n["prompt_hash"],
                "prompt_hash_match": prompt_match,
                "max_tokens_requested": c["max_tokens_requested"],
                "out_tokens_chunked": c["actual_output_tokens"],
                "out_tokens_nochunk": n["actual_output_tokens"],
                "out_hash_chunked": c["output_token_ids_hash"],
                "out_hash_nochunk": n["output_token_ids_hash"],
                "out_hash_match": hash_match,
                "token_ids_exact_match": exact,
                "first_divergence_index": "" if ci is None else ci,
                "common_prefix_len": common,
                "finish_reason_chunked": c["finish_reason"],
                "finish_reason_nochunk": n["finish_reason"],
                "prefill_steps_chunked": c["prefill_step_count"],
                "chunk_sizes_chunked": c["chunk_sizes"],
                "sum_chunked": c["sum_prefill_scheduled"],
                "prefill_steps_nochunk": n["prefill_step_count"],
                "chunk_sizes_nochunk": n["chunk_sizes"],
                "sum_nochunk": n["sum_prefill_scheduled"],
                "verdict": "PASS" if ok else "FAIL",
                "fail_reason": "; ".join(why),
            }
        )

    path = os.path.join(OUT, "e3_correctness.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()) if out_rows else [])
        if out_rows:
            w.writeheader()
            w.writerows(out_rows)
    print(f"[m17] wrote {path} ({len(out_rows)} paired cases)")

    if not out_rows:
        print("[m17][FAIL] 没有可比对的 measure case。")
        return False
    if failures:
        print(f"[m17][FAIL] {len(failures)}/{len(out_rows)} 个 case 不一致：")
        for k, why in failures:
            print(f"  - {k}: {why}")
        return False
    print(
        f"[m17][PASS] {len(out_rows)}/{len(out_rows)} 个 case 的输出 token 逐一相同。"
    )
    return True


def summarize_e1_output_len():
    """I5-b 汇总：按 max_tokens 比较 off 与 hot 的首 token / 完成延迟。

    统计口径与 M17 其余部分一致：median difference 的 Bootstrap 95% CI
    （10000 次，固定 seed=0，仅 stdlib）。**不使用 σ**；CI 含 0 只能写
    「未观察到稳定差异」。写独立 CSV，不并入 prefix_cache_*.csv。
    """
    os.makedirs(OUT, exist_ok=True)
    rows = _measure(_load_shards("e1_output_len_"))
    if not rows:
        print("[m17] 未找到 e1_output_len_*.jsonl 的 measure 样本 —— I5-b 尚未执行。")
        return False

    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault(
            (r["prefix_caching"], r["kind"], r["max_tokens_requested"]), []
        ).append(r)

    def col(v, key):
        return [x[key] for x in v if x.get(key) is not None]

    path = os.path.join(OUT, "apc_output_length.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "prefix_caching",
                "kind",
                "max_tokens",
                "n",
                "ft_median_ms",
                "ft_stdev_ms",
                "ft_p95_ms",
                "completion_median_ms",
                "completion_stdev_ms",
                "completion_p95_ms",
                "post_ft_median_ms",
                "post_ft_stdev_ms",
                "tpot_median_ms",
                "hit_tokens",
                "scheduled_prompt_tokens",
                "actual_prompt_tokens",
                "actual_output_tokens",
                "prefill_step_count",
                "metric_scope",
            ]
        )
        for k in sorted(groups, key=lambda x: (x[0], x[1], x[2])):
            v = groups[k]
            ft = _stats(col(v, "in_process_first_token_latency_ms"))
            cp = _stats(col(v, "request_completion_latency_ms"))
            pf = _stats(col(v, "post_first_token_latency_ms"))
            tp = _stats(col(v, "tpot_ms"))
            w.writerow(
                [
                    k[0],
                    k[1],
                    k[2],
                    len(v),
                    ft.get("median"),
                    ft.get("stdev"),
                    ft.get("p95"),
                    cp.get("median"),
                    cp.get("stdev"),
                    cp.get("p95"),
                    pf.get("median"),
                    pf.get("stdev"),
                    tp.get("median", "") if tp else "",
                    v[0]["hit_tokens"],
                    v[0]["scheduled_prompt_tokens"],
                    v[0]["actual_prompt_tokens"],
                    v[0]["actual_output_tokens"],
                    v[0]["prefill_step_count"],
                    "in-process",
                ]
            )
    print(f"[m17] wrote {path}")

    path2 = os.path.join(OUT, "apc_output_length_effect.csv")
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "max_tokens",
                "off_completion_median_ms",
                "hot_completion_median_ms",
                "completion_median_diff_ms",
                "completion_ci95_lo_ms",
                "completion_ci95_hi_ms",
                "completion_ci_excludes_zero",
                "completion_rel_reduction_pct",
                "off_ft_median_ms",
                "hot_ft_median_ms",
                "ft_median_diff_ms",
                "ft_ci95_lo_ms",
                "ft_ci95_hi_ms",
                "ft_ci_excludes_zero",
                "off_post_ft_median_ms",
                "hot_post_ft_median_ms",
                "off_tpot_median_ms",
                "hot_tpot_median_ms",
                "n_per_group",
                "metric_scope",
            ]
        )
        for mt in I5B_MAX_TOKENS:
            ko, kh = ("off", "off", mt), ("on", "hot", mt)
            if ko not in groups or kh not in groups:
                continue
            o, h = groups[ko], groups[kh]
            oc = col(o, "request_completion_latency_ms")
            hc = col(h, "request_completion_latency_ms")
            of = col(o, "in_process_first_token_latency_ms")
            hf = col(h, "in_process_first_token_latency_ms")
            cd, clo, chi = _boot_ci_median_diff(oc, hc)
            fd, flo, fhi = _boot_ci_median_diff(of, hf)
            om, hm = statistics.median(oc), statistics.median(hc)
            otp, htp = col(o, "tpot_ms"), col(h, "tpot_ms")
            w.writerow(
                [
                    mt,
                    round(om, 4),
                    round(hm, 4),
                    cd,
                    clo,
                    chi,
                    "yes" if (clo > 0 or chi < 0) else "no",
                    round((om - hm) / om * 100, 2),
                    round(statistics.median(of), 4),
                    round(statistics.median(hf), 4),
                    fd,
                    flo,
                    fhi,
                    "yes" if (flo > 0 or fhi < 0) else "no",
                    round(statistics.median(col(o, "post_first_token_latency_ms")), 4),
                    round(statistics.median(col(h, "post_first_token_latency_ms")), 4),
                    round(statistics.median(otp), 4) if otp else "",
                    round(statistics.median(htp), 4) if htp else "",
                    len(oc),
                    "in-process",
                ]
            )
    print(f"[m17] wrote {path2}")
    print(f"[m17] I5-b summarize done. measure samples: {len(rows)}")
    return True


def main():
    p = argparse.ArgumentParser(
        description="M17 prefix caching & chunked prefill (需 GPU；一个进程一个引擎)"
    )
    p.add_argument(
        "--shard",
        choices=[
            "e1-on",
            "e1-off",
            "e3",
            "e3-conc",
            "summarize",
            # 新增（实现完成，待 GPU 执行）
            "e3-correctness",
            "e3-correctness-summarize",
            "e1-output-len-on",
            "e1-output-len-off",
            "e1-output-len-summarize",
        ],
        required=True,
    )
    p.add_argument("--e3-chunked", choices=["on", "off"], help="--shard e3 必填")
    p.add_argument("--e3-mbt", type=int, help="--shard e3 必填")
    p.add_argument(
        "--conc-mode",
        choices=["chunked", "nochunk"],
        help="--shard e3-conc / e3-correctness 必填",
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
    elif a.shard == "e3-correctness":
        if a.conc_mode is None:
            p.error("--shard e3-correctness 必须提供 --conc-mode {chunked,nochunk}")
        _dump(run_e3_correctness(a.conc_mode), f"e3_correctness_{a.conc_mode}")
    elif a.shard == "e3-correctness-summarize":
        if not summarize_e3_correctness():
            raise SystemExit(1)  # 不一致必须让调用方看见非零退出
    elif a.shard == "e1-output-len-on":
        _dump(run_e1_output_len(prefix_on=True), "e1_output_len_on")
    elif a.shard == "e1-output-len-off":
        _dump(run_e1_output_len(prefix_on=False), "e1_output_len_off")
    elif a.shard == "e1-output-len-summarize":
        if not summarize_e1_output_len():
            raise SystemExit(1)
    else:
        summarize()


if __name__ == "__main__":
    main()
