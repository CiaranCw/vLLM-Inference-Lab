"""M19 · Speculative Decoding 第一阶段：普通 Decode Baseline vs N-gram。

============================================================================
运行需 GPU（会建真实 vLLM 引擎）。必须用 --shard 分片跑：一个进程只建一个引擎
（vLLM 的 KV 显存 `del engine + empty_cache()` 归还不掉，同进程建第二个引擎会失败；
M6 已实证，M17 沿用）。

    export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
    python scripts/m19_spec_decode.py --shard smoke    --mode ngram      # 小样本自检
    python scripts/m19_spec_decode.py --shard baseline                   # 引擎 1
    python scripts/m19_spec_decode.py --shard ngram                      # 引擎 2
    python scripts/m19_spec_decode.py --shard summarize                  # 不建引擎、不需 GPU

理论 / 源码定位 / 冻结设计：notes/m19/speculative_decoding.md
============================================================================

两侧严格单变量：除 `speculative_config` 外所有引擎参数逐字相同。特别注意两个
容易变成隐藏变量的开关，本脚本已在两侧都显式写死：

  1. `async_scheduling=False`——上游在 `speculative_config.method == "ngram"` 且
     `async_scheduling is None` 时会自动把它关掉并 warning
     （`vllm/config/vllm.py:976-986`）。若 baseline 侧留 None（可能被开启），
     开 ngram 就会顺带改变 scheduler 行为，对照即不成立。
  2. `disable_log_stats=False`——投机解码计数有三道 `log_stats` 门禁
     （`scheduler.py:2597` / `scheduler.py:2558` / `llm_engine.py:452`），
     关掉就读不到 proposed/accepted。baseline 侧同样打开，保证开销对称。

另外关闭 APC（`enable_prefix_caching=False`）：本实验重复跑相同 prompt，
开着会让后续 repeat 命中缓存，把 prefill 成本和 ngram 的上下文都搅乱。

计数来源（只读，不改产品源码）：注册一个自定义 `StatLoggerBase` 到
`LLMEngine.from_engine_args(stat_loggers=[...])`（`vllm/v1/engine/llm_engine.py:215`），
逐 step 收 `SchedulerStats.spec_decoding_stats`（`vllm/v1/metrics/stats.py:190`，
类型见 `vllm/v1/spec_decode/metrics.py:17-49`）。
不用 `get_metrics()`，因为那是进程级累计的 Prometheus counter，读取不清零、
难以按样本切窗口；自定义 logger 拿到的是per-step 原始对象，可按样本精确归属。

⚠ `num_accepted_tokens` 的语义（`vllm/v1/core/sched/scheduler.py:1845-1869`）：
    num_accepted = max(len(generated_token_ids) - 1, 0)
即「本 step 该请求产出的 token 数 − 1」，不含那个总会产出的 bonus/修正 token。
所以 acceptance_length = 1 + accepted/drafts（与上游示例
`examples/features/speculative_decoding/spec_decode_offline.py:212` 一致）。
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
from vllm.v1.metrics.loggers import StatLoggerBase

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "m19")
RAW = os.path.join(OUT, "raw")

# ------------------------------ 冻结配置（跑前写死） ------------------------------
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
SEED = 0
MAX_MODEL_LEN = 4096
GPU_MEM_UTIL = 0.6
MAX_NUM_SEQS = 64

PROMPT_LEN = 256  # 两个 workload 的 prompt token 数完全相同，隔离 prefill 成本
OUTPUT_LEN = 128  # 每个请求的输出 token 数（ignore_eos=True，恰好等于此值）
BATCH_SIZES = [1, 4, 16]
WORKLOADS = ["high_repeat", "low_repeat"]
WARMUP = 5
REPEATS = 15

# N-gram 配置：照抄上游自带测试的可用组合
# （tests/v1/e2e/spec_decode/test_spec_decode.py:165-170）。
# ⚠ prompt_lookup_min/max 必须两个都显式写死：只给一个或都不给，
#   上游会静默改写成 5/5（vllm/config/speculative.py:633-659），自变量就变了。
NGRAM_CONFIG = {
    "method": "ngram",
    "prompt_lookup_min": 3,
    "prompt_lookup_max": 5,
    "num_speculative_tokens": 3,
}

# Smoke 用的极小样本量（只验证配置真的生效、计数字段可观测）
SMOKE_BATCH = 1
SMOKE_REPEATS = 2
SMOKE_OUTPUT_LEN = 32

# ---- 输出一致性 probe（独立于主实验；存完整 token ids 以便定位分歧点）----
# 主实验只落 hash，发现 hash 不等时无法回答"差在哪一位"。本 probe 专门补这一步，
# 写独立文件，不触碰主实验已落盘的性能数据。
EQUIV_WARMUP = 1
EQUIV_REPEATS = 3

# ---- workload 文本源（确定性；两侧、两 workload 均由同一函数构造，落盘 hash 可核对）----
# high_repeat：高度重复的结构化模式 → 贪心续写极易复读，ngram 能从上下文命中
_HIGH_REPEAT_UNIT = (
    "步骤 A：读取请求。步骤 B：分配显存。步骤 C：执行前向。步骤 D：返回结果。"
)
# low_repeat：内容多变、几乎不自我重复 → ngram 难以命中
# ⚠ 必须一次就够长（≥ PROMPT_LEN + 余量）。首版曾用 `_LOW_REPEAT_TEXT * 3` 凑长度，
#   结果 prompt 里同一段话出现三遍 → 上下文其实高度重复，smoke 实测 acceptance 也是
#   1.0，"低重复"对照根本不成立。这是被 smoke 抓出来的真实设计缺陷，已改为不重复。
_LOW_REPEAT_TEXT = (
    "苏州园林的花窗把光线切成细碎的格子，投在青砖地上。"
    "北方的冬天来得早，河面结冰的那天孩子们会跑去看。"
    "咖啡烘焙师说，浅烘留住果酸，深烘换来焦糖般的醇厚。"
    "他在旧书店翻到一本一九三七年的植物图鉴，扉页有钢笔字。"
    "潮汐发电依赖月球引力，成本高但输出可预测。"
    "编曲时她把贝斯线往后挪了半拍，整首歌忽然松弛下来。"
    "戈壁滩的夜空没有云，银河像被谁泼上去的一道白。"
    "老木匠不用钉子，榫卯咬合处留一丝余量给木头呼吸。"
    "疫苗冷链在最后一公里最容易失效，村医靠冰袋硬扛。"
    "围棋的定式是经验的结晶，也是想象力的牢笼。"
    "海鸥跟着渔船飞了三海里，直到船舱盖上最后一块木板。"
    "他修好了那台一九九八年的收音机，只为听一段短波。"
    "青藏高原的紫外线强得离谱，牧民的帐篷顶都晒成了灰白。"
    "程序员在凌晨三点发现问题出在时区，而不是算法。"
    "小城的电影院只剩一块幕布，放映员兼职卖爆米花。"
    "候鸟迁徙靠地磁定位，磁暴会让它们偏离航线数百公里。"
    "祖母腌菜从不称重，全凭手感，坛口用荷叶封住。"
    "深海热泉旁的管虫不靠阳光，靠硫化物维生。"
    "他把最后一枚邮票贴上信封，投进了那个绿色的邮筒。"
    "钢琴调律师说，湿度每变化百分之十，音准就得重来一次。"
    "沙漠公路两侧的草方格是人工铺的，一格一格压住流沙。"
    "旧唱片的沙沙声不是缺陷，是母带磁粉的呼吸。"
)


def _h(ids: list[int]) -> str:
    """token 序列的短哈希（记录身份，不把全量 ids 写进 CSV）。"""
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()[:12]


# ============================ 投机解码计数采集（只读） ============================
class SpecStatCollector(StatLoggerBase):
    """自定义 StatLogger：逐 step 累计 `SpecDecodingStats`，按样本窗口归属。

    引擎内部会用 `factory(vllm_config, engine_index)` 构造它
    （`vllm/v1/metrics/loggers.py:376-378`），拿不到实例引用，
    因此用类变量登记单例供实验代码读取。

    不修改任何 vLLM 产品源码，只用上游公开的 `stat_loggers=` 扩展点。
    """

    instance: SpecStatCollector | None = None

    def __init__(self, vllm_config, engine_index: int = 0):
        self.vllm_config = vllm_config
        self.engine_index = engine_index
        self.reset()
        SpecStatCollector.instance = self

    def reset(self):
        self.num_drafts = 0
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        self.accepted_per_pos: list[int] = []
        self.draft_per_pos: list[int] = []
        self.steps_with_spec_stats = 0
        self.num_spec_tokens: int | None = None

    def record(
        self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx: int = 0
    ):
        if scheduler_stats is None:
            return
        s = getattr(scheduler_stats, "spec_decoding_stats", None)
        if s is None:
            return
        self.steps_with_spec_stats += 1
        self.num_spec_tokens = s.num_spec_tokens
        self.num_drafts += s.num_drafts
        self.num_draft_tokens += s.num_draft_tokens
        self.num_accepted_tokens += s.num_accepted_tokens
        for name, src in (
            ("accepted_per_pos", s.num_accepted_tokens_per_pos),
            ("draft_per_pos", s.num_draft_tokens_per_pos),
        ):
            dst = getattr(self, name)
            if len(dst) < len(src):
                dst.extend([0] * (len(src) - len(dst)))
            for i, v in enumerate(src):
                dst[i] += v

    def log_engine_initialized(self):
        pass

    def snapshot(self) -> dict:
        acc, prop = self.num_accepted_tokens, self.num_draft_tokens
        return {
            "num_drafts": self.num_drafts,
            "proposed_tokens": prop,
            "accepted_tokens": acc,
            "rejected_tokens": prop - acc,
            "acceptance_rate": (round(acc / prop, 6) if prop else None),
            # acceptance_length = 1 + accepted/drafts，与上游示例同口径
            "acceptance_length": (
                round(1 + acc / self.num_drafts, 6) if self.num_drafts else None
            ),
            "accepted_per_pos": list(self.accepted_per_pos),
            "draft_per_pos": list(self.draft_per_pos),
            "steps_with_spec_stats": self.steps_with_spec_stats,
            "num_spec_tokens_cfg": self.num_spec_tokens,
        }


# ============================ 引擎与内省（沿用 M17 模式） ============================
def build_engine(spec: bool) -> LLMEngine:
    """两侧唯一的差别就是 speculative_config；其余参数逐字相同。"""
    args = EngineArgs(
        model=MODEL,
        dtype="auto",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        tensor_parallel_size=1,
        enforce_eager=True,
        seed=SEED,
        max_num_seqs=MAX_NUM_SEQS,
        enable_prefix_caching=False,
        # 两侧都显式关：否则 ngram 侧会被上游自动关掉，成为隐藏的第二变量
        async_scheduling=False,
        # 两侧都打开统计：投机计数依赖它，baseline 侧同样打开以保持开销对称
        disable_log_stats=False,
        # dict 每次新建：上游会原地 update 注入 target config（arg_utils.py:1727-1732）
        speculative_config=dict(NGRAM_CONFIG) if spec else None,
    )
    return LLMEngine.from_engine_args(
        args,
        usage_context=UsageContext.ENGINE_CONTEXT,
        stat_loggers=[SpecStatCollector],
        enable_multiprocessing=False,
    )


def effective_cfg(engine: LLMEngine) -> dict:
    """把真正生效的配置读回来落盘——不靠"我传了什么"，靠"引擎实际是什么"。"""
    vc = engine.vllm_config
    sc, sched = vc.speculative_config, vc.scheduler_config
    return {
        "speculative_enabled": sc is not None,
        "spec_method": getattr(sc, "method", None),
        "num_speculative_tokens": getattr(sc, "num_speculative_tokens", None),
        "prompt_lookup_min": getattr(sc, "prompt_lookup_min", None),
        "prompt_lookup_max": getattr(sc, "prompt_lookup_max", None),
        "async_scheduling": bool(sched.async_scheduling),
        "enable_chunked_prefill": bool(sched.enable_chunked_prefill),
        "max_num_batched_tokens": int(sched.max_num_batched_tokens),
        "max_num_seqs": int(sched.max_num_seqs),
        "enable_prefix_caching": bool(vc.cache_config.enable_prefix_caching),
        "enforce_eager": bool(vc.model_config.enforce_eager),
    }


class TokenFactory:
    """确定性构造 prompt：两个独立进程必然得到逐一相同的 token ids（落盘 hash 可核对）。"""

    def __init__(self, engine: LLMEngine):
        tok = getattr(engine, "tokenizer", None)
        if tok is None or not hasattr(tok, "encode"):
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL)
        self.tok = tok
        self._pool: dict[str, list[int]] = {}

    def _base(self, workload: str) -> list[int]:
        if workload not in self._pool:
            if workload == "high_repeat":
                text = _HIGH_REPEAT_UNIT * 40
            elif workload == "low_repeat":
                # 只用一遍：重复它就等于把"低重复"对照做没了（见文本常量处的说明）
                text = _LOW_REPEAT_TEXT
            else:
                raise ValueError(f"未知 workload: {workload}")
            ids = self.tok.encode(text, add_special_tokens=False)
            if len(ids) < PROMPT_LEN + 64:
                raise RuntimeError(
                    f"{workload} 文本过短：{len(ids)} < {PROMPT_LEN + 64}，"
                    "无法构造固定长度 prompt"
                )
            self._pool[workload] = ids
        return self._pool[workload]

    def prompt(self, workload: str, req_idx: int) -> list[int]:
        """恰好 PROMPT_LEN 个 token；req_idx 只改开头一小段，保证批内各请求不同。"""
        head = self.tok.encode(f"样本{req_idx}：", add_special_tokens=False)
        ids = (head + self._base(workload))[:PROMPT_LEN]
        assert len(ids) == PROMPT_LEN, f"prompt 长度不符: {len(ids)}"
        return ids


def _same_req(observed: str, rid: str) -> bool:
    """vLLM 会给 add_request 的 id 追加后缀（"r0" -> "r0-8ef2061f"），故容错匹配。"""
    return observed == rid or observed.startswith(rid + "-")


def _repetition_ratio(ids: list[int], n: int = 4) -> float | None:
    """输出自身的 n-gram 重复度：1 − 去重 n-gram 数 / 总 n-gram 数。"""
    if len(ids) <= n:
        return None
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 6)


def _ngram_in_context_ratio(out_ids, prompt_ids, n: int = 4) -> float | None:
    """输出的 n-gram 有多大比例已经出现在 prompt 里。

    这是与引擎的 acceptance 计数正交的第二证据源：acceptance 来自 vLLM 内部
    的 scheduler 统计，本指标完全由落盘的 token 序列离线算出。若某 workload 的
    acceptance 高而本指标低（或反之），说明我对"为什么命中"的解释站不住，
    必须回去查，而不是直接下结论。
    """
    if len(out_ids) <= n or len(prompt_ids) <= n:
        return None
    ctx = {tuple(prompt_ids[i : i + n]) for i in range(len(prompt_ids) - n + 1)}
    grams = [tuple(out_ids[i : i + n]) for i in range(len(out_ids) - n + 1)]
    return round(sum(1 for g in grams if g in ctx) / len(grams), 6)


# ============================ 一个样本 = 一批并发请求 ============================
def run_one_sample(engine, tf, collector, workload, batch, out_len, keep_ids=False):
    """加 batch 个请求并跑到全部结束，返回逐样本观测。

    engine_steps 直接计数 `engine.step()` 调用次数。在当前
    `enforce_eager=True + async_scheduling=False` 配置下，它是 execution /
    model / scheduling round 的强 proxy，适合观察串行 round 数变化；
    不能一般化为严格的一次纯 Decode Target Forward（chunked prefill /
    boundary / scheduling 都可能计入）。这是「投机解码是否减少了串行
    execution depth」的直接证据（而不是从延迟反推）。
    """
    collector.reset()
    torch.cuda.reset_peak_memory_stats()
    rids = [f"m19-{workload}-b{batch}-r{i}" for i in range(batch)]
    prompts = {rid: tf.prompt(workload, i) for i, rid in enumerate(rids)}
    for rid in rids:
        engine.add_request(
            rid,
            {"prompt_token_ids": prompts[rid]},
            SamplingParams(
                temperature=0.0, max_tokens=out_len, ignore_eos=True, seed=SEED
            ),
        )

    outputs: dict[str, dict] = {}
    semantics_violation = ""
    steps = 0
    first_token_s = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    while engine.has_unfinished_requests() and steps < 100000:
        outs = engine.step()
        steps += 1
        for o in outs:
            for rid in rids:
                if not _same_req(o.request_id, rid) or not o.outputs:
                    continue
                co = o.outputs[0]
                # output_kind 默认 CUMULATIVE（sampling_params.py:301），
                # token_ids 是引擎内部会继续 append 的同一个 list 对象
                # （output_processor.py:389-390）→ 必须拷贝。
                ids = list(co.token_ids)
                prev = outputs.get(rid)
                if prev is not None and len(ids) < len(prev["token_ids"]):
                    semantics_violation = (
                        f"{rid}: token_ids 长度回退 "
                        f"{len(prev['token_ids'])}->{len(ids)}（疑似 DELTA 语义）"
                    )
                outputs[rid] = {
                    "token_ids": ids,
                    "finished": bool(o.finished),
                    "finish_reason": co.finish_reason,
                }
        if first_token_s is None and outputs:
            torch.cuda.synchronize()
            first_token_s = time.perf_counter() - t0
    torch.cuda.synchronize()
    completion_s = time.perf_counter() - t0

    per_req = [outputs.get(rid, {"token_ids": [], "finished": False}) for rid in rids]
    out_lens = [len(x["token_ids"]) for x in per_req]
    total_out = sum(out_lens)
    ft_ms = None if first_token_s is None else round(first_token_s * 1000, 4)
    comp_ms = round(completion_s * 1000, 4)
    post_ms = None if ft_ms is None else round(comp_ms - ft_ms, 4)
    # amortized TPOT：首 token 之后每个输出 token 的摊销耗时（按最长请求）。
    # 这不是 streaming ITL：Spec Decode 一轮可能一次产出多个 token（bursty），
    # 本实验没有逐 token timestamp。
    tpot = (
        round(post_ms / (max(out_lens) - 1), 4)
        if post_ms is not None and max(out_lens, default=0) > 1
        else None
    )

    ok = (
        all(x["finished"] for x in per_req)
        and all(n == out_len for n in out_lens)
        and ft_ms is not None
        and not semantics_violation
    )
    reasons = []
    if not all(x["finished"] for x in per_req):
        reasons.append("存在未正常结束的请求")
    if any(n != out_len for n in out_lens):
        reasons.append(f"输出长度 {out_lens} != {out_len}")
    if ft_ms is None:
        reasons.append("未观察到首 token")
    if semantics_violation:
        reasons.append(semantics_violation)

    spec = collector.snapshot()
    ids_field = (
        {"output_token_ids": [x["token_ids"] for x in per_req]} if keep_ids else {}
    )
    return {
        **ids_field,
        "workload": workload,
        "batch_size": batch,
        "max_tokens_requested": out_len,
        "engine_steps": steps,
        "prompt_len": PROMPT_LEN,
        "prompt_hashes": [_h(prompts[rid]) for rid in rids],
        "batch_prompt_hash": _h([t for rid in rids for t in prompts[rid]]),
        "actual_output_tokens": out_lens,
        "total_output_tokens": total_out,
        "output_token_hashes": [_h(x["token_ids"]) for x in per_req],
        "batch_output_hash": _h([t for x in per_req for t in x["token_ids"]]),
        "output_repetition_4gram": [_repetition_ratio(x["token_ids"]) for x in per_req],
        "output_4gram_in_prompt_ratio": [
            _ngram_in_context_ratio(x["token_ids"], prompts[rid])
            for rid, x in zip(rids, per_req)
        ],
        "finish_reasons": [x.get("finish_reason") for x in per_req],
        "in_process_first_token_latency_ms": ft_ms,
        "request_completion_latency_ms": comp_ms,
        "post_first_token_latency_ms": post_ms,
        "tpot_ms": tpot,
        "output_throughput_tok_s": round(total_out / completion_s, 4),
        "total_throughput_tok_s": round(
            (total_out + PROMPT_LEN * batch) / completion_s, 4
        ),
        "steps_per_output_token": round(steps / max(out_lens), 6) if out_lens else None,
        "peak_mem_allocated_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "mem_reserved_mib": round(torch.cuda.memory_reserved() / 1024**2, 2),
        "success": ok,
        "invalid_reason": "; ".join(reasons),
        **spec,
    }


def run_side(spec: bool, smoke: bool = False, equiv: bool = False):
    """跑一侧（baseline 或 ngram）。一个进程只建一个引擎。"""
    samples = []
    engine = build_engine(spec)
    eff = effective_cfg(engine)
    print(f"[m19] effective config: {json.dumps(eff, ensure_ascii=False)}")
    collector = SpecStatCollector.instance
    if collector is None:
        raise RuntimeError(
            "SpecStatCollector 未被引擎构造 —— stat_loggers 注册失败，"
            "投机计数不可观测。请检查 disable_log_stats 与 vLLM 版本。"
        )
    tf = TokenFactory(engine)
    try:
        torch.cuda.synchronize()
        post_build_mem = round(torch.cuda.memory_allocated() / 1024**2, 2)
        batches = [SMOKE_BATCH] if smoke else BATCH_SIZES
        n_warm = 0 if smoke else (EQUIV_WARMUP if equiv else WARMUP)
        n_meas = SMOKE_REPEATS if smoke else (EQUIV_REPEATS if equiv else REPEATS)
        out_len = SMOKE_OUTPUT_LEN if smoke else OUTPUT_LEN
        for workload in WORKLOADS:
            for batch in batches:
                for r in range(n_warm + n_meas):
                    phase = "warmup" if r < n_warm else "measure"
                    s = run_one_sample(
                        engine, tf, collector, workload, batch, out_len, keep_ids=equiv
                    )
                    s.update(
                        {
                            "experiment": "M19_spec_decode",
                            "metric_scope": "in-process",
                            "config": "ngram" if spec else "baseline",
                            "effective_cfg": eff,
                            "repeat": r,
                            "phase": phase,
                            "post_build_mem_allocated_mib": post_build_mem,
                        }
                    )
                    samples.append(s)
                print(
                    f"[m19] {'ngram' if spec else 'baseline'} {workload} batch={batch} "
                    f"done ({n_warm}w+{n_meas}m)"
                )
    finally:
        del engine
        torch.cuda.empty_cache()
    return samples


# ============================ 落盘 / 汇总 ============================
def _dump(samples, tag):
    os.makedirs(RAW, exist_ok=True)
    path = os.path.join(RAW, f"{tag}.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(s, ensure_ascii=False) + "\n" for s in samples)
    print(f"[m19] shard written: {path} ({len(samples)} samples)")


def _load(*tags):
    """按精确文件名加载分片。

    刻意不做前缀通配：一致性 probe 的分片也叫 m19_*，前缀匹配会把它混进主汇总，
    让 CSV 里的 n 与设计不符却不报错。
    """
    out = []
    for tag in tags:
        path = os.path.join(RAW, f"{tag}.jsonl")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            out.extend(json.loads(line) for line in f if line.strip())
    return out


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
    """median(a) - median(b) 的 Bootstrap 95% CI（与 M17 同算法、同种子）。"""
    import random

    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.median(ra) - statistics.median(rb))
    diffs.sort()
    return (
        round(statistics.median(a) - statistics.median(b), 4),
        round(diffs[int(0.025 * (iters - 1))], 4),
        round(diffs[int(0.975 * (iters - 1))], 4),
    )


def summarize():
    """由已落盘 JSONL 离线汇总为 CSV（不建引擎、不需 GPU）。"""
    os.makedirs(OUT, exist_ok=True)
    rows = _measure(_load("m19_baseline", "m19_ngram"))
    if not rows:
        print("[m19] 未找到 measure 样本 —— 实验尚未执行。")
        return False

    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["config"], r["workload"], r["batch_size"]), []).append(r)

    def col(v, k):
        return [x[k] for x in v if x.get(k) is not None]

    # ---------------- 主汇总 ----------------
    path = os.path.join(OUT, "spec_decode_summary.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "config",
                "workload",
                "batch_size",
                "n",
                "ft_median_ms",
                "ft_stdev_ms",
                "ft_p95_ms",
                "completion_median_ms",
                "completion_stdev_ms",
                "completion_p95_ms",
                "tpot_median_ms",
                "tpot_stdev_ms",
                "tpot_p95_ms",
                "output_tps_median",
                "output_tps_stdev",
                "total_tps_median",
                "engine_steps_median",
                "engine_steps_stdev",
                "steps_per_output_token_median",
                "proposed_tokens_median",
                "accepted_tokens_median",
                "acceptance_rate_median",
                "acceptance_length_median",
                "num_drafts_median",
                "output_repetition_4gram_median",
                "output_4gram_in_prompt_ratio_median",
                "peak_mem_allocated_mib_median",
                "mem_reserved_mib_median",
                "total_output_tokens",
                "metric_scope",
            ]
        )
        for k in sorted(groups, key=lambda x: (x[0], x[1], x[2])):
            v = groups[k]
            ft = _stats(col(v, "in_process_first_token_latency_ms"))
            cp = _stats(col(v, "request_completion_latency_ms"))
            tp = _stats(col(v, "tpot_ms"))
            ot = _stats(col(v, "output_throughput_tok_s"))
            tt = _stats(col(v, "total_throughput_tok_s"))
            st = _stats(col(v, "engine_steps"))
            sp = _stats(col(v, "steps_per_output_token"))
            ar = _stats(col(v, "acceptance_rate"))
            al = _stats(col(v, "acceptance_length"))
            pr = _stats(col(v, "proposed_tokens"))
            ac = _stats(col(v, "accepted_tokens"))
            nd = _stats(col(v, "num_drafts"))
            rep = _stats(
                [
                    x
                    for s in v
                    for x in (s.get("output_repetition_4gram") or [])
                    if x is not None
                ]
            )
            inctx = _stats(
                [
                    x
                    for s in v
                    for x in (s.get("output_4gram_in_prompt_ratio") or [])
                    if x is not None
                ]
            )
            pm = _stats(col(v, "peak_mem_allocated_mib"))
            mr = _stats(col(v, "mem_reserved_mib"))
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
                    tp.get("median"),
                    tp.get("stdev"),
                    tp.get("p95"),
                    ot.get("median"),
                    ot.get("stdev"),
                    tt.get("median"),
                    st.get("median"),
                    st.get("stdev"),
                    sp.get("median"),
                    pr.get("median", ""),
                    ac.get("median", ""),
                    ar.get("median", ""),
                    al.get("median", ""),
                    nd.get("median", ""),
                    rep.get("median", ""),
                    inctx.get("median", ""),
                    pm.get("median"),
                    mr.get("median"),
                    sum(col(v, "total_output_tokens")),
                    "in-process",
                ]
            )
    print(f"[m19] wrote {path}")

    # ---------------- baseline vs ngram 的效应量 ----------------
    path2 = os.path.join(OUT, "spec_decode_effect.csv")
    with open(path2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "workload",
                "batch_size",
                "n_per_group",
                "acceptance_rate_ngram",
                "acceptance_length_ngram",
                "base_completion_median_ms",
                "ngram_completion_median_ms",
                "completion_median_diff_ms",
                "completion_ci95_lo_ms",
                "completion_ci95_hi_ms",
                "completion_ci_excludes_zero",
                "completion_rel_change_pct",
                "base_tpot_median_ms",
                "ngram_tpot_median_ms",
                "tpot_median_diff_ms",
                "tpot_ci95_lo_ms",
                "tpot_ci95_hi_ms",
                "tpot_ci_excludes_zero",
                "base_output_tps_median",
                "ngram_output_tps_median",
                "output_tps_rel_change_pct",
                "base_engine_steps_median",
                "ngram_engine_steps_median",
                "engine_steps_rel_change_pct",
                "base_ft_median_ms",
                "ngram_ft_median_ms",
                "metric_scope",
            ]
        )
        for wl in WORKLOADS:
            for b in BATCH_SIZES:
                kb, kn = ("baseline", wl, b), ("ngram", wl, b)
                if kb not in groups or kn not in groups:
                    continue
                vb, vn = groups[kb], groups[kn]
                bc = col(vb, "request_completion_latency_ms")
                nc = col(vn, "request_completion_latency_ms")
                bt = col(vb, "tpot_ms")
                nt = col(vn, "tpot_ms")
                cd, clo, chi = _boot_ci_median_diff(bc, nc)
                td, tlo, thi = _boot_ci_median_diff(bt, nt)
                bcm, ncm = statistics.median(bc), statistics.median(nc)
                bsm = statistics.median(col(vb, "engine_steps"))
                nsm = statistics.median(col(vn, "engine_steps"))
                bom = statistics.median(col(vb, "output_throughput_tok_s"))
                nom = statistics.median(col(vn, "output_throughput_tok_s"))
                arn = col(vn, "acceptance_rate")
                aln = col(vn, "acceptance_length")
                w.writerow(
                    [
                        wl,
                        b,
                        min(len(vb), len(vn)),
                        round(statistics.median(arn), 6) if arn else "",
                        round(statistics.median(aln), 6) if aln else "",
                        round(bcm, 4),
                        round(ncm, 4),
                        cd,
                        clo,
                        chi,
                        "yes" if (clo > 0 or chi < 0) else "no",
                        round((bcm - ncm) / bcm * 100, 2),
                        round(statistics.median(bt), 4),
                        round(statistics.median(nt), 4),
                        td,
                        tlo,
                        thi,
                        "yes" if (tlo > 0 or thi < 0) else "no",
                        round(bom, 4),
                        round(nom, 4),
                        round((nom - bom) / bom * 100, 2),
                        round(bsm, 4),
                        round(nsm, 4),
                        round((nsm - bsm) / bsm * 100, 2),
                        round(
                            statistics.median(
                                col(vb, "in_process_first_token_latency_ms")
                            ),
                            4,
                        ),
                        round(
                            statistics.median(
                                col(vn, "in_process_first_token_latency_ms")
                            ),
                            4,
                        ),
                        "in-process",
                    ]
                )
    print(f"[m19] wrote {path2}")

    # ---------------- 逐位置接受率（机制证据） ----------------
    path3 = os.path.join(OUT, "acceptance_by_position.csv")
    with open(path3, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "workload",
                "batch_size",
                "position",
                "accepted",
                "drafted",
                "accept_ratio_at_pos",
                "n_samples",
            ]
        )
        for wl in WORKLOADS:
            for b in BATCH_SIZES:
                v = groups.get(("ngram", wl, b))
                if not v:
                    continue
                npos = max((len(x.get("accepted_per_pos") or []) for x in v), default=0)
                for p in range(npos):
                    acc = sum((x.get("accepted_per_pos") or [0] * npos)[p] for x in v)
                    dr = sum((x.get("draft_per_pos") or [0] * npos)[p] for x in v)
                    w.writerow(
                        [wl, b, p, acc, dr, round(acc / dr, 6) if dr else "", len(v)]
                    )
    print(f"[m19] wrote {path3}")

    # ---------------- baseline vs ngram 输出一致性 ----------------
    path4 = os.path.join(OUT, "output_equivalence.csv")
    with open(path4, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "workload",
                "batch_size",
                "repeat",
                "prompt_hash_match",
                "output_hash_match",
                "base_output_hash",
                "ngram_output_hash",
                "base_out_lens",
                "ngram_out_lens",
            ]
        )
        idx = {
            (r["config"], r["workload"], r["batch_size"], r["repeat"]): r for r in rows
        }
        for wl in WORKLOADS:
            for b in BATCH_SIZES:
                for r in range(REPEATS + WARMUP):
                    kb, kn = ("baseline", wl, b, r), ("ngram", wl, b, r)
                    if kb not in idx or kn not in idx:
                        continue
                    rb, rn = idx[kb], idx[kn]
                    w.writerow(
                        [
                            wl,
                            b,
                            r,
                            rb["batch_prompt_hash"] == rn["batch_prompt_hash"],
                            rb["batch_output_hash"] == rn["batch_output_hash"],
                            rb["batch_output_hash"],
                            rn["batch_output_hash"],
                            rb["actual_output_tokens"],
                            rn["actual_output_tokens"],
                        ]
                    )
    print(f"[m19] wrote {path4}")
    print(f"[m19] summarize done. measure samples: {len(rows)}")
    return True


def _first_divergence(a: list, b: list):
    """返回 (首个不同的下标, 公共前缀长度)；完全相等时下标为 None。"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i, i
    return (None, n) if len(a) == len(b) else (n, n)


def summarize_equivalence():
    """逐请求比对 baseline / ngram 的完整输出 token 序列，定位首个分歧下标。

    主实验只落 hash，只能回答"等不等"；本文件回答"差在哪一位、差多少"——
    这决定了该把它读作实现错误（首位就分歧 / 长度不对）还是数值差异
    （长尾分歧，与 M5 观察到的 compiled-vs-eager 第 5 token 起分叉同类）。
    """
    os.makedirs(OUT, exist_ok=True)
    sides = {}
    for mode in ("baseline", "ngram"):
        path = os.path.join(RAW, f"m19_equiv_{mode}.jsonl")
        if not os.path.exists(path):
            print(f"[m19] 缺少 {path} —— 一致性 probe 尚未执行。")
            return False
        with open(path) as f:
            sides[mode] = _measure([json.loads(x) for x in f if x.strip()])

    idx = {
        mode: {(r["workload"], r["batch_size"], r["repeat"]): r for r in rows}
        for mode, rows in sides.items()
    }
    keys = sorted(set(idx["baseline"]) & set(idx["ngram"]))
    out_rows = []
    for k in keys:
        rb, rn = idx["baseline"][k], idx["ngram"][k]
        if rb["batch_prompt_hash"] != rn["batch_prompt_hash"]:
            raise SystemExit(f"[m19][FAIL] {k} 两侧 prompt 不同，对照不成立")
        for i, (ab, an) in enumerate(
            zip(rb["output_token_ids"], rn["output_token_ids"])
        ):
            di, common = _first_divergence(ab, an)
            out_rows.append(
                {
                    "workload": k[0],
                    "batch_size": k[1],
                    "repeat": k[2],
                    "req_index": i,
                    "base_len": len(ab),
                    "ngram_len": len(an),
                    "exact_match": di is None,
                    "first_divergence_index": "" if di is None else di,
                    "common_prefix_len": common,
                    "common_prefix_frac": round(common / max(len(ab), 1), 6),
                    "base_hash": _h(ab),
                    "ngram_hash": _h(an),
                }
            )

    path = os.path.join(OUT, "output_equivalence_detail.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[m19] wrote {path} ({len(out_rows)} 个请求级比对)")

    n_match = sum(1 for r in out_rows if r["exact_match"])
    print(f"[m19] 完全一致 {n_match}/{len(out_rows)}")
    for wl in WORKLOADS:
        for b in BATCH_SIZES:
            v = [r for r in out_rows if r["workload"] == wl and r["batch_size"] == b]
            if not v:
                continue
            m = sum(1 for r in v if r["exact_match"])
            div = [r["first_divergence_index"] for r in v if not r["exact_match"]]
            extra = (
                f" 首个分歧下标 min={min(div)} median={statistics.median(div)} "
                f"max={max(div)}（输出长 {v[0]['base_len']}）"
                if div
                else ""
            )
            print(f"  {wl:<12} batch={b:<3} 一致 {m}/{len(v)}{extra}")
    # 长度必须一致；长度不一致属于功能问题而非数值差异
    bad_len = [r for r in out_rows if r["base_len"] != r["ngram_len"]]
    if bad_len:
        print(f"[m19][FAIL] {len(bad_len)} 个请求两侧输出长度不同：{bad_len[:3]}")
        return False

    _batch_shape_control(sides)
    return True


def _batch_shape_control(sides):
    """正交对照：同一侧内部，仅改变 batch 组成是否就会改变输出？

    这是判定 baseline↔ngram 分歧性质的关键控制组。req_index=0 的 prompt 只取决于
    workload，与 batch 无关，因此可以跨 batch 直接比对同一个请求的输出：
      · 若 baseline 自己换个 batch 就会分歧 → 证明 batch-shape 可以导致
        greedy divergence，并与 baseline↔ngram 现象一致；现有证据未发现明确
        Spec Decode correctness bug，但不是「已严格证明每一个 divergence
        都与 Spec Decode 无关」；
      · 若只有 ngram 侧分歧 → 才需要优先怀疑投机解码路径。
    同时报告同配置下跨 repeat 的可复现性，用来排除"随机性"这一竞争解释。
    """
    rows_out = []
    print("\n[m19] 正交对照 —— 仅改变 batch 组成（同一侧内部，req_index=0）：")
    for mode in ("baseline", "ngram"):
        by = {}
        for r in sides[mode]:
            by.setdefault((r["workload"], r["batch_size"]), {})[r["repeat"]] = r[
                "output_token_ids"
            ][0]
        for wl in WORKLOADS:
            base1 = by.get((wl, 1), {})
            for b in BATCH_SIZES[1:]:
                other = by.get((wl, b), {})
                shared = sorted(set(base1) & set(other))
                divs = [_first_divergence(base1[r], other[r])[0] for r in shared]
                same = sum(1 for d in divs if d is None)
                nz = [d for d in divs if d is not None]
                rows_out.append(
                    {
                        "config": mode,
                        "workload": wl,
                        "batch_a": 1,
                        "batch_b": b,
                        "n_compared": len(shared),
                        "n_identical": same,
                        "first_divergence_indices": nz,
                    }
                )
                extra = f"  首个分歧下标={nz}" if nz else ""
                print(
                    f"  {mode:<8} {wl:<12} batch1 vs batch{b:<3}: "
                    f"一致 {same}/{len(shared)}{extra}"
                )
        # 同配置跨 repeat 的可复现性
        for (wl, b), d in sorted(by.items()):
            outs = [d[r] for r in sorted(d)]
            rows_out.append(
                {
                    "config": mode,
                    "workload": wl,
                    "batch_a": b,
                    "batch_b": b,
                    "n_compared": len(outs),
                    "n_identical": sum(1 for o in outs if o == outs[0]),
                    "first_divergence_indices": [],
                }
            )
    path = os.path.join(OUT, "batch_shape_control.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        for r in rows_out:
            r["first_divergence_indices"] = json.dumps(r["first_divergence_indices"])
            w.writerow(r)
    print(f"[m19] wrote {path}")


def main():
    p = argparse.ArgumentParser(
        description="M19 spec decode: baseline vs ngram（需 GPU；一个进程一个引擎）"
    )
    p.add_argument(
        "--shard",
        choices=[
            "baseline",
            "ngram",
            "smoke",
            "equiv",
            "summarize",
            "summarize-equiv",
        ],
        required=True,
    )
    p.add_argument(
        "--mode", choices=["baseline", "ngram"], help="--shard smoke/equiv 必填"
    )
    a = p.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if a.shard == "baseline":
        _dump(run_side(spec=False), "m19_baseline")
    elif a.shard == "ngram":
        _dump(run_side(spec=True), "m19_ngram")
    elif a.shard == "equiv":
        if a.mode is None:
            p.error("--shard equiv 必须提供 --mode {baseline,ngram}")
        _dump(
            run_side(spec=(a.mode == "ngram"), equiv=True),
            f"m19_equiv_{a.mode}",
        )
    elif a.shard == "summarize-equiv":
        if not summarize_equivalence():
            raise SystemExit(1)
    elif a.shard == "smoke":
        if a.mode is None:
            p.error("--shard smoke 必须提供 --mode {baseline,ngram}")
        s = run_side(spec=(a.mode == "ngram"), smoke=True)
        os.makedirs(os.path.join(OUT, "smoke"), exist_ok=True)
        sp = os.path.join(OUT, "smoke", f"smoke_{a.mode}.jsonl")
        with open(sp, "w") as f:
            f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in s)
        print(f"\n[m19][SMOKE] {a.mode} -> {sp}")
        for x in s:
            print(
                f"  {x['workload']:<12} b={x['batch_size']} steps={x['engine_steps']:<4} "
                f"out={x['actual_output_tokens']} proposed={x['proposed_tokens']} "
                f"accepted={x['accepted_tokens']} rate={x['acceptance_rate']} "
                f"acc_len={x['acceptance_length']} rep4={x['output_repetition_4gram']} "
                f"in_prompt={x['output_4gram_in_prompt_ratio']} "
                f"ok={x['success']} {x['invalid_reason']}"
            )
    else:
        if not summarize():
            raise SystemExit(1)


if __name__ == "__main__":
    main()
