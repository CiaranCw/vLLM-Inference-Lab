#!/usr/bin/env python3
"""M18 · W4A16（AWQ / GPTQ）量化对比 —— 单配置 driver。

设计冻结见 `notes/m18/quantization_w4a16.md`。要点：
  - **一个进程只建一个引擎**（AGENTS.md §4：KV 显存 del + empty_cache 也归还不掉）；
  - 同一进程内**分离** perf 与 quality 两阶段（perf 绝不开 prompt_logprobs）；
  - **显式冻结激活 dtype=bfloat16**：量化 checkpoint 的 config 声明 float16、BF16 基线声明
    bfloat16，用 dtype=auto 会把量化效应和激活 dtype 混在一起（Gate 0-c 已验证可强制）；
  - 显存/质量/性能三类测量直接**复用 M10 的实现**（`m10_quant.py`），保证口径可对接。

用法（每配置一个独立进程）：
    python scripts/m18_quant.py --model 0.5b --config Q0
    python scripts/m18_quant.py --model 1.5b --config Q1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m6_benchmark as m6  # noqa: E402
import m10_quant as m10  # noqa: E402  复用显存/质量/性能实现，避免重复造轮子
from vllm import LLMEngine  # noqa: E402
from vllm.engine.arg_utils import EngineArgs  # noqa: E402
from vllm.usage.usage_lib import UsageContext  # noqa: E402

OUT = "outputs/m18"

MODELS = {
    "0.5b": {
        "Q0": "Qwen/Qwen2.5-0.5B-Instruct",
        "Q1": "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        "Q2": "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
    },
    "1.5b": {
        "Q0": "Qwen/Qwen2.5-1.5B-Instruct",
        "Q1": "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        "Q2": "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
    },
}

# 公共 engine 参数：与 M10 的 COMMON 逐项一致，唯一有意的差别是 dtype 由 "auto" 改为显式
# "bfloat16"（见设计冻结 §3 V1）。**除模型与量化外不得再有第二个变量。**
COMMON = dict(
    dtype="bfloat16",
    max_model_len=2048,
    gpu_memory_utilization=0.6,
    tensor_parallel_size=1,
    enforce_eager=True,
    seed=0,
    max_num_seqs=64,
    max_num_batched_tokens=4096,
    enable_prefix_caching=False,
    async_scheduling=False,
    disable_log_stats=True,
)

# 冻结网格（设计 §4.2）：prompt 长度与 decode 场景与 M10 相同；
# 样本量取本项目通用口径 5 warmup + 15 measure（M10 是 8+30，故**不与 M10 绝对值相减**）。
PREFILL_LENGTHS = [128, 512, 2000]
DECODE_MATRIX = [(64, [1, 64]), (1024, [1, 8])]
WARMUP, REPS = 5, 15


def build(model_id: str) -> LLMEngine:
    args = EngineArgs(model=model_id, **COMMON)
    return LLMEngine.from_engine_args(args, usage_context=UsageContext.LLM_CLASS)


def dump_config(engine, model_key: str, cfg: str, model_id: str) -> dict:
    """把**生效**配置从引擎读回（不信 CLI 参数，只信引擎回读）。"""
    vc = engine.vllm_config
    mc, cc = vc.model_config, vc.cache_config
    hf_quant = None
    try:
        hf_quant = getattr(mc.hf_config, "quantization_config", None)
    except Exception:
        pass
    return {
        "config": cfg,
        "model_key": model_key,
        "model": model_id,
        "common_frozen": COMMON,
        "effective": {
            "dtype": str(mc.dtype),
            "quantization": mc.quantization,
            "hf_quantization_config": hf_quant,
            "kv_cache_dtype": str(cc.cache_dtype),
            "num_gpu_blocks": cc.num_gpu_blocks,
            "block_size": cc.block_size,
            "kv_cache_size_tokens": getattr(cc, "kv_cache_size_tokens", None),
            "attention_backend": os.environ.get("VLLM_ATTENTION_BACKEND", "<unset:default>"),
            "max_model_len": mc.max_model_len,
        },
        "grid": {
            "prefill_lengths": PREFILL_LENGTHS,
            "decode_matrix": [[c, b] for c, b in DECODE_MATRIX],
            "warmup": WARMUP,
            "measure": REPS,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--config", required=True, choices=["Q0", "Q1", "Q2"])
    ap.add_argument("--out-dir", default=OUT)
    args = ap.parse_args()

    model_id = MODELS[args.model][args.config]
    tag = f"{args.model}_{args.config}"
    for d in ("raw", "configs", "memory", "quality"):
        os.makedirs(os.path.join(args.out_dir, d), exist_ok=True)

    print(f"[m18 {tag}] building engine: {model_id}", flush=True)
    t0 = time.perf_counter()
    engine = build(model_id)
    init_s = round(time.perf_counter() - t0, 3)

    spy = m6.ScheduleSpy(m6.get_scheduler(engine))
    tf = m6.TokenFactory(engine)

    conf = dump_config(engine, args.model, args.config, model_id)
    conf["engine_init_s_single_obs"] = init_s
    json.dump(conf, open(os.path.join(args.out_dir, "configs", f"{tag}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[m18 {tag}] dtype={conf['effective']['dtype']} "
          f"quant={conf['effective']['quantization']} "
          f"kv_tokens={conf['effective']['kv_cache_size_tokens']} init={init_s}s", flush=True)

    # ---- 显存（复用 M10 的真实 tensor 分类统计）----
    mem = m10.mem_audit(engine)
    mem["config"] = tag
    json.dump(mem, open(os.path.join(args.out_dir, "memory", f"{tag}.json"), "w"),
              ensure_ascii=False, indent=2)
    ts = mem.get("tensor_sum", {})
    print(f"[m18 {tag}] weight_by_dtype={ts.get('weight_bytes_by_dtype')} "
          f"embed_lm_head_MiB={ts.get('embed_lm_head_MiB')}", flush=True)

    # ---- 阶段 1：quality（prompt_logprobs，不计时）----
    q = m10.quality_ppl(engine, tf, quick=False)
    q["config"] = tag
    json.dump(q, open(os.path.join(args.out_dir, "quality", f"{tag}.json"), "w"),
              ensure_ascii=False, indent=2)
    print(f"[m18 {tag}] PPL overall={q['overall']['ppl']} zh={q['zh']['ppl']} "
          f"en={q['en']['ppl']} scored={q['overall']['scored_tokens']}", flush=True)

    # ---- 阶段 2：perf（绝不开 prompt_logprobs）----
    raw_path = os.path.join(args.out_dir, "raw", f"{tag}_perf.jsonl")
    with open(raw_path, "w", encoding="utf-8") as raw:
        prefill = m10.perf_prefill(engine, tf, tag, PREFILL_LENGTHS, WARMUP, REPS, raw)
        decode = m10.perf_decode(engine, spy, tf, tag, DECODE_MATRIX, WARMUP, REPS, raw)
    spy.restore()

    perf = {"config": tag, "model": model_id, "prefill": prefill, "decode": decode}
    json.dump(perf, open(os.path.join(args.out_dir, "configs", f"{tag}_perf.json"), "w"),
              ensure_ascii=False, indent=2)
    for p in prefill:
        print(f"  prefill n={p['prompt_tokens']}: median={p['median_ms']}ms", flush=True)
    for d in decode:
        print(f"  decode ctx={d['context_len']} b={d['batch_size']}: median={d['median_ms']}ms "
              f"thru={d['throughput_tok_s_median']} valid={d['all_valid']}", flush=True)
    print(f"[m18 {tag}] done -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
