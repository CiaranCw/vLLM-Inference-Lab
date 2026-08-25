"""M5 执行模式验证 driver。

目标：在本机（RTX 5080 / sm_120 / torch 2.11+cu130 / vLLM ee0da84）上，
用同一个单请求分别以 enforce_eager=True / False 运行，落盘引擎实际解析后的
compilation_config（执行模式的真相来源），并跑通一次生成用于对照输出。

它只做两件事：
  1) 构建引擎后，从存活引擎里读出 vllm_config.compilation_config（不是猜、是实际值）；
  2) 跑一次单请求，记录 output_token_ids（供 eager vs compiled 正确性对照）。

配合 VLLM_LAB_TRACE=1 时，源码里的 exec/capture、exec/dispatch 钩子会额外输出
“捕获了哪些 cudagraph 尺寸”“每步用的运行时模式 + padding”。

用法：
  # eager（当前基线，默认）
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python scripts/m5_exec_mode.py --enforce-eager --out outputs/m5/eager_config.json

  # compiled（vLLM 默认 O2：torch.compile + cudagraph）
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python scripts/m5_exec_mode.py --no-enforce-eager --out outputs/m5/compiled_config.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

import vllm
from vllm import LLM, SamplingParams

DEFAULT_PROMPT = "请用一句话解释大模型推理中的 KV Cache。"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM M5 execution-mode probe")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=8)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/m5/exec_config.json")
    return p.parse_args()


def _get(obj, name, default=None):
    try:
        return getattr(obj, name)
    except Exception:
        return default


def dump_compilation_config(llm: LLM) -> dict:
    """从存活引擎里读实际解析后的执行模式配置（真相来源）。"""
    # V1：LLM -> LLMEngine(.llm_engine) -> vllm_config
    engine = _get(llm, "llm_engine")
    vconf = _get(engine, "vllm_config")
    if vconf is None:
        # 兜底：尝试其它路径
        vconf = _get(_get(engine, "engine_core"), "vllm_config")
    if vconf is None:
        return {"error": "无法从引擎读取 vllm_config"}

    cc = _get(vconf, "compilation_config")
    mc = _get(vconf, "model_config")

    def name_of(v):
        return getattr(v, "name", None) or (str(v) if v is not None else None)

    info = {
        "optimization_level": name_of(_get(vconf, "optimization_level")),
        "model_config.enforce_eager": _get(mc, "enforce_eager"),
        "compilation": {
            "mode": name_of(_get(cc, "mode")),
            "cudagraph_mode": name_of(_get(cc, "cudagraph_mode")),
            "backend": _get(cc, "backend"),
            "use_inductor_graph_partition": _get(cc, "use_inductor_graph_partition"),
            "ir_enable_torch_wrap": _get(cc, "ir_enable_torch_wrap"),
            "custom_ops": _get(cc, "custom_ops"),
            "splitting_ops": _get(cc, "splitting_ops"),
            "cudagraph_capture_sizes": _get(cc, "cudagraph_capture_sizes"),
            "max_capture_size": _get(cc, "max_cudagraph_capture_size", None),
            "cudagraph_num_of_warmups": _get(cc, "cudagraph_num_of_warmups"),
        },
    }
    return info


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("[m5] vllm_file:", os.path.realpath(vllm.__file__))
    print("[m5] gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print("[m5] enforce_eager(requested):", args.enforce_eager)
    print("=" * 80)

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    t_init = time.perf_counter() - t0

    resolved = dump_compilation_config(llm)
    print("[m5] resolved compilation config:")
    print(json.dumps(resolved, ensure_ascii=False, indent=2, default=str))

    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, seed=args.seed)

    t1 = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params)
    t_gen = time.perf_counter() - t1

    out = outputs[0]
    completion = out.outputs[0]
    result = {
        "requested_enforce_eager": args.enforce_eager,
        "resolved": resolved,
        "num_prompt_tokens": len(out.prompt_token_ids),
        "output_token_ids": list(completion.token_ids),
        "num_output_tokens": len(completion.token_ids),
        "output_text": completion.text,
        "finish_reason": completion.finish_reason,
        "timings_s": {"engine_init": round(t_init, 4), "generate": round(t_gen, 4)},
        "vllm_file": os.path.realpath(vllm.__file__),
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    print("=" * 80)
    print("[m5] output_token_ids =", result["output_token_ids"])
    print("[m5] output_text      =", repr(result["output_text"]))
    print("[m5] timings_s        =", result["timings_s"])
    print("[m5] saved ->", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()
