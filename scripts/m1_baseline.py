"""M1 最小离线推理基线（single request）。

目标：用一个固定的单请求跑通 vLLM V1 主链路，并把环境指纹、配置、
输入输出、耗时和显存等结构化保存，作为后续 Trace / 对照的基线。

用法示例：
    # 第二次（缓存命中后）正式基线，带 DEBUG 日志
    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_LOGGING_LEVEL=DEBUG \
        python scripts/m1_baseline.py 2>&1 | tee logs/m1/baseline_debug.log

    # Prefill/Decode 实验（只生成 2 个 token）
    VLLM_LAB_TRACE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python scripts/m1_baseline.py --max-tokens 2

本脚本不加载/评测能力，只做主链路观察。禁止量化格式与模型兜底。
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

# 固定 prompt（可用 --prompt 覆盖）
DEFAULT_PROMPT = "请用一句话解释大模型推理中的 KV Cache。"


def gpu_mem_snapshot() -> dict:
    """设备级显存快照（MiB）。"""
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "device_free_MiB": round(free / 2**20, 1),
        "device_total_MiB": round(total / 2**20, 1),
        "device_used_MiB": round((total - free) / 2**20, 1),
        "proc_allocated_MiB": round(torch.cuda.memory_allocated() / 2**20, 1),
        "proc_reserved_MiB": round(torch.cuda.memory_reserved() / 2**20, 1),
    }


def env_fingerprint() -> dict:
    return {
        "sys_executable": sys.executable,
        "vllm_file": os.path.realpath(vllm.__file__),
        "vllm_version": getattr(vllm, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "env": {
            k: os.environ.get(k)
            for k in (
                "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_LOGGING_LEVEL",
                "VLLM_LAB_TRACE",
                "VLLM_ATTENTION_BACKEND",
                "VLLM_USE_FLASHINFER_SAMPLER",
            )
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM M1 minimal baseline")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="outputs/m1/baseline_result.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    config = {
        "model": args.model,
        "prompt": args.prompt,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": 1,
        "temperature": 0.0,
        "seed": args.seed,
        "enforce_eager": args.enforce_eager,
        "dtype": "auto",
    }

    print("=" * 80)
    print("[m1_baseline] env:", json.dumps(env_fingerprint(), ensure_ascii=False))
    print("[m1_baseline] config:", json.dumps(config, ensure_ascii=False))
    print("=" * 80)

    mem_before = gpu_mem_snapshot()

    # ---- Engine 初始化 ----
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
    mem_after_init = gpu_mem_snapshot()

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    # ---- generate ----
    t1 = time.perf_counter()
    outputs = llm.generate([args.prompt], sampling_params)
    t_gen = time.perf_counter() - t1
    mem_after_gen = gpu_mem_snapshot()

    out = outputs[0]
    completion = out.outputs[0]
    result = {
        "env": env_fingerprint(),
        "config": config,
        "prompt": args.prompt,
        "prompt_token_ids": list(out.prompt_token_ids),
        "num_prompt_tokens": len(out.prompt_token_ids),
        "output_token_ids": list(completion.token_ids),
        "num_output_tokens": len(completion.token_ids),
        "output_text": completion.text,
        "finish_reason": completion.finish_reason,
        "timings_s": {
            "engine_init": round(t_init, 4),
            "generate": round(t_gen, 4),
            "total": round(t_init + t_gen, 4),
        },
        "gpu_memory": {
            "before": mem_before,
            "after_init": mem_after_init,
            "after_generate": mem_after_gen,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("[m1_baseline] num_prompt_tokens =", result["num_prompt_tokens"])
    print("[m1_baseline] num_output_tokens =", result["num_output_tokens"])
    print("[m1_baseline] output_token_ids  =", result["output_token_ids"])
    print("[m1_baseline] output_text       =", repr(result["output_text"]))
    print("[m1_baseline] finish_reason     =", result["finish_reason"])
    print("[m1_baseline] timings_s         =", result["timings_s"])
    print("[m1_baseline] mem after_init    =", mem_after_init)
    print("[m1_baseline] saved ->", args.out)
    print("=" * 80)


if __name__ == "__main__":
    main()
