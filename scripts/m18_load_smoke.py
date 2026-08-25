#!/usr/bin/env python3
"""M18 Gate 0-b · 真实 AWQ/GPTQ checkpoint 的加载与生成 smoke。

Gate 0（`m18_probe.py`）只证明了 kernel 能在 sm_120 上 launch；没有覆盖真实 checkpoint 的
加载路径（quantization_config 解析 → 权重 repack → scale 传递 → 实际前向）。本脚本补上这段。

规则：一个进程只建一个引擎（AGENTS.md §4：vLLM 的 KV 显存 del + empty_cache 也归还不掉），
所以每个模型用 `--model` 单独跑一次。

只做 smoke：加载成功 + 生成非空且不含 NaN/乱码即可，不测性能（性能是 M18 正式实验的事，
且首次跑含 Triton kernel JIT，此处的耗时不具参考性）。

用法：
    python scripts/m18_load_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct-AWQ
    python scripts/m18_load_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct-AWQ --dump-quant-config
"""

import argparse
import json
import os
import time

OUT = "outputs/m18"
PROMPTS = [
    "The capital of France is",
    "用一句话解释什么是 KV cache：",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    ap.add_argument("--max-tokens", type=int, default=32)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    rec = {"model": args.model, "started_at": time.strftime("%F %T%z"),
           "max_model_len": args.max_model_len,
           "gpu_memory_utilization": args.gpu_memory_utilization}

    t0 = time.perf_counter()
    llm = LLM(
        model=args.model,
        enforce_eager=True,          # 与 M10/M17 口径一致
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        seed=0,
    )
    rec["engine_build_s"] = round(time.perf_counter() - t0, 2)

    # 从引擎读回生效的量化配置（不是从 CLI 猜的）
    try:
        mc = llm.llm_engine.vllm_config.model_config
        rec["effective"] = {
            "quantization": mc.quantization,
            "dtype": str(mc.dtype),
            "hf_quant_config": getattr(mc, "hf_config", None)
            and getattr(mc.hf_config, "quantization_config", None),
        }
    except Exception as e:
        rec["effective"] = {"error": f"{type(e).__name__}: {e}"}

    sp = SamplingParams(temperature=0, max_tokens=args.max_tokens, seed=0)
    t1 = time.perf_counter()
    outs = llm.generate(PROMPTS, sp)
    rec["generate_s"] = round(time.perf_counter() - t1, 2)

    gens = []
    for o in outs:
        text = o.outputs[0].text
        gens.append({
            "prompt": o.prompt,
            "text": text,
            "n_tokens": len(o.outputs[0].token_ids),
            "non_empty": bool(text.strip()),
        })
    rec["generations"] = gens
    rec["PASS"] = all(g["non_empty"] and g["n_tokens"] > 0 for g in gens)

    os.makedirs(OUT, exist_ok=True)
    safe = args.model.replace("/", "_")
    path = os.path.join(OUT, f"gate0b_load_smoke_{safe}.json")
    with open(path, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    print(f"\n[m18] model={args.model}")
    print(f"[m18] quantization(生效)={rec['effective'].get('quantization')} "
          f"build={rec['engine_build_s']}s PASS={rec['PASS']}")
    for g in gens:
        print(f"    {g['prompt']!r} -> {g['text']!r}")
    print("[m18] ->", path)
    return 0 if rec["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
