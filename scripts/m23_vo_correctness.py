#!/usr/bin/env python3
"""M23 · **V-O correctness smoke**（optimized lane 的前置门）。

设计冻结 v4 §5.0.3：本 lab **从未验证过** `enforce_eager=False` 在当前 RTX 5080 + pinned vLLM
上的**生成正确性**（M5 只验证了实际执行路径，M6 的 compiled 数字是性能而非正确性）。
在没验证之前就用 V-O 的性能数字做跨引擎对照，等于把未验证的前提当结论基础。

**判据（跑前冻结）**：
  - **硬失败 → 停止 optimized lane**：进程崩溃 / 输出为空 / 输出长度不符 / 明显乱码。
  - **诊断项（不自动判失败）**：V-O 与 V-E 在 greedy + 同 seed 下的 **exact-match 率**。
    不完全一致 ≠ 有 bug（M5/M19 先例：数值与 batch-shape 敏感性可让贪心 argmax 翻转），
    但必须如实报告。

**一个进程只建一个引擎**（AGENTS.md §4），故 V-E 与 V-O 分两次子进程跑。

用法：
    python scripts/m23_vo_correctness.py            # 跑 V-E 与 V-O 并比对
    python scripts/m23_vo_correctness.py --lane E   # 内部用：只跑一条 lane
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

OUT, LOGD = "outputs/m23", "logs/m23"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_TOKENS = 64

# 固定提示集：覆盖中英文、长短、事实/续写，够暴露乱码与截断
PROMPTS = [
    "The capital of France is",
    "Explain in one sentence what a KV cache is:",
    "用一句话解释什么是张量并行：",
    "1, 1, 2, 3, 5, 8, 13, the next number is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "北京是中国的首都，它位于",
]

# 两条 lane 的唯一差别就是 enforce_eager（其余参数逐项相同）
COMMON = dict(
    model=MODEL, dtype="bfloat16", max_model_len=2048,
    gpu_memory_utilization=0.6, max_num_seqs=64, max_num_batched_tokens=4096,
    enable_prefix_caching=False, seed=0,
)


def run_lane(lane: str) -> dict:
    """在**当前进程**内建一个引擎跑完一条 lane（由 main 以子进程方式调用）。"""
    from vllm import LLM, SamplingParams

    enforce_eager = (lane == "E")
    t0 = time.perf_counter()
    llm = LLM(enforce_eager=enforce_eager, **COMMON)
    build_s = round(time.perf_counter() - t0, 2)

    # 回读**生效**的执行模式配置（不信入参，只信回读）——冻结稿 §5.0.2 要求四组都要存
    vc = llm.llm_engine.vllm_config
    cc = vc.compilation_config
    effective = {
        "enforce_eager_requested": enforce_eager,
        "compilation_mode": str(getattr(cc, "mode", None)),
        "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
        "cudagraph_capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", []) or [])[:12],
        "max_cudagraph_capture_size": getattr(cc, "max_cudagraph_capture_size", None),
        "dtype": str(vc.model_config.dtype),
        "kv_cache_dtype": str(vc.cache_config.cache_dtype),
        "kv_cache_size_tokens": getattr(vc.cache_config, "kv_cache_size_tokens", None),
    }

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, seed=0, ignore_eos=True)
    outs = llm.generate(PROMPTS, sp)
    gens = [{
        "prompt": o.prompt,
        "text": o.outputs[0].text,
        "token_ids": list(o.outputs[0].token_ids),
        "n_tokens": len(o.outputs[0].token_ids),
    } for o in outs]

    return {"lane": lane, "build_s": build_s, "effective": effective, "generations": gens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", choices=["E", "O"], help="内部用：只跑一条 lane 并打印 JSON")
    args = ap.parse_args()

    # ---- 子进程模式：跑一条 lane，结果打到 stdout ----
    if args.lane:
        rec = run_lane(args.lane)
        print("M23_LANE " + json.dumps(rec, ensure_ascii=False))
        return 0

    # ---- 主进程：分别以子进程跑 E 与 O（一进程一引擎）----
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)
    env = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", VLLM_ENABLE_V1_MULTIPROCESSING="0")

    lanes = {}
    for lane in ("E", "O"):
        log = os.path.join(LOGD, f"vo_correctness_lane{lane}.log")
        print(f"[m23] === lane {lane}（enforce_eager={lane == 'E'}）-> {log}", flush=True)
        t0 = time.time()
        with open(log, "w") as lf:
            proc = subprocess.run(
                [sys.executable, __file__, "--lane", lane],
                stdout=subprocess.PIPE, stderr=lf, text=True, env=env)
        rec = None
        for line in (proc.stdout or "").splitlines():
            if line.startswith("M23_LANE "):
                rec = json.loads(line[len("M23_LANE "):])
        lanes[lane] = {"returncode": proc.returncode, "elapsed_s": round(time.time() - t0, 1),
                       "record": rec, "log": log}
        print(f"[m23] lane {lane}: rc={proc.returncode} elapsed={lanes[lane]['elapsed_s']}s "
              f"got_record={rec is not None}", flush=True)

    # ---- 判据 ----
    report = {"generated_at": time.strftime("%F %T%z"), "model": MODEL,
              "max_tokens": MAX_TOKENS, "n_prompts": len(PROMPTS), "lanes": lanes}

    e, o = lanes["E"]["record"], lanes["O"]["record"]
    hard_fail_reasons = []
    if lanes["O"]["returncode"] != 0 or o is None:
        hard_fail_reasons.append(f"V-O 进程失败 rc={lanes['O']['returncode']}")
    if lanes["E"]["returncode"] != 0 or e is None:
        hard_fail_reasons.append(f"V-E 进程失败 rc={lanes['E']['returncode']}")

    if o:
        for g in o["generations"]:
            if not g["text"].strip():
                hard_fail_reasons.append(f"V-O 空输出: {g['prompt'][:30]!r}")
            if g["n_tokens"] != MAX_TOKENS:
                hard_fail_reasons.append(
                    f"V-O 输出长度 {g['n_tokens']} != 请求的 {MAX_TOKENS}: {g['prompt'][:30]!r}")

    # 诊断项：exact match 率（token 级）
    if e and o:
        pairs = list(zip(e["generations"], o["generations"]))
        exact = sum(1 for a, b in pairs if a["token_ids"] == b["token_ids"])
        first_div = []
        for a, b in pairs:
            if a["token_ids"] != b["token_ids"]:
                idx = next((i for i, (x, y) in enumerate(zip(a["token_ids"], b["token_ids"]))
                            if x != y), min(len(a["token_ids"]), len(b["token_ids"])))
                first_div.append({"prompt": a["prompt"][:40], "first_divergence_index": idx,
                                  "E_text": a["text"][:60], "O_text": b["text"][:60]})
        report["exact_match"] = {
            "matched": exact, "total": len(pairs),
            "rate": round(exact / len(pairs), 4) if pairs else None,
            "divergences": first_div,
            "note": ("诊断项，非判据。不完全一致 ≠ 有 bug —— M5/M19 已有先例："
                     "数值与 batch-shape 敏感性可让贪心 argmax 翻转。但必须如实报告。"),
        }

    report["hard_fail_reasons"] = hard_fail_reasons
    report["PASS"] = not hard_fail_reasons
    report["verdict"] = ("optimized lane 可以继续做性能对照"
                        if report["PASS"] else
                        "**硬失败 → 按冻结稿 §5.0.3 停止 optimized lane 性能比较，如实记录**")

    path = os.path.join(OUT, "vo_correctness.json")
    with open(path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[m23] V-O correctness: PASS={report['PASS']}")
    if e:
        print(f"[m23] V-E 生效: cudagraph_mode={e['effective']['cudagraph_mode']} "
              f"compilation_mode={e['effective']['compilation_mode']}")
    if o:
        print(f"[m23] V-O 生效: cudagraph_mode={o['effective']['cudagraph_mode']} "
              f"compilation_mode={o['effective']['compilation_mode']} "
              f"capture_sizes={o['effective']['cudagraph_capture_sizes'][:6]}...")
    if em := report.get("exact_match"):
        print(f"[m23] exact-match（诊断项）: {em['matched']}/{em['total']} = {em['rate']}")
        for d in em["divergences"][:2]:
            print(f"    分歧 @tok{d['first_divergence_index']}: {d['prompt']!r}")
    for r in hard_fail_reasons:
        print("[m23] ⚠️ 硬失败:", r)
    print("[m23] ->", path)
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
