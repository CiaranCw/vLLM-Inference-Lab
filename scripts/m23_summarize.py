#!/usr/bin/env python3
"""M23 · 离线汇总（不占 GPU、不 import 引擎）。

硬约束（冻结稿 v4/v5 §5.0.2）：
  - 正式跨引擎比较只允许同 lane：`T-E vs V-E` 与 `T-O vs V-O`。
  - 允许 `V-O vs V-E`、`T-O vs T-E` 作为同引擎 execution-mode ablation。
  - 禁止交叉 lane（`T-E vs V-O` / `T-O vs V-E`）做 engine speedup。
  - 差值不得称为「纯 engine intrinsic speedup」。
  - 显存只报 total（host nvidia-smi 整卡口径），分项 `unavailable`，禁止残差推导。

产出：`outputs/m23/cross_engine.csv`、`outputs/m23/cross_engine_summary.json`

用法：python scripts/m23_summarize.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics

OUT, RAW = "outputs/m23", "outputs/m23/raw"
LANES = {"E": ("V-E", "T-E"), "O": ("V-O", "T-O")}
OUT_TOKENS, N_PROMPTS = 128, 64


def bootstrap_median_diff(a, b, n=10000, seed=0):
    """median(a) − median(b) 的 Bootstrap 95% CI（与 M17/M18/M20 同口径）。"""
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
    return {"median_diff": statistics.median(a) - statistics.median(b),
            "ci95_lo": lo, "ci95_hi": hi, "ci_excludes_zero": not (lo <= 0 <= hi)}


def load(group):
    p = os.path.join(RAW, f"bench_{group}.json")
    d = json.load(open(p))
    r = d["record"]
    if not r or "error" in r:
        raise SystemExit(f"{group} 无有效记录：{(r or {}).get('error')}")
    return d, r


def metrics(d, r):
    single, batch = r["single_latency_ms"], r["batch_wall_ms"]
    return {
        "group": r["group"], "engine": r["engine"], "lane": r["lane"],
        "engine_version": r.get("engine_version"),
        # ---- Primary metrics ----
        "cold_start_s": r["cold_start_s"],
        "single_e2e_latency_median_ms": round(statistics.median(single), 2),
        "single_e2e_latency_p95_ms": round(sorted(single)[int(len(single) * 0.95) - 1], 2),
        "batch64_wall_median_ms": round(statistics.median(batch), 2),
        "request_throughput_req_s": round(N_PROMPTS / (statistics.median(batch) / 1e3), 2),
        "output_token_throughput_tok_s": round(
            N_PROMPTS * OUT_TOKENS / (statistics.median(batch) / 1e3), 1),
        "gpu_total_mem_peak_mib": d["gpu_total_memory_mib"]["peak_during_run"],
        "gpu_total_mem_baseline_mib": d["gpu_total_memory_mib"]["baseline_before_container"],
        # ---- 校验 ----
        "out_len_all_equal": r["out_len_all_equal"], "n_out_checked": r["n_out_checked"],
        "n_single_samples": len(single), "n_batch_samples": len(batch),
        "_single_raw": single, "_batch_raw": batch,
    }


def main():
    rows, raws = [], {}
    for g in ("V-E", "T-E", "V-O", "T-O"):
        d, r = load(g)
        m = metrics(d, r)
        raws[g] = {"single": m.pop("_single_raw"), "batch": m.pop("_batch_raw"),
                   "effective": r.get("effective", {})}
        rows.append(m)

    csv_path = os.path.join(OUT, "cross_engine.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", csv_path)

    # ---- lane 内对照（T vs V），不跨 lane ----
    comps = {}
    for lane, (v, t) in LANES.items():
        cmp_ = {}
        for key, unit in (("single", "single_e2e_latency_ms"), ("batch", "batch64_wall_ms")):
            bs = bootstrap_median_diff(raws[t][key], raws[v][key])
            vm = statistics.median(raws[v][key])
            cmp_[unit] = {
                "V_median_ms": round(vm, 2),
                "T_median_ms": round(statistics.median(raws[t][key]), 2),
                "T_minus_V_median_ms": round(bs["median_diff"], 2),
                "ci95_lo": round(bs["ci95_lo"], 2), "ci95_hi": round(bs["ci95_hi"], 2),
                "ci_excludes_zero": bs["ci_excludes_zero"],
                "T_rel_vs_V_pct": round(100 * bs["median_diff"] / vm, 1),
            }
        rv = next(r for r in rows if r["group"] == v)
        rt = next(r for r in rows if r["group"] == t)
        cmp_["cold_start_s"] = {"V": rv["cold_start_s"], "T": rt["cold_start_s"],
                                "T_minus_V_s": round(rt["cold_start_s"] - rv["cold_start_s"], 2)}
        cmp_["gpu_total_mem_peak_mib"] = {
            "V": rv["gpu_total_mem_peak_mib"], "T": rt["gpu_total_mem_peak_mib"],
            "T_minus_V_mib": rt["gpu_total_mem_peak_mib"] - rv["gpu_total_mem_peak_mib"],
            "note": "整卡口径；分项 unavailable，禁止残差推导"}
        cmp_["output_token_throughput_tok_s"] = {
            "V": rv["output_token_throughput_tok_s"], "T": rt["output_token_throughput_tok_s"]}
        comps[f"T-{lane} vs V-{lane}"] = cmp_

    summary = {
        "generated_at": __import__("time").strftime("%F %T%z"),
        "design": "2×2 execution-mode（冻结稿 v4 §5.0）",
        "workload": {"shape": "in256/out128", "n_prompts_batch": N_PROMPTS,
                     "single_samples": rows[0]["n_single_samples"],
                     "batch_samples": rows[0]["n_batch_samples"]},
        "groups": rows,
        "effective_configs": {g: raws[g]["effective"] for g in raws},
        "lane_comparisons": comps,
        "wording_discipline": [
            "正式跨引擎比较只允许 T-E vs V-E 与 T-O vs V-O。",
            "允许 V-O vs V-E、T-O vs T-E 作为同引擎 execution-mode ablation。",
            "**禁止交叉 lane**（T-E vs V-O / T-O vs V-E）做 engine speedup。",
            "任一差值**不得**称为「纯 engine intrinsic speedup」——"
            "OFF lane 的差值仍混着两套 runtime 的实现差异；ON lane 还叠加了两侧优化栈内容本就不同。",
            "本对照是 **stack-vs-stack**，不是 kernel 级 A/B：两个容器的 CUDA/PyTorch/驱动栈与"
            "引擎版本均不同（vLLM 0.24.0+092c4842.dev / TRT-LLM 1.3.0rc24）。",
            "显存只报整卡 total；分项 unavailable（TRT executor 在独立 runtime，主进程 torch 计数为 0）。",
            "V-O 的一切结论必须带 limitation：V-E/V-O 各自 across-process 6/6 确定性，"
            "但同引擎 V-O vs V-E exact-match 仅 1/6 —— 两条 execution path **不数值等价**，"
            "只能说未观察到明显 functional degradation。",
        ],
    }
    p = os.path.join(OUT, "cross_engine_summary.json")
    json.dump(summary, open(p, "w"), ensure_ascii=False, indent=2)
    print("->", p)

    # ---- 打印 ----
    print(f"\n{'group':>6} {'cold_s':>8} {'single_med_ms':>14} {'batch64_ms':>11} "
          f"{'req/s':>8} {'tok/s':>9} {'peak_MiB':>9}")
    for r in rows:
        print(f"{r['group']:>6} {r['cold_start_s']:>8.1f} "
              f"{r['single_e2e_latency_median_ms']:>14.1f} {r['batch64_wall_median_ms']:>11.1f} "
              f"{r['request_throughput_req_s']:>8.1f} {r['output_token_throughput_tok_s']:>9.1f} "
              f"{r['gpu_total_mem_peak_mib']:>9}")
    for name, c in comps.items():
        print(f"\n=== {name} （lane 内对照）===")
        for k in ("single_e2e_latency_ms", "batch64_wall_ms"):
            x = c[k]
            print(f"  {k}: V={x['V_median_ms']} T={x['T_median_ms']} "
                  f"ΔT−V={x['T_minus_V_median_ms']:+} ({x['T_rel_vs_V_pct']:+}%) "
                  f"CI[{x['ci95_lo']}, {x['ci95_hi']}] 排除0={x['ci_excludes_zero']}")
        print(f"  cold_start: V={c['cold_start_s']['V']}s T={c['cold_start_s']['T']}s "
              f"(Δ{c['cold_start_s']['T_minus_V_s']:+}s)")
        print(f"  peak total GPU mem: V={c['gpu_total_mem_peak_mib']['V']} "
              f"T={c['gpu_total_mem_peak_mib']['T']} MiB "
              f"(Δ{c['gpu_total_mem_peak_mib']['T_minus_V_mib']:+})")
        print(f"  output tok/s: V={c['output_token_throughput_tok_s']['V']} "
              f"T={c['output_token_throughput_tok_s']['T']}")


if __name__ == "__main__":
    main()
