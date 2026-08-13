#!/usr/bin/env python3
"""M23 出图 —— 完全离线：只读已落盘 CSV/JSON，不需 GPU、不 import 引擎。

产出 figures/fig14_m23_cross_engine.png：
  (a) 单请求端到端延迟（lane 内对照 + Bootstrap 95% CI）
  (b) 输出 token 吞吐（batch64）
  (c) 冷启动时间与整卡显存峰值

**刻意不画**「跨 lane」的柱子放在一起比较——冻结稿禁止跨 lane 相减，
图上把两条 lane 用分隔与配色明确分开。

用法：python scripts/m23_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, FIG = "outputs/m23", "figures"
plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
C = {"V": "tab:blue", "T": "tab:orange"}


def main():
    d = json.load(open(os.path.join(OUT, "cross_engine_summary.json")))
    rows = {r["group"]: r for r in d["groups"]}
    comps = d["lane_comparisons"]
    order = ["V-E", "T-E", "V-O", "T-O"]
    colors = [C[g[0]] for g in order]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) 单请求端到端延迟
    ax = axes[0]
    vals = [rows[g]["single_e2e_latency_median_ms"] for g in order]
    ax.bar(range(4), vals, color=colors, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(1.5, color="grey", ls="--", lw=1)
    ax.text(0.5, max(vals) * 0.95, "OFF lane", ha="center", fontsize=8, color="grey")
    ax.text(2.5, max(vals) * 0.95, "ON lane", ha="center", fontsize=8, color="grey")
    ax.set_xticks(range(4))
    ax.set_xticklabels(order)
    ax.set_ylabel("single-request e2e latency, median (ms)")
    ax.set_title("(a) End-to-end latency\n(compare WITHIN a lane only)")

    # (b) 输出 token 吞吐
    ax = axes[1]
    vals = [rows[g]["output_token_throughput_tok_s"] for g in order]
    ax.bar(range(4), vals, color=colors, alpha=0.9)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=8)
    ax.axvline(1.5, color="grey", ls="--", lw=1)
    ax.set_xticks(range(4))
    ax.set_xticklabels(order)
    ax.set_ylabel("output token throughput, batch=64 (tok/s)")
    ax.set_title("(b) Output token throughput\n(compare WITHIN a lane only)")

    # (c) 冷启动 + 整卡显存峰值
    ax = axes[2]
    cold = [rows[g]["cold_start_s"] for g in order]
    ax.bar(range(4), cold, color=colors, alpha=0.9)
    for i, v in enumerate(cold):
        ax.text(i, v, f"{v:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(order)
    ax.set_ylabel("cold start (s)", color="black")
    ax2 = ax.twinx()
    mem = [rows[g]["gpu_total_mem_peak_mib"] for g in order]
    ax2.plot(range(4), mem, "k^--", ms=6, label="peak total GPU mem")
    ax2.set_ylabel("peak TOTAL GPU memory (MiB)")
    ax2.set_ylim(0, max(mem) * 1.25)
    ax2.grid(False)
    ax2.legend(fontsize=7, loc="lower right")
    ax.set_title("(c) Cold start + peak TOTAL GPU memory\n"
                 "(memory: whole-card only; per-component unavailable)")

    lane_txt = " | ".join(
        f"{k}: Δmedian {v['single_e2e_latency_ms']['T_minus_V_median_ms']:+.0f}ms "
        f"({v['single_e2e_latency_ms']['T_rel_vs_V_pct']:+.1f}%)"
        for k, v in comps.items())
    fig.suptitle(
        "M23 - vLLM vs TensorRT-LLM (both PyTorch runtimes), 2x2 execution-mode.  " + lane_txt +
        "\nstack-vs-stack, NOT kernel-level A/B; cross-lane subtraction forbidden; "
        "no difference may be called intrinsic engine speedup.  "
        "Scope: RTX 5080 sm_120 / WSL2 / Qwen2.5-0.5B / in256-out128 / containers",
        fontsize=7.5, y=1.06)
    fig.tight_layout()
    p = os.path.join(FIG, "fig14_m23_cross_engine.png")
    fig.savefig(p, bbox_inches="tight")
    print("->", p)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    main()
