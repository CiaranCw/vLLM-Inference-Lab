"""M19 · 从既有实验数据离线生成核心图（无需 GPU / 模型 / vllm）。

读取 M19 的正式 CSV，产出 figures/ 下 2 张图：
  fig7_m19_acceptance.png  接受率（含逐位置衰减）+ 每输出 token 的 engine step 数
  fig8_m19_speedup.png     完成延迟 / TPOT / 输出吞吐 的 baseline↔ngram 对照

只画两张：M19 主线的两条主结论是「额外并行 verification 换更少串行 execution rounds」
与「Acceptance 重要但非充分；Coverage + Round Cost 共同决定 speedup」。
其余（不变量、输出一致性）是布尔判定或表格事实，画图不增信息。

**只读正式 CSV**：不重跑实验、不 import vllm、不使用任何人造数据。
复现：source .venv/bin/activate && python scripts/m19_figures.py
数据边界：单卡 RTX 5080 / sm_120 / WSL2 / Qwen2.5-0.5B / enforce_eager / in-process。
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

BASE_C, NGRAM_C = "#d62728", "#1f77b4"
HIGH_C, LOW_C = "#2ca02c", "#ff7f0e"
BATCHES = [1, 4, 16]
WORKLOADS = ["high_repeat", "low_repeat"]


def _read(rel):
    with open(os.path.join(ROOT, rel)) as f:
        return list(csv.DictReader(f))


def _key(rows, *fields):
    return {tuple(r[f] for f in fields): r for r in rows}


def fig7():
    """接受率 + 机制（engine step 数）。"""
    summary = _key(
        _read("outputs/m19/spec_decode_summary.csv"), "config", "workload", "batch_size"
    )
    pos = _read("outputs/m19/acceptance_by_position.csv")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.5))
    x = range(len(BATCHES))
    width = 0.36

    # 左：整体接受率
    for i, (wl, c) in enumerate(((WORKLOADS[0], HIGH_C), (WORKLOADS[1], LOW_C))):
        vals = [
            float(summary[("ngram", wl, str(b))]["acceptance_rate_median"])
            for b in BATCHES
        ]
        ax1.bar([j + i * width for j in x], vals, width, label=wl, color=c)
        for j, v in zip(x, vals):
            ax1.text(j + i * width, v + 0.015, f"{v:.2f}", ha="center", fontsize=8)
    ax1.set_ylim(0, 1.18)
    ax1.set_xticks([j + width / 2 for j in x])
    ax1.set_xticklabels([f"batch={b}" for b in BATCHES])
    ax1.set(
        ylabel="acceptance rate (accepted / proposed)",
        title="N-gram acceptance by workload/config\n(not 'batch raises acceptance'; prompt composition confounds)",
    )
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=8)

    # 中：逐位置接受率（draft 越靠后越难命中）
    for wl, c in ((WORKLOADS[0], HIGH_C), (WORKLOADS[1], LOW_C)):
        for b, ls in zip(BATCHES, ("-", "--", ":")):
            rows = sorted(
                (r for r in pos if r["workload"] == wl and r["batch_size"] == str(b)),
                key=lambda r: int(r["position"]),
            )
            if not rows:
                continue
            ax2.plot(
                [int(r["position"]) for r in rows],
                [float(r["accept_ratio_at_pos"]) for r in rows],
                marker="o",
                ls=ls,
                color=c,
                label=f"{wl} b={b}",
            )
    ax2.set_ylim(0, 1.08)
    ax2.set_xticks([0, 1, 2])
    ax2.set(
        xlabel="draft position (0 = first speculative token)",
        ylabel="accept ratio at position",
        title="acceptance decays along the draft\n(high_repeat stays saturated)",
    )
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, ncol=2)

    # 右：每输出 token 需要几次 engine execution round（强 proxy，非严格 target forward）
    # 三组并列：baseline（两 workload 恒为 1.0，画一条即可）+ ngram 的两个 workload。
    w3 = 0.27
    series = [
        ("baseline (same for both workloads)", WORKLOADS[0], "baseline", BASE_C, ""),
        (f"ngram · {WORKLOADS[0]}", WORKLOADS[0], "ngram", HIGH_C, "//"),
        (f"ngram · {WORKLOADS[1]}", WORKLOADS[1], "ngram", LOW_C, "//"),
    ]
    for i, (label, wl, cfg, c, hatch) in enumerate(series):
        vals = [
            float(summary[(cfg, wl, str(b))]["steps_per_output_token_median"])
            for b in BATCHES
        ]
        ax3.bar(
            [j + i * w3 for j in x],
            vals,
            w3,
            label=label,
            color=c,
            hatch=hatch,
            edgecolor="white",
        )
        for j, v in zip(x, vals):
            ax3.text(j + i * w3, v + 0.02, f"{v:.2f}", ha="center", fontsize=7.5)
    ax3.axhline(1.0, ls="--", lw=0.8, color="grey")
    ax3.set_ylim(0, 1.22)
    ax3.set_xticks([j + w3 for j in x])
    ax3.set_xticklabels([f"batch={b}" for b in BATCHES])
    ax3.set(
        ylabel="engine execution rounds per output token",
        title="mechanism: fewer serial execution rounds\n(baseline ≡ 1.0 by construction; not pure target-forward count)",
    )
    ax3.grid(alpha=0.3, axis="y")
    ax3.legend(fontsize=7)

    fig.suptitle(
        "M19 · N-gram speculative decoding: acceptance and serial execution-depth reduction",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig7_m19_acceptance.png"), dpi=130)
    plt.close(fig)


def fig8():
    """性能：完成延迟 / TPOT / 输出吞吐。"""
    eff = _key(_read("outputs/m19/spec_decode_effect.csv"), "workload", "batch_size")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.5, 4.5))
    x = range(len(BATCHES))
    width = 0.36

    def paired(ax, base_col, ngram_col, ylabel, title, sig_col=None):
        for i, (cfg, col, c) in enumerate(
            (("baseline", base_col, BASE_C), ("ngram", ngram_col, NGRAM_C))
        ):
            for wl, alpha in ((WORKLOADS[0], 1.0), (WORKLOADS[1], 0.55)):
                vals = [float(eff[(wl, str(b))][col]) for b in BATCHES]
                off = i * width
                ax.plot(
                    [j + off for j in x],
                    vals,
                    marker="o" if wl == WORKLOADS[0] else "s",
                    ls="-" if cfg == "baseline" else "--",
                    color=c,
                    alpha=alpha,
                    label=f"{cfg} · {wl}",
                )
        ax.set_xticks([j + width / 2 for j in x])
        ax.set_xticklabels([f"batch={b}" for b in BATCHES])
        ax.set(ylabel=ylabel, title=title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    paired(
        ax1,
        "base_completion_median_ms",
        "ngram_completion_median_ms",
        "request completion latency (ms)",
        "completion latency (all diffs: Bootstrap 95% CI excludes 0)",
    )
    paired(
        ax2,
        "base_tpot_median_ms",
        "ngram_tpot_median_ms",
        "amortized TPOT (ms / output token; not streaming ITL)",
        "amortized TPOT (all diffs: CI excludes 0)",
    )

    # 右：相对变化（完成延迟降幅 / 吞吐增幅），并标出接受率
    for i, (wl, c) in enumerate(((WORKLOADS[0], HIGH_C), (WORKLOADS[1], LOW_C))):
        red = [float(eff[(wl, str(b))]["completion_rel_change_pct"]) for b in BATCHES]
        ax3.bar([j + i * width for j in x], red, width, label=wl, color=c)
        for j, b in zip(x, BATCHES):
            ar = float(eff[(wl, str(b))]["acceptance_rate_ngram"])
            ax3.text(
                j + i * width,
                red[j] + 1.2,
                f"{red[j]:.0f}%\nacc={ar:.2f}",
                ha="center",
                fontsize=7.5,
            )
    ax3.set_ylim(0, 95)
    ax3.set_xticks([j + width / 2 for j in x])
    ax3.set_xticklabels([f"batch={b}" for b in BATCHES])
    ax3.set(
        ylabel="completion latency reduction (%)",
        title="latency reduction co-varies with acceptance\n(but acceptance is not sufficient; batch≤16, still CPU-submit bound)",
    )
    ax3.grid(alpha=0.3, axis="y")
    ax3.legend(fontsize=8)

    fig.suptitle(
        "M19 · baseline vs N-gram: latency and throughput "
        "(single RTX 5080 / eager / Qwen2.5-0.5B / in-process)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig8_m19_speedup.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig7()
    fig8()
    print("figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
