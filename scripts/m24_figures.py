#!/usr/bin/env python3
"""M24 离线出图（回传后在本机跑，不需 GPU、不 import vllm）。

    python scripts/m24_figures.py --tag h20_formal

产出 figures/fig15_m24_dcp.png：
  左  · KV 可服务 token（DCP=1 vs 2）—— 复制因子 2→1
  中  · 稳态 decode 单步 vs prompt_len（batch=1/8）
  右  · DCP=2 − DCP=1 的 median difference + Bootstrap 95% CI

只读正式 CSV。这不是 Ring Attention。
"""

from __future__ import annotations

import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")

DCP_COLORS = {1: "#1f77b4", 2: "#d62728"}


def _read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.isfile(p):
        return None
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="h20_formal")
    a = p.parse_args()
    mem = _read(f"outputs/m24/{a.tag}_memory_kv.csv")
    dec = _read(f"outputs/m24/{a.tag}_decode.csv")
    eff = _read(f"outputs/m24/{a.tag}_decode_effect.csv")
    if not mem or not dec or not eff:
        raise SystemExit(
            f"缺 outputs/m24/{a.tag}_{{memory_kv,decode,decode_effect}}.csv"
        )
    dcps = sorted({int(r["dcp"]) for r in dec})
    if dcps != [1, 2]:
        raise SystemExit(f"需要 DCP=1 与 2 两档，实际 {dcps} —— 不画伪图")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"matplotlib 不可用: {e}")

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    # 左：KV tokens
    xs, ys = [], []
    for dcp in (1, 2):
        row = next((r for r in mem if int(r["dcp"]) == dcp), None)
        if row and _f(row.get("sched_kv_cache_size_tokens")) is not None:
            xs.append(f"DCP={dcp}")
            ys.append(_f(row["sched_kv_cache_size_tokens"]) / 1e6)
    ax1.bar(xs, ys, color=[DCP_COLORS[1], DCP_COLORS[2]], width=0.55)
    ax1.set(
        ylabel="KV cache size (million tokens)",
        title="D1: KV token capacity\n(replication 2→1; not more GPUs)",
    )
    ax1.grid(axis="y", alpha=0.3)
    if len(ys) == 2 and ys[0]:
        ax1.annotate(
            f"{ys[1] / ys[0]:.2f}×",
            xy=(1, ys[1]),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    # 中：steady step vs prompt_len
    prompts = sorted({int(r["prompt_len"]) for r in dec})
    for dcp in (1, 2):
        for batch, ls, mk in ((1, "-", "o"), (8, "--", "s")):
            xs, ys, es = [], [], []
            for p_len in prompts:
                row = next(
                    (
                        r
                        for r in dec
                        if int(r["dcp"]) == dcp
                        and int(r["prompt_len"]) == p_len
                        and int(r["batch_size"]) == batch
                    ),
                    None,
                )
                if row and _f(row["steady_step_ms_median"]) is not None:
                    xs.append(p_len)
                    ys.append(_f(row["steady_step_ms_median"]))
                    es.append(_f(row["steady_step_ms_stdev"]) or 0)
            if xs:
                ax2.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker=mk,
                    ls=ls,
                    capsize=3,
                    color=DCP_COLORS[dcp],
                    label=f"DCP={dcp} batch={batch}",
                )
    ax2.set(
        xlabel="prompt_len (context tokens)",
        ylabel="steady decode step median (ms)",
        title="D2: per-step wall latency\n(pure-decode steps, batch stable)",
    )
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=8)

    # 右：effect + CI
    labels, diffs, yerr_lo, yerr_hi = [], [], [], []
    for row in eff:
        p_len, b = int(row["prompt_len"]), int(row["batch_size"])
        d = _f(row["median_dcp2_minus_dcp1_ms"])
        lo, hi = _f(row["ci95_lo"]), _f(row["ci95_hi"])
        if d is None or lo is None or hi is None:
            continue
        labels.append(f"P={p_len}\nb={b}")
        diffs.append(d)
        yerr_lo.append(d - lo)
        yerr_hi.append(hi - d)
    ax3.axhline(0.0, color="0.5", lw=0.8)
    ax3.errorbar(
        range(len(diffs)),
        diffs,
        yerr=[yerr_lo, yerr_hi],
        fmt="o",
        capsize=3,
        color=DCP_COLORS[2],
    )
    ax3.set_xticks(range(len(labels)))
    ax3.set_xticklabels(labels, fontsize=8)
    ax3.set(
        ylabel="median(DCP=2) − median(DCP=1)  (ms)",
        title="DCP=2 minus DCP=1\n(positive = DCP=2 slower; CI excl. 0)",
    )
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"M24 DCP {a.tag}  ·  4×H20 / 1.5B / eager  ·  not Ring Attention",
        fontsize=11,
        y=1.02,
    )
    os.makedirs(OUT, exist_ok=True)
    dest = os.path.join(OUT, "fig15_m24_dcp.png")
    fig.tight_layout()
    fig.savefig(dest, dpi=120, bbox_inches="tight")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
