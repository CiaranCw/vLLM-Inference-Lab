#!/usr/bin/env python3
"""M18 出图 —— 完全离线：只读已落盘 CSV，不需要 GPU、不 import vllm。

产出 figures/fig13_m18_w4a16.png：
  (a) 权重显存：整体降幅 vs「可量化部分」降幅（解释为什么达不到理论 75%）
  (b) 速度：相对 BF16 基线的变化（正=更慢），标注 CI 是否排除 0
  (c) 质量：ΔPPL%

用法：python scripts/m18_figures.py
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT, FIG = "outputs/m18", "figures"
MODELS = ["0.5b", "1.5b"]
QCFG = ["Q1", "Q2"]
LABEL = {"Q1": "AWQ", "Q2": "GPTQ"}

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def read(name):
    with open(os.path.join(OUT, name)) as f:
        return list(csv.DictReader(f))


def main():
    mem, perf, qual = read("memory.csv"), read("perf.csv"), read("quality.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    # ---- (a) 权重显存 ----
    ax = axes[0]
    labels, whole, part = [], [], []
    for m in MODELS:
        for c in QCFG:
            r = next(x for x in mem if x["model"] == m and x["config"] == c)
            labels.append(f"{m}\n{LABEL[c]}")
            whole.append(float(r["weight_total_vs_Q0_pct"]))
            part.append(float(r["quantizable_part_vs_Q0_pct"]))
    idx = range(len(labels))
    ax.bar([i - 0.2 for i in idx], whole, width=0.4, label="whole model", color="tab:blue")
    ax.bar([i + 0.2 for i in idx], part, width=0.4, label="quantizable part only",
           color="tab:cyan")
    ax.axhline(-75, color="crimson", ls=":", lw=1, label="4-bit theoretical (-75%)")
    for i, (w, p) in enumerate(zip(whole, part)):
        ax.text(i - 0.2, w - 2, f"{w:.0f}%", ha="center", va="top", fontsize=7)
        ax.text(i + 0.2, p - 2, f"{p:.0f}%", ha="center", va="top", fontsize=7)
    ax.set_xticks(list(idx))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("weight memory change vs BF16 (%)")
    ax.set_title("(a) Whole-model saving is diluted\nby the unquantized embed/lm_head")
    ax.legend(fontsize=7, loc="lower right")

    # ---- (b) 速度 ----
    ax = axes[1]
    grids = [("prefill", "128", "", "pf128"), ("prefill", "2000", "", "pf2000"),
             ("decode", "64", "1", "dec c64b1"), ("decode", "1024", "8", "dec c1024b8")]
    width = 0.2
    for j, (m, c) in enumerate([(m, c) for m in MODELS for c in QCFG]):
        vals, sig = [], []
        for exp, a1, a2, _ in grids:
            r = next(x for x in perf if x["model"] == m and x["config"] == c
                     and x["experiment"] == exp
                     and (x["prompt_tokens"] == a1 if exp == "prefill" else x["context_len"] == a1)
                     and (True if exp == "prefill" else x["batch_size"] == a2))
            vals.append(float(r["rel_change_vs_Q0_pct"]))
            sig.append(r["ci_excludes_zero"] == "yes")
        pos = [i + (j - 1.5) * width for i in range(len(grids))]
        # 不用 hatch 区分显著性：hatch 只能整条 series 设，会把「某一格不显著」
        # 误画成「整条 series 不显著」。改为逐柱标 ns（CI 含 0）。
        ax.bar(pos, vals, width=width, label=f"{m} {LABEL[c]}", alpha=0.9)
        for x, v, s in zip(pos, vals, sig):
            if not s:
                ax.text(x, v + 0.6, "ns", ha="center", fontsize=6, color="grey")
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(range(len(grids)))
    ax.set_xticklabels([g[3] for g in grids], fontsize=7)
    ax.set_ylabel("latency change vs BF16 (%)   positive = slower")
    ax.set_title("(b) W4A16 is slower everywhere here\n(ns = Bootstrap 95% CI covers 0)")
    ax.legend(fontsize=7)

    # ---- (c) 质量 ----
    ax = axes[2]
    labels, dppl = [], []
    for m in MODELS:
        for c in QCFG:
            r = next(x for x in qual if x["model"] == m and x["config"] == c)
            labels.append(f"{m}\n{LABEL[c]}")
            dppl.append(float(r["delta_ppl_pct"]))
    ax.bar(range(len(labels)), dppl, color="tab:orange", width=0.55)
    for i, v in enumerate(dppl):
        ax.text(i, v, f"{v:+.2f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("token-weighted PPL change vs BF16 (%)")
    ax.set_title("(c) Quality cost\n(tiled corpus: relative delta only)")
    ax.margins(y=0.2)

    fig.suptitle("M18 - W4A16 (AWQ/GPTQ, 4bit/group128, Marlin) vs BF16.  "
                 "Scope: single RTX 5080 sm_120 / WSL2 / enforce_eager / in-process / "
                 "activation dtype frozen to bfloat16", fontsize=8, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG, "fig13_m18_w4a16.png")
    fig.savefig(path, bbox_inches="tight")
    print("->", path)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    main()
