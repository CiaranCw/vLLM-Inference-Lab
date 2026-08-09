"""M22 · 从既有实验数据离线生成核心图（无需 GPU / 模型 / vllm）。

    python scripts/m22_figures.py --tag h20_formal

产出 figures/fig9_m22_tp_scaling.png：
  左  · steady-state decode 单步延迟 vs batch（TP=1/2/4）—— 回答「小 batch 时 TP 为何更慢」
  中  · 吞吐 speedup 与 scaling efficiency vs batch —— 全部由 D2 离线派生，未额外跑实验
  右  · prefill 首 token 延迟 vs prompt 长度 —— 计算受限侧 TP 是否真能提速

**只读正式 CSV**，不重跑实验、不使用任何人造数据。
若只有一档 TP（例如本地预演），则跳过对比图并明确提示，**不画伪图**。
数据边界随 --tag 决定：`local_preflight` 是 RTX5080 单卡预演，**不是** H20 结果，
两者绝不放进同一张图。
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)
TP_COLORS = {1: "#d62728", 2: "#ff7f0e", 4: "#1f77b4"}


def _read(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build(tag: str) -> bool:
    dec = _read(f"outputs/m22/{tag}_decode.csv")
    pre = _read(f"outputs/m22/{tag}_prefill.csv")
    if not dec:
        print(f"skip：outputs/m22/{tag}_decode.csv 不存在（实验未执行）")
        return False
    tps = sorted({int(r["tp"]) for r in dec})
    if len(tps) < 2:
        print(
            f"skip fig9：{tag} 只有 TP={tps} 一档数据，"
            "TP 对比图无从谈起 —— 不生成伪图。"
        )
        return False

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.6))

    # 左：steady-state decode 单步延迟 vs batch
    batches = sorted({int(r["batch_size"]) for r in dec})
    for tp in tps:
        xs, ys, es = [], [], []
        for b in batches:
            row = next(
                (r for r in dec if int(r["tp"]) == tp and int(r["batch_size"]) == b),
                None,
            )
            if row and _f(row["steady_step_ms_median"]) is not None:
                xs.append(b)
                ys.append(_f(row["steady_step_ms_median"]))
                es.append(_f(row["steady_step_ms_stdev"]) or 0)
        if xs:
            ax1.errorbar(
                xs,
                ys,
                yerr=es,
                marker="o",
                capsize=3,
                label=f"TP={tp}",
                color=TP_COLORS.get(tp),
            )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(batches)
    ax1.set_xticklabels([str(b) for b in batches])
    ax1.set(
        xlabel="batch size (concurrent requests)",
        ylabel="steady-state decode step latency (ms)",
        title="decode: per-step wall latency\n(pure-decode steps only, batch verified constant)",
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # 中：speedup / efficiency（由 D2 离线派生）
    for tp in [t for t in tps if t != 1]:
        xs, sp, ef = [], [], []
        for b in batches:
            row = next(
                (r for r in dec if int(r["tp"]) == tp and int(r["batch_size"]) == b),
                None,
            )
            if row and _f(row.get("speedup_vs_tp1")) is not None:
                xs.append(b)
                sp.append(_f(row["speedup_vs_tp1"]))
                ef.append(_f(row["efficiency_vs_tp1"]))
        if xs:
            ax2.plot(
                xs, sp, marker="o", color=TP_COLORS.get(tp), label=f"TP={tp} speedup"
            )
            ax2.plot(
                xs,
                ef,
                marker="s",
                ls="--",
                color=TP_COLORS.get(tp),
                alpha=0.65,
                label=f"TP={tp} efficiency",
            )
    ax2.axhline(1.0, ls=":", lw=1, color="grey")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(batches)
    ax2.set_xticklabels([str(b) for b in batches])
    ax2.set(
        xlabel="batch size",
        ylabel="speedup / efficiency  (vs TP=1)",
        title="throughput speedup and scaling efficiency\n(derived offline from the same D2 runs)",
    )
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)

    # 右：prefill 首 token 延迟 vs prompt 长度
    if pre:
        plens = sorted({int(r["prompt_len"]) for r in pre})
        for tp in tps:
            xs, ys, es = [], [], []
            for p in plens:
                row = next(
                    (
                        r
                        for r in pre
                        if int(r["tp"]) == tp and int(r["prompt_len"]) == p
                    ),
                    None,
                )
                if row and _f(row["first_token_ms_median"]) is not None:
                    xs.append(p)
                    ys.append(_f(row["first_token_ms_median"]))
                    es.append(_f(row["first_token_ms_stdev"]) or 0)
            if xs:
                ax3.errorbar(
                    xs,
                    ys,
                    yerr=es,
                    marker="o",
                    capsize=3,
                    label=f"TP={tp}",
                    color=TP_COLORS.get(tp),
                )
        ax3.set_xscale("log", base=2)
        ax3.set_xticks(plens)
        ax3.set_xticklabels([str(p) for p in plens])
        ax3.set(
            xlabel="prompt length (tokens)",
            ylabel="in-process first-token latency (ms)",
            title="prefill: compute-bound side\n(batch=1, only prompt length varies)",
        )
        ax3.grid(alpha=0.3)
        ax3.legend(fontsize=8)
    else:
        ax3.text(0.5, 0.5, "no prefill data", ha="center", va="center")
        ax3.set_axis_off()

    fig.suptitle(
        f"M22 · Tensor Parallel scaling ({tag}) —— in-process, enforce_eager",
        fontweight="bold",
    )
    fig.tight_layout()
    path = os.path.join(OUT, f"fig9_m22_tp_scaling_{tag}.png")
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"figure written: {path}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="h20_formal")
    a = ap.parse_args()
    build(a.tag)
