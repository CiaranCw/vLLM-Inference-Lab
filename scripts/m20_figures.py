"""M20 · 离线出图（无需 GPU / 不 import vllm）。

    python scripts/m20_figures.py

产出 figures/fig10_m20_same_runtime.png：
  左 TTFT p50/p99 vs offered QPS（C0 vs C1）
  右 E2E p50/p99 vs offered QPS
"""
from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SWEEP = os.path.join(ROOT, "outputs", "m20", "c_same_runtime_sweep.csv")
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)


def main():
    if not os.path.exists(SWEEP):
        print(f"skip: {SWEEP} missing")
        return 1
    rows = list(csv.DictReader(open(SWEEP)))
    by = {}
    for r in rows:
        by.setdefault(r["backend"], []).append(r)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = {
        "c0": ("#1f77b4", "C0 vllm serve"),
        "c1": ("#d62728", "C1 Triton+vLLM"),
    }
    for be, (color, label) in styles.items():
        rs = sorted(by.get(be, []), key=lambda x: float(x["offered_qps"]))
        if not rs:
            continue
        xs = [float(r["offered_qps"]) for r in rs]
        ax1.plot(xs, [float(r["ttft_p50_ms"]) for r in rs], "o-", color=color, label=f"{label} p50")
        ax1.plot(
            xs,
            [float(r["ttft_p99_ms"]) for r in rs],
            "s--",
            color=color,
            alpha=0.7,
            label=f"{label} p99",
        )
        ax2.plot(xs, [float(r["e2e_p50_ms"]) for r in rs], "o-", color=color, label=f"{label} p50")
        ax2.plot(
            xs,
            [float(r["e2e_p99_ms"]) for r in rs],
            "s--",
            color=color,
            alpha=0.7,
            label=f"{label} p99",
        )
    ax1.set(xlabel="offered QPS (Poisson)", ylabel="TTFT (ms)", title="same-runtime TTFT")
    ax2.set(xlabel="offered QPS (Poisson)", ylabel="E2E (ms)", title="same-runtime E2E latency")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(
        "M20-C: container vllm serve vs Triton+vLLM backend\n"
        "(Qwen2.5-0.5B, in256/out128, eager, same EngineArgs)",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(OUT, "fig10_m20_same_runtime.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
