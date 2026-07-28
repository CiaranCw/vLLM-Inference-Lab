"""M16 · 从既有实验 CSV 离线生成核心图（无需 GPU / 模型）。

读取 M6/M10/M12 的 CSV，产出 figures/ 下 4 张图，供 README 直观呈现：
  fig1_m6_decode.png    decode 单步延迟 & 吞吐 vs batch（eager vs compiled）
  fig2_m6_prefill.png   prefill 延迟 vs 输入长度（eager vs compiled）
  fig3_m10_quant.png    量化对比：decode 延迟(反常变慢) + 长 prefill 延迟(fp8 变快)
  fig4_m12_capacity.png 在线容量：offered QPS → P99 TTFT 拐点 + SLO 线 + 最大稳定 QPS

复现：source .venv/bin/activate && python scripts/m16_figures.py
数据源固定（本机 sm_120 / eager / Qwen2.5-0.5B），图仅由 CSV 生成，不重跑实验。
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


def _read(path):
    with open(os.path.join(ROOT, path)) as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# fig1: M6 decode latency & throughput vs batch (eager vs compiled)
# --------------------------------------------------------------------------- #
def fig1():
    rows = [r for r in _read("outputs/m6/baseline.csv") if r["experiment"] == "decode"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for mode, color in (("eager", "#d62728"), ("compiled", "#1f77b4")):
        m = sorted(
            [r for r in rows if r["mode"] == mode], key=lambda r: int(r["dim"])
        )
        b = [int(r["dim"]) for r in m]
        lat = [_f(r["latency_ms_median"]) for r in m]
        std = [_f(r["latency_ms_stdev"]) for r in m]
        thr = [_f(r["throughput_tok_s_median"]) for r in m]
        ax1.errorbar(b, lat, yerr=std, marker="o", capsize=3, label=mode, color=color)
        ax2.plot(b, thr, marker="o", label=mode, color=color)
    ax1.set(
        xlabel="batch size", ylabel="decode step latency (ms)",
        title="Decode step latency (near-flat)",
    )
    ax1.set_xscale("log", base=2)
    ax1.axhline(7, ls="--", lw=0.8, color="grey")
    ax1.text(1, 7.15, "~7ms eager floor = CPU submit (M7)", fontsize=8, color="grey")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax2.set(
        xlabel="batch size", ylabel="throughput (tok/s)",
        title="Decode throughput (near-linear; compiled ~2.5-2.85x)",
    )
    ax2.set_xscale("log", base=2)
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.suptitle(
        "M6 · Decode: eager vs compiled (RTX 5080 / Qwen2.5-0.5B)", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_m6_decode.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig2: M6 prefill latency vs input length (eager vs compiled)
# --------------------------------------------------------------------------- #
def fig2():
    rows = [r for r in _read("outputs/m6/baseline.csv") if r["experiment"] == "prefill"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for mode, color in (("eager", "#d62728"), ("compiled", "#1f77b4")):
        m = sorted(
            [r for r in rows if r["mode"] == mode], key=lambda r: int(r["dim"])
        )
        x = [int(r["dim"]) for r in m]
        lat = [_f(r["latency_ms_median"]) for r in m]
        ax.plot(x, lat, marker="o", label=mode, color=color)
    ax.axvspan(128, 256, alpha=0.08, color="green")
    ax.text(140, 20, "compiled speedup only ≤128\n(≥256 exceeds capture cap → NONE≈eager)",
            fontsize=8)
    ax.set(
        xlabel="prompt length (tokens)", ylabel="prefill latency (ms)",
        title="M6 · Prefill latency vs input length",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_m6_prefill.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig3: M10 quantization — decode slowdown anomaly + prefill fp8 speedup
# --------------------------------------------------------------------------- #
def fig3():
    rows = _read("outputs/m10/latency.csv")
    configs = ["C0T", "C1", "C2", "C2dyn"]  # C0T = same-backend baseline
    labels = {"C0T": "BF16 (TRITON)", "C1": "FP8 W8A8", "C2": "FP8 KV", "C2dyn": "FP8 KV dyn"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # decode ctx64 across batches
    batches = ["ctx64_b1", "ctx64_b8", "ctx64_b64"]
    xlab = ["b1", "b8", "b64"]
    width = 0.2
    for i, cfg in enumerate(configs):
        vals = []
        for b in batches:
            hit = [r for r in rows if r["config"] == cfg and r["dim"] == b]
            vals.append(_f(hit[0]["median_ms"]) if hit else 0)
        ax1.bar([x + i * width for x in range(len(batches))], vals, width, label=labels[cfg])
    ax1.set_xticks([x + 1.5 * width for x in range(len(batches))])
    ax1.set_xticklabels(xlab)
    ax1.set(ylabel="decode latency (ms)", title="Decode ctx64 (FP8 W8A8 SLOWER: +36% @b1)")
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=8)

    # prefill len2000
    vals = []
    for cfg in configs:
        hit = [r for r in rows if r["config"] == cfg and r["dim"] == "len2000"]
        vals.append(_f(hit[0]["median_ms"]) if hit else 0)
    ax2.bar([labels[c] for c in configs], vals, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"])
    ax2.set(ylabel="prefill latency (ms)", title="Prefill len2000 (FP8 W8A8 FASTER: -27%)")
    ax2.grid(alpha=0.3, axis="y")
    ax2.tick_params(axis="x", labelrotation=20, labelsize=8)
    fig.suptitle(
        "M10 · Quantization is workload-dependent (honest anomaly)", fontweight="bold"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_m10_quant.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig4: M12 online capacity — offered QPS -> P99 TTFT knee
# --------------------------------------------------------------------------- #
def fig4():
    rows = [r for r in _read("outputs/m12/sweep.csv") if r["shape"] == "256/128"]
    coarse = {}
    fine = {}
    for r in rows:
        rid = r["run_id"]
        q = _f(r["target_qps"])
        p99 = _f(r["p99_ttft_ms"])
        if q is None or p99 is None:
            continue  # skip burst (qps=inf)
        if "coarse" in rid:
            coarse[q] = p99
        elif "fine" in rid:
            fine.setdefault(q, []).append(p99)  # worst-seed later

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    if coarse:
        xs = sorted(coarse)
        ax.plot(xs, [coarse[x] for x in xs], marker="o", color="#1f77b4",
                label="coarse (1 seed)")
    if fine:
        xs = sorted(fine)
        ax.plot(xs, [max(fine[x]) for x in xs], marker="s", color="#d62728",
                label="fine (3-seed worst)")
    ax.axhline(300, ls="--", color="black")
    ax.text(2, 320, "SLO: P99 TTFT ≤ 300 ms", fontsize=9)
    ax.axvspan(40, 44, alpha=0.15, color="orange")
    ax.text(40.3, ax.get_ylim()[1] * 0.55, "max stable\nQPS ≈ 40\n(capacity (40,44])", fontsize=8)
    ax.set(
        xlabel="offered QPS (Poisson)", ylabel="P99 TTFT (ms)",
        title="M12 · Online capacity: overload = tail-latency, not throughput collapse",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_m12_capacity.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    print("figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
