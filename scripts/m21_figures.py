#!/usr/bin/env python3
"""M21 出图 —— 完全离线：只读已落盘数据，不需要 GPU、不 import vllm、不重跑实验。

产出：
  figures/fig11_m21_capacity.png            容量阶梯 + 采样窗口敏感性
  figures/fig12_m21_bottleneck_signature.png 四类瓶颈的指标签名

用法：python scripts/m21_figures.py
"""

import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs/m21"
RAW = os.path.join(OUT, "raw")
FIG = "figures"
SLO_MS = 300.0

plt.rcParams.update({"font.size": 9, "figure.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def worst_by_qps(runs, key):
    d = {}
    for r in runs:
        v = r.get(key)
        if v is None:
            continue
        d.setdefault(r["offered_qps"], []).append(v)
    return {q: max(vs) for q, vs in sorted(d.items())}


def median_by_qps(runs, key):
    d = {}
    for r in runs:
        v = r.get(key)
        if v is None:
            continue
        d.setdefault(r["offered_qps"], []).append(v)
    return {q: statistics.median(vs) for q, vs in sorted(d.items())}


def fig_capacity():
    long_runs = load_jsonl(os.path.join(RAW, "e3_shapeA_60s_runs.jsonl"))
    short_runs = load_jsonl(os.path.join(RAW, "e3_shapeA_short200_runs.jsonl"))
    if not long_runs:
        print("[fig] 缺少 60s 窗口数据，跳过 fig11")
        return

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    # (a) 容量拐点：worst-seed P99 TTFT
    ax = axes[0]
    for runs, label, style in ((long_runs, "60s window (M12 protocol)", "-o"),
                               (short_runs, "~5s window (200 samples)", "--s")):
        if not runs:
            continue
        w = worst_by_qps(runs, "client_p99_ttft_ms")
        ax.plot(list(w), list(w.values()), style, label=label, ms=4)
    ax.axhline(SLO_MS, color="crimson", lw=1, ls=":", label=f"frozen SLO {SLO_MS:.0f}ms")
    ax.set_xlabel("offered QPS (req/s)")
    ax.set_ylabel("worst-seed client P99 TTFT (ms)")
    ax.set_yscale("log")
    ax.set_title("(a) Capacity knee is highly sensitive\nto measurement window")
    ax.legend(fontsize=7)

    # (b) 服务端直方图分位数 vs 客户端真值
    ax = axes[1]
    cli = median_by_qps(long_runs, "client_p99_ttft_ms")
    srv = median_by_qps(long_runs, "server_hist_ttft_p99_ms")
    ax.plot(list(cli), list(cli.values()), "-o", ms=4, label="client ground truth P99 (np.percentile)")
    ax.plot(list(srv), list(srv.values()), "-^", ms=4,
            label="server histogram P99 (histogram_quantile)")
    ax.set_xlabel("offered QPS (req/s)")
    ax.set_ylabel("P99 TTFT (ms)")
    ax.set_title("(b) Server histogram P99 != client P99")
    ax.legend(fontsize=7)

    # (c) 队列与 GPU 忙碌
    ax = axes[2]
    wait = worst_by_qps(long_runs, "waiting_max")
    ax.plot(list(wait), list(wait.values()), "-o", color="tab:orange", ms=4, label="peak waiting")
    ax.set_xlabel("offered QPS (req/s)")
    ax.set_ylabel("peak queue length (requests)", color="tab:orange")
    ax2 = ax.twinx()
    gu = median_by_qps(long_runs, "dcgm_gpu_util_mean")
    ax2.plot(list(gu), list(gu.values()), "-^", color="tab:blue", ms=4, label="GPU util (window mean)")
    ax2.set_ylabel("window-mean DCGM GPU util, seed median (%)", color="tab:blue")
    ax2.set_ylim(0, 100)
    ax2.grid(False)
    ax.set_title("(c) Overload queues up first;\nGPU never saturates")

    fig.suptitle("M21 - single RTX 5080 / WSL2 / Qwen2.5-0.5B / enforce_eager / Shape A (in256,out128)",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG, "fig11_m21_capacity.png")
    fig.savefig(path, bbox_inches="tight")
    print("->", path)


def fig_bottleneck():
    p = os.path.join(OUT, "e4_bottleneck.json")
    if not os.path.exists(p):
        print("[fig] 缺少 e4_bottleneck.json，跳过 fig12")
        return
    runs = json.load(open(p))["runs"]
    runs = [r for r in runs if not r.get("server_start_failed")]
    if not runs:
        print("[fig] e4 无有效场景，跳过 fig12")
        return

    order = {"B1_cpu_submit": 0, "B2_gpu_compute": 1, "B3_kv_pressure": 2,
             "B3b_kv_long_output": 3, "B4_queue": 4}
    runs.sort(key=lambda r: (order.get(r["scenario"], 9),
                             -float(r.get("kv_ladder_step") or 1)))

    def label(r):
        base = {"B1_cpu_submit": "B1 cpu\nsubmit", "B2_gpu_compute": "B2 gpu\ncompute",
                "B3_kv_pressure": "B3 kv", "B3b_kv_long_output": "B3b kv\nlong-out",
                "B4_queue": "B4 queue"}[r["scenario"]]
        step = r.get("kv_ladder_step")
        return f"{base}\nutil={step}" if step else base

    labels = [label(r) for r in runs]
    panels = [
        ("peak GPU util, DCGM (%)", [r.get("dcgm_gpu_util_max") for r in runs], "tab:blue"),
        ("peak server-process CPU (%, 100=1 core)",
         [r.get("proc_cpu_percent_max") for r in runs], "tab:green"),
        ("peak KV cache usage (fraction)", [r.get("kv_usage_max") for r in runs], "tab:red"),
        ("peak queue length (requests)", [r.get("waiting_max") for r in runs], "tab:orange"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.2))
    for ax, (title, vals, color) in zip(axes, panels):
        vals = [0 if v is None else v for v in vals]
        ax.bar(range(len(vals)), vals, color=color, alpha=0.85)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_title(title, fontsize=8)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7)
        ax.margins(y=0.18)
    fig.suptitle("M21 - metric signature of each bottleneck class (which metric saturates first).  "
                 "Scope: single RTX 5080 / WSL2 / Qwen2.5-0.5B / enforce_eager",
                 fontsize=8, y=1.03)
    fig.tight_layout()
    path = os.path.join(FIG, "fig12_m21_bottleneck_signature.png")
    fig.savefig(path, bbox_inches="tight")
    print("->", path)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    fig_capacity()
    fig_bottleneck()
