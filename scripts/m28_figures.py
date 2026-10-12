#!/usr/bin/env python3
"""M28 · 从已落盘的派生 CSV 离线生成图（无需 GPU / 模型 / vllm）。

  fig_m28_offload_ttft.png   A：各路径 online TTFT median（误差棒到 p95）vs 前缀长度
  fig_m28_faults.png         B：B1 / B3 各演练从注入到 health 非 200 / EngineCore 退出 / API 退出 / 端口关闭的时间，
                             以及在途请求完整完成数

复现：python scripts/m28_figures.py [--derived-dir outputs/m28] [--fig-dir figures]
边界：单卡 RTX 5080 / sm_120 / WSL2 / Qwen2.5-0.5B / enforce_eager / 串行 HTTP 流式客户端。
"""

import argparse
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SERIES = [("offload", "cold", "offload: cold (full recompute)", "#ff7f0e", "o"),
          ("offload", "gpu_hot", "offload: GPU hot hit", "#1f77b4", "s"),
          ("offload", "cpu_hit", "offload: GPU miss + CPU hit", "#2ca02c", "^"),
          ("base", "recompute_evicted", "no connector: recompute after GPU eviction", "#d62728", "x")]
EVENTS = [("t_health_non200_s", "/health != 200", "#9467bd"), ("t_engine_exit_s", "EngineCore exited", "#d62728"),
          ("t_api_exit_s", "API process exited", "#1f77b4"), ("t_port_closed_s", "port closed", "#7f7f7f")]


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def _f(v):
    return None if v in ("", None, "None") else float(v)


def fig_a(rows, out):
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for arm, path, label, color, mk in SERIES:
        pts = sorted((int(r["p_len"]), _f(r["ttft_median_ms"]), _f(r["ttft_p95_ms"]))
                     for r in rows if r["arm"] == arm and r["path"] == path and _f(r["ttft_median_ms"]) is not None)
        if not pts:
            continue
        xs, med, p95 = zip(*pts)
        ax.errorbar(xs, med, yerr=[[0] * len(xs), [p - m for p, m in zip(p95, med)]], label=label,
                    color=color, marker=mk, capsize=3, lw=1.4)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("shared prefix length P (tokens)")
    ax.set_ylabel("online TTFT median (ms); bar top = p95")
    ax.set_title("M28-A: OffloadingConnector paths (n=15 per point, serial requests)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.01, "RTX 5080 / WSL2 / Qwen2.5-0.5B / enforce_eager; every sample path proven by Prometheus counter deltas",
             fontsize=7, color="gray")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_b(rows, out):
    rows = [r for r in rows if r["drill"] in ("b1", "b3") and not r.get("error")]
    if not rows:
        return False
    rows.sort(key=lambda r: (r["drill"], _f(r["shutdown_timeout"]) or -1, int(r["rep"])))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 0.45 * len(rows) + 1.8), gridspec_kw={"width_ratios": [3, 1]},
                                 sharey=True)
    ys = range(len(rows))
    for key, label, color in EVENTS:
        vals = [(_f(r[key]), y) for r, y in zip(rows, ys) if _f(r[key]) is not None]
        if vals:
            ax.scatter([v for v, _ in vals], [y for _, y in vals], label=label, color=color, s=28)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8)
    ax.set_xlabel("seconds after fault injection")
    ax.set_title("M28-B: observable events after injection")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(fontsize=7, loc="lower right")
    bx.barh(list(ys), [int(r["n_completed_full"]) for r in rows], color="#2ca02c")
    bx.set_xlim(0, max(int(r["n_inflight"]) for r in rows))
    bx.set_xlabel("in-flight requests fully completed")
    bx.grid(alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--derived-dir", default=os.path.join(ROOT, "outputs", "m28"))
    ap.add_argument("--fig-dir", default=os.path.join(ROOT, "figures"))
    a = ap.parse_args()
    os.makedirs(a.fig_dir, exist_ok=True)
    lat = _read(os.path.join(a.derived_dir, "m28_a_latency.csv"))
    if lat:
        fig_a(lat, os.path.join(a.fig_dir, "fig_m28_offload_ttft.png"))
        print("[m28-figures] A 图已生成")
    else:
        print("[m28-figures] 无 A CSV（未执行），不画 A 图")
    b = _read(os.path.join(a.derived_dir, "m28_b_drills.csv"))
    if b and fig_b(b, os.path.join(a.fig_dir, "fig_m28_faults.png")):
        print("[m28-figures] B 图已生成")
    else:
        print("[m28-figures] 无 B CSV（未执行），不画 B 图")
    return 0


if __name__ == "__main__":
    sys.exit(main())
