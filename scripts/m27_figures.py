#!/usr/bin/env python3
"""M27 · 离线出图（只读已落盘 CSV；不需 GPU、不 import vllm、不重跑实验）。

    python scripts/m27_figures.py --tag h20_formal

缺任一所需 CSV → 打印 PENDING 并以退出码 2 结束，**不生成占位图**。
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402

FIG_DIR = os.path.join(C.ROOT, "figures")


def load_csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fig_e1(tag, rows, plt):
    rows = sorted(rows, key=lambda r: int(r["context"]))
    L = [int(r["context"]) for r in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(L, [fnum(r["nixl_bytes_median"]) / 2**20 for r in rows], "o-", label="NIXL measured")
    ax[0].plot(L, [fnum(r["theory_bytes_blocks"]) / 2**20 for r in rows], "s--", label="theory (blocks)")
    ax[0].set(xscale="log", xlabel="context tokens", ylabel="MiB / request", title="KV bytes")
    ax[0].legend()
    ax[1].plot(L, [fnum(r["nixl_xfer_ms_median"]) for r in rows], "o-", label="xfer")
    ax[1].plot(L, [fnum(r["nixl_post_ms_median"]) for r in rows], "s-", label="post")
    ax[1].set(xscale="log", xlabel="context tokens", ylabel="ms", title="NIXL time (median)")
    ax[1].legend()
    ax[2].plot(L, [fnum(r["bw_GBps_median"]) for r in rows], "o-")
    ax[2].set(xscale="log", xlabel="context tokens", ylabel="GB/s", title="bytes / xfer time")
    fig.suptitle(f"M27 E1 transfer scaling ({tag}; 1P1D / TP1 / Qwen2.5-7B / eager)")
    fig.tight_layout()
    return fig


def fig_e2(tag, rows, plt):
    arms = sorted({r["arm"] for r in rows})
    plens = sorted({int(r["p_len"]) for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    w = 0.8 / max(len(arms), 1)
    for k, arm in enumerate(arms):
        ys = []
        for pl in plens:
            r = next((x for x in rows if x["arm"] == arm and int(x["p_len"]) == pl), None)
            ys.append(fnum(r["max_stall_ms_median"]) if r else None)
        ax[0].bar([i + k * w for i in range(len(plens))], [y or 0 for y in ys], w, label=arm)
        ts = []
        for pl in plens:
            r = next((x for x in rows if x["arm"] == arm and int(x["p_len"]) == pl), None)
            ts.append(fnum(r["inj_ttft_ms_median"]) if r else None)
        ax[1].plot(plens, ts, "o-", label=arm)
    ax[0].set_xticks([i + w * (len(arms) - 1) / 2 for i in range(len(plens))], [str(p) for p in plens])
    ax[0].set(xlabel="injected prefill tokens", ylabel="ms", title="decode max stall (median)")
    ax[0].legend(fontsize=8)
    ax[1].set(xlabel="injected prefill tokens", ylabel="ms", title="injected request TTFT (median)")
    ax[1].legend(fontsize=8)
    fig.suptitle(f"M27 E2 controlled interference ({tag})")
    fig.tight_layout()
    return fig


def fig_e3(tag, rows, plt):
    shapes = sorted({r["shape"] for r in rows})
    arms = sorted({r["arm"] for r in rows})
    fig, ax = plt.subplots(2, len(shapes), figsize=(4.2 * len(shapes), 7), squeeze=False)
    for j, sh in enumerate(shapes):
        for arm in arms:
            pts = {}
            for r in rows:
                if r["shape"] == sh and r["arm"] == arm:
                    pts.setdefault(fnum(r["load_frac"]), []).append(r)
            fr = sorted(pts)
            med = lambda k: [sorted(fnum(x[k]) for x in pts[f])[len(pts[f]) // 2] for f in fr]  # noqa: E731
            ax[0][j].plot(fr, med("ttft_p99"), "o-", label=arm)
            ax[1][j].plot(fr, med("itl_p99"), "o-", label=arm)
        ax[0][j].set(title=f"{sh}: TTFT p99", ylabel="ms")
        ax[1][j].set(title=f"{sh}: ITL p99", xlabel="load fraction of C_cal", ylabel="ms")
    ax[0][0].legend(fontsize=7)
    fig.suptitle(f"M27 E3 online Poisson ({tag}; median over 3 seeds)")
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    specs = [("e1_transfer", "fig16_m27_e1_transfer", fig_e1),
             ("e2_interference", "fig17_m27_e2_interference", fig_e2),
             ("e3_online", "fig18_m27_e3_online", fig_e3)]
    have = {k: load_csv(os.path.join(C.OUT, f"{a.tag}_{k}.csv")) for k, _, _ in specs}
    missing = [k for k, v in have.items() if not v]
    if len(missing) == len(specs):
        print(f"[figures] PENDING：outputs/m27/{a.tag}_*.csv 全部缺失（实验未执行），不生成任何图")
        return 2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    for k, fname, fn in specs:
        if not have[k]:
            print(f"[figures] PENDING：缺 {a.tag}_{k}.csv，跳过 {fname}")
            continue
        fig = fn(a.tag, have[k], plt)
        out = os.path.join(FIG_DIR, f"{fname}_{a.tag}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        print(f"[figures] wrote {os.path.relpath(out, C.ROOT)}")
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
