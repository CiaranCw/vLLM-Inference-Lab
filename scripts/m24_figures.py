#!/usr/bin/env python3
"""M24 离线出图（回传后在本机跑，不需 GPU、不 import vllm）。"""

from __future__ import annotations

import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="h20_formal")
    a = p.parse_args()
    src = os.path.join(ROOT, "outputs", "m24", f"{a.tag}_decode.csv")
    if not os.path.isfile(src):
        raise SystemExit(f"缺 {src}。先跑 summarize。")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(f"matplotlib 不可用: {e}")

    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for dcp, marker in (("1", "o"), ("2", "s")):
        xs, ys = [], []
        for r in rows:
            if r["dcp"] == dcp and r["batch_size"] == "1":
                xs.append(int(r["prompt_len"]))
                ys.append(float(r["steady_step_ms_median"]))
        if xs:
            ax.plot(xs, ys, marker=marker, label=f"DCP={dcp} batch=1")
    ax.set_xlabel("prompt_len (context)")
    ax.set_ylabel("steady decode step median (ms)")
    ax.set_title(f"M24 DCP {a.tag} — not Ring Attention")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
    dest = os.path.join(ROOT, "figures", f"fig_m24_dcp_{a.tag}.png")
    fig.tight_layout()
    fig.savefig(dest, dpi=120)
    print("wrote", dest)


if __name__ == "__main__":
    raise SystemExit(main())
