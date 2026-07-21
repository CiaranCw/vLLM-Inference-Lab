"""M9 · isolated microbenchmark：量化"合并 H2D 拷贝"的机制与天花板。

对照（脱离 vLLM 引擎，纯 H2D 发行成本）：
  two_copies：两块独立 pinned int64 缓冲，各 copy_(non_blocking)  —— 现状（req_indices + query_pos）
  one_copy  ：一块合并 pinned 缓冲，一次 copy_(2N)                —— M9 合并后

用 triton.testing.do_bench 计时（内部 CUDA event + warmup），计时区间仅含拷贝调用。
目的：证明"少发一次 H2D"的每步节省量级（进而说明 E2E 天花板）。不追求大收益。
"""
from __future__ import annotations

import csv
import os

import torch
from triton.testing import do_bench

DEV = "cuda"
SIZES = [1, 8, 64, 256, 1024]  # total_num_scheduled_tokens 量级（decode 通常很小）


def main():
    rows = []
    for n in SIZES:
        # 两块独立 pinned 缓冲（模拟 req_indices / query_pos）
        a_cpu = torch.zeros(n, dtype=torch.int64, pin_memory=True)
        b_cpu = torch.zeros(n, dtype=torch.int64, pin_memory=True)
        a_gpu = torch.zeros(n, dtype=torch.int64, device=DEV)
        b_gpu = torch.zeros(n, dtype=torch.int64, device=DEV)
        # 合并缓冲（2N 连续）
        c_cpu = torch.zeros(2 * n, dtype=torch.int64, pin_memory=True)
        c_gpu = torch.zeros(2 * n, dtype=torch.int64, device=DEV)

        def two_copies():
            a_gpu.copy_(a_cpu, non_blocking=True)
            b_gpu.copy_(b_cpu, non_blocking=True)

        def one_copy():
            c_gpu.copy_(c_cpu, non_blocking=True)

        for name, fn in [("two_copies", two_copies), ("one_copy", one_copy)]:
            med, p20, p80 = do_bench(fn, warmup=50, rep=200, quantiles=[0.5, 0.2, 0.8])
            rows.append({"n": n, "impl": name,
                         "latency_us_median": round(med * 1e3, 4),
                         "latency_us_p20": round(p20 * 1e3, 4),
                         "latency_us_p80": round(p80 * 1e3, 4)})
            print(f"n={n:>5} {name:<11} median={med*1e3:8.4f}us")
        # 每 n 的节省
        two = next(r for r in rows if r["n"] == n and r["impl"] == "two_copies")["latency_us_median"]
        one = next(r for r in rows if r["n"] == n and r["impl"] == "one_copy")["latency_us_median"]
        print(f"  -> saving(two-one) = {two-one:.4f}us/step")

    os.makedirs("outputs/m9", exist_ok=True)
    with open("outputs/m9/microbench.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("[m9_microbench] -> outputs/m9/microbench.csv")


if __name__ == "__main__":
    main()
