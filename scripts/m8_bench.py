"""M8 · microbenchmark（规模曲线）。

对照三种 slot_mapping **生成**实现（同一输入/口径）：
  - 自写 Triton（m8_kernels.slot_mapping_gen）
  - vLLM 官方 `_compute_slot_mapping_kernel`
  - PyTorch 向量化 reference（gather 计算，作清晰性/正确性基线，非性能目标）

计时：`triton.testing.do_bench`（内部 CUDA event + warmup），计时区间**仅含 kernel/op 调用**；
输入张量、输出缓冲在计时区外分配（两个 kernel 复用预分配 out；torch 版天然分配中间张量，如实标注）。

指标：`latency_us`（median/p20/p80）与 `effective_payload_bandwidth_GBps`
      = (positions 读 8B + slot 写 8B) × num_tokens / time
      —— **逻辑有效载荷带宽，仅用于实现间相对比较**，不代表硬件 DRAM 带宽
      （未计 block_table 读、事务放大、L2、地址计算等）。目标：写对、无数量级退化、趋势合理、差距可解释。
"""

from __future__ import annotations

import csv
import json
import os
import sys

import torch
import triton
from triton.testing import do_bench

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m8_kernels as K

from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

DEV = "cuda"
PAD = K.PAD_ID
BS = 16
SIZES = [128, 512, 1024, 2048, 4096, 8192, 16384]


def build(num_tokens):
    """单请求场景：positions 0..n-1，block_table 覆盖足够物理块。"""
    max_blocks = (num_tokens + BS - 1) // BS
    g = torch.Generator().manual_seed(123)
    bt = (torch.randperm(max_blocks * 2, generator=g)[:max_blocks] + 1).reshape(1, max_blocks).to(torch.int32).to(DEV)
    qsl = torch.tensor([0, num_tokens], dtype=torch.int32, device=DEV)
    positions = torch.arange(num_tokens, dtype=torch.int64, device=DEV)
    return positions, bt, qsl


def main():
    rows = []
    for n in SIZES:
        positions, bt, qsl = build(n)
        out_buf = torch.full((n,), PAD, dtype=torch.int64, device=DEV)  # 两个 kernel 复用

        def run_mine():
            K.slot_mapping_gen(positions, bt, qsl, BS, n, tile=1024, out=out_buf)

        def run_vllm():
            _compute_slot_mapping_kernel[(2,)](
                n, n, qsl, positions, bt, bt.stride(0), BS, out_buf,
                TOTAL_CP_WORLD_SIZE=1, TOTAL_CP_RANK=0, CP_KV_CACHE_INTERLEAVE_SIZE=1,
                PAD_ID=PAD, BLOCK_SIZE=1024)

        bt0 = bt[0]

        def run_torch():
            blk_idx = positions // BS
            off = positions % BS
            _ = bt0[blk_idx].to(torch.int64) * BS + off

        impls = {"mine_triton": run_mine, "vllm_triton": run_vllm, "torch_ref": run_torch}
        for name, fn in impls.items():
            # 中位数、20%分位数、80%分位数（转换成微秒），用中位数是因为微秒级 Kernel 容易被干扰导致出现异常值大幅影响平均值
            med, p20, p80 = do_bench(fn, warmup=25, rep=100, quantiles=[0.5, 0.2, 0.8])
            med_us, p20_us, p80_us = med * 1e3, p20 * 1e3, p80 * 1e3
            payload_bytes = (8 + 8) * n  # 读 positions int64 + 写 slot int64
            gbps = payload_bytes / (med_us * 1e-6) / 1e9
            rows.append({
                "num_tokens": n, "impl": name,
                "latency_us_median": round(med_us, 3),
                "latency_us_p20": round(p20_us, 3),
                "latency_us_p80": round(p80_us, 3),
                "eff_payload_bandwidth_GBps": round(gbps, 2),
            })
            print(f"n={n:>6} {name:<12} median={med_us:8.3f}us  eff_payload={gbps:7.2f} GB/s")

    os.makedirs("outputs/m8", exist_ok=True)
    with open("outputs/m8/microbench.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    # summary：每 impl 在各 size 的中位延迟
    summary = {"config": {"block_size": BS, "sizes": SIZES, "timing": "triton.testing.do_bench",
                          "metric_note": "eff_payload_bandwidth = logical K/V-index payload only, not HW DRAM BW"},
               "rows": rows}
    with open("outputs/m8/bench_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n[m8_bench] -> outputs/m8/microbench.csv, outputs/m8/bench_summary.json")


if __name__ == "__main__":
    main()
