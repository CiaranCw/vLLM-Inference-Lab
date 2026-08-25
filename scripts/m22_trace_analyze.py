"""M22 · P1 通信证据：离线解析 profiler trace（不需 GPU、不 import vllm）。

    python scripts/m22_trace_analyze.py --trace-dir outputs/m22/traces --tag h20_formal

在数据回传之后离线跑，不占用 H20 机时。

--------------------------------------------------------------------------
分析单位是 per workload / per rank / per kernel，不是「引擎选了哪个后端」。
理由（见 notes/m22/tensor_parallel.md R5）：`should_custom_ar()`
（custom_all_reduce.py:230-243）是逐次调用判定的——每次都要看这一次的 payload
大小 / 16 字节对齐 / 弱连续性，不满足就落到下一个后端。所以同一个引擎里，
小 payload 的 decode all-reduce 与大 payload 的 prefill all-reduce
完全可能走不同 kernel。本脚本按 trace 里的真实 kernel 名分类，
不拿后端选择日志一锤定音。

沿用 M7 的 profiler 纪律：
  · GPU 时间取 `cat=="kernel"` 事件的区间并集，不对 duration 简单求和
    （嵌套/重叠会双计）；
  · 通信与计算可能重叠，故只报「通信 kernel 占 GPU busy 的比例」，
    不把它换算成「通信造成的净 wall-time 损失」；
  · 未识别的 kernel 归入 `unclassified` 并列出 top 名字，不静默丢弃。
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# kernel 名 → 通信路径分类。依据 notes/m22/tensor_parallel.md R5 的符号表。
# 顺序有意义：先匹配到的优先。
COMM_PATTERNS = [
    ("custom_all_reduce", re.compile(r"cross_device_reduce", re.IGNORECASE)),
    (
        "torch_symm_mem",
        re.compile(r"multimem_all_reduce|two_shot_all_reduce", re.IGNORECASE),
    ),
    ("nccl", re.compile(r"nccl", re.IGNORECASE)),
    (
        "other_collective",
        re.compile(r"all_?reduce|all_?gather|reduce_?scatter|broadcast", re.IGNORECASE),
    ),
]
# 从 vLLM 的 rank 后缀反解 rank：dp0_pp0_tp{N}_dcp0_ep0_rank{M}
RANK_RE = re.compile(r"tp(\d+)_dcp\d+_ep\d+_rank(\d+)")


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def classify(name: str) -> str:
    for label, pat in COMM_PATTERNS:
        if pat.search(name):
            return label
    return "compute"


def interval_union(intervals) -> float:
    """区间并集总长（M7 口径）。intervals: [(start_us, end_us), ...]"""
    if not intervals:
        return 0.0
    xs = sorted(intervals)
    total, cs, ce = 0.0, xs[0][0], xs[0][1]
    for s, e in xs[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + (ce - cs)


def analyze_one(path: str) -> dict | None:
    try:
        with _open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"trace": os.path.basename(path), "error": repr(e), "ok": False}

    events = data.get("traceEvents", data if isinstance(data, list) else [])
    by_class_intervals = defaultdict(list)
    by_class_count = defaultdict(int)
    by_class_dur_sum = defaultdict(float)  # 仅供对照，不用于占比
    kernel_names = defaultdict(int)
    all_kernel_intervals = []

    for ev in events:
        if ev.get("cat") != "kernel":
            continue
        name = ev.get("name", "")
        ts, dur = ev.get("ts"), ev.get("dur")
        if ts is None or dur is None:
            continue
        cls = classify(name)
        by_class_intervals[cls].append((ts, ts + dur))
        by_class_count[cls] += 1
        by_class_dur_sum[cls] += dur
        all_kernel_intervals.append((ts, ts + dur))
        if cls != "compute":
            kernel_names[name] += 1

    if not all_kernel_intervals:
        return {
            "trace": os.path.basename(path),
            "error": "no kernel events",
            "ok": False,
        }

    gpu_busy_us = interval_union(all_kernel_intervals)
    comm_intervals = [
        iv for c, ivs in by_class_intervals.items() if c != "compute" for iv in ivs
    ]
    comm_union_us = interval_union(comm_intervals)

    m = RANK_RE.search(os.path.basename(path))
    per_class = {}
    for c in by_class_intervals:
        per_class[c] = {
            "n_kernels": by_class_count[c],
            "union_us": round(interval_union(by_class_intervals[c]), 2),
            "naive_sum_us": round(by_class_dur_sum[c], 2),
        }
    return {
        "trace": os.path.basename(path),
        "tp_rank": int(m.group(1)) if m else None,
        "global_rank": int(m.group(2)) if m else None,
        "ok": True,
        "n_kernel_events": len(all_kernel_intervals),
        "gpu_busy_union_us": round(gpu_busy_us, 2),
        "comm_union_us": round(comm_union_us, 2),
        # ★ 只报占比，不换算成 wall-time penalty（通信与计算可能重叠）
        "comm_share_of_gpu_busy": round(comm_union_us / gpu_busy_us, 6)
        if gpu_busy_us
        else None,
        "per_class": per_class,
        "comm_kernel_names": dict(
            sorted(kernel_names.items(), key=lambda x: -x[1])[:12]
        ),
        "backends_observed": sorted(
            c for c in by_class_intervals if c not in ("compute",)
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="M22 P1 trace 分析（离线）")
    ap.add_argument(
        "--trace-dir", default=os.path.join(ROOT, "outputs", "m22", "traces")
    )
    ap.add_argument("--tag", default="h20_formal")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(
        glob.glob(os.path.join(a.trace_dir, "*.pt.trace.json"))
        + glob.glob(os.path.join(a.trace_dir, "*.pt.trace.json.gz"))
    )
    if not files:
        print(
            f"[m22-trace] {a.trace_dir} 下没有 trace 文件 —— "
            "P1 未执行或 profiler 不可用。**记为 observation gap，不得用延迟差反推通信时间。**"
        )
        return 0

    rows = [r for r in (analyze_one(p) for p in files) if r]
    out = a.out or os.path.join(ROOT, "outputs", "m22", f"{a.tag}_comm_profile.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "trace",
                "workload_prefix",
                "tp_rank",
                "global_rank",
                "ok",
                "n_kernel_events",
                "gpu_busy_union_us",
                "comm_union_us",
                "comm_share_of_gpu_busy",
                "backends_observed",
                "custom_ar_n",
                "custom_ar_union_us",
                "nccl_n",
                "nccl_union_us",
                "symm_mem_n",
                "symm_mem_union_us",
                "top_comm_kernels",
                "error",
            ]
        )
        for r in rows:
            pc = r.get("per_class", {})

            def g(cls, key, _pc=pc):
                return _pc.get(cls, {}).get(key, 0)

            # workload 前缀 = start_profile 时传的 profile_prefix（tag_tpN_case）
            wl = (
                r["trace"].split("_dp0_pp0_tp")[0]
                if "_dp0_pp0_tp" in r["trace"]
                else ""
            )
            w.writerow(
                [
                    r["trace"],
                    wl,
                    r.get("tp_rank"),
                    r.get("global_rank"),
                    r["ok"],
                    r.get("n_kernel_events"),
                    r.get("gpu_busy_union_us"),
                    r.get("comm_union_us"),
                    r.get("comm_share_of_gpu_busy"),
                    "|".join(r.get("backends_observed", [])),
                    g("custom_all_reduce", "n_kernels"),
                    g("custom_all_reduce", "union_us"),
                    g("nccl", "n_kernels"),
                    g("nccl", "union_us"),
                    g("torch_symm_mem", "n_kernels"),
                    g("torch_symm_mem", "union_us"),
                    json.dumps(r.get("comm_kernel_names", {}), ensure_ascii=False),
                    r.get("error", ""),
                ]
            )
    print(f"[m22-trace] wrote {out}  ({len(rows)} traces)")
    for r in rows:
        if not r["ok"]:
            print(f"  [FAIL] {r['trace']}: {r.get('error')}")
            continue
        print(
            f"  {r['trace'][:52]:<52} rank={r.get('global_rank')} "
            f"comm_share={r['comm_share_of_gpu_busy']} "
            f"backends={r['backends_observed']}"
        )
    print(
        "\n[m22-trace] 提醒：comm_share 是**通信 kernel 占 GPU busy 的比例**，"
        "不是通信造成的净 wall-time 损失（通信与计算可能重叠）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
