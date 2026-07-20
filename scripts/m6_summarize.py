"""M6 · 汇总器 —— 读取 outputs/m6/raw/*.jsonl 逐样本数据，过滤无效样本，
按 (experiment, mode, prefix, 维度) 分组统计，输出 baseline.csv 与 summary JSON。

用法：
  python scripts/m6_summarize.py --raw-dir outputs/m6/raw --out-dir outputs/m6 [--quick]
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics

import numpy as np


def stats(vals: list[float]) -> dict:
    if not vals:
        return {"sample_count": 0}
    return {
        "sample_count": len(vals),
        "median": round(float(np.median(vals)), 4),
        "mean": round(float(np.mean(vals)), 4),
        "stdev": round(float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0, 4),
        "p95": round(float(np.percentile(vals, 95)), 4),
        "min": round(float(min(vals)), 4),
        "max": round(float(max(vals)), 4),
    }


def load(raw_dir: str, quick: bool) -> list[dict]:
    rows = []
    for p in sorted(glob.glob(os.path.join(raw_dir, "*.jsonl"))):
        if quick != ("_quick" in os.path.basename(p)):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def key_of(r: dict):
    if r["experiment"] == "prefill":
        return ("prefill", r["mode"], r["prefix_caching"], r["target_prompt_tokens"])
    if r["experiment"] == "decode":
        return ("decode", r["mode"], r["prefix_caching"], r["batch_size"])
    if r["experiment"] == "prefix":
        return ("prefix", r["mode"], r["prefix_caching"], r.get("prefix_kind", ""))
    return (r["experiment"], r["mode"], r["prefix_caching"], "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="outputs/m6/raw")
    ap.add_argument("--out-dir", default="outputs/m6")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    rows = load(args.raw_dir, args.quick)
    # 只统计 measure 且 valid 的样本
    valid = [r for r in rows if r.get("phase") == "measure" and r.get("valid_sample")]
    dropped = len(rows) - len(valid)

    groups: dict = {}
    for r in valid:
        groups.setdefault(key_of(r), []).append(r)

    summary = []
    for (exp, mode, prefix, dim), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        lat = stats([r["latency_ms"] for r in rs])
        row = {"experiment": exp, "mode": mode, "prefix_caching": prefix, "dim": dim}
        for k, v in lat.items():
            row["latency_ms_" + k] = v
        if exp == "decode":
            th = stats([r["throughput_tokens_per_s"] for r in rs if r.get("throughput_tokens_per_s")])
            row["throughput_tok_s_median"] = th.get("median")
            row["throughput_tok_s_p95"] = th.get("p95")
            row["seq_len"] = rs[0].get("seq_len")
        if exp == "prefix":
            row["prefix_hit_tokens_median"] = int(np.median([r.get("prefix_hit_tokens") or 0 for r in rs]))
            row["scheduled_tokens_median"] = int(np.median([r.get("scheduled_tokens") or 0 for r in rs]))
        summary.append(row)

    os.makedirs(os.path.join(args.out_dir, "summary"), exist_ok=True)
    sfx = "_quick" if args.quick else ""
    sum_json = os.path.join(args.out_dir, "summary", f"summary{sfx}.json")
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump({"dropped_invalid_or_warmup": dropped, "groups": summary}, f, ensure_ascii=False, indent=2)

    # baseline.csv（宽表）
    fields = ["experiment", "mode", "prefix_caching", "dim",
              "latency_ms_sample_count", "latency_ms_median", "latency_ms_mean",
              "latency_ms_stdev", "latency_ms_p95", "latency_ms_min", "latency_ms_max",
              "throughput_tok_s_median", "throughput_tok_s_p95", "seq_len",
              "prefix_hit_tokens_median", "scheduled_tokens_median"]
    csv_path = os.path.join(args.out_dir, f"baseline{sfx}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in summary:
            w.writerow(row)

    print(f"[summarize] total rows={len(rows)} valid_measure={len(valid)} dropped={dropped}")
    print(f"[summarize] groups={len(summary)}")
    print(f"[summarize] -> {csv_path}")
    print(f"[summarize] -> {sum_json}")


if __name__ == "__main__":
    main()
