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
            # seq_len 如实汇报为“实际出现的集合”（window=1 时应为单值）
            row["seq_len"] = ",".join(str(x) for x in sorted({r.get("seq_len") for r in rs if r.get("seq_len") is not None}))
        if exp == "prefix":
            row["prefix_hit_tokens_median"] = int(np.median([r.get("prefix_hit_tokens") or 0 for r in rs]))
            row["scheduled_tokens_median"] = int(np.median([r.get("scheduled_tokens") or 0 for r in rs]))
        summary.append(row)

    # —— prefix 配对分析（cold_i - hot_i，同 repeat 逐配对；判断命中是否真的降低了延迟）——
    prefix_paired = None
    cold = {r["repeat"]: r for r in valid if r.get("experiment") == "prefix" and r.get("prefix_kind") == "cold"}
    hot = {r["repeat"]: r for r in valid if r.get("experiment") == "prefix" and r.get("prefix_kind") == "hot"}
    common = sorted(set(cold) & set(hot))
    if common:
        diffs = [cold[i]["latency_ms"] - hot[i]["latency_ms"] for i in common]  # 正=hot 更快
        prefix_paired = {
            "n_pairs": len(diffs),
            "cold_minus_hot_ms_mean": round(float(np.mean(diffs)), 4),
            "cold_minus_hot_ms_median": round(float(np.median(diffs)), 4),
            "cold_minus_hot_ms_stdev": round(float(statistics.pstdev(diffs)) if len(diffs) > 1 else 0.0, 4),
            "cold_minus_hot_ms_min": round(float(min(diffs)), 4),
            "cold_minus_hot_ms_max": round(float(max(diffs)), 4),
            "pairs_hot_faster": sum(1 for d in diffs if d > 0),
            "note": "均值≈0 且 min/max 跨零 → 命中减少计算 token 但未带来可测的延迟下降（落在噪声内）",
        }

    os.makedirs(os.path.join(args.out_dir, "summary"), exist_ok=True)
    sfx = "_quick" if args.quick else ""
    sum_json = os.path.join(args.out_dir, "summary", f"summary{sfx}.json")
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump({"dropped_invalid_or_warmup": dropped, "groups": summary,
                   "prefix_paired": prefix_paired}, f, ensure_ascii=False, indent=2)
    if prefix_paired:
        print(f"[summarize] prefix paired cold-hot: mean={prefix_paired['cold_minus_hot_ms_mean']}ms "
              f"hot_faster={prefix_paired['pairs_hot_faster']}/{prefix_paired['n_pairs']}")

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
