#!/usr/bin/env python3
"""M18 · 离线汇总（不占 GPU、不 import vllm）。

从已落盘的 raw/configs/memory/quality 生成三张 CSV：
  outputs/m18/memory.csv     显存与 KV 容量（含「量化部分」的单独口径）
  outputs/m18/perf.csv       prefill/decode 逐点 median + 相对 Q0 的 Bootstrap 95% CI
  outputs/m18/quality.csv    token 加权 PPL 与 ΔPPL%

统计口径与 M17/M20 一致：median difference 的 Bootstrap 95% CI（固定 seed=0）。
CI 含 0 只能写「未观察到超出噪声的差异」，不得写成「没有差异」。

用法：python scripts/m18_summarize.py
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics

OUT = "outputs/m18"
MODELS = ["0.5b", "1.5b"]
CONFIGS = ["Q0", "Q1", "Q2"]
BASELINE = "Q0"


def bootstrap_median_diff(a: list[float], b: list[float], n=10000, seed=0):
    """median(a) − median(b) 的 Bootstrap 95% CI（与 M17/M20 同口径）。"""
    if not a or not b:
        return None
    rng = random.Random(seed)
    diffs = []
    for _ in range(n):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.median(sa) - statistics.median(sb))
    diffs.sort()
    lo, hi = diffs[int(0.025 * n)], diffs[int(0.975 * n) - 1]
    return {
        "median_diff": statistics.median(a) - statistics.median(b),
        "ci95_lo": lo, "ci95_hi": hi,
        "ci_excludes_zero": not (lo <= 0 <= hi),
    }


def load_raw(tag: str):
    """按 (experiment, grid) 聚合逐样本延迟；只用 valid=True 的样本。"""
    path = os.path.join(OUT, "raw", f"{tag}_perf.jsonl")
    buckets: dict[tuple, list[float]] = {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if not r.get("valid", True):
                continue
            if r["experiment"] == "prefill":
                key = ("prefill", r["prompt_tokens"], None)
            else:
                key = ("decode", r["context_len"], r["batch_size"])
            buckets.setdefault(key, []).append(r["latency_ms"])
    return buckets


def write_memory():
    rows = []
    for m in MODELS:
        base = None
        for c in CONFIGS:
            mem = json.load(open(os.path.join(OUT, "memory", f"{m}_{c}.json")))["tensor_sum"]
            cfg = json.load(open(os.path.join(OUT, "configs", f"{m}_{c}.json")))["effective"]
            by = mem["weight_bytes_by_dtype"]
            total = sum(by.values())
            embed = mem["embed_lm_head_MiB"]
            # 「量化部分」= 总权重 − 未被量化的 embed/lm_head。用它才能看出 4bit 的真实压缩率，
            # 否则会被 embed 稀释（M10 已见过同一现象：FP8 只降 36% 而非 50%）。
            quantizable = total - embed
            row = {
                "model": m, "config": c,
                "weight_total_MiB": round(total, 2),
                "weight_bf16_MiB": round(by.get("torch.bfloat16", 0.0), 2),
                "weight_int32_MiB": round(by.get("torch.int32", 0.0), 2),
                "embed_lm_head_MiB": round(embed, 2),
                "embed_share_of_total_pct": round(100 * embed / total, 2),
                "quantizable_part_MiB": round(quantizable, 2),
                "weight_scale_MiB": mem["weight_scale_MiB"],
                "kv_cache_size_tokens": cfg["kv_cache_size_tokens"],
                "num_gpu_blocks": cfg["num_gpu_blocks"],
                "dtype": cfg["dtype"], "quantization": cfg["quantization"],
            }
            if c == BASELINE:
                base = row
                row["weight_total_vs_Q0_pct"] = 0.0
                row["quantizable_part_vs_Q0_pct"] = 0.0
                row["kv_tokens_vs_Q0_pct"] = 0.0
            else:
                row["weight_total_vs_Q0_pct"] = round(
                    100 * (total - base["weight_total_MiB"]) / base["weight_total_MiB"], 2)
                row["quantizable_part_vs_Q0_pct"] = round(
                    100 * (quantizable - base["quantizable_part_MiB"])
                    / base["quantizable_part_MiB"], 2)
                row["kv_tokens_vs_Q0_pct"] = round(
                    100 * (cfg["kv_cache_size_tokens"] - base["kv_cache_size_tokens"])
                    / base["kv_cache_size_tokens"], 2)
            rows.append(row)
    path = os.path.join(OUT, "memory.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", path)
    return rows


def write_perf():
    rows = []
    for m in MODELS:
        raws = {c: load_raw(f"{m}_{c}") for c in CONFIGS}
        for key in sorted(raws[BASELINE]):
            exp, a1, a2 = key
            base_samples = raws[BASELINE][key]
            for c in CONFIGS:
                s = raws[c].get(key, [])
                row = {
                    "model": m, "config": c, "experiment": exp,
                    "prompt_tokens": a1 if exp == "prefill" else "",
                    "context_len": a1 if exp == "decode" else "",
                    "batch_size": a2 if exp == "decode" else "",
                    "n": len(s),
                    "median_ms": round(statistics.median(s), 4) if s else None,
                    "stdev_ms": round(statistics.pstdev(s), 4) if s else None,
                    "p95_ms": round(sorted(s)[min(len(s) - 1, int(len(s) * 0.95))], 4) if s else None,
                }
                if exp == "decode" and s:
                    row["throughput_tok_s_median"] = round(a2 / (row["median_ms"] / 1e3), 2)
                if c == BASELINE:
                    row.update({"median_diff_vs_Q0_ms": 0.0, "ci95_lo": 0.0, "ci95_hi": 0.0,
                                "ci_excludes_zero": "", "rel_change_vs_Q0_pct": 0.0})
                else:
                    bs = bootstrap_median_diff(s, base_samples)
                    base_med = statistics.median(base_samples)
                    row.update({
                        "median_diff_vs_Q0_ms": round(bs["median_diff"], 4),
                        "ci95_lo": round(bs["ci95_lo"], 4),
                        "ci95_hi": round(bs["ci95_hi"], 4),
                        "ci_excludes_zero": "yes" if bs["ci_excludes_zero"] else "no",
                        "rel_change_vs_Q0_pct": round(100 * bs["median_diff"] / base_med, 2),
                    })
                rows.append(row)
    path = os.path.join(OUT, "perf.csv")
    with open(path, "w", newline="") as f:
        cols = ["model", "config", "experiment", "prompt_tokens", "context_len", "batch_size",
                "n", "median_ms", "stdev_ms", "p95_ms", "throughput_tok_s_median",
                "median_diff_vs_Q0_ms", "ci95_lo", "ci95_hi", "ci_excludes_zero",
                "rel_change_vs_Q0_pct"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print("->", path)
    return rows


def write_quality():
    rows = []
    for m in MODELS:
        base = None
        for c in CONFIGS:
            q = json.load(open(os.path.join(OUT, "quality", f"{m}_{c}.json")))
            row = {"model": m, "config": c,
                   "scored_tokens": q["overall"]["scored_tokens"],
                   "ppl_overall": q["overall"]["ppl"],
                   "ppl_zh": q["zh"]["ppl"], "ppl_en": q["en"]["ppl"],
                   "mean_nll_overall": q["overall"]["mean_nll"]}
            if c == BASELINE:
                base = row
                row.update({"delta_ppl_pct": 0.0, "delta_ppl_zh_pct": 0.0, "delta_ppl_en_pct": 0.0})
            else:
                for k, src in (("delta_ppl_pct", "ppl_overall"),
                               ("delta_ppl_zh_pct", "ppl_zh"),
                               ("delta_ppl_en_pct", "ppl_en")):
                    row[k] = round(100 * (row[src] - base[src]) / base[src], 3)
            rows.append(row)
    path = os.path.join(OUT, "quality.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("->", path)
    return rows


if __name__ == "__main__":
    write_memory()
    write_perf()
    write_quality()
