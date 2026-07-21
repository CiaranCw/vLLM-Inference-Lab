"""M10 · 汇总：读各配置 configs/memory/quality/perf → latency.csv / summary.csv / quality 对照 / KV 倍率。"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os


def load(out_dir, sub, cfg):
    p = os.path.join(out_dir, sub, f"{cfg}.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/m10")
    args = ap.parse_args()
    od = args.out_dir
    cfgs = sorted({os.path.basename(p)[:-5] for p in glob.glob(os.path.join(od, "configs", "*.json"))
                   if not p.endswith("_perf.json")})

    # ---- latency.csv（prefill + decode 每测点）----
    rows = []
    for cfg in cfgs:
        perf = load(od, "configs", cfg + "_perf") or {}
        for pf in perf.get("prefill", []):
            rows.append({"config": cfg, "experiment": "prefill", "dim": f"len{pf['prompt_tokens']}",
                         "median_ms": pf["median_ms"], "p95_ms": pf["p95_ms"], "stdev_ms": pf["stdev_ms"],
                         "n": pf["sample_count"]})
        for d in perf.get("decode", []):
            rows.append({"config": cfg, "experiment": "decode", "dim": f"ctx{d['context_len']}_b{d['batch_size']}",
                         "median_ms": d["median_ms"], "p95_ms": d["p95_ms"], "stdev_ms": d["stdev_ms"],
                         "n": d["sample_count"], "throughput_tok_s": d.get("throughput_tok_s_median"),
                         "seq_len": d.get("seq_len")})
    if rows:
        with open(os.path.join(od, "latency.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["config", "experiment", "dim", "median_ms", "p95_ms", "stdev_ms",
                                              "n", "throughput_tok_s", "seq_len"], extrasaction="ignore")
            w.writeheader(); w.writerows(rows)

    # ---- 质量对照（token 加权 PPL + Δ%）----
    ppl = {c: (load(od, "quality", c) or {}).get("overall", {}).get("ppl") for c in cfgs}
    def delta(a, b):
        if ppl.get(a) is None or ppl.get(b) is None:
            return None
        return round((ppl[a] - ppl[b]) / ppl[b] * 100, 2)
    quality_cmp = {"ppl_overall": ppl,
                   "C1_vs_C0(FP8 Linear)": delta("C1", "C0"),
                   "C0T_vs_C0(backend)": delta("C0T", "C0"),
                   "C2_vs_C0T(FP8 KV 净)": delta("C2", "C0T"),
                   "C2_vs_C0(总体)": delta("C2", "C0"),
                   "C2dyn_vs_C0T(动态scale KV)": delta("C2dyn", "C0T")}

    # ---- 显存 + KV 容量 ----
    mem = {}
    for cfg in cfgs:
        conf = load(od, "configs", cfg) or {}
        m = load(od, "memory", cfg) or {}
        mem[cfg] = {"kv_cache_size_tokens": conf.get("kv_cache_size_tokens"),
                    "num_gpu_blocks": conf.get("num_gpu_blocks"),
                    "weight_bytes_by_dtype_MiB": m.get("tensor_sum", {}).get("weight_bytes_by_dtype"),
                    "kv_cache_tensor_MiB": m.get("tensor_sum", {}).get("kv_cache_tensor_MiB"),
                    "device_used_MiB": m.get("snapshot", {}).get("device_used_MiB")}
    kv_ratio = None
    if mem.get("C2", {}).get("kv_cache_size_tokens") and mem.get("C0T", {}).get("kv_cache_size_tokens"):
        kv_ratio = {"tokens_C2_over_C0T": round(mem["C2"]["kv_cache_size_tokens"] / mem["C0T"]["kv_cache_size_tokens"], 3),
                    "kv_tensor_MiB_C2_over_C0T": round(mem["C2"]["kv_cache_tensor_MiB"] / mem["C0T"]["kv_cache_tensor_MiB"], 3)
                    if mem["C2"].get("kv_cache_tensor_MiB") and mem["C0T"].get("kv_cache_tensor_MiB") else None}

    summary = {"configs": cfgs, "quality": quality_cmp, "memory": mem, "kv_capacity_ratio_C2_over_C0T": kv_ratio}
    json.dump(summary, open(os.path.join(od, "summary.json"), "w"), ensure_ascii=False, indent=2)
    # summary.csv（关键指标一览）
    with open(os.path.join(od, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["config", "ppl_overall", "kv_tokens", "kv_tensor_MiB", "device_used_MiB"])
        for c in cfgs:
            w.writerow([c, ppl.get(c), mem[c]["kv_cache_size_tokens"], mem[c]["kv_cache_tensor_MiB"], mem[c]["device_used_MiB"]])
    print("[summarize] quality:", json.dumps(quality_cmp, ensure_ascii=False))
    print("[summarize] kv_ratio C2/C0T:", kv_ratio)
    print(f"[summarize] -> {od}/summary.json, latency.csv, summary.csv")


if __name__ == "__main__":
    main()
