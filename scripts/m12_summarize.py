"""M12 · 汇总：读 bench summary + detailed + metrics jsonl + metadata + 冻结 config →
逐 run 派生指标、队列稳定性、run_stable/qps_stable、最大稳定 QPS 区间、Shape 对照、Burst。

指标口径与阈值全部来自 outputs/m12/config.json（跑前冻结），本脚本不引入新阈值。
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os

import numpy as np

OUT = "outputs/m12"


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def read_metrics(run_id):
    p = os.path.join(OUT, "metrics", f"{run_id}.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p) if l.strip()]


def queue_analysis(rows, drain_time_s, Q):
    arr = [r for r in rows if r.get("phase") == "arrival" and r.get("num_requests_waiting") is not None]
    if len(arr) < Q["min_samples"]:
        return {"queue_metrics_invalid": True, "n_arrival_samples": len(arr)}
    t = np.array([r["time_since_run_start_s"] for r in arr], float)
    w = np.array([r["num_requests_waiting"] for r in arr], float)
    run = np.array([r["num_requests_running"] for r in arr], float)
    kv = np.array([r.get("kv_cache_usage_perc") or 0 for r in arr], float)
    half = arr[len(arr) // 2:]
    th = np.array([r["time_since_run_start_s"] for r in half], float)
    wh = np.array([r["num_requests_waiting"] for r in half], float)
    slope = float(np.polyfit(th, wh, 1)[0]) if len(half) >= 2 and np.ptp(th) > 0 else 0.0
    k = max(1, len(arr) // 5)
    head20 = float(w[:k].mean()); tail20 = float(w[-k:].mean())
    end_waiting = float(w[-1]); peak_waiting = float(w.max())
    qpt = slope > Q["slope_thr"] # Queue Positive Trend，队列后半段存在明显持续正趋势
    qlg = (tail20 - head20 >= Q["abs_diff"]) and (tail20 / max(head20, Q["base_floor"]) >= Q["ratio"]) # Queue Level Growth，队列后段平均水平明显高于前段
    qnd = end_waiting > Q["end_waiting_thr"] # Queue Not Drained at End，队列末尾仍有明显积压
    queue_unstable = qpt and (qlg or qnd)
    drain_unstable = drain_time_s > Q["drain_budget_s"]
    # counter/histogram run-local delta
    pre = [r.get("num_preemptions_total") or 0 for r in rows]
    preempt_delta = (max(pre) - min(pre)) if pre else 0
    return {"queue_metrics_invalid": False, "n_arrival_samples": len(arr),
            "waiting_slope_req_s": round(slope, 4), "head20_waiting": round(head20, 2),
            "tail20_waiting": round(tail20, 2), "end_waiting": end_waiting, "peak_waiting": peak_waiting,
            "peak_running": float(run.max()), "peak_kv_usage": round(float(kv.max()), 4),
            "preempt_delta": preempt_delta,
            "queue_unstable": bool(queue_unstable), "drain_unstable": bool(drain_unstable),
            "queue_stable": bool((not queue_unstable) and (not drain_unstable))}


def per_request(detailed, slo):
    """从 detailed 重建逐请求，算 slo_pass（成功且 ttft/tpot/e2el 全达标）与 realized_qps。"""
    il = detailed.get("itls"); # Token 间延迟
    ol = detailed.get("output_lens"); st = detailed.get("start_times")
    tt = detailed.get("ttfts"); err = detailed.get("errors")
    if not (il and ol and tt):
        return None
    n = len(tt); completed = 0; good = 0
    for i in range(n):
        if err and i < len(err) and err[i]:
            continue
        completed += 1
        out = ol[i] if i < len(ol) else 0
        itl = il[i] if i < len(il) else []
        ttft_ms = tt[i] * 1000
        e2el_ms = (tt[i] + sum(itl)) * 1000
        tpot_ms = (sum(itl) / (out - 1) * 1000) if out > 1 else 0.0
        if (ttft_ms <= slo["ttft_p99_ms"] and tpot_ms <= slo["tpot_p99_ms"] and e2el_ms <= slo["e2el_p99_ms"]):
            good += 1
    realized = None
    if st and len(st) >= 2:
        span = max(st) - min(st)
        realized = round((len(st) - 1) / span, 3) if span > 0 else None
    return {"completed": completed, "good": good, "realized_offered_qps": realized}


def run_stable(rec, slo, cfg):
    return bool(
        rec.get("exit_code") == 0 and not rec.get("timeout")
        and not rec.get("queue_metrics_invalid", True)
        and rec.get("success_ratio", 0) >= cfg["success_ratio_min"]
        and rec.get("slo_pass_ratio_all", 0) >= cfg["slo_pass_ratio_min"]
        and rec.get("p99_ttft_ms", 1e9) <= slo["ttft_p99_ms"]
        and rec.get("p99_tpot_ms", 1e9) <= slo["tpot_p99_ms"]
        and rec.get("p99_e2el_ms", 1e9) <= slo["e2el_p99_ms"]
        and rec.get("p99_itl_ms", 1e9) <= slo["itl_p99_guardrail_ms"]
        and rec.get("queue_stable", False)
        and rec.get("drain_time_s", 1e9) <= cfg["queue_thresholds"]["drain_budget_s"]
        and rec.get("preempt_delta", 1e9) <= cfg["preemption_limit"]
        and rec.get("timeout_count", 0) <= cfg["timeout_limit"])


def build_run(md_path, cfg):
    md = load(md_path) # Orchestrator Metadata
    if not md:
        return None
    rid = md["run_id"]
    res = load(md.get("result_path", "")) # 官方 Bench Summary + Detailed
    rows = read_metrics(rid) # 服务端 Metrics 时间序列
    slo = cfg["experimental_slo"]; Q = cfg["queue_thresholds"]
    rec = {"run_id": rid, "shape": f"{md['shape']['in']}/{md['shape']['out']}",
           "target_qps": md["target_offered_qps"], "seed": md["seed"], "num_prompts": md["num_prompts"],
           "exit_code": md.get("exit_code"), "timeout": md.get("timeout"),
           "drain_time_s": md.get("drain_time_s"), "server_restart_count": md.get("server_restart_count")}
    if res:
        rec.update({"completed": res.get("completed"), "failed": res.get("failed"),
                    "request_throughput": round(res.get("request_throughput", 0), 3),
                    "request_goodput": round(res.get("request_goodput", 0) or 0, 3),
                    "output_throughput": round(res.get("output_throughput", 0), 2),
                    "total_token_throughput": round(res.get("total_token_throughput", 0), 2),
                    "p50_ttft_ms": res.get("p50_ttft_ms"), "p95_ttft_ms": res.get("p95_ttft_ms"),
                    "p99_ttft_ms": res.get("p99_ttft_ms"), "p99_tpot_ms": res.get("p99_tpot_ms"),
                    "p99_itl_ms": res.get("p99_itl_ms"), "p99_e2el_ms": res.get("p99_e2el_ms")})
        rec["success_ratio"] = round((res.get("completed", 0)) / max(md["num_prompts"], 1), 4)
        rec["timeout_count"] = res.get("failed", 0)
        pr = per_request(res, slo)
        if pr:
            rec["slo_pass_ratio_all"] = round(pr["good"] / max(md["num_prompts"], 1), 4)
            rec["realized_offered_qps"] = pr["realized_offered_qps"]
    rec.update(queue_analysis(rows, rec.get("drain_time_s") or 0, Q))
    rec["run_stable"] = run_stable(rec, slo, cfg)
    return rec


def main():
    cfg = load(os.path.join(OUT, "config.json"))
    if not cfg:
        print("no config.json; run m12_run_sweep first"); return
    recs = [build_run(p, cfg) for p in sorted(glob.glob(os.path.join(OUT, "metadata", "*.json")))]
    recs = [r for r in recs if r]

    # sweep.csv
    cols = ["run_id", "shape", "target_qps", "seed", "num_prompts", "success_ratio", "slo_pass_ratio_all",
            "request_throughput", "request_goodput", "realized_offered_qps",
            "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "p99_tpot_ms", "p99_itl_ms", "p99_e2el_ms",
            "peak_waiting", "waiting_slope_req_s", "end_waiting", "drain_time_s", "queue_stable",
            "preempt_delta", "timeout", "run_stable"]
    with open(os.path.join(OUT, "sweep.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
        for r in recs:
            w.writerow(r)

    # fine 多 seed → qps_stable（三 seed 全过），最大稳定 QPS 区间
    fine = [r for r in recs if "fine" in r["run_id"]]
    by_rate = {}
    for r in fine:
        by_rate.setdefault(r["target_qps"], []).append(r)
    qps_verdict = {}
    for rate, rs in sorted(by_rate.items()):
        seeds_ok = all(r["run_stable"] for r in rs)
        p99s = [r.get("p99_ttft_ms") for r in rs if r.get("p99_ttft_ms") is not None]
        qps_verdict[rate] = {"seeds": len(rs), "all_stable": seeds_ok,
                             "worst_seed_p99_ttft_ms": max(p99s) if p99s else None,
                             "median_p99_ttft_ms": round(float(np.median(p99s)), 1) if p99s else None}
    stable_rates = sorted([r for r, v in qps_verdict.items() if v["all_stable"]])
    unstable_rates = sorted([r for r, v in qps_verdict.items() if not v["all_stable"]])
    max_stable = max(stable_rates) if stable_rates else None
    first_unstable = min([r for r in unstable_rates if max_stable is None or r > max_stable], default=None)
    capacity = {"max_tested_stable_offered_qps": max_stable,
                "first_tested_unstable_offered_qps": first_unstable,
                "capacity_bracket_qps": [max_stable, first_unstable], "qps_resolution": 0.25}

    coarse_a = [r for r in recs if r["run_id"].startswith("shape_a_coarse")]
    coarse_b = [r for r in recs if r["run_id"].startswith("shape_b_coarse")]
    burst = [r for r in recs if "burst" in r["run_id"]]
    summary = {"experimental_slo": cfg["experimental_slo"], "low_load_p99": cfg.get("low_load_p99"),
               "capacity_shape_a": capacity, "qps_verdict_fine": qps_verdict,
               "coarse_a": [{k: r.get(k) for k in ("target_qps", "success_ratio", "request_throughput",
                            "request_goodput", "p95_ttft_ms", "p99_ttft_ms", "peak_waiting", "end_waiting",
                            "drain_time_s", "queue_stable", "run_stable")} for r in coarse_a],
               "coarse_b": [{k: r.get(k) for k in ("target_qps", "success_ratio", "request_throughput",
                            "output_throughput", "total_token_throughput", "p99_ttft_ms", "peak_waiting",
                            "queue_stable")} for r in coarse_b],
               "burst": [{k: r.get(k) for k in ("p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "p99_e2el_ms",
                          "peak_waiting", "peak_running", "peak_kv_usage", "drain_time_s",
                          "request_throughput", "output_throughput")} for r in burst]}
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print("[summarize] SLO:", cfg["experimental_slo"])
    print("[summarize] capacity Shape A:", capacity)
    print("[summarize] fine verdict:", json.dumps(qps_verdict, ensure_ascii=False))
    print(f"[summarize] -> {OUT}/sweep.csv, summary.json  ({len(recs)} runs)")


if __name__ == "__main__":
    main()
