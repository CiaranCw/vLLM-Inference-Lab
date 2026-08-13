#!/usr/bin/env python3
"""M21 · E2（指标真值校验）+ E3（容量阶梯）—— 共用一次 server 生命周期。

E2：每个 measure 窗口都核对服务端 counter 增量 vs 客户端 bench 汇总（不变量 I1–I3、I6），
    并量化「服务端直方图**近似**分位数」与「客户端 exact percentile」的偏差（结论，非判据）。
    注意：客户端 exact percentile 是 client/SLO 口径的 ground truth，**不是** server metric 的 ground truth。
E3：QPS 阶梯 → 服务端指标视角复核 M12 的容量拐点（客户端口径也一并记录，便于对照）。

纪律：
  - 服务端配置 = M12 冻结命令，唯一自变量是 offered QPS（与 seed）。
  - warmup 与 measure **分开跑、分开统计**；counter 快照只圈 measure 窗口。
  - 不新造 SLO，直接用 M12 冻结的 P99 TTFT ≤ 300ms。

用法：
  python scripts/m21_capacity_sweep.py                       # 默认 Shape A, 3 seed 阶梯
  python scripts/m21_capacity_sweep.py --shape B --seeds 0   # Shape B 单 seed
"""

import argparse
import csv
import json
import os
import time

import m21_common as C

DEFAULT_RATES = [8, 16, 24, 32, 40, 44, 48]
WARMUP_PROMPTS = 30
MEASURE_PROMPTS = 200

# 服务端窗口聚合：对整个 measure 窗口做 increase()，再取直方图分位数
HIST_Q = ('histogram_quantile({q}, sum by (le) (increase({metric}_bucket[{w}s])))')

WINDOW_EXPRS = {
    "waiting": "sum(vllm:num_requests_waiting)",
    "running": "sum(vllm:num_requests_running)",
    "kv_usage": "max(vllm:kv_cache_usage_perc)",
    "dcgm_gpu_util": "max(DCGM_FI_DEV_GPU_UTIL)",
    "nvml_gpu_util": "max(m21_nvml_gpu_util_percent)",
    "dcgm_fb_used_mib": "max(DCGM_FI_DEV_FB_USED)",
    "proc_cpu_percent": "m21_target_cpu_percent",
    "host_cpu_percent": "m21_host_cpu_percent",
    "proc_rss_bytes": "m21_target_rss_bytes",
}


def counters_snapshot():
    m = C.scrape_vllm_metrics()
    return {k: m.get(k) for k in
            ("vllm:prompt_tokens_total", "vllm:generation_tokens_total",
             "vllm:request_success_total", "vllm:num_preemptions_total")}


def window_metrics(t0, t1):
    """对 [t0, t1] 窗口做聚合；直方图分位数在 t1 时刻按窗口长度求值。"""
    w = max(int(t1 - t0) + 1, 5)
    out = {}
    for key, expr in WINDOW_EXPRS.items():
        agg = C.prom_range_agg(expr, t0, t1)
        out[f"{key}_max"] = agg["max"]
        out[f"{key}_mean"] = agg["mean"]
        out[f"{key}_samples"] = agg["n"]
    for name, metric in (("ttft", "vllm:time_to_first_token_seconds"),
                         ("itl", "vllm:inter_token_latency_seconds"),
                         ("e2el", "vllm:e2e_request_latency_seconds")):
        for q in (0.5, 0.95, 0.99):
            expr = HIST_Q.format(q=q, metric=metric, w=w)
            v = C.prom_scalar_at(expr, t1)
            out[f"server_hist_{name}_p{int(q * 100)}_ms"] = None if v is None else v * 1000.0
    return out


def client_metrics(result):
    if not result:
        return {}
    keys = ["completed", "duration", "request_throughput", "output_throughput",
            "total_input_tokens", "total_output_tokens",
            "mean_ttft_ms", "median_ttft_ms", "p95_ttft_ms", "p99_ttft_ms",
            "median_itl_ms", "p99_itl_ms", "median_tpot_ms", "p99_tpot_ms",
            "median_e2el_ms", "p99_e2el_ms"]
    return {f"client_{k}": result.get(k) for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="A", choices=list(C.SHAPES))
    ap.add_argument("--rates", default=",".join(str(r) for r in DEFAULT_RATES))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--measure-prompts", type=int, default=MEASURE_PROMPTS)
    ap.add_argument("--measure-seconds", type=float, default=None,
                    help="按到达秒数定样本量(num_prompts=rate*seconds)，M12 用的是 60s。"
                         "短窗口会系统性低估 P99 尾部，做容量判定必须用这个模式。")
    ap.add_argument("--warmup-prompts", type=int, default=WARMUP_PROMPTS)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    C.acquire_lock("m21_capacity_sweep")
    rates = [float(x) if "." in x else int(x) for x in args.rates.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    tag = args.tag or f"e3_shape{args.shape}"

    gpu0 = C.gpu_snapshot()
    print(f"[m21] GPU before: {gpu0}", flush=True)
    if "error" in gpu0:
        raise SystemExit(f"nvidia-smi 读取失败，停止：{gpu0}")

    server = C.Server()
    ready = server.start(os.path.join(C.LOGD, f"{tag}_server.log"))
    print(f"[m21] server ready {ready}s pid={server.pid()}", flush=True)

    runs = []
    runs_path = os.path.join(C.RAW, f"{tag}_runs.jsonl")
    os.makedirs(C.RAW, exist_ok=True)
    try:
        # 这里的 JIT = Triton GPU-kernel JIT（编译器），与 NVIDIA Triton Inference Server 无关
        print("[m21] global warmup (Triton kernel JIT 冷启动与 measure 分离)...", flush=True)
        C.warmup(10)
        time.sleep(5)

        for seed in seeds:
            for rate in rates:
                rid = f"{tag}_qps{rate}_s{seed}"
                n_measure = (int(round(rate * args.measure_seconds))
                             if args.measure_seconds else args.measure_prompts)
                # ---- warmup（结果不进汇总）----
                C.run_bench(args.shape, rate, args.warmup_prompts, rid + "_warmup",
                            seed=seed, timeout_s=900)
                C.wait_idle(60)
                time.sleep(3)

                # ---- measure ----
                before = counters_snapshot()
                bench = C.run_bench(args.shape, rate, n_measure, rid,
                                    seed=seed, timeout_s=1800)
                drained, drain_s = C.wait_idle(90)
                after = counters_snapshot()

                t0, t1 = bench["wall_start_ts"], bench["wall_end_ts"]
                time.sleep(3)  # 等最后一轮 scrape 落库
                rec = {"run_id": rid, "phase": "measure", "shape": args.shape,
                       "offered_qps": rate, "seed": seed,
                       "num_prompts": n_measure,
                       "measure_seconds_target": args.measure_seconds,
                       "exit_code": bench["exit_code"], "timeout": bench["timeout"],
                       "duration_s": bench["duration_s"], "drained": drained,
                       "drain_s": drain_s,
                       "wall_start_ts": t0, "wall_end_ts": t1,
                       "counter_delta": {k: (after[k] - before[k]) for k in before},
                       **client_metrics(bench["result"]),
                       **window_metrics(t0, t1)}
                rec["success_rate"] = (rec.get("client_completed") or 0) / n_measure
                runs.append(rec)
                with open(runs_path, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[m21] {rid}: client_p99_ttft={rec.get('client_p99_ttft_ms')}ms "
                      f"server_hist_p99={rec.get('server_hist_ttft_p99_ms')}ms "
                      f"waiting_max={rec.get('waiting_max')} "
                      f"gpu_util_max={rec.get('dcgm_gpu_util_max')}", flush=True)
    finally:
        server.stop()
        print(f"[m21] server stopped; GPU after: {C.gpu_snapshot()}", flush=True)

    # ---------------- 落盘 ----------------
    if runs:
        cols = sorted({k for r in runs for k in r if not isinstance(r[k], dict)})
        csv_path = os.path.join(C.OUT, f"{tag}.csv")
        os.makedirs(C.OUT, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["counter_delta_prompt_tokens",
                                              "counter_delta_generation_tokens",
                                              "counter_delta_request_success",
                                              "counter_delta_preemptions"] + cols)
            w.writeheader()
            for r in runs:
                d = r["counter_delta"]
                row = {k: r.get(k) for k in cols}
                row.update({
                    "counter_delta_prompt_tokens": d["vllm:prompt_tokens_total"],
                    "counter_delta_generation_tokens": d["vllm:generation_tokens_total"],
                    "counter_delta_request_success": d["vllm:request_success_total"],
                    "counter_delta_preemptions": d["vllm:num_preemptions_total"],
                })
                w.writerow(row)
        print(f"[m21] -> {csv_path}  ({len(runs)} runs)")
        print(f"[m21] -> {runs_path}")


if __name__ == "__main__":
    raise SystemExit(main())
