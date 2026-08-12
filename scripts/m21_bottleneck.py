#!/usr/bin/env python3
"""M21 · E4 瓶颈签名矩阵 —— 回答「瓶颈是 GPU 计算 / 显存 / KV / CPU / 队列」。

四个场景各自制造一类瓶颈，看**哪个指标先饱和**，产出「症状 → 指标签名 → 根因」对照表。
每个场景一个独立 server 生命周期（配置不同，必须重启；也避免 KV 显存残留互相污染）。

| 场景 | 构造 | 预期签名（执行前假设，允许被推翻） |
|---|---|---|
| B1 CPU 提交受限 | Shape A，低 QPS（并发≈1） | GPU util 低 + ITL 卡在 ~7ms 地板（承接 M7） |
| B2 GPU 计算受限 | Shape B(in1024/out32) 高 QPS，prefill 重 | GPU util 显著更高 |
| B3 KV 受限     | 压低 --gpu-memory-utilization + 满并发长输出 | kv_usage→1 且 preemption>0（正交双证据） |
| B4 队列受限    | Shape A 超容量 QPS | waiting 持续堆积、TTFT 尾爆，但吞吐仍跟随 |

诚实约束：B3 需要实测调参才能造出压力；**若某场景造不出干净对照，如实记录为「未能构造」**，
不得把别的模块的旧结论搬过来充数。

用法：python scripts/m21_bottleneck.py
"""

import json
import os
import time

import m21_common as C

MEASURE_SECONDS = 60
WARMUP_PROMPTS = 20

SCENARIOS = [
    {"name": "B1_cpu_submit", "shape": "A", "rate": 1, "extra_args": [],
     "hypothesis": "GPU util 低但延迟有地板 → CPU 提交受限(M7)"},
    {"name": "B2_gpu_compute", "shape": "B", "rate": 24, "extra_args": [],
     "hypothesis": "prefill 重 → GPU util 明显高于 B1/B4"},
    # B3 的 util 由 --gpu-memory-utilization 压低制造 KV 压力；多档递降直到出现抢占
    {"name": "B3_kv_pressure", "shape": "A", "rate": 40,
     "extra_args": ["--gpu-memory-utilization", "0.40"],
     # 实测：util 0.40/0.38/0.36 下 kv_usage 峰值仅 ~5%（vLLM 的显存预算按自身用量算，
     # 不含桌面进程占用的 4.2GB），必须压到更低才可能出现抢占
     "kv_ladder": ["0.20", "0.15", "0.12", "0.10"],
     "hypothesis": "kv_usage 逼近 1 且 num_preemptions 上升"},
    # B3 的预算侧路线到 util=0.10 仍只有 59% KV 使用率（未出现抢占），
    # 故改从**需求侧**构造：长输出让每条序列的 KV 随解码不断增长
    {"name": "B3b_kv_long_output", "shape": "C", "rate": 4,
     "extra_args": ["--gpu-memory-utilization", "0.20"],
     "kv_ladder": ["0.20", "0.15", "0.12", "0.10"],
     "hypothesis": "长输出把 KV 需求推到池子上限 → kv_usage→1 且出现抢占"},
    {"name": "B4_queue", "shape": "A", "rate": 64, "extra_args": [],
     "hypothesis": "waiting 持续堆积、TTFT 尾爆，吞吐仍跟随 offered"},
]

WINDOW = {
    "dcgm_gpu_util": "max(DCGM_FI_DEV_GPU_UTIL)",
    "nvml_gpu_util": "max(m21_nvml_gpu_util_percent)",
    "proc_cpu_percent": "m21_target_cpu_percent",
    "host_cpu_percent": "m21_host_cpu_percent",
    "kv_usage": "max(vllm:kv_cache_usage_perc)",
    "waiting": "sum(vllm:num_requests_waiting)",
    "running": "sum(vllm:num_requests_running)",
    "dcgm_fb_used_mib": "max(DCGM_FI_DEV_FB_USED)",
}


def counters():
    m = C.scrape_vllm_metrics()
    return {k: m.get(k) for k in ("vllm:num_preemptions_total", "vllm:request_success_total",
                                  "vllm:prompt_tokens_total", "vllm:generation_tokens_total")}


def measure(scn, extra_args, run_id):
    """起 server → warmup → 60s 压测 → 收指标 → 停 server。"""
    server = C.Server(extra_args=extra_args)
    log_path = os.path.join(C.LOGD, f"e4_{run_id}_server.log")
    rec = {"run_id": run_id, "scenario": scn["name"], "shape": scn["shape"],
           "offered_qps": scn["rate"], "extra_args": extra_args,
           "hypothesis": scn["hypothesis"]}
    try:
        rec["server_ready_s"] = server.start(log_path, expect_failure=True)
        if rec["server_ready_s"] is None:
            rec["server_start_failed"] = True
            rec["note"] = "server 启动失败（多半是显存预算过低），见 log"
            return rec
        rec["server_pid"] = server.pid()
        C.warmup(8)
        C.run_bench(scn["shape"], scn["rate"], WARMUP_PROMPTS, run_id + "_warmup", timeout_s=900)
        C.wait_idle(90)
        time.sleep(3)

        n = int(round(scn["rate"] * MEASURE_SECONDS))
        before = counters()
        bench = C.run_bench(scn["shape"], scn["rate"], n, run_id, timeout_s=2400)
        drained, drain_s = C.wait_idle(180)
        after = counters()
        t0, t1 = bench["wall_start_ts"], bench["wall_end_ts"]
        time.sleep(3)

        res = bench["result"] or {}
        rec.update({
            "num_prompts": n, "duration_s": bench["duration_s"],
            "exit_code": bench["exit_code"], "drained": drained, "drain_s": drain_s,
            "counter_delta": {k: (after[k] - before[k]) for k in before},
            "client_completed": res.get("completed"),
            "client_request_throughput": res.get("request_throughput"),
            "client_output_throughput": res.get("output_throughput"),
            "client_median_ttft_ms": res.get("median_ttft_ms"),
            "client_p99_ttft_ms": res.get("p99_ttft_ms"),
            "client_median_itl_ms": res.get("median_itl_ms"),
            "client_p99_itl_ms": res.get("p99_itl_ms"),
            "client_median_tpot_ms": res.get("median_tpot_ms"),
            "client_p99_e2el_ms": res.get("p99_e2el_ms"),
        })
        for key, expr in WINDOW.items():
            agg = C.prom_range_agg(expr, t0, t1)
            rec[f"{key}_max"] = agg["max"]
            rec[f"{key}_mean"] = agg["mean"]
        rec["success_rate"] = (rec.get("client_completed") or 0) / n
    finally:
        server.stop()
        rec["gpu_after"] = C.gpu_snapshot()
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append",
                    choices=[s["name"] for s in SCENARIOS])
    ap.add_argument("--kv-ladder", default=None,
                    help="覆盖 kv_ladder，逗号分隔，如 0.12,0.10")
    args = ap.parse_args()
    scenarios = ([s for s in SCENARIOS if s["name"] in args.only]
                 if args.only else SCENARIOS)
    if args.kv_ladder:
        for s in scenarios:
            if s.get("kv_ladder"):
                s["kv_ladder"] = args.kv_ladder.split(",")
    C.acquire_lock("m21_bottleneck")
    print("[e4] GPU before:", C.gpu_snapshot(), flush=True)
    out = []
    out_path = os.path.join(C.OUT, "e4_bottleneck.json")
    jsonl = os.path.join(C.RAW, "e4_runs.jsonl")

    for scn in scenarios:
        if scn.get("kv_ladder"):
            # 逐档压低显存预算，直到真的出现抢占；每一档都落盘（含没造出压力的）
            got = False
            for util in scn["kv_ladder"]:
                rid = f"{scn['name']}_util{util}"
                rec = measure(scn, ["--gpu-memory-utilization", util], rid)
                rec["kv_ladder_step"] = util
                preempt = (rec.get("counter_delta") or {}).get("vllm:num_preemptions_total", 0)
                rec["preemption_observed"] = bool(preempt and preempt > 0)
                out.append(rec)
                with open(jsonl, "a") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[e4] {rid}: kv_max={rec.get('kv_usage_max')} "
                      f"preempt={preempt} started={not rec.get('server_start_failed')}", flush=True)
                if rec["preemption_observed"]:
                    got = True
                    break
            if not got:
                print(f"[e4] {scn['name']} 未能造出抢占 —— 如实记录，不得假装成功",
                      flush=True)
        else:
            rec = measure(scn, scn["extra_args"], scn["name"])
            out.append(rec)
            with open(jsonl, "a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[e4] {scn['name']}: gpu_util_max={rec.get('dcgm_gpu_util_max')} "
                  f"proc_cpu_max={rec.get('proc_cpu_percent_max')} "
                  f"waiting_max={rec.get('waiting_max')} "
                  f"itl_med={rec.get('client_median_itl_ms')}", flush=True)

    if args.only:
        # 只跑部分场景时，合并已有结果，避免覆盖掉先前跑过的场景
        if os.path.exists(out_path):
            old = json.load(open(out_path))["runs"]
            keep = [r for r in old if r["scenario"] not in {s["name"] for s in scenarios}]
            out = keep + out
    C.save_json(out_path, {"generated_at": time.strftime("%F %T%z"),
                           "measure_seconds": MEASURE_SECONDS, "runs": out})
    print(f"[e4] -> {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
