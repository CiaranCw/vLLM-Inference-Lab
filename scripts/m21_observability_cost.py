#!/usr/bin/env python3
"""M21 · E7 「监控本身要花多少钱」—— 单变量 A/B：监控栈 ON vs OFF。

动机（诚实记录发现过程）：
  E3 的 60s 窗口阶梯里，achieved 吞吐在 ~35 req/s 就饱和，而 M12 同配置测到 ~46.6 req/s。
  两次实验之间最大的已知差异，就是 M21 多跑了一整套监控栈（Prometheus 1s 抓取 + Grafana
  + dcgm-exporter 1s 采集 + host exporter）。不能靠猜，所以做这个对照。

单变量：只切换监控栈开关，server 全程同一个进程、同一份配置、同一批 seed。
  ON  = prometheus + grafana + dcgm-exporter 三个容器运行（host exporter 由 Prometheus 抓取时才干活）
  OFF = 三个容器全部停止（host exporter 进程留着，但无人抓取 → collect() 不会被调用）

顺序交错（seed 奇偶换 ON/OFF 先后），避免时间漂移与顺序偏置。

用法：python scripts/m21_observability_cost.py
"""

import json
import os
import subprocess
import time

import m21_common as C

DOCKER = os.path.expanduser("~/bin/docker")
COMPOSE = ["-f", "monitoring/docker-compose.yml"]
SERVICES = ["prometheus", "grafana", "dcgm-exporter"]

QPS_POINTS = [32, 40]
SEEDS = [0, 1, 2]
MEASURE_SECONDS = 60
WARMUP_PROMPTS = 30


def stack(state):
    """切换监控栈；返回该操作的原始输出（落盘留证）。"""
    action = ["start"] if state == "on" else ["stop"]
    p = subprocess.run([DOCKER, "compose"] + COMPOSE + action + SERVICES,
                       capture_output=True, text=True, timeout=180)
    time.sleep(8 if state == "on" else 4)
    return {"rc": p.returncode, "stdout": p.stdout[-400:], "stderr": p.stderr[-400:]}


def running_containers():
    p = subprocess.run([DOCKER, "ps", "--filter", "name=m21-", "--format", "{{.Names}}"],
                       capture_output=True, text=True, timeout=60)
    return sorted(x for x in p.stdout.split() if x)


def main():
    C.acquire_lock("m21_observability_cost")
    print("[e7] GPU before:", C.gpu_snapshot(), flush=True)
    server = C.Server()
    ready = server.start(os.path.join(C.LOGD, "e7_server.log"))
    print(f"[e7] server ready {ready}s pid={server.pid()}", flush=True)

    runs = []
    jsonl = os.path.join(C.RAW, "e7_obs_cost.jsonl")
    try:
        C.warmup(10)
        C.wait_idle(90)
        for seed in SEEDS:
            order = ["on", "off"] if seed % 2 == 0 else ["off", "on"]
            for qps in QPS_POINTS:
                for cond in order:
                    sw = stack(cond)
                    containers = running_containers()
                    rid = f"e7_qps{qps}_s{seed}_{cond}"
                    C.run_bench("A", qps, WARMUP_PROMPTS, rid + "_warmup", seed=seed,
                                timeout_s=900)
                    C.wait_idle(120)
                    time.sleep(3)
                    n = qps * MEASURE_SECONDS
                    bench = C.run_bench("A", qps, n, rid, seed=seed, timeout_s=2400)
                    C.wait_idle(240)
                    res = bench["result"] or {}
                    rec = {"run_id": rid, "condition": cond, "offered_qps": qps,
                           "seed": seed, "num_prompts": n,
                           "containers_running": containers, "stack_switch": sw,
                           "client_completed": res.get("completed"),
                           "client_request_throughput": res.get("request_throughput"),
                           "client_output_throughput": res.get("output_throughput"),
                           "client_median_ttft_ms": res.get("median_ttft_ms"),
                           "client_p99_ttft_ms": res.get("p99_ttft_ms"),
                           "client_median_itl_ms": res.get("median_itl_ms"),
                           "client_p99_itl_ms": res.get("p99_itl_ms"),
                           "client_median_e2el_ms": res.get("median_e2el_ms"),
                           "client_p99_e2el_ms": res.get("p99_e2el_ms"),
                           "duration_s": bench["duration_s"]}
                    runs.append(rec)
                    with open(jsonl, "a") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    print(f"[e7] {rid}: achieved={rec['client_request_throughput']:.1f} "
                          f"p99_ttft={rec['client_p99_ttft_ms']:.0f}ms "
                          f"containers={len(containers)}", flush=True)
    finally:
        stack("on")   # 恢复监控栈
        server.stop()
        print("[e7] server stopped; containers:", running_containers(), flush=True)

    # 汇总：按 (qps, condition) 取中位数
    import statistics
    summary = {}
    for qps in QPS_POINTS:
        for cond in ("on", "off"):
            rs = [r for r in runs if r["offered_qps"] == qps and r["condition"] == cond]
            if not rs:
                continue
            summary[f"qps{qps}_{cond}"] = {
                "n": len(rs),
                "achieved_median": statistics.median(
                    [r["client_request_throughput"] for r in rs]),
                "p99_ttft_median_ms": statistics.median([r["client_p99_ttft_ms"] for r in rs]),
                "p99_ttft_worst_ms": max(r["client_p99_ttft_ms"] for r in rs),
                "median_itl_median_ms": statistics.median(
                    [r["client_median_itl_ms"] for r in rs]),
            }
    C.save_json(os.path.join(C.OUT, "e7_observability_cost.json"),
                {"generated_at": time.strftime("%F %T%z"),
                 "design": "单变量：仅切换 prometheus/grafana/dcgm-exporter 三容器；server 全程同一进程",
                 "measure_seconds": MEASURE_SECONDS, "runs": runs, "summary": summary})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
