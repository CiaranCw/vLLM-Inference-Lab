#!/usr/bin/env python3
"""M21 Gate 1 / E1 —— 监控栈自证。

判据（notes/m21/monitoring_and_capacity.md §4·E1）：
  G1-1 Prometheus 四个 target（vllm/dcgm/host/prometheus）全部 health=up
  G1-2 告警规则全部加载成功且无 firing（空载态）
  G1-3 关键 PromQL 都能取到数（指标名与 dashboard 里写的一致）
  G1-4 Grafana datasource + dashboard 均由 provisioning 成功加载

副产物：把 vLLM /metrics 原文落盘，供后续核对指标名（防止面板写错名字却静默显示空图）。

用法：python scripts/m21_gate1.py
"""

import json
import os
import time

import m21_common as C

# 单位约定（2026-08-12 audit）：vLLM 的 *_seconds_* 直方图返回秒，PromQL 不做换算，
# 因此这里的 key 一律带显式单位后缀，避免再出现「秒值被存进 _ms 字段」那类错误
# （E6 request_timeout 首版就栽在这上面）。
CHECK_EXPRS = {
    "vllm_running": "sum(vllm:num_requests_running)",
    "vllm_waiting": "sum(vllm:num_requests_waiting)",
    "vllm_kv_usage": "max(vllm:kv_cache_usage_perc)",
    "vllm_ttft_p99_seconds": ("histogram_quantile(0.99, sum by (le) "
                      "(rate(vllm:time_to_first_token_seconds_bucket[30s])))"),
    "vllm_itl_p99_seconds": ("histogram_quantile(0.99, sum by (le) "
                     "(rate(vllm:inter_token_latency_seconds_bucket[30s])))"),
    "vllm_success_rate": "sum(rate(vllm:request_success_total[30s]))",
    "vllm_prompt_tok_rate": "sum(rate(vllm:prompt_tokens_total[30s]))",
    "vllm_gen_tok_rate": "sum(rate(vllm:generation_tokens_total[30s]))",
    "vllm_preemptions": "sum(vllm:num_preemptions_total)",
    "dcgm_gpu_util": "max(DCGM_FI_DEV_GPU_UTIL)",
    "dcgm_fb_used_mib": "max(DCGM_FI_DEV_FB_USED)",  # DCGM 该字段单位为 MiB
    "nvml_gpu_util": "max(m21_nvml_gpu_util_percent)",
    "nvml_fb_used_bytes": "max(m21_nvml_fb_used_bytes)",
    "proc_cpu_percent": "m21_target_cpu_percent",
    "proc_rss_bytes": "m21_target_rss_bytes",
    "target_up": "m21_target_up",
}


def grafana_check():
    out = {}
    try:
        ds = json.loads(C.http_get("http://127.0.0.1:3000/api/datasources", timeout=10))
        out["datasources"] = [d["name"] for d in ds]
    except Exception as e:
        out["datasources_error"] = f"{type(e).__name__}: {e}"
    try:
        s = json.loads(C.http_get(
            "http://127.0.0.1:3000/api/search?type=dash-db", timeout=10))
        out["dashboards"] = [{"title": d["title"], "uid": d["uid"]} for d in s]
    except Exception as e:
        out["dashboards_error"] = f"{type(e).__name__}: {e}"
    return out


def main():
    os.makedirs(C.LOGD, exist_ok=True)
    C.acquire_lock("m21_gate1")
    report = {"started_at": time.strftime("%F %T%z"),
              "gpu_before": C.gpu_snapshot()}
    print("[gate1] GPU before:", report["gpu_before"], flush=True)

    server = C.Server()
    log_path = os.path.join(C.LOGD, "gate1_server.log")
    print("[gate1] starting vllm serve (M12 冻结配置)...", flush=True)
    report["server_cmd"] = server.cmd
    report["server_ready_s"] = server.start(log_path)
    report["server_pid"] = server.pid()
    print(f"[gate1] server ready in {report['server_ready_s']}s pid={server.pid()}", flush=True)

    try:
        C.warmup(5)
        # 冷启动观察：warmup 首请求含 Triton GPU-kernel JIT / 图构建，TTFT 会远高于稳态
        # （AGENTS.md §5）。此处 Triton = GPU kernel 编译器，不是 NVIDIA Triton Inference Server。
        # 先在 rate 窗口内取一次读数，作为「冷启动确实会打破 SLO 告警」的证据。
        time.sleep(15)
        report["cold_start_ttft_p99_in_window_s"] = C.prom_scalar(CHECK_EXPRS["vllm_ttft_p99_seconds"])
        report["alerts_firing_cold_window"] = [
            a["labels"]["alertname"] for a in C.prom_alerts() if a["state"] == "firing"]

        # 再等 warmup 样本滑出 30s rate 窗口，才判「空载无告警」。
        time.sleep(45)

        # ---- 落盘 vLLM /metrics 原文，供核对指标名 ----
        raw = C.http_get(f"http://{C.HOST}:{C.PORT}/metrics", timeout=15)
        raw_path = os.path.join(C.LOGD, "vllm_metrics_sample.txt")
        with open(raw_path, "w") as f:
            f.write(raw)
        names = sorted({ln.split("{")[0].split(" ")[0]
                        for ln in raw.splitlines() if ln and not ln.startswith("#")})
        report["vllm_metric_sample_path"] = raw_path
        report["vllm_metric_series_count"] = len(names)

        # ---- G1-1 targets ----
        targets = C.prom_targets()
        report["targets"] = targets
        report["G1_1_all_targets_up"] = all(v["health"] == "up" for v in targets.values()) \
            and set(targets) >= {"vllm", "dcgm", "host", "prometheus"}

        # ---- G1-2 rules ----
        rules = json.loads(C.http_get(f"{C.PROM}/api/v1/rules", timeout=10))
        loaded = {g["name"]: [r["name"] for r in g["rules"]] for g in rules["data"]["groups"]}
        report["alert_rules_loaded"] = loaded
        report["alert_rule_count"] = sum(len(v) for v in loaded.values())
        firing = [a["labels"]["alertname"] for a in C.prom_alerts() if a["state"] == "firing"]
        report["alerts_firing_idle"] = firing
        report["G1_2_rules_loaded_no_firing"] = report["alert_rule_count"] >= 7 and not firing

        # ---- G1-3 PromQL ----
        vals, missing = {}, []
        for k, expr in CHECK_EXPRS.items():
            v = C.prom_scalar(expr)
            vals[k] = v
            if v is None:
                missing.append(k)
        report["promql_values"] = vals
        report["promql_missing"] = missing
        report["G1_3_all_exprs_resolve"] = not missing

        # ---- G1-4 grafana ----
        g = grafana_check()
        report["grafana"] = g
        report["G1_4_grafana_provisioned"] = ("Prometheus" in g.get("datasources", [])) and \
            any(d["uid"] == "m21-vllm-capacity" for d in g.get("dashboards", []))
    finally:
        server.stop()
        report["gpu_after_stop"] = C.gpu_snapshot()

    report["PASS"] = all(report.get(k) for k in
                         ("G1_1_all_targets_up", "G1_2_rules_loaded_no_firing",
                          "G1_3_all_exprs_resolve", "G1_4_grafana_provisioned"))
    path = C.save_json(os.path.join(C.OUT, "e1_targets.json"), report)

    print("\n=== Gate 1 ===")
    for k in ("G1_1_all_targets_up", "G1_2_rules_loaded_no_firing",
              "G1_3_all_exprs_resolve", "G1_4_grafana_provisioned", "PASS"):
        print(f"  {k}: {report.get(k)}")
    if missing := report.get("promql_missing"):
        print("  取不到数的表达式:", missing)
    print("  targets:", {k: v["health"] for k, v in report["targets"].items()})
    print("  ->", path)
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
