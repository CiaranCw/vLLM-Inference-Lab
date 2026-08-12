#!/usr/bin/env python3
"""M21 · E6 故障定位演练 —— 回答「OOM / 请求超时 / worker 退出怎样定位」。

**安全约束（所有者 2026-08-11 明确授权时规定，逐条落实在代码里）**：
  1. 只操作本脚本自己 Popen 出来、且 PID 已落盘的 server 进程；
  2. 禁止 pkill/killall 等模糊匹配；kill 前用 /proc/<pid>/cmdline 校验身份（见 Server.kill_self_pid）；
  3. 禁止 nvidia-smi --reset、禁止重启系统/WSL/Docker daemon、禁止碰任何其他进程；
  4. OOM 必须是**可恢复的进程级 CUDA OOM**，事后逐条验证：进程已退出 / 显存已回收 / server 可重启；
  5. 出现 GPU/driver 异常（NVML 报错、显存不回收）→ 立即中止后续演练并上报。

每个演练都要拿到两类证据：① 原始报错或原始指标；② 告警是否**真的 firing**。

用法：python scripts/m21_fault_drills.py            # 全部
      python scripts/m21_fault_drills.py --only oom
"""

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request

import m21_common as C

NVML_FREE_TOL_MIB = 800   # 显存回收判据容差（桌面进程本身会波动）
OOM_UTIL = "0.97"         # 制造进程级 CUDA OOM 的显存预算


def watch_alerts(seconds, interval=2.0):
    """轮询 Prometheus alerts，返回 {alertname: 首次 firing 的时间偏移秒}。"""
    seen = {}
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            for a in C.prom_alerts():
                if a["state"] == "firing":
                    name = a["labels"]["alertname"]
                    seen.setdefault(name, round(time.time() - t0, 1))
        except Exception:
            pass
        time.sleep(interval)
    return seen


def nvml_fb_used_mib():
    v = C.prom_scalar("max(m21_nvml_fb_used_bytes)")
    return None if v is None else round(v / 1024 / 1024)


def wait_reclaim(baseline_mib, timeout_s=240, interval=5):
    """轮询等显存回到基线附近，返回 (是否回收, 耗时秒, 最终读数)。

    实测：WSL2 下 SIGKILL 之后显存**不是立刻**归还的（首版只等 10s 判定为「未回收」，
    是判据太急，不是系统有泄漏）。这里改成轮询并记录真实回收耗时。
    """
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout_s:
        last = nvml_fb_used_mib()
        if last is not None and baseline_mib is not None and \
                last - baseline_mib < NVML_FREE_TOL_MIB:
            return True, round(time.time() - t0, 1), last
        time.sleep(interval)
    return False, round(time.time() - t0, 1), last


# --------------------------------------------------------------------- D1 worker exit
def drill_worker_exit():
    rec = {"drill": "worker_exit",
           "purpose": "worker 进程被杀时，监控上长什么样、怎么确认不是端口问题"}
    baseline_idle = nvml_fb_used_mib()
    rec["nvml_fb_used_mib_before_server"] = baseline_idle

    server = C.Server()
    rec["server_ready_s"] = server.start(os.path.join(C.LOGD, "e6_worker_exit_server.log"))
    rec["server_pid"] = server.pid()
    C.warmup(5)
    time.sleep(8)
    rec["nvml_fb_used_mib_serving"] = nvml_fb_used_mib()
    rec["up_vllm_before_kill"] = C.prom_scalar('up{job="vllm"}')
    rec["target_up_before_kill"] = C.prom_scalar("m21_target_up")

    # kill 之前先把子进程 PID 落盘：SIGKILL 父进程不会连带杀死持有显存的 EngineCore
    rec["child_pids_before_kill"] = server.child_pids()

    killed, pid, cmdline = server.kill_self_pid(sig=9)
    rec["killed"] = killed
    rec["killed_pid"] = pid
    rec["killed_cmdline_checked"] = cmdline[:200]
    if not killed:
        rec["error"] = "PID 身份校验未通过，按安全约束不发信号"
        return rec

    alerts = watch_alerts(75)
    rec["alerts_firing"] = alerts
    rec["up_vllm_after_kill"] = C.prom_scalar('up{job="vllm"}')
    rec["target_up_after_kill"] = C.prom_scalar("m21_target_up")
    # 第一次观察：只杀父进程后，显存**不会**自动回收（这正是本演练最有价值的发现）
    ok1, secs1, last1 = wait_reclaim(baseline_idle, timeout_s=90)
    rec["parent_only_kill"] = {
        "gpu_memory_reclaimed": ok1,
        "waited_s": secs1,
        "nvml_fb_used_mib": last1,
        "orphan_engine_cores_alive": [
            {"pid": c.get("pid"), "alive": os.path.exists(f"/proc/{c.get('pid')}")}
            for c in rec["child_pids_before_kill"] if c.get("pid")],
    }

    # 恢复动作：清理**先前已记录**的 EngineCore 子进程 PID，再验证显存回收
    rec["orphan_cleanup"] = C.Server.kill_recorded_pids(rec["child_pids_before_kill"])
    ok2, secs2, last2 = wait_reclaim(baseline_idle, timeout_s=180)
    rec["nvml_fb_used_mib_after_cleanup"] = last2
    rec["gpu_memory_reclaim_seconds"] = secs2
    rec["gpu_memory_reclaimed"] = ok2

    # 恢复：确认能重新拉起
    server2 = C.Server()
    rec["restart_ready_s"] = server2.start(os.path.join(C.LOGD, "e6_worker_exit_server.log"),
                                           expect_failure=True)
    rec["restart_ok"] = rec["restart_ready_s"] is not None
    server2.stop()
    rec["PASS"] = bool(rec["killed"] and rec["gpu_memory_reclaimed"] and rec["restart_ok"]
                       and "VLLMServerDown" in alerts)
    return rec


# --------------------------------------------------------------------- D2 timeout
def drill_timeout(overload_qps=64, probe_timeout_s=0.5, probes=60):
    """过载下用短 timeout 的探针客户端打请求，看客户端超时与服务端队列指标的对应关系。"""
    rec = {"drill": "request_timeout", "overload_qps": overload_qps,
           "probe_timeout_s": probe_timeout_s, "probes": probes,
           "purpose": "区分「客户端超时」与「服务端真的堆积」"}
    server = C.Server()
    rec["server_ready_s"] = server.start(os.path.join(C.LOGD, "e6_timeout_server.log"))
    try:
        C.warmup(8)
        C.wait_idle(60)

        # 背景过载负载
        n = int(overload_qps * 45)
        bench_result = {}

        def _bg():
            bench_result["r"] = C.run_bench("A", overload_qps, n, "e6_timeout_load",
                                            timeout_s=1200)

        th = threading.Thread(target=_bg, daemon=True)
        th.start()
        time.sleep(12)  # 等队列真正堆起来

        # 探针：短 timeout 的普通请求
        url = f"http://{C.HOST}:{C.PORT}/v1/completions"
        body = json.dumps({"model": C.MODEL, "prompt": "probe", "max_tokens": 16,
                           "temperature": 0}).encode()
        timeouts, oks, errs = 0, 0, []
        t0 = time.time()
        for _ in range(probes):
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            try:
                urllib.request.urlopen(req, timeout=probe_timeout_s).read()
                oks += 1
            except urllib.error.URLError as e:
                timeouts += 1
                errs.append(f"{type(e).__name__}: {e}")
            except Exception as e:  # socket.timeout 等
                timeouts += 1
                errs.append(f"{type(e).__name__}: {e}")
            time.sleep(0.2)
        t1 = time.time()

        rec["probe_ok"] = oks
        rec["probe_timeout_count"] = timeouts
        rec["probe_error_samples"] = errs[:3]
        rec["server_waiting_max_during_probe"] = C.prom_range_agg(
            "sum(vllm:num_requests_waiting)", t0, t1)["max"]
        rec["server_running_max_during_probe"] = C.prom_range_agg(
            "sum(vllm:num_requests_running)", t0, t1)["max"]
        rec["server_hist_ttft_p99_ms_during_probe"] = C.prom_scalar_at(
            "histogram_quantile(0.99, sum by (le) "
            "(increase(vllm:time_to_first_token_seconds_bucket[30s])))", t1)
        rec["alerts_firing"] = watch_alerts(25)
        th.join(timeout=600)
        res = (bench_result.get("r") or {}).get("result") or {}
        rec["background_load_completed"] = res.get("completed")
        rec["background_load_p99_ttft_ms"] = res.get("p99_ttft_ms")
        C.wait_idle(180)
    finally:
        server.stop()
    rec["PASS"] = bool(rec.get("probe_timeout_count", 0) > 0
                       and rec.get("server_waiting_max_during_probe") is not None)
    return rec


# --------------------------------------------------------------------- D3 OOM
def drill_oom():
    """进程级、可恢复的 CUDA OOM：把显存预算调到远超实际可用，令 server 启动期失败。"""
    rec = {"drill": "oom", "gpu_memory_utilization": OOM_UTIL,
           "purpose": "OOM 的原始报错长什么样、事后如何确认显存已回收且可恢复"}
    baseline = nvml_fb_used_mib()
    rec["nvml_fb_used_mib_before"] = baseline
    rec["nvidia_smi_before"] = C.gpu_snapshot()

    log_path = os.path.join(C.LOGD, "e6_oom_server.log")
    server = C.Server(extra_args=["--gpu-memory-utilization", OOM_UTIL])
    ready = server.start(log_path, health_timeout=240, expect_failure=True)
    rec["server_ready_s"] = ready
    rec["server_started_unexpectedly"] = ready is not None
    if ready is not None:
        # 没 OOM 成功：如实记录，不重试更激进的参数（安全优先）
        server.stop()
        rec["note"] = "预算 0.97 下 server 仍启动成功，未构造出 OOM —— 如实记录"
        rec["PASS"] = False
        return rec

    # 抓原始报错
    with open(log_path) as f:
        tail = f.read()[-20000:]
    keys = [ln for ln in tail.splitlines()
            if any(k in ln for k in ("OutOfMemory", "out of memory", "ValueError",
                                     "free memory", "gpu_memory_utilization", "Error"))]
    rec["error_lines"] = keys[-12:]
    rec["log_path"] = log_path

    ok, secs, last = wait_reclaim(baseline)
    rec["nvml_fb_used_mib_after"] = last
    rec["gpu_memory_reclaim_seconds"] = secs
    rec["nvidia_smi_after"] = C.gpu_snapshot()
    rec["gpu_memory_reclaimed"] = ok
    rec["process_exited"] = not server.alive()

    # 恢复验证：正常配置能否重新起来
    ok_server = C.Server()
    try:
        rec["recovery_ready_s"] = ok_server.start(os.path.join(C.LOGD, "e6_oom_recovery.log"))
        rec["recovery_ok"] = True
    except Exception as e:
        rec["recovery_ok"] = False
        rec["recovery_error"] = f"{type(e).__name__}: {e}"
    finally:
        ok_server.stop()

    rec["PASS"] = bool(rec["process_exited"] and rec["gpu_memory_reclaimed"]
                       and rec.get("recovery_ok"))
    return rec


# --------------------------------------------------------------------- D4 preemption
def drill_preemption(kv_util="0.12", qps=4, seconds=60, shape="C"):
    """压低显存预算制造 KV 压力 → 验证抢占计数上升且告警 firing。"""
    # 参数来自 E4 实测：只压显存预算到 0.10 仍只有 59% KV 使用率；
    # 必须同时从需求侧构造（Shape C = in128/out1900 长输出），util=0.12 时才真的触发抢占。
    rec = {"drill": "preemption", "gpu_memory_utilization": kv_util, "shape": shape,
           "offered_qps": qps, "purpose": "KV 不足的可观测签名 = kv_usage 高 + 抢占计数上升"}
    server = C.Server(extra_args=["--gpu-memory-utilization", kv_util])
    ready = server.start(os.path.join(C.LOGD, "e6_preemption_server.log"),
                         health_timeout=240, expect_failure=True)
    rec["server_ready_s"] = ready
    if ready is None:
        rec["note"] = f"显存预算 {kv_util} 下 server 起不来 —— 如实记录，未构造出抢占"
        rec["PASS"] = False
        return rec
    try:
        C.warmup(8)
        C.wait_idle(90)
        before = C.scrape_vllm_metrics()
        alerts_box = {}

        def _watch():
            alerts_box["a"] = watch_alerts(seconds + 20)

        th = threading.Thread(target=_watch, daemon=True)
        th.start()
        bench = C.run_bench(shape, qps, int(qps * seconds), "e6_preemption_load",
                            timeout_s=2400)
        drained, _ = C.wait_idle(240)
        after = C.scrape_vllm_metrics()
        th.join(timeout=60)

        t0, t1 = bench["wall_start_ts"], bench["wall_end_ts"]
        rec["preemption_delta"] = (after.get("vllm:num_preemptions_total", 0)
                                   - before.get("vllm:num_preemptions_total", 0))
        rec["kv_usage_max"] = C.prom_range_agg("max(vllm:kv_cache_usage_perc)", t0, t1)["max"]
        rec["waiting_max"] = C.prom_range_agg("sum(vllm:num_requests_waiting)", t0, t1)["max"]
        rec["alerts_firing"] = alerts_box.get("a", {})
        res = bench["result"] or {}
        rec["client_completed"] = res.get("completed")
        rec["client_p99_ttft_ms"] = res.get("p99_ttft_ms")
        rec["drained"] = drained
    finally:
        server.stop()
    rec["PASS"] = bool(rec.get("preemption_delta", 0) > 0)
    return rec


DRILLS = {"worker_exit": drill_worker_exit, "timeout": drill_timeout,
          "oom": drill_oom, "preemption": drill_preemption}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", choices=sorted(DRILLS))
    ap.add_argument("--kv-util", default="0.12", help="preemption 演练用的显存预算")
    args = ap.parse_args()

    names = args.only or ["worker_exit", "oom", "timeout", "preemption"]
    C.acquire_lock("m21_fault_drills")
    print("[e6] GPU before:", C.gpu_snapshot(), flush=True)

    out, jsonl = [], os.path.join(C.RAW, "e6_drills.jsonl")
    for name in names:
        print(f"\n[e6] ===== {name} =====", flush=True)
        fn = DRILLS[name]
        rec = fn(kv_util=args.kv_util) if name == "preemption" else fn()
        rec["finished_at"] = time.strftime("%F %T%z")
        out.append(rec)
        with open(jsonl, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[e6] {name}: PASS={rec.get('PASS')} alerts={rec.get('alerts_firing')}", flush=True)

        gpu = C.gpu_snapshot()
        if "error" in gpu:
            print("[e6] nvidia-smi 异常，按安全约束中止后续演练:", gpu, flush=True)
            break

    path = C.save_json(os.path.join(C.OUT, "e6_drills.json"),
                       {"generated_at": time.strftime("%F %T%z"), "drills": out})
    print(f"\n[e6] -> {path}")


if __name__ == "__main__":
    raise SystemExit(main())
