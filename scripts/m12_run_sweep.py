"""M12 · 在线/开环 Serving 与 SLO —— orchestrator。

职责（见批准计划）：起停真实 OpenAI server(Popen, 冻结命令) + /health + 全局 warmup +
metrics sampler(0.5s) + calibration(rate=1) + 冻结 SLO 入 config.json +
Shape A 粗扫 → Knee 精扫(3 seed, 每 seed 重启 server) + Burst(inf) + Shape B +
每 run: vllm bench serve --save-result --save-detailed --goodput + drain barrier + timeout + metadata；
finally 关 bench/sampler/server、验证端口释放。

指标为真在线值（HTTP+流式）。零 vLLM 源码改动。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.request

HOST = "127.0.0.1"
PORT = 8014
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
OUT = "outputs/m12"
LOGD = "logs/m12"
ENV = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")

SERVER_CMD = [
    "vllm", "serve", MODEL, "--host", HOST, "--port", str(PORT), "--seed", "0",
    "--enforce-eager", "--dtype", "auto", "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.6", "--max-num-seqs", "64",
    "--max-num-batched-tokens", "4096", "--no-enable-prefix-caching",
    "--no-async-scheduling", "--enable-chunked-prefill", "--disable-uvicorn-access-log",
]
SHAPES = {"A": (256, 128), "B": (1024, 32)}

# 冻结阈值（跑前写入 config.json）
QUEUE = dict(sample_interval_s=0.5, min_samples=20, slope_thr=0.5, abs_diff=5,
             ratio=2, base_floor=1, end_waiting_thr=5, drain_budget_s=20)
SLO_MULT = dict(ttft=3, tpot=2, e2el=2, itl=3)
SLO_FLOOR = dict(ttft_p99_ms=300, tpot_p99_ms=20, e2el_p99_ms=2000, itl_p99_guardrail_ms=30)
SUCCESS_MIN, SLO_PASS_MIN, PREEMPT_LIMIT, TIMEOUT_LIMIT = 0.99, 0.95, 0, 0
FINE_SEEDS = [0, 1, 2]
COARSE_RATES_A = [1, 2, 4, 8, 16, 24, 32, 48]
COARSE_RATES_B = [1, 2, 4, 8, 16, 24, 32]


def _frange(a, b, step):
    xs = []; x = a
    while x <= b + 1e-9:
        xs.append(round(x, 2)); x += step
    return xs

_GAUGE = ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc")


def _metrics_url():
    return f"http://{HOST}:{PORT}/metrics"


def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def health_code():
    try:
        with urllib.request.urlopen(f"http://{HOST}:{PORT}/health", timeout=3) as r:
            return r.status
    except Exception:
        return 0


def parse_metrics(text):
    """解析 Prometheus 文本 → 关键指标（gauge 取值；counter/histogram 取 total）。"""
    d = {}
    for name in _GAUGE:
        m = re.search(rf"^vllm:{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)", text, re.M)
        if m:
            d[name] = float(m.group(1))
    m = re.search(r"^vllm:num_preemptions_total(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M)
    d["num_preemptions_total"] = float(m.group(1)) if m else 0.0
    m = re.search(r"^vllm:request_queue_time_seconds_count(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M)
    d["request_queue_time_count"] = float(m.group(1)) if m else 0.0
    m = re.search(r"^vllm:request_queue_time_seconds_sum(?:\{[^}]*\})?\s+([0-9.eE+-]+)", text, re.M)
    d["request_queue_time_sum"] = float(m.group(1)) if m else 0.0
    return d


class Server:
    def __init__(self):
        self.proc = None
        self.restart_count = 0

    def _port_free(self):
        s = socket.socket(); r = s.connect_ex((HOST, PORT)); s.close(); return r != 0

    def start(self, log_path):
        if not self._port_free():
            raise RuntimeError(f"port {PORT} already in use")
        self.lf = open(log_path, "a")
        self.proc = subprocess.Popen(SERVER_CMD, env=ENV, stdout=self.lf, stderr=subprocess.STDOUT)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 120:
            if health_code() == 200:
                return round(time.perf_counter() - t0, 2)
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early rc={self.proc.returncode} (see log)")
            time.sleep(2)
        raise RuntimeError("server health timeout 120s")

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=20)
            except Exception:
                self.proc.kill(); self.proc.wait(timeout=10)
        self.proc = None
        # 等端口释放
        for _ in range(20):
            if self._port_free():
                break
            time.sleep(0.5)

    def restart(self, log_path):
        self.stop(); self.restart_count += 1; return self.start(log_path)


class Sampler(threading.Thread):
    def __init__(self, path, run_start):
        super().__init__(daemon=True)
        self.path = path; self.run_start = run_start
        self.phase = "idle"; self._stopev = threading.Event(); self.rows = []

    def run(self):
        f = open(self.path, "w")
        while not self._stopev.is_set():
            try:
                m = parse_metrics(http_get(_metrics_url()))
                now = time.perf_counter() # 不受系统时钟校准影响，因此 Monotonic Time 更可靠
                rec = {"timestamp_monotonic_s": round(now, 3),
                       "time_since_run_start_s": round(now - self.run_start, 3),
                       "phase": self.phase, **{k: m.get(k) for k in
                       ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
                        "num_preemptions_total", "request_queue_time_count", "request_queue_time_sum")}}
                self.rows.append(rec)
                f.write(json.dumps(rec) + "\n"); f.flush()
            except Exception:
                pass
            self._stopev.wait(QUEUE["sample_interval_s"])
        f.close()

    def stop(self):
        self._stopev.set()


def wait_idle(timeout, need=3):
    """轮询 /metrics 直到 running==0 && waiting==0 连续 need 次；返回 (drained, elapsed)。"""
    t0 = time.perf_counter(); ok = 0
    while time.perf_counter() - t0 < timeout:
        try:
            m = parse_metrics(http_get(_metrics_url()))
            if (m.get("num_requests_running", 1) == 0) and (m.get("num_requests_waiting", 1) == 0):
                ok += 1
                if ok >= need:
                    return True, round(time.perf_counter() - t0, 2)
            else:
                ok = 0
        except Exception:
            ok = 0
        time.sleep(0.5)
    return False, round(time.perf_counter() - t0, 2)


def warmup(n=10):
    url = f"http://{HOST}:{PORT}/v1/completions"
    body = json.dumps({"model": MODEL, "prompt": "warmup", "max_tokens": 8, "temperature": 0}).encode()
    for _ in range(n):
        try:
            urllib.request.urlopen(urllib.request.Request(url, data=body,
                                   headers={"Content-Type": "application/json"}), timeout=30).read()
        except Exception:
            pass
    wait_idle(30)


def run_bench(shape, rate, seed, num_prompts, run_id, goodput, sampler, timeout_s):
    inl, outl = SHAPES[shape]
    result_file = f"{run_id}.json"
    cmd = ["vllm", "bench", "serve", 
           "--backend", "openai", 
           "--model", MODEL,
           "--host", HOST, 
           "--port", str(PORT), 
           "--endpoint", "/v1/completions",
           # 固定随机数据集
           "--dataset-name", "random", 
           "--random-input-len", str(inl), 
           "--random-output-len", str(outl),
           "--random-range-ratio", "0", 
           # 固定请求数量
           "--num-prompts", str(num_prompts),
           # 开环到达参数
           "--request-rate", str(rate), 
           "--burstiness", "1.0", 
           # 固定输出长度，即使生成 EOS 也继续到指定长度
           "--ignore-eos",
           # 固定 Sampling
           "--temperature", "0", 
           "--top-p", "1", 
           "--seed", str(seed), 
           # 关闭 Bench 内部 Warmup
           "--num-warmups", "0",
           # 请求在线指标，汇总四个指标的 P50/P95/P99
           "--percentile-metrics", "ttft,tpot,itl,e2el", 
           "--metric-percentiles", "50,95,99",
           # 保存
           "--save-result", 
           "--save-detailed", 
           "--result-dir", os.path.join(OUT, "raw"),
           "--result-filename", result_file]
    if goodput:
        cmd += ["--goodput"] + goodput
    meta = {"run_id": run_id, "shape": {"in": inl, "out": outl}, "target_offered_qps": rate,
            "seed": seed, "num_prompts": num_prompts}
    if sampler:
        sampler.phase = "arrival"
    t0 = time.perf_counter()
    log_path = os.path.join(LOGD, "per_run", f"{run_id}.log")
    with open(log_path, "w") as lg:
        try:
            rc = subprocess.call(cmd, env=ENV, stdout=lg, stderr=subprocess.STDOUT, timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = -1; timed_out = True
    meta["arrival_bench_s"] = round(time.perf_counter() - t0, 2)
    meta["exit_code"] = rc; meta["timeout"] = timed_out
    # drain barrier
    if sampler:
        sampler.phase = "drain"
    drained, drain_t = wait_idle(QUEUE["drain_budget_s"] + 10)
    meta["drain_time_s"] = drain_t; meta["drained"] = drained
    if sampler:
        sampler.phase = "idle"
    # 结果文件路径
    meta["result_path"] = os.path.join(OUT, "raw", result_file)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "tail", "fineonly"])
    ap.add_argument("--fine-rates", default="")
    args = ap.parse_args()
    for d in ("raw", "metrics", "metadata", "calibration"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    os.makedirs(os.path.join(LOGD, "per_run"), exist_ok=True)
    server = Server()
    all_meta = []

    def do_run(shape, rate, seed, num_prompts, tag, goodput):
        run_id = f"shape_{shape.lower()}_{tag}_qps{rate}_seed{seed}"
        run_start = time.perf_counter()
        smp = Sampler(os.path.join(OUT, "metrics", f"{run_id}.jsonl"), run_start); smp.start()
        eff_rate = 0.5 if rate == "inf" else float(rate)
        timeout_s = 300 if rate == "inf" else (num_prompts / max(eff_rate, 0.5) + 120)
        try:
            meta = run_bench(shape, rate, seed, num_prompts, run_id, goodput, smp, timeout_s)
        finally:
            smp.stop(); smp.join(timeout=3)
        meta["server_restart_count"] = server.restart_count
        json.dump(meta, open(os.path.join(OUT, "metadata", f"{run_id}.json"), "w"), ensure_ascii=False, indent=2)
        all_meta.append(meta)
        print(f"[m12] {run_id}: exit={meta['exit_code']} timeout={meta['timeout']} drain={meta['drain_time_s']}s")
        return meta

    try:
        ready = server.start(os.path.join(LOGD, "server.log"))
        print(f"[m12] server ready {ready}s (restart={server.restart_count})")
        warmup(10)
        # config.json（含冻结阈值 + goodput 占位；SLO 数值在 calibration 后填）
        cfg = {"server_cmd": SERVER_CMD, "server_config_hash": hashlib.sha256(" ".join(SERVER_CMD).encode()).hexdigest()[:16],
               "shapes": SHAPES, "queue_thresholds": QUEUE, "slo_multipliers": SLO_MULT, "slo_floors": SLO_FLOOR,
               "success_ratio_min": SUCCESS_MIN, "slo_pass_ratio_min": SLO_PASS_MIN,
               "preemption_limit": PREEMPT_LIMIT, "timeout_limit": TIMEOUT_LIMIT, "fine_seeds": FINE_SEEDS,
               "server_ready_time_s": ready}

        # --- Calibration (rate=1) + 冻结 SLO（仅 stage=all；tail 复用已冻结 config.json）---
        if args.stage in ("all",):
            cal_prompts = 20 if args.quick else 100
            m = do_run("A", 1, 0, cal_prompts, "calib", None)
            try:
                cr = json.load(open(m["result_path"]))
                slo = {"ttft_p99_ms": max(SLO_FLOOR["ttft_p99_ms"], round(cr["p99_ttft_ms"] * SLO_MULT["ttft"])),
                       "tpot_p99_ms": max(SLO_FLOOR["tpot_p99_ms"], round(cr["p99_tpot_ms"] * SLO_MULT["tpot"])),
                       "e2el_p99_ms": max(SLO_FLOOR["e2el_p99_ms"], round(cr["p99_e2el_ms"] * SLO_MULT["e2el"])),
                       "itl_p99_guardrail_ms": max(SLO_FLOOR["itl_p99_guardrail_ms"], round(cr["p99_itl_ms"] * SLO_MULT["itl"]))}
                cfg["low_load_p99"] = {k: cr.get(f"p99_{k}_ms") for k in ("ttft", "tpot", "e2el", "itl")}
            except Exception as e:  # noqa
                slo = dict(SLO_FLOOR); slo["_calib_error"] = repr(e); cfg["low_load_p99"] = {}
            cfg["experimental_slo"] = slo
            json.dump(cfg, open(os.path.join(OUT, "config.json"), "w"), ensure_ascii=False, indent=2)
        else:
            # tail：复用跑前已冻结的 SLO（不重新 calibration，保证阈值一致）
            frozen = json.load(open(os.path.join(OUT, "config.json")))
            slo = frozen["experimental_slo"]
        goodput = [f"ttft:{slo['ttft_p99_ms']}", f"tpot:{slo['tpot_p99_ms']}", f"e2el:{slo['e2el_p99_ms']}"]
        print(f"[m12] SLO: {slo}")

        # --- Shape A coarse (wide, single seed) → 记录探索性稳定(achieved≥0.9×rate 且 success≥0.99) ---
        coarse_stab = {}

        def _explor_stable(meta, rate):
            rp = meta.get("result_path", "")
            if not os.path.exists(rp):
                return False
            r = json.load(open(rp))
            succ = (r.get("completed", 0)) / max(meta["num_prompts"], 1)
            return (succ >= 0.99) and (r.get("request_throughput", 0) >= 0.9 * rate) and (not meta.get("timeout"))

        if args.stage in ("all",):
            coarse = [1, 4] if args.quick else COARSE_RATES_A
            for r in coarse:
                npr = 60 if args.quick else min(max(60, math.ceil(r * 30)), 1000)
                m = do_run("A", r, 0, npr, "coarse", goodput)
                coarse_stab[r] = _explor_stable(m, r)
                print(f"[m12] coarse qps{r}: exploratory_stable={coarse_stab[r]}")

        if not args.quick and args.stage in ("all",):
            # --- 动态生成 Knee 精扫档（计划算法）: last_stable_like → first_overload 之间细扫 ---
            stable_like = [r for r in COARSE_RATES_A if coarse_stab.get(r)]
            unstable = [r for r in COARSE_RATES_A if not coarse_stab.get(r)]
            last_stable = max(stable_like) if stable_like else COARSE_RATES_A[0]
            first_over = min([r for r in unstable if r > last_stable], default=None)
            if first_over is None:
                fine_rates = [last_stable]  # 未见过载：容量≥最高档，如实标注
            else:
                step = 1.0 if (first_over - last_stable) > 6 else 0.5
                fine_rates = _frange(last_stable, first_over, step)
            print(f"[m12] knee: last_stable={last_stable} first_over={first_over} fine_rates={fine_rates}")
            json.dump({"coarse_exploratory_stable": coarse_stab, "last_stable": last_stable,
                       "first_over": first_over, "fine_rates": fine_rates},
                      open(os.path.join(OUT, "knee.json"), "w"), ensure_ascii=False, indent=2)
            for seed in FINE_SEEDS:
                server.restart(os.path.join(LOGD, "server.log")); warmup(10)
                for r in fine_rates:
                    do_run("A", r, seed, max(300, math.ceil(r * 60)), "fine", goodput)

        # --- fineonly：手动指定 fine 档细扫 knee（复用已冻结 SLO）---
        if args.stage == "fineonly":
            rates = [float(x) for x in args.fine_rates.split(",") if x.strip()]
            print(f"[m12] fineonly rates={rates}")
            for seed in FINE_SEEDS:
                server.restart(os.path.join(LOGD, "server.log")); warmup(10)
                for r in rates:
                    do_run("A", r, seed, max(300, math.ceil(r * 60)), "fine", goodput)

        if not args.quick and args.stage in ("all", "tail"):
            # --- Burst ---
            server.restart(os.path.join(LOGD, "server.log")); warmup(10)
            do_run("A", "inf", 0, 300, "burst", goodput)
            # --- Shape B coarse ---
            server.restart(os.path.join(LOGD, "server.log")); warmup(10)
            for r in COARSE_RATES_B:
                do_run("B", r, 0, min(max(60, math.ceil(r * 30)), 1000), "coarse", goodput)
    finally:
        server.stop()
        print(f"[m12] server stopped; port_free={server._port_free()}; runs={len(all_meta)}")


if __name__ == "__main__":
    main()
