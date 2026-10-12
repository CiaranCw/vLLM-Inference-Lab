#!/usr/bin/env python3
"""M28-B · Serving 可靠性演练：B1 EngineCore 子进程 SIGKILL / B3 SIGTERM × shutdown_timeout / B2 SIGSTOP。

与 M21 E6 的区别：M21 杀的是 API 父进程且无在途请求；B1 杀的是 **EngineCore 子进程且有在途请求**。

安全约束（notes/m28/M28_PLAN.md §6，逐条落在代码里）：
  1. 只对本脚本 Popen（新会话）出来的进程发信号；发信号前再次核对 sid / create_time / 角色 / ppid；
  2. 目标 EngineCore 必须恰好 1 个，否则不注入，记 BLOCKED；
  3. 禁止 pkill/killall/按名字匹配；禁止 nvidia-smi --reset；
  4. SIGSTOP 必有 SIGCONT 兜底（finally + atexit + SIGINT/SIGTERM handler）；
  5. 每次演练前后：nvidia-smi / 进程树 / 端口 / /dev/shm 快照；结束后确认无存活子进程、端口释放、显存回收；
  6. 人工停止：kill -INT <runner pid>（见锁文件）→ runner 自己收尾；runner 已死则
     `python scripts/m28_faults.py --cleanup-orphans [--really]` 只按落盘记录清理。

用法：
  python scripts/m28_faults.py --dry-run
  python scripts/m28_faults.py --mock --out-dir /tmp/x --drill b1 --drill b3 --drill b2 --authorize-b2
  python scripts/m28_faults.py --drill b1 --drill b3 --i-have-gpu-authorization --i-have-fault-authorization
"""

import argparse
import atexit
import json
import os
import signal
import sys
import tempfile
import threading
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m28_common as C  # noqa: E402

B = C.FROZEN["B"]

LOG_MARKERS = {
    "engine_exited_unexpectedly": "engine core exited unexpectedly",
    "engine_dead_error": "EngineDeadError",
    "api_shutdown_triggered": "[shutdown] API server: shutdown triggered",
    "api_stopping_engine": "[shutdown] API server: stopping engine client",
    "engine_shutdown_start": "[shutdown] EngineCore: start mode=",
    "engine_aborting": "[shutdown] EngineCore: aborting in-flight requests",
    "engine_draining": "[shutdown] EngineCore: draining in-flight requests",
    "engine_teardown": "[shutdown] EngineCore: request processing complete",
    "force_killing": "[shutdown] Process manager: force killing remaining processes",
    "engine_signal_received": "[shutdown] EngineCore: trigger received signal=",
    "traceback": "Traceback (most recent call last)",
}

_ACTIVE = []  # 供 atexit / signal handler 兜底的 Server 列表


def _emergency(*_):
    for srv in list(_ACTIVE):
        try:
            srv.resume_stopped()
        except Exception as e:
            print(f"[m28-B] 兜底 SIGCONT 失败: {type(e).__name__}: {e}", file=sys.stderr, flush=True)


def _on_signal(signum, frame):
    _emergency()
    raise SystemExit(f"runner 收到信号 {signal.Signals(signum).name}，进入收尾")


def scan_log(path):
    out, first = {k: 0 for k in LOG_MARKERS}, {}
    try:
        with open(path, errors="replace") as f:
            for line in f:
                for k, pat in LOG_MARKERS.items():
                    if pat in line:
                        out[k] += 1
                        first.setdefault(k, line.strip()[:300])
    except OSError as e:
        out["error"] = str(e)
    out["first_lines"] = first
    return out


def summarize_req(r):
    keep = ("stream", "http_status", "finish_reason", "error_event", "exception", "t_send",
            "t_first_token", "t_end", "done_marker", "sse_tail", "n_sse_events", "max_tokens",
            "online_ttft_ms", "e2e_ms", "raw_body")
    d = {k: r.get(k) for k in keep}
    d["n_tokens"] = len(r.get("token_ids") or [])
    return d


class Inflight:
    def __init__(self, port, key, max_tokens):
        self.port, self.max_tokens = port, max_tokens
        self.specs = [(True, k) for k in range(B["n_stream"])] + \
                     [(False, k) for k in range(B["n_nonstream"])]
        self.prompts = [C.trial_prompt(B["prompt_len"], 0, suffix_len=0, salt=f"b|{key}|{s}|{k}")
                        for s, k in self.specs]
        self.results = [None] * len(self.specs)
        self.first_flags = [False] * len(self.specs)
        self.first = threading.Semaphore(0)
        self.threads = []

    def _run(self, i):
        stream = self.specs[i][0]

        def _cb():
            self.first_flags[i] = True
            self.first.release()

        self.results[i] = C.completion(self.port, self.prompts[i], self.max_tokens, stream=stream,
                                       timeout=600, on_first_token=_cb if stream else None)
        if stream and not self.first_flags[i]:
            self.first.release()  # 未拿到首 token 也要唤醒等待方；first_flags 如实为 False

    def start(self):
        for i in range(len(self.specs)):
            t = threading.Thread(target=self._run, args=(i,), daemon=True)
            t.start()
            self.threads.append(t)

    def wait_inflight(self, timeout=120):
        """证据 1：所有 streaming 请求都拿到首 token；证据 2：running 指标 == 在途总数。"""
        t0 = time.perf_counter()
        for _ in range(B["n_stream"]):
            self.first.acquire(timeout=max(0.1, timeout - (time.perf_counter() - t0)))
        got = sum(1 for i, (s, _) in enumerate(self.specs) if s and self.first_flags[i])
        running, t1 = None, time.perf_counter()
        while time.perf_counter() - t1 < 15:
            try:
                running = C.metric_snapshot(self.port).get("running")
            except Exception as e:
                running = f"<error {type(e).__name__}: {e}>"
            if running == len(self.specs):
                break
            time.sleep(0.2)
        unfinished = sum(1 for r in self.results if r is None)
        return {"stream_first_token_events": got, "running_gauge": running, "n_total": len(self.specs), "unfinished_requests": unfinished,
                "ok": got == B["n_stream"] and running == len(self.specs) and unfinished == len(self.specs)}

    def join(self, timeout):
        t0 = time.perf_counter()
        for t in self.threads:
            t.join(timeout=max(0.1, timeout - (time.perf_counter() - t0)))
        return [summarize_req(r) if r else {"missing": True} for r in self.results]


class Observer(threading.Thread):
    def __init__(self, srv, engine_pid, interval, max_s):
        super().__init__(daemon=True)
        self.srv, self.engine_pid, self.interval, self.max_s = srv, engine_pid, interval, max_s
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        t0 = time.perf_counter()
        quiet_since = None
        while not self._halt.is_set() and time.perf_counter() - t0 < self.max_s:
            code, exc = C.health_code(self.srv.port, timeout=1.0)
            s = {"ts": C.now(), "health": code, "health_exc": exc, "api_alive": self.srv.api_alive(),
                 "engine_alive": C.pid_alive(self.engine_pid), "port_listening": C.port_listening(self.srv.port)}
            self.samples.append(s)
            dead = not (s["api_alive"] or s["engine_alive"] or s["port_listening"])
            quiet_since = (quiet_since or time.perf_counter()) if dead else None
            if quiet_since and time.perf_counter() - quiet_since > 3:
                break
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=10)
        return self.samples


class Prober(threading.Thread):
    """在故障后发新请求，记录拒绝形态（状态码 / 错误体 / 连接异常）。"""

    def __init__(self, port, t0, offsets=None, interval=None, max_s=60, key="probe"):
        super().__init__(daemon=True)
        self.port, self.t0, self.offsets, self.interval, self.max_s = port, t0, offsets, interval, max_s
        self.prompt = C.trial_prompt(32, 0, suffix_len=0, salt=f"probe|{key}")
        self.results = []
        self._halt = threading.Event()

    def _one(self):
        r = C.completion(self.port, self.prompt, 8, stream=False, timeout=30)
        d = summarize_req(r)
        d["offset_s"] = round(r["t_send"] - self.t0, 3) if r.get("t_send") else None
        self.results.append(d)
        return d

    def run(self):
        if self.offsets is not None:
            for off in self.offsets:
                delay = self.t0 + off - C.now()
                if delay > 0 and self._halt.wait(delay):
                    return
                self._one()
            return
        refused = 0
        while not self._halt.is_set() and C.now() - self.t0 < self.max_s and len(self.results) < 400:
            d = self._one()
            refused = refused + 1 if (d.get("exception") or "").startswith("ConnectionRefusedError") else 0
            if refused >= 3:
                break
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=40)
        return self.results


class Ctx:
    def __init__(self, args, out_dir, log_dir):
        self.args, self.out_dir, self.log_dir = args, out_dir, log_dir
        self.writer = C.JsonlWriter(os.path.join(out_dir, "m28_b_drills.jsonl"),
                                    backup_existing=args.backup_existing, append=args.append)
        self.life = C.JsonlWriter(os.path.join(out_dir, "m28_lifecycle.jsonl"), append=True)

    def server(self, label, shutdown_timeout=None):
        if self.args.mock:
            cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "m28_mock_server.py"),
                   "--port", str(self.args.port), "--arm", "fault"]
            if shutdown_timeout is not None:
                cmd += ["--shutdown-timeout", str(shutdown_timeout)]
        else:
            cmd = C.vllm_serve_cmd(C.b_server_args(shutdown_timeout), port=self.args.port)
        srv = C.Server(cmd, C.server_env(), os.path.join(self.log_dir, f"m28_{label}_server.log"),
                       self.args.port, label, active_file=os.path.join(self.out_dir, "m28_active_server.json"))
        _ACTIVE.append(srv)
        return srv


def pre_snapshot(port):
    return {"gpu": C.nvidia_smi(), "dev_shm": C.dev_shm_listing(), "port_free": C.port_free(port),
            "host_mem": C.host_mem()}


def post_checks(ctx, srv, rec, base_mib):
    rec["cleanup_actions"] = srv.cleanup()
    rec["cleanup_needed"] = any(a.get("sent") for a in rec["cleanup_actions"])
    rec["api_returncode"] = srv.returncode()
    rec["alive_known_after_cleanup"] = srv.alive_known()
    rec["port_free_after"] = C.port_free(ctx.args.port)
    if ctx.args.mock:
        rec["gpu_reclaim"] = {"skipped": "mock"}
    else:
        ok, secs, last = C.wait_gpu_reclaim(base_mib)
        rec["gpu_reclaim"] = {"ok": ok, "waited_s": secs, "mem_used_mib": last, "baseline_mib": base_mib}
    rec["dev_shm_after"] = C.dev_shm_listing()
    rec["log_markers"] = scan_log(srv.log_path)
    if srv in _ACTIVE:
        _ACTIVE.remove(srv)


def ready_server(ctx, srv, rec):
    rec["ready_s"] = srv.start(ready_timeout=B["server_ready_timeout_s"])
    rec["api_pid"] = srv.pid
    rec["tree_ready"] = srv.refresh_tree()
    cores = srv.engine_cores()
    rec["engine_cores_ready"] = cores
    rec["gpu_ready"] = C.nvidia_smi()
    rec["cmd"] = srv.cmd
    rec["log_path"] = os.path.relpath(srv.log_path, C.ROOT)
    if len(cores) != 1:
        raise RuntimeError(f"BLOCKED: 期望恰好 1 个 {C.ENGINE_CORE_TITLE} 子进程，实际 {len(cores)}：{cores}")
    # 预热（含 JIT），不计入
    w = Inflight(ctx.args.port, f"{rec['drill']}|{rec['rep']}|warm", 16)
    w.start()
    rec["warmup"] = [r.get("finish_reason") for r in w.join(300)]
    return cores[0]["pid"]


def drill(ctx, kind, rep, shutdown_timeout=None):
    label = f"{kind}_t{shutdown_timeout}_r{rep}" if kind == "b3" else f"{kind}_r{rep}"
    cfg = B[kind.upper()]
    rec = {"module": "m28", "exp": "B", "drill": kind, "rep": rep, "label": label, "mock": ctx.args.mock,
           "plan_version": C.FROZEN["plan_version"], "shutdown_timeout": shutdown_timeout,
           "t_begin": C.now(), "pre": pre_snapshot(ctx.args.port),
           "env_fingerprint": C.env_fingerprint(with_vllm_import=False)}
    base_mib = C.gpu_used_mib(rec["pre"]["gpu"])
    srv = ctx.server(label, shutdown_timeout)
    gpu = C.GpuPoller(B["gpu_poll_s"]) if not ctx.args.mock else None
    try:
        engine_pid = ready_server(ctx, srv, rec)
        rec["engine_pid"] = engine_pid
        if kind == "b3":
            cal = Inflight(ctx.args.port, f"{label}|cal", cfg["max_tokens"])
            cal.start()
            res = cal.join(600)
            ts = [r["t_send"] for r in res if r.get("t_send")]
            te = [r["t_end"] for r in res if r.get("t_end")]
            rec["calibration"] = {"duration_s": (max(te) - min(ts)) if ts and te else None,
                                  "n_ok": sum(1 for r in res if r.get("finish_reason") == "length"),
                                  "requests": res}
        if gpu:
            gpu.start()
        load = Inflight(ctx.args.port, f"{label}|load", cfg["max_tokens"])
        load.start()
        rec["inflight_evidence"] = load.wait_inflight()
        if not rec["inflight_evidence"]["ok"]:
            raise RuntimeError(f"BLOCKED: 在途证据不足，不注入：{rec['inflight_evidence']}")
        time.sleep(cfg["fault_after_first_token_s"])
        if kind == "b1":
            sig, target, role = signal.SIGKILL, engine_pid, "engine_core"
        elif kind == "b3":
            sig, target, role = signal.SIGTERM, srv.pid, "api"
        else:
            sig, target, role = signal.SIGSTOP, engine_pid, "engine_core"
        obs = Observer(srv, engine_pid, B["health_poll_s"], cfg["observe_s"])
        obs.start()
        rec["fault"] = srv.signal(target, sig, role)
        t_fault = rec["fault"]["ts"]
        if not rec["fault"]["sent"]:
            raise RuntimeError(f"BLOCKED: 目标身份核对未通过，未发信号：{rec['fault']}")
        if kind == "b1":
            prober = Prober(ctx.args.port, t_fault, offsets=cfg["probe_new_request_at_s"], key=label)
        elif kind == "b3":
            prober = Prober(ctx.args.port, t_fault, interval=cfg["probe_interval_s"], max_s=cfg["observe_s"],
                            key=label)
        else:
            prober = Prober(ctx.args.port, t_fault, offsets=[1.0, cfg["stop_s"] - 2.0], key=label)
        prober.start()
        if kind == "b2":
            time.sleep(cfg["stop_s"])
            rec["resume"] = srv.signal(engine_pid, signal.SIGCONT, "engine_core")
            if not rec["resume"]["sent"]:
                rec["resume_fallback"] = srv.resume_stopped()
        rec["requests"] = load.join(cfg["observe_s"] + 60)
        prober.join(timeout=cfg["observe_s"] + 40)
        rec["probes"] = prober.stop()
        obs.join(timeout=cfg["observe_s"] + 5)
        rec["timeline"] = obs.stop()
        rec["t_fault"] = t_fault
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    finally:
        _emergency()
        if gpu:
            rec["gpu_samples"] = gpu.stop()
        rec["api_returncode_before_cleanup"] = srv.returncode()
        rec["alive_known_before_cleanup"] = srv.alive_known()
        post_checks(ctx, srv, rec, base_mib)
    if kind == "b1" and "error" not in rec:
        rec["restart"] = restart_check(ctx, label, base_mib)
    rec["t_end"] = C.now()
    ctx.writer.write(rec)
    ctx.life.write({"module": "m28", "exp": "B", "phase": "lifecycle", "label": label, "mock": ctx.args.mock,
                    "t_start": rec["t_begin"], "t_end": rec["t_end"], "api_pid": rec.get("api_pid"),
                    "error": rec.get("error")})
    status = "ERROR " + rec["error"] if "error" in rec else "ok"
    print(f"[m28-B] {label}: {status}", flush=True)
    return rec


def restart_check(ctx, label, base_mib):
    srv = ctx.server(label + "_restart")
    out = {}
    try:
        out["ready_s"] = srv.start(ready_timeout=B["server_ready_timeout_s"])
        out["tree"] = srv.refresh_tree()
        r = C.completion(ctx.args.port, C.trial_prompt(64, 0, suffix_len=0, salt="restart"), 8, stream=False)
        out["request"] = summarize_req(r)
        out["ok"] = r.get("finish_reason") == "length" and r.get("http_status") == 200
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["ok"] = False
    finally:
        post_checks(ctx, srv, out, base_mib)
    return out


def dry_run():
    print("M28-B 冻结矩阵（plan_version=%s）" % C.FROZEN["plan_version"])
    print("B1 cmd:", " ".join(C.vllm_serve_cmd(C.b_server_args())))
    for t in B["B3"]["shutdown_timeouts"]:
        print(f"B3 t={t} cmd:", " ".join(C.vllm_serve_cmd(C.b_server_args(t))))
    print(json.dumps(B, ensure_ascii=False, indent=2))
    kv = B["num_gpu_blocks_override"] * 16 * C.FROZEN["kv_bytes_per_token_theory"]
    print(f"GPU KV 池 {kv / 2**20:.0f} MiB；在途 {B['n_stream']}+{B['n_nonstream']} 请求 × "
          f"(prompt {B['prompt_len']} + 输出 ≤{B['B1']['max_tokens']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--drill", action="append", choices=["b1", "b2", "b3"])
    ap.add_argument("--repeats", type=int, default=None, help="覆盖冻结 repeats（仅 mock 演练用）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--port", type=int, default=C.FROZEN["port"])
    ap.add_argument("--backup-existing", action="store_true")
    ap.add_argument("--append", action="store_true", help="向已有 m28_b_drills.jsonl 追加（分批执行 B1/B3 时）")
    ap.add_argument("--i-have-gpu-authorization", action="store_true")
    ap.add_argument("--i-have-fault-authorization", action="store_true", help="所有者已授权 B1/B3 故障注入")
    ap.add_argument("--authorize-b2", action="store_true", help="所有者已单独授权 B2 SIGSTOP")
    ap.add_argument("--cleanup-orphans", action="store_true", help="人工停止路径：只按落盘记录清理")
    ap.add_argument("--really", action="store_true", help="与 --cleanup-orphans 合用才真正发信号")
    a = ap.parse_args()
    if a.dry_run:
        dry_run()
        return 0
    if a.cleanup_orphans:
        out_dir = a.out_dir or C.RAW
        print(json.dumps(C.cleanup_from_record(os.path.join(out_dir, "m28_active_server.json"),
                                               dry_run=not a.really), ensure_ascii=False, indent=2))
        return 0
    drills = a.drill or ["b1", "b3"]
    if "b2" in drills and not a.authorize_b2:
        print("[m28-B] B2 需要单独授权（--authorize-b2），本轮跳过 B2")
        drills = [d for d in drills if d != "b2"]
    if a.mock:
        out_dir = a.out_dir or tempfile.mkdtemp(prefix="m28_mock_")
        if os.path.realpath(out_dir).startswith(os.path.realpath(C.OUT)):
            raise SystemExit("mock 产物不得写入 outputs/m28")
        log_dir = os.path.join(out_dir, "logs")
    else:
        out_dir, log_dir = a.out_dir or C.RAW, C.LOGD
        if not a.i_have_fault_authorization:
            print("[m28-B] 缺少 --i-have-fault-authorization（B1/B3 需所有者一次明确授权），不启动")
            return 2
        ok, reasons, snap = C.gpu_gate(a.i_have_gpu_authorization, a.port)
        print("[m28-B] nvidia-smi:", json.dumps(snap, ensure_ascii=False))
        if not ok:
            print("[m28-B] GPU 门禁未通过，不启动：", *reasons, sep="\n  - ")
            return 2
    C.acquire_lock(f"m28_faults {' '.join(drills)}", raw_dir=out_dir)
    atexit.register(_emergency)
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    ctx = Ctx(a, out_dir, log_dir)
    n_err = 0
    for d in drills:
        cfg = B[d.upper()]
        reps = a.repeats if (a.repeats and a.mock) else cfg["repeats"]
        timeouts = cfg["shutdown_timeouts"] if d == "b3" else [None]
        for t in timeouts:
            for rep in range(reps):
                rec = drill(ctx, d, rep, t)
                n_err += "error" in rec
                if rec.get("gpu_reclaim", {}).get("ok") is False or rec.get("alive_known_after_cleanup"):
                    print("[m28-B] 资源未回收/仍有存活进程，按安全约束中止后续演练", flush=True)
                    return 3
    print("[m28-B] 产物目录:", out_dir, "错误演练数:", n_err)
    return 1 if n_err else 0


if __name__ == "__main__":
    sys.exit(main())
