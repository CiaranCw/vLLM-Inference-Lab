#!/usr/bin/env python3
"""M27 · 拓扑启动器：按 arm 拉起 colocated 副本或 xPyD 实例 + 代理，回读生效配置。

    python scripts/m27_launch.py --tag h20_smoke --exp gate1 --arm pd_1p1d --hold   # 手动排障用

库用法（run_all / workload 调用）：
    with Topology(tag, exp, arm, gpu_ids, model_dir) as topo:
        topo.proxy_url ...

生效配置两源核对（AGENTS.md「布尔判据要有独立的第二证据源」）：
  源 1：GET /server_info?config_format=json（需 VLLM_SERVER_DEV_MODE=1，只绑 127.0.0.1）
  源 2：启动日志 "Initializing a V1 LLM engine ... with config:" / "Setting KV cache layout to" /
        "Using <X> backend" / "GPU KV cache size"
任一关键键缺失或与请求值不一致 → 停掉全部进程并抛错（不做隐式回退）。

进程管理：每个实例 start_new_session=True，停止时只对**自己 Popen 出来的进程组**发
SIGTERM→(60s)→SIGKILL（M21 教训：只 kill API server 不会杀掉持显存的 EngineCore 子进程）。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402

HEALTH_TIMEOUT_S = 1200
MISSING = "__MISSING__"

# (effective key, server_info path, expected-from-engine-key)
READBACK_KEYS = [
    ("max_model_len", ("model_config", "max_model_len"), "max_model_len"),
    ("dtype", ("model_config", "dtype"), "dtype"),
    ("seed", ("model_config", "seed"), "seed"),
    ("enforce_eager", ("model_config", "enforce_eager"), "enforce_eager"),
    ("quantization", ("model_config", "quantization"), "quantization"),
    ("tensor_parallel_size", ("parallel_config", "tensor_parallel_size"), "tensor_parallel_size"),
    ("gpu_memory_utilization", ("cache_config", "gpu_memory_utilization"), "gpu_memory_utilization"),
    ("block_size", ("cache_config", "block_size"), "block_size"),
    ("enable_prefix_caching", ("cache_config", "enable_prefix_caching"), "enable_prefix_caching"),
    ("kv_cache_dtype", ("cache_config", "cache_dtype"), "kv_cache_dtype"),
    ("max_num_seqs", ("scheduler_config", "max_num_seqs"), "max_num_seqs"),
    ("max_num_batched_tokens", ("scheduler_config", "max_num_batched_tokens"), None),
    ("enable_chunked_prefill", ("scheduler_config", "enable_chunked_prefill"), "enable_chunked_prefill"),
    ("async_scheduling", ("scheduler_config", "async_scheduling"), "async_scheduling"),
    ("stream_interval", ("scheduler_config", "stream_interval"), "stream_interval"),
    ("attention_backend", ("attention_config", "backend"), "attention_backend"),
    ("speculative_config", ("speculative_config",), "speculative_config"),
]

LOG_PATTERNS = {
    "engine_config": re.compile(r"Initializing a V1 LLM engine \(v([^)]+)\) with config: (.*)"),
    "kv_layout": re.compile(r"[Ss]etting KV cache layout to (HND|NHD)"),
    "attn_backend": re.compile(r"Using (?:AttentionBackendEnum\.)?(\w+) backend\."),
    "kv_cache_tokens": re.compile(r"GPU KV cache size: ([\d,]+) tokens"),
    "nixl_compat_pass": re.compile(r"NIXL compatibility check passed \(hash: (\w+)\)"),
    "nixl_compat_mismatch": re.compile(r"NIXL compatibility hash mismatch"),
    "kv_load_fail": re.compile(r"Failing (\d+) request\(s\) due to KV load failure"),
    "kv_load_recovered": re.compile(r"Recovered from KV load failure"),
    "traceback": re.compile(r"Traceback \(most recent call last\)"),
    "nixl_available": re.compile(r"NIXL is available"),
    "nixl_not_available": re.compile(r"NIXL is not available"),
}
ATTN_CLASS_TO_NAME = {"FlashAttentionBackend": "FLASH_ATTN"}
ENGINE_CFG_FIELDS = ("max_seq_len", "enforce_eager", "enable_prefix_caching",
                     "enable_chunked_prefill", "dtype", "seed", "tensor_parallel_size",
                     "kv_cache_dtype", "quantization")


def _get(d, path):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return MISSING
        cur = cur[k]
    return cur


def _norm(key, v):
    if v is MISSING:
        return v
    if key == "dtype":
        return "bfloat16" if "bfloat16" in str(v) else str(v)
    if key == "attention_backend" and isinstance(v, str):
        # server_info dumps the enum *value* (class path); logs print AttentionBackendEnum.NAME
        return ATTN_CLASS_TO_NAME.get(v.split(".")[-1], v.split(".")[-1])
    return v


def parse_log_evidence(log_path: str) -> dict:
    ev = {k: [] for k in LOG_PATTERNS}
    if not os.path.exists(log_path):
        return {"log_missing": True}
    with open(log_path, errors="replace") as f:
        for line in f:
            for k, pat in LOG_PATTERNS.items():
                m = pat.search(line)
                if m:
                    ev[k].append(m.groups() if m.groups() else True)
    out = {"log_missing": False}
    if ev["engine_config"]:
        ver, cfg = ev["engine_config"][-1]
        out["engine_version_log"] = ver
        fields = {}
        for f in ENGINE_CFG_FIELDS:
            m = re.search(rf"\b{f}=([^,]+)", cfg)
            fields[f] = m.group(1).strip() if m else MISSING
        out["engine_config_log"] = fields
    out["kv_layout_log"] = sorted({g[0] for g in ev["kv_layout"]})
    out["attn_backend_log"] = sorted({g[0] for g in ev["attn_backend"]})
    out["kv_cache_tokens_log"] = [int(g[0].replace(",", "")) for g in ev["kv_cache_tokens"]]
    out["nixl_compat_pass"] = sorted({g[0] for g in ev["nixl_compat_pass"]})
    out["nixl_compat_mismatch"] = len(ev["nixl_compat_mismatch"])
    out["kv_load_fail_events"] = sum(int(g[0]) for g in ev["kv_load_fail"])
    out["kv_load_recovered_events"] = len(ev["kv_load_recovered"])
    out["tracebacks"] = len(ev["traceback"])
    out["nixl_available_log"] = bool(ev["nixl_available"]) and not ev["nixl_not_available"]
    return out


def http_json(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def http_text(url, timeout=30):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def health_ok(port) -> bool:
    try:
        with urllib.request.urlopen(f"http://{C.HOST}:{port}/health", timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


class Topology:
    def __init__(self, tag, exp, arm, gpu_ids, model_dir, allow_fault=False):
        C.formal_guard(tag, C.parse_arm(arm)["n_gpus"])
        self.tag, self.exp, self.arm = tag, exp, arm
        self.model_dir = model_dir
        self.allow_fault = allow_fault
        self.engine = C.effective_engine(tag)
        self.instances = C.build_instances(arm, gpu_ids)
        self.kind = C.parse_arm(arm)["kind"]
        self.procs: dict[str, subprocess.Popen] = {}
        self.logs: dict[str, str] = {}
        self.proxy = None
        self.log_dir = os.path.join(C.LOGS, tag)
        self.prefix = f"{tag}_{exp}_{arm}"
        self.proxy_log = os.path.join(C.RAW, f"{self.prefix}_proxy.jsonl")
        self.lifecycle = os.path.join(C.RAW, f"{tag}_lifecycle.jsonl")
        self.proxy_url = f"http://{C.HOST}:{C.PROXY_PORT}"

    # ------------------------------------------------------------------ helpers
    def ports(self, role=None):
        return [i["http_port"] for i in self.instances if role is None or i["role"] == role]

    def metrics_rows(self, inst_name):
        inst = next(i for i in self.instances if i["name"] == inst_name)
        return C.parse_prom(http_text(f"http://{C.HOST}:{inst['http_port']}/metrics", timeout=10))

    def snapshot(self) -> dict:
        return {i["name"]: C.metrics_snapshot(self.metrics_rows(i["name"])) for i in self.instances}

    def all_idle(self) -> bool:
        snap = self.snapshot()
        return all((s.get("num_requests_running") or 0) == 0 and
                   (s.get("num_requests_waiting") or 0) == 0 for s in snap.values())

    def wait_idle(self, timeout=120, settle=0.5) -> bool:
        t0 = time.time()
        ok = 0
        while time.time() - t0 < timeout:
            try:
                ok = ok + 1 if self.all_idle() else 0
            except Exception:
                ok = 0
            if ok >= 2:
                time.sleep(settle)
                return True
            time.sleep(0.25)
        return False

    # ------------------------------------------------------------------ lifecycle
    def _check_free(self):
        ports = [C.PROXY_PORT] + [i["http_port"] for i in self.instances] + \
                [i["side_channel_port"] for i in self.instances]
        busy = [p for p in ports if not C.port_free(p)]
        if busy:
            raise RuntimeError(f"端口被占用（不 kill 别人的进程）：{busy}")
        apps = [a for a in C.gpu_compute_apps() if not a.startswith("error")]
        if apps:
            raise RuntimeError(f"GPU 上已有计算进程，停下问所有者：{apps}")

    def start(self):
        self._check_free()
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(C.CONFIGS, exist_ok=True)
        cmds = {}
        for inst in self.instances:
            cmd = C.serve_cmd(self.model_dir, inst, self.engine)
            env = C.instance_env(inst)
            log = os.path.join(self.log_dir, f"{self.exp}_{self.arm}_{inst['name']}.log")
            self.logs[inst["name"]] = log
            lf = open(log, "w")
            lf.write(f"# cmd: {json.dumps(cmd)}\n# gpu={inst['gpu']} role={inst['role']}\n")
            lf.flush()
            self.procs[inst["name"]] = subprocess.Popen(
                cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True,
                cwd=C.ROOT)
            cmds[inst["name"]] = cmd
        C.append_jsonl(self.lifecycle, {"event": "start", "tag": self.tag, "exp": self.exp,
                                        "arm": self.arm, "instances": self.instances,
                                        "pids": {k: p.pid for k, p in self.procs.items()}})
        t0 = time.time()
        pending = set(self.procs)
        while pending and time.time() - t0 < HEALTH_TIMEOUT_S:
            for name in list(pending):
                proc = self.procs[name]
                if proc.poll() is not None:
                    self.stop(reason=f"{name} exited early rc={proc.returncode}")
                    raise RuntimeError(f"{name} 提前退出 rc={proc.returncode}；见 {self.logs[name]}")
                port = next(i["http_port"] for i in self.instances if i["name"] == name)
                if health_ok(port):
                    pending.discard(name)
            time.sleep(2)
        if pending:
            self.stop(reason=f"health timeout {sorted(pending)}")
            raise RuntimeError(f"health 超时 {HEALTH_TIMEOUT_S}s：{sorted(pending)}")
        startup_s = round(time.time() - t0, 1)
        eff = self.readback(cmds, startup_s)
        if not eff["effective_matches_requested"]:
            self.stop(reason="effective config mismatch")
            raise RuntimeError(f"生效配置与请求不符：{eff['mismatches'][:6]}；见 "
                               f"{os.path.relpath(eff['_path'], C.ROOT)}")
        self._start_proxy()
        return self

    def readback(self, cmds, startup_s) -> dict:
        per_inst, mismatches = {}, []
        for inst in self.instances:
            name = inst["name"]
            si = http_json(f"http://{C.HOST}:{inst['http_port']}/server_info?config_format=json", 60)
            C.save_json(os.path.join(C.CONFIGS, f"{self.prefix}_{name}_server_info.json"), si)
            prom = http_text(f"http://{C.HOST}:{inst['http_port']}/metrics", 30)
            os.makedirs(C.METRICS, exist_ok=True)
            with open(os.path.join(C.METRICS, f"{self.prefix}_{name}_startup.prom"), "w") as f:
                f.write(prom)
            vc = si.get("vllm_config") if isinstance(si.get("vllm_config"), dict) else {}
            eff = {k: _norm(k, _get(vc, path)) for k, path, _ in READBACK_KEYS}
            eff["shutdown_timeout"] = vc.get("shutdown_timeout", MISSING)
            ktc = vc.get("kv_transfer_config") if isinstance(vc, dict) else None
            eff["kv_connector"] = (ktc or {}).get("kv_connector")
            eff["kv_role"] = (ktc or {}).get("kv_role")
            eff["kv_load_failure_policy"] = (ktc or {}).get("kv_load_failure_policy")
            eff["kv_engine_id"] = (ktc or {}).get("engine_id")
            eff["vllm_env_kv_layout"] = (si.get("vllm_env") or {}).get("VLLM_KV_CACHE_LAYOUT", MISSING)
            logev = parse_log_evidence(self.logs[name])
            req = {k: (inst["mbt"] if k == "max_num_batched_tokens" else self.engine[ek])
                   for k, _, ek in READBACK_KEYS}
            req = {k: _norm(k, v) for k, v in req.items()}
            want_ktc = C.kv_transfer_config(inst["role"])
            req["kv_connector"] = (want_ktc or {}).get("kv_connector")
            req["kv_role"] = (want_ktc or {}).get("kv_role")
            req["kv_load_failure_policy"] = (want_ktc or {}).get("kv_load_failure_policy")
            req["shutdown_timeout"] = C.SHUTDOWN_TIMEOUT_S
            req["vllm_env_kv_layout"] = C.FROZEN["kv_cache_layout"]
            for k, want in req.items():
                if eff.get(k) != want:
                    mismatches.append({"instance": name, "key": k, "requested": want,
                                       "effective": eff.get(k), "source": "server_info"})
            ecl = logev.get("engine_config_log") or {}
            log_checks = {
                "max_seq_len": str(req["max_model_len"]),
                "enforce_eager": str(req["enforce_eager"]),
                "enable_prefix_caching": str(req["enable_prefix_caching"]),
                "enable_chunked_prefill": str(req["enable_chunked_prefill"]),
                "tensor_parallel_size": str(req["tensor_parallel_size"]),
            }
            for k, want in log_checks.items():
                if ecl.get(k) != want:
                    mismatches.append({"instance": name, "key": k, "requested": want,
                                       "effective": ecl.get(k, MISSING), "source": "log"})
            if logev.get("kv_layout_log") != [C.FROZEN["kv_cache_layout"]]:
                mismatches.append({"instance": name, "key": "kv_layout", "requested": "HND",
                                   "effective": logev.get("kv_layout_log"), "source": "log"})
            if inst["role"] in ("P", "D") and not logev.get("nixl_available_log"):
                mismatches.append({"instance": name, "key": "nixl_available", "requested": True,
                                   "effective": False, "source": "log"})
            if self.engine["attention_backend"] not in (logev.get("attn_backend_log") or []):
                mismatches.append({"instance": name, "key": "attn_backend",
                                   "requested": self.engine["attention_backend"],
                                   "effective": logev.get("attn_backend_log"), "source": "log"})
            per_inst[name] = {"instance": inst, "requested": req, "effective": eff,
                              "log_evidence": logev, "cmd": cmds[name]}
        out = {
            "tag": self.tag, "exp": self.exp, "arm": self.arm, "kind": self.kind,
            "startup_s": startup_s, "instances": per_inst,
            "effective_matches_requested": not mismatches, "mismatches": mismatches,
            "env_fingerprint": C.env_fingerprint(), "frozen_plan_version": C.FROZEN["plan_version"],
            "model_dir": self.model_dir,
        }
        path = os.path.join(C.CONFIGS, f"{self.prefix}_effective.json")
        C.save_json(path, out)
        out["_path"] = path
        return out

    def _start_proxy(self):
        cmd = [sys.executable, os.path.join(C.ROOT, "scripts", "m27_proxy.py"),
               "--log", self.proxy_log, "--port", str(C.PROXY_PORT)]
        if self.kind == "colo":
            cmd += ["--mode", "passthrough", "--backends", ",".join(map(str, self.ports("colo")))]
        else:
            cmd += ["--mode", "pd", "--prefill", ",".join(map(str, self.ports("P"))),
                    "--decode", ",".join(map(str, self.ports("D")))]
        if self.allow_fault:
            cmd.append("--allow-fault-injection")
        log = os.path.join(self.log_dir, f"{self.exp}_{self.arm}_proxy.log")
        lf = open(log, "w")
        self.proxy = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                      start_new_session=True, cwd=C.ROOT)
        t0 = time.time()
        while time.time() - t0 < 60:
            if self.proxy.poll() is not None:
                self.stop(reason="proxy exited")
                raise RuntimeError(f"proxy 提前退出；见 {log}")
            if health_ok(C.PROXY_PORT):
                return
            time.sleep(0.5)
        self.stop(reason="proxy health timeout")
        raise RuntimeError("proxy health 超时")

    def _stop_group(self, proc) -> str:
        if proc is None or proc.poll() is not None:
            return "already_exited"
        try:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=60)
        except ProcessLookupError:
            return "gone"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
                return "sigkill_group"
            return "sigterm_group"
        # Let vLLM's API server shut its EngineCore down before signalling
        # surviving processes in this launcher-owned process group.
        for _ in range(30):
            active = []
            for app in C.gpu_compute_apps():
                try:
                    pid = int(app.split(",", 1)[0])
                    if os.getpgid(pid) == proc.pid:
                        active.append(pid)
                except (ValueError, ProcessLookupError, PermissionError):
                    continue
            if not active:
                return "sigterm_graceful"
            time.sleep(1)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return "sigterm_graceful"
        return "sigterm_orphan_group"

    def stop(self, reason="normal"):
        res = {"proxy": self._stop_group(self.proxy)}
        for name, p in self.procs.items():
            res[name] = self._stop_group(p)
        busy = []
        for _ in range(60):
            busy = [i["http_port"] for i in self.instances if not C.port_free(i["http_port"])]
            if not busy:
                break
            time.sleep(1)
        log_ev = {n: parse_log_evidence(lp) for n, lp in self.logs.items()}
        C.append_jsonl(self.lifecycle, {"event": "stop", "tag": self.tag, "exp": self.exp,
                                        "arm": self.arm, "reason": reason, "result": res,
                                        "ports_still_busy": busy, "log_evidence": log_ev})
        self.procs, self.proxy = {}, None
        return res

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop(reason="exception" if exc[0] else "normal")
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--exp", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--model", default=os.environ.get("M27_MODEL", ""))
    p.add_argument("--hold", action="store_true", help="启动后保持，Ctrl-C 停止（排障用）")
    a = p.parse_args()
    if not a.model or not os.path.isdir(a.model):
        raise SystemExit("--model / M27_MODEL 必须是本地模型目录（不联网下载）")
    gpu_ids = C.gpu_map(a.tag)
    topo = Topology(a.tag, a.exp, a.arm, gpu_ids, a.model).start()
    print(f"[launch] up: proxy={topo.proxy_url} instances={topo.instances}")
    try:
        while a.hold:
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        print("[launch] stop:", topo.stop(reason="cli"))


if __name__ == "__main__":
    main()
