#!/usr/bin/env python3
"""M20-A · Native serving production semantics（轻量 smoke，非大规模 benchmark）。

Shards:
  a1-stream   stream=true vs stream=false（少量请求；协议/首包时机）
  a2-prefix   enable-prefix-caching serving smoke（共享 prefix；引用 M17，不扫 APC）
  a3-health   /health → kill → FAIL → restart → PASS
  all         a1 → a2 → a3（顺序；每 shard 独立起停 server）

不复用 M12 QPS 网格；不做 Bootstrap；不做 admission 正式实验。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m20"
RAW = OUT / "raw"
LOG = ROOT / "logs" / "m20"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
HOST = "127.0.0.1"
PORT = 8020  # 避开 M12 的 8014
BASE = f"http://{HOST}:{PORT}"

# 对齐 M12 server 指纹（notes/m12/serving.md §1）；A2 仅改 prefix
SERVER_BASE = [
    sys.executable,
    "-m",
    "vllm.entrypoints.openai.api_server",
    "--model",
    MODEL,
    "--enforce-eager",
    "--dtype",
    "auto",
    "--max-model-len",
    "2048",
    "--gpu-memory-utilization",
    "0.6",
    "--max-num-seqs",
    "64",
    "--max-num-batched-tokens",
    "4096",
    "--no-async-scheduling",
    "--enable-chunked-prefill",
    "--disable-uvicorn-access-log",
    "--port",
    str(PORT),
    "--host",
    HOST,
]


def _env():
    e = os.environ.copy()
    e["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    e["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    e.pop("VLLM_LAB_TRACE", None)
    e.pop("VLLM_LAB_M9_COALESCE", None)
    return e


def health_code(timeout=3.0):
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=timeout) as r:
            return r.status
    except Exception:
        return None


def wait_healthy(proc, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early rc={proc.returncode}")
        if health_code() == 200:
            return time.time() - t0
        time.sleep(0.5)
    raise RuntimeError("health timeout")


def start_server(extra_args, log_name):
    LOG.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    log_path = LOG / log_name
    cmd = SERVER_BASE + extra_args
    lf = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, stdout=lf, stderr=subprocess.STDOUT, env=_env(), cwd=str(ROOT)
    )
    ready_s = wait_healthy(proc)
    return proc, lf, log_path, ready_s, cmd


def stop_server(proc, lf):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    lf.close()


def chat(messages, stream: bool, max_tokens=32, temperature=0.0):
    body = json.dumps(
        {
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    first_byte_s = None
    chunks = []
    with urllib.request.urlopen(req, timeout=120) as r:
        if not stream:
            data = r.read()
            first_byte_s = time.perf_counter() - t0
            return {
                "mode": "non_stream",
                "http_status": r.status,
                "t_first_byte_ms": round(first_byte_s * 1000, 3),
                "t_e2e_ms": round((time.perf_counter() - t0) * 1000, 3),
                "content_type": r.headers.get("Content-Type"),
                "body": json.loads(data.decode()),
            }
        # streaming: SSE lines
        while True:
            line = r.readline()
            if not line:
                break
            if first_byte_s is None:
                first_byte_s = time.perf_counter() - t0
            s = line.decode("utf-8", errors="replace").strip()
            if not s:
                continue
            chunks.append(s)
            if s == "data: [DONE]":
                break
    return {
        "mode": "stream",
        "http_status": 200,
        "t_first_byte_ms": None
        if first_byte_s is None
        else round(first_byte_s * 1000, 3),
        "t_e2e_ms": round((time.perf_counter() - t0) * 1000, 3),
        "n_sse_lines": len(chunks),
        "first_sse": chunks[0] if chunks else None,
        "note": "streaming ITL 需逐 token timestamp；本 smoke 只记 first-byte 与 E2E",
    }


def metrics_snippet():
    try:
        with urllib.request.urlopen(f"{BASE}/metrics", timeout=5) as r:
            text = r.read().decode("utf-8", errors="replace")
        keys = []
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            name = line.split("{", 1)[0].split(" ", 1)[0]
            if name not in keys:
                keys.append(name)
            if len(keys) >= 25:
                break
        return {"ok": True, "n_chars": len(text), "sample_metric_names": keys}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def shard_a1():
    proc, lf, log_path, ready_s, cmd = start_server(
        ["--no-enable-prefix-caching"], "a1_server.log"
    )
    rows = []
    try:
        # warmup 1
        chat([{"role": "user", "content": "ping"}], stream=False, max_tokens=8)
        for i in range(3):
            rows.append(
                {
                    "shard": "a1",
                    "trial": i,
                    **chat(
                        [{"role": "user", "content": f"用一句话介绍 KV Cache。#{i}"}],
                        stream=False,
                        max_tokens=24,
                    ),
                }
            )
        for i in range(3):
            rows.append(
                {
                    "shard": "a1",
                    "trial": i,
                    **chat(
                        [{"role": "user", "content": f"用一句话介绍 KV Cache。#{i}"}],
                        stream=True,
                        max_tokens=24,
                    ),
                }
            )
        out = {
            "shard": "a1-stream",
            "ready_s": ready_s,
            "cmd": cmd,
            "log": str(log_path),
            "semantics": {
                "non_stream": "单次 JSON 响应；t_first_byte≈整包到达，不能拆 streaming ITL",
                "stream": "SSE data: 行；t_first_byte=首包；ITL 需逐 token 时间戳（本 smoke 不做）",
            },
            "trials": rows,
            "metrics": metrics_snippet(),
        }
    finally:
        stop_server(proc, lf)
    path = RAW / "a1_stream_smoke.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"[a1] wrote {path}")
    ns = [r for r in rows if r["mode"] == "non_stream"]
    st = [r for r in rows if r["mode"] == "stream"]
    print(
        f"[a1] non_stream first_byte_ms med≈{sorted(x['t_first_byte_ms'] for x in ns)[len(ns)//2]}"
    )
    print(
        f"[a1] stream     first_byte_ms med≈{sorted(x['t_first_byte_ms'] for x in st)[len(st)//2]}"
    )
    return True


def shard_a2():
    # shared long prefix + distinct suffixes（serving 行为 smoke，非 M17 性能网格）
    prefix = ("这是一段用于验证 online prefix caching 的共享前缀。" * 40)[:800]
    proc, lf, log_path, ready_s, cmd = start_server(
        ["--enable-prefix-caching"], "a2_server.log"
    )
    rows = []
    try:
        chat([{"role": "user", "content": prefix + " warmup"}], stream=False, max_tokens=8)
        for i in range(4):
            msg = prefix + f" 请用五个字回答编号{i}。"
            rows.append(
                {
                    "trial": i,
                    "prompt_chars": len(msg),
                    **chat(
                        [{"role": "user", "content": msg}],
                        stream=False,
                        max_tokens=16,
                    ),
                }
            )
        out = {
            "shard": "a2-prefix",
            "ready_s": ready_s,
            "cmd": cmd,
            "log": str(log_path),
            "prefix_caching": "enabled via --enable-prefix-caching",
            "performance_claim": "NONE — APC 性能见 notes/m17；此处只验证 serving 可启用与可观测",
            "trials": rows,
            "metrics": metrics_snippet(),
        }
    finally:
        stop_server(proc, lf)
    path = RAW / "a2_prefix_smoke.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"[a2] wrote {path}")
    return True


def shard_a3():
    proc, lf, log_path, ready_s, cmd = start_server(
        ["--no-enable-prefix-caching"], "a3_server.log"
    )
    timeline = []
    try:
        h1 = health_code()
        timeline.append({"step": "running", "health": h1})
        assert h1 == 200, h1
        m1 = metrics_snippet()
        timeline.append({"step": "metrics_while_up", "metrics_ok": m1.get("ok")})
        stop_server(proc, lf)
        time.sleep(1)
        h2 = health_code()
        timeline.append({"step": "after_stop", "health": h2})
        assert h2 != 200, h2
        # restart
        proc2, lf2, log_path2, ready_s2, cmd2 = start_server(
            ["--no-enable-prefix-caching"], "a3_server_restart.log"
        )
        try:
            h3 = health_code()
            timeline.append(
                {"step": "after_restart", "health": h3, "ready_s": ready_s2}
            )
            assert h3 == 200, h3
            chat([{"role": "user", "content": "recover?"}], stream=False, max_tokens=8)
            timeline.append({"step": "generate_after_restart", "ok": True})
        finally:
            stop_server(proc2, lf2)
            log_path = log_path2
            cmd = cmd2
            ready_s = ready_s2
        out = {
            "shard": "a3-health",
            "initial_ready_s": ready_s,
            "cmd": cmd,
            "log": str(log_path),
            "timeline": timeline,
            "scope": "single kill/restart smoke only; OOM/port/collision deferred",
        }
    except Exception:
        if proc.poll() is None:
            stop_server(proc, lf)
        raise
    path = RAW / "a3_health_restart_smoke.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(f"[a3] wrote {path}")
    print("[a3] timeline:", json.dumps(timeline, ensure_ascii=False))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shard",
        choices=["a1-stream", "a2-prefix", "a3-health", "all"],
        default="all",
    )
    a = ap.parse_args()
    # refuse if another server already on PORT
    if health_code() == 200:
        print(f"ERROR: something already healthy on {BASE}", file=sys.stderr)
        return 2
    nvidia = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
    )
    apps = [x for x in nvidia.stdout.splitlines() if x.strip()]
    if apps:
        print("ERROR: GPU compute apps present; refuse to start:", apps, file=sys.stderr)
        return 3
    shards = {
        "a1-stream": shard_a1,
        "a2-prefix": shard_a2,
        "a3-health": shard_a3,
    }
    order = (
        ["a1-stream", "a2-prefix", "a3-health"] if a.shard == "all" else [a.shard]
    )
    for name in order:
        print(f"\n===== {name} =====")
        shards[name]()
    print("\n[m20-a] all requested shards done (lightweight smoke only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
