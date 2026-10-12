#!/usr/bin/env python3
"""M28 CPU 演练用 mock server —— **不是 vLLM，不产生任何可引用的实验结论**。

用途只有一个：在不碰 GPU 的前提下，端到端演练 m28_offload.py / m28_faults.py 的
HTTP/SSE 解析、指标增量记账、进程树归属核对、信号注入与兜底清理、validator 的检出能力。
它模仿的接口形态（/health、/metrics 计数器名、SSE 错误事件、EngineCore 子进程名）
来自 notes/m28/source_review.md 的源码阅读；语义是简化模型，与真实 vLLM 行为无关。
产物一律带 mock=true，且 runner 拒绝把 mock 产物写进 outputs/m28。
"""

import argparse
import collections
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BLOCK = 16
KV_PER_TOKEN = 12288


def child_main():
    try:
        import setproctitle
        setproctitle.setproctitle("VLLM::EngineCore")
    except ImportError:
        pass
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(1)


class State:
    def __init__(self, arm, gpu_blocks, cpu_blocks, shutdown_timeout):
        self.arm = arm
        self.prefix = arm in ("offload", "base", "evict")
        self.connector = arm in ("offload", "evict")
        self.gpu = collections.OrderedDict()
        self.cpu = collections.OrderedDict()
        self.gpu_cap, self.cpu_cap = gpu_blocks, cpu_blocks
        self.c = collections.Counter()
        self.lock = threading.Lock()
        self.running = 0
        self.engine_dead = False
        self.shutting_down = False
        self.abort_all = threading.Event()
        self.shutdown_timeout = shutdown_timeout
        self.child = None

    def hashes(self, ids):
        out, h = [], b""
        for i in range(len(ids) // BLOCK):
            h = hashlib.sha256(h + json.dumps(ids[i * BLOCK:(i + 1) * BLOCK]).encode()).digest()
            out.append(h)
        return out

    def admit(self, ids):
        with self.lock:
            hs = self.hashes(ids)
            max_hit = (len(ids) - 1) // BLOCK
            local = 0
            if self.prefix:
                while local < max_hit and hs[local] in self.gpu:
                    self.gpu.move_to_end(hs[local])
                    local += 1
                self.c["local_queries"] += len(ids)
                self.c["local_hits"] += local * BLOCK
            ext = 0
            if self.connector:
                while local + ext < max_hit and hs[local + ext] in self.cpu:
                    ext += 1
                self.c["ext_queries"] += len(ids) - local * BLOCK
                self.c["ext_hits"] += ext * BLOCK
                if ext:
                    self.c["load_bytes"] += ext * BLOCK * KV_PER_TOKEN
                    self.c["load_time"] += ext * 1e-5
            if self.prefix:
                for h in hs:
                    self.gpu[h] = True
                    self.gpu.move_to_end(h)
                while len(self.gpu) > self.gpu_cap:
                    self.gpu.popitem(last=False)
            if self.connector:
                for h in hs:
                    if h not in self.cpu:
                        self.cpu[h] = True
                        self.c["store_bytes"] += BLOCK * KV_PER_TOKEN
                while len(self.cpu) > self.cpu_cap:
                    self.cpu.popitem(last=False)
            self.c["prompt_tokens"] += len(ids)
            return (local + ext) * BLOCK, len(ids) - (local + ext) * BLOCK

    def metrics(self):
        lines = []
        names = {"local_queries": "vllm:prefix_cache_queries_total", "local_hits": "vllm:prefix_cache_hits_total",
                 "prompt_tokens": "vllm:prompt_tokens_total", "gen": "vllm:generation_tokens_total",
                 "success": "vllm:request_success_total"}
        if self.connector:
            names.update({"ext_queries": "vllm:external_prefix_cache_queries_total",
                          "ext_hits": "vllm:external_prefix_cache_hits_total",
                          "load_bytes": "vllm:kv_offload_load_bytes_total",
                          "load_time": "vllm:kv_offload_load_time_total",
                          "store_bytes": "vllm:kv_offload_store_bytes_total"})
        for k, n in names.items():
            lines.append(f'{n}{{engine="0",model_name="mock"}} {float(self.c[k])}')
        lines.append(f'vllm:num_requests_running{{engine="0"}} {float(self.running)}')
        return "\n".join(lines) + "\n"


def make_handler(st, server_ref):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.send_response(503 if st.engine_dead else 200)
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.path == "/metrics":
                body = st.metrics().encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json(404, {"error": "not found"})

        def _engine_stopped(self):
            try:
                import psutil
                return psutil.Process(st.child.pid).status() == psutil.STATUS_STOPPED
            except Exception:
                return False

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            err = {"error": {"message": "EngineCore encountered an issue. See stack trace (above) for the root cause.",
                             "type": "InternalServerError", "param": None, "code": 500}}
            if st.engine_dead:
                return self._json(500, err)
            ids, mt, stream = req["prompt"], req["max_tokens"], req.get("stream")
            if st.shutting_down:
                return self._json(500, {"error": {"message": "mock: rejected during shutdown",
                                                  "type": "InternalServerError", "code": 500}})
            cached, compute = st.admit(ids)
            seed = hashlib.sha256(json.dumps(ids).encode()).digest()
            toks = [1000 + seed[i % 32] * 7 + i for i in range(mt)]
            with st.lock:
                st.running += 1
            time.sleep(0.003 + compute * 2e-7 + cached * 2e-8)
            usage = {"prompt_tokens": len(ids), "completion_tokens": mt, "total_tokens": len(ids) + mt,
                     "prompt_tokens_details": {"cached_tokens": cached}}
            finish = "length"
            try:
                if not stream:
                    out = []
                    for t in toks:
                        while self._engine_stopped():
                            time.sleep(0.05)
                        if st.engine_dead:
                            return self._json(500, err)
                        if st.abort_all.is_set():
                            finish = "abort"
                            break
                        out.append(t)
                        time.sleep(0.005)
                    with st.lock:
                        st.c["success"] += 1
                    lp = {"tokens": [f"token_id:{t}" for t in out], "token_logprobs": [-0.5] * len(out)}
                    return self._json(200, {"choices": [{"text": "", "logprobs": lp, "finish_reason": finish}],
                                            "usage": usage})
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i, t in enumerate(toks):
                    while self._engine_stopped():
                        time.sleep(0.05)
                    if st.engine_dead:
                        self.wfile.write(f"data: {json.dumps(err)}\n\n".encode())
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                        return
                    if st.abort_all.is_set():
                        finish = "abort"
                        break
                    ch = {"choices": [{"text": "", "logprobs": {"tokens": [f"token_id:{t}"], "token_logprobs": [-0.5]},
                                       "finish_reason": "length" if i == mt - 1 else None}]}
                    self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                    self.wfile.flush()
                    time.sleep(0.005)
                if finish == "abort":
                    ch = {"choices": [{"text": "", "logprobs": None, "finish_reason": "abort"}]}
                    self.wfile.write(f"data: {json.dumps(ch)}\n\n".encode())
                with st.lock:
                    st.c["success"] += 1
                self.wfile.write(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with st.lock:
                    st.running -= 1
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--port", type=int, default=8028)
    ap.add_argument("--arm", default="offload")
    ap.add_argument("--shutdown-timeout", type=int, default=0)
    a = ap.parse_args()
    if a.child:
        return child_main()
    cpu_blocks = (1 << 30) * 4 // (BLOCK * KV_PER_TOKEN)
    if a.arm == "evict":
        cpu_blocks = (1 << 28) // (BLOCK * KV_PER_TOKEN)
    st = State(a.arm, 1280, cpu_blocks, a.shutdown_timeout)
    print(f"mock server arm={a.arm} (NOT vLLM)", flush=True)
    st.child = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child"])
    httpd = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(st, None))
    httpd.daemon_threads = True

    def monitor():
        st.child.wait()
        if not st.shutting_down:
            print("[shutdown] MPClient: engine core exited unexpectedly; starting cleanup", flush=True)
            st.engine_dead = True
            time.sleep(5.0)  # 模仿 watchdog 周期
            httpd.shutdown()

    def shutdown():
        print("[shutdown] API server: shutdown triggered", flush=True)
        st.shutting_down = True
        if st.shutdown_timeout == 0:
            print("[shutdown] EngineCore: aborting in-flight requests count=%d" % st.running, flush=True)
            st.abort_all.set()
        else:
            t0 = time.time()
            while st.running and time.time() - t0 < st.shutdown_timeout:
                time.sleep(0.05)
            if st.running:
                print("[shutdown] Process manager: force killing remaining processes count=1", flush=True)
                st.abort_all.set()
        time.sleep(0.2)
        if st.child.poll() is None:
            st.child.kill()
        httpd.shutdown()

    threading.Thread(target=monitor, daemon=True).start()
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=shutdown, daemon=True).start())
    httpd.serve_forever()
    httpd.server_close()
    if st.child.poll() is None:
        st.child.kill()
        st.child.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
