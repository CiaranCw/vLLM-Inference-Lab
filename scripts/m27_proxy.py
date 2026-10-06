#!/usr/bin/env python3
"""M27 · fail-closed 代理：colocated 透传 / P→D 解耦路由，并逐请求落盘时间点。

    python scripts/m27_proxy.py --mode pd --prefill 8400 --decode 8401 --log outputs/m27/raw/<tag>_<exp>_<arm>_proxy.jsonl
    python scripts/m27_proxy.py --mode passthrough --backends 8400,8401 --log ...

与上游 toy_proxy_server.py 的关键差异（tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:235-237）：
toy proxy 在 P 没返回 kv_transfer_params 时**静默**把原请求发给 D，D 会自己做完整 prefill，
PD 实验因此退化成「D 上的 colocated」却不报错。本代理在这种情况下返回 502，绝不转发。

colocated 也走本代理（passthrough），以抵消「多一跳 HTTP」这个隐藏变量。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import json
import os
import sys
import time
import uuid

import aiohttp
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402

REQUIRED_KV_KEYS = ("remote_engine_id", "remote_block_ids", "remote_host", "remote_port")
P_SIDE_KV_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


def validate_kv_params(kv) -> str | None:
    """返回 None 表示合法；否则返回拒绝原因（fail-closed）。"""
    if not isinstance(kv, dict) or not kv:
        return "missing_kv_transfer_params"
    if kv.get("do_remote_prefill") is not True:
        return "do_remote_prefill_not_true"
    for k in REQUIRED_KV_KEYS:
        if kv.get(k) in (None, "", []):
            return f"missing_{k}"
    blocks = kv["remote_block_ids"]
    n_blocks = sum(len(g) for g in blocks) if blocks and isinstance(blocks[0], list) else len(blocks)
    if n_blocks == 0:
        return "empty_remote_block_ids"
    return None


def n_remote_blocks(kv) -> int | None:
    try:
        b = kv["remote_block_ids"]
        return sum(len(g) for g in b) if b and isinstance(b[0], list) else len(b)
    except Exception:
        return None


class Proxy:
    def __init__(self, a):
        self.mode = a.mode
        self.log = a.log
        self.allow_fault = a.allow_fault_injection
        self.backends = [int(x) for x in (a.backends or "").split(",") if x.strip()]
        self.prefill = [int(x) for x in (a.prefill or "").split(",") if x.strip()]
        self.decode = [int(x) for x in (a.decode or "").split(",") if x.strip()]
        if self.mode == "passthrough" and not self.backends:
            raise SystemExit("passthrough 需要 --backends")
        if self.mode == "pd" and (not self.prefill or not self.decode):
            raise SystemExit("pd 需要 --prefill 与 --decode")
        self.rr_b = itertools.cycle(self.backends) if self.backends else None
        self.rr_p = itertools.cycle(self.prefill) if self.prefill else None
        self.rr_d = itertools.cycle(self.decode) if self.decode else None
        self.session: aiohttp.ClientSession | None = None
        self.counters = {"ok": 0, "rejected": 0, "upstream_error": 0}

    async def start(self, app):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, sock_connect=10),
            connector=aiohttp.TCPConnector(limit=0, force_close=True))
        self.record({"event": "upstream_transport_config", "force_close": True,
                     "reason": "avoid stale upstream keep-alive socket reuse"})

    async def stop(self, app):
        if self.session:
            await self.session.close()

    def record(self, row):
        row = dict(row)
        row["mode"] = self.mode
        C.append_jsonl(self.log, row)

    async def health(self, request):
        ports = self.backends or (self.prefill + self.decode)
        bad = []
        for p in ports:
            try:
                async with self.session.get(f"http://{C.HOST}:{p}/health",
                                            timeout=aiohttp.ClientTimeout(total=3)) as r:
                    if r.status != 200:
                        bad.append(p)
            except Exception:
                bad.append(p)
        return web.json_response({"ok": not bad, "bad": bad}, status=200 if not bad else 503)

    async def stats(self, request):
        return web.json_response({"mode": self.mode, **self.counters})

    async def _stream_to_client(self, request, port, body, rid, rec):
        url = f"http://{C.HOST}:{port}/v1/completions"
        headers = {"X-Request-Id": rid}
        resp = None
        async with self.session.post(url, json=body, headers=headers) as up:
            rec["stream_status"] = up.status
            rec["t_stream_headers"] = time.time()
            if up.status != 200:
                text = await up.text()
                rec["error"] = f"stream_upstream_{up.status}: {text[:300]}"
                self.counters["upstream_error"] += 1
                return web.json_response({"error": rec["error"], "m27_proxy_request_id": rid},
                                         status=502)
            resp = web.StreamResponse(status=200, headers={
                "Content-Type": up.headers.get("Content-Type", "text/event-stream"),
                "X-M27-Proxy-Request-Id": rid})
            await resp.prepare(request)
            first = True
            async for chunk in up.content.iter_any():
                if first:
                    rec["t_stream_first_byte"] = time.time()
                    first = False
                await resp.write(chunk)
            rec["t_stream_done"] = time.time()
            await resp.write_eof()
        return resp

    async def completions(self, request):
        rid = request.headers.get("X-Request-Id") or f"m27-{uuid.uuid4().hex}"
        body = await request.json()
        rec = {"proxy_request_id": rid, "t_recv": time.time(),
               "client_request_tag": body.pop("m27_tag", None)}
        fault = body.pop("m27_fault", None)
        try:
            if self.mode == "passthrough":
                port = next(self.rr_b)
                rec["backend_port"] = port
                resp = await self._stream_to_client(request, port, body, rid, rec)
            else:
                resp = await self._pd(request, body, rid, rec, fault)
            if rec.get("error") is None:
                self.counters["ok"] += 1
            return resp
        except Exception as e:  # 不吞：落盘 + 502
            rec["error"] = f"proxy_exception: {type(e).__name__}: {e}"
            self.counters["upstream_error"] += 1
            return web.json_response({"error": rec["error"], "m27_proxy_request_id": rid},
                                     status=502)
        finally:
            rec["t_end"] = time.time()
            self.record(rec)

    async def _pd(self, request, body, rid, rec, fault):
        p_port, d_port = next(self.rr_p), next(self.rr_d)
        rec.update(p_port=p_port, d_port=d_port)
        p_body = copy.deepcopy(body)
        p_body["kv_transfer_params"] = dict(P_SIDE_KV_PARAMS)
        p_body["stream"] = False
        p_body["max_tokens"] = 1
        p_body.pop("stream_options", None)
        p_body.pop("min_tokens", None)
        p_body.pop("return_token_ids", None)
        rec["t_p_send"] = time.time()
        async with self.session.post(f"http://{C.HOST}:{p_port}/v1/completions",
                                     json=p_body, headers={"X-Request-Id": rid}) as r:
            rec["p_status"] = r.status
            p_text = await r.text()
        rec["t_p_done"] = time.time()
        if rec["p_status"] != 200:
            rec["error"] = f"prefill_upstream_{rec['p_status']}: {p_text[:300]}"
            self.counters["upstream_error"] += 1
            return web.json_response({"error": rec["error"], "m27_proxy_request_id": rid},
                                     status=502)
        p_json = json.loads(p_text)
        kv = p_json.get("kv_transfer_params")
        if fault == "drop_kv_params":
            if not self.allow_fault:
                rec["error"] = "fault_injection_not_allowed"
                self.counters["rejected"] += 1
                return web.json_response({"error": rec["error"]}, status=400)
            rec["fault_injected"] = "drop_kv_params"
            kv = None
        reason = validate_kv_params(kv)
        rec["kv_reject_reason"] = reason
        rec["n_remote_blocks"] = n_remote_blocks(kv) if kv else None
        rec["remote_engine_id"] = (kv or {}).get("remote_engine_id")
        if reason is not None:
            rec["error"] = f"fail_closed:{reason}"
            self.counters["rejected"] += 1
            return web.json_response({"error": rec["error"], "m27_proxy_request_id": rid},
                                     status=502)
        d_body = copy.deepcopy(body)
        d_body["kv_transfer_params"] = kv
        rec["t_d_send"] = time.time()
        return await self._stream_to_client(request, d_port, d_body, rid, rec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["passthrough", "pd"], required=True)
    p.add_argument("--port", type=int, default=C.PROXY_PORT)
    p.add_argument("--backends", default="")
    p.add_argument("--prefill", default="")
    p.add_argument("--decode", default="")
    p.add_argument("--log", required=True)
    p.add_argument("--allow-fault-injection", action="store_true",
                   help="仅 E0 fail-closed 自测使用；formal E1–E4 不得开启")
    a = p.parse_args()
    px = Proxy(a)
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app.on_startup.append(px.start)
    app.on_cleanup.append(px.stop)
    app.router.add_post("/v1/completions", px.completions)
    app.router.add_get("/health", px.health)
    app.router.add_get("/proxy_stats", px.stats)
    web.run_app(app, host=C.HOST, port=a.port, access_log=None, print=None)


if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
    main()
