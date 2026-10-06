#!/usr/bin/env python3
"""M27 · 客户端工作负载：E0 正确性 / E1 传输扫描 / E2 受控干扰 / E3–E5 在线 Poisson / 速率校准。

由 m27_run_all.py 在已启动的 Topology 上调用（一个 arm 一个 Topology，一个进程一个引擎）。
所有请求都发给 proxy（colocated 透传 / PD 解耦），客户端用 token id 作 prompt，
`return_token_ids=true` 逐 chunk 计 token —— 这是在线口径（metric_scope=online_client），
允许 TTFT / ITL 命名（与 in-process 口径区分，AGENTS.md §6）。

ITL 定义：同一请求相邻两个**含 token 的 chunk** 的到达间隔（stream_interval=1 下通常 1 token/chunk；
多 token chunk 的数量单独记 n_multi_token_chunks，不拆分、不插值）。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402

VOCAB_LO, VOCAB_HI = 1000, 150000  # 避开 Qwen2.5 特殊 token（>=151643）
METRICS_SETTLE_TIMEOUT_S = 20
METRICS_SAMPLE_PERIOD_S = 1.0
E2_MIN_D_TOKENS_AFTER_INJECTION = 16
CAL_START_RATE = 0.5
CAL_STEP = math.sqrt(2)
CAL_MAX_STEPS = 16
CAL_HORIZON_S = 30
CAL_TTFT_MULT = 10
CAL_ACHIEVED_FRAC = 0.95
E3_WARMUP_REQUESTS = 16
E0_N = 5
E0_PROMPT = 2048
E0_OUTPUT = 32


# --------------------------------------------------------------------------- prompts
def prompt_ids(key: str, n: int) -> list[int]:
    """确定性 prompt：同一 key 在所有 arm 上逐 token 相同（prompt identity 不变量的基础）。"""
    seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def pctl(xs, q):
    """exact percentile（线性插值，与 numpy 默认一致）；空 → None。"""
    if not xs:
        return None
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * q / 100.0
    f, c = math.floor(k), math.ceil(k)
    return s[f] + (s[c] - s[f]) * (k - f)


# --------------------------------------------------------------------------- client
async def stream_completion(session, base_url, ids, max_tokens, label, on_tokens=None,
                            fault=None, timeout_s=1800) -> dict:
    body = {
        "model": C.SERVED_MODEL_NAME, "prompt": ids, "max_tokens": max_tokens,
        "temperature": 0, "top_p": 1, "ignore_eos": True, "stream": True,
        "stream_options": {"include_usage": True}, "return_token_ids": True,
        "m27_tag": label,
    }
    if fault:
        body["m27_fault"] = fault
    rec = {"label": label, "prompt_len": len(ids), "prompt_sha256": C.sha256_ids(ids),
           "max_tokens": max_tokens, "t_send_unix": time.time(), "success": False,
           "error": None, "metric_scope": "online_client"}
    t0 = time.perf_counter()
    times, ntoks, out_ids = [], [], []
    try:
        async with session.post(f"{base_url}/v1/completions", json=body,
                                timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
            rec["http_status"] = r.status
            rec["proxy_request_id"] = r.headers.get("X-M27-Proxy-Request-Id")
            if r.status != 200:
                txt = await r.text()
                rec["error"] = f"http_{r.status}: {txt[:300]}"
                try:
                    rec["proxy_request_id"] = rec["proxy_request_id"] or json.loads(txt).get(
                        "m27_proxy_request_id")
                except Exception:
                    pass
                return rec
            buf = b""
            async for chunk in r.content.iter_any():
                now = time.perf_counter()
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line.startswith(b"data:"):
                        continue
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        continue
                    obj = json.loads(data)
                    if obj.get("usage"):
                        rec["usage"] = obj["usage"]
                    for ch in obj.get("choices") or []:
                        if ch.get("prompt_token_ids"):
                            rec["prompt_sha256_returned"] = C.sha256_ids(ch["prompt_token_ids"])
                        tids = ch.get("token_ids") or []
                        if tids:
                            times.append((now - t0) * 1000.0)
                            ntoks.append(len(tids))
                            out_ids.extend(tids)
                            if on_tokens:
                                on_tokens(len(out_ids))
                        if ch.get("finish_reason"):
                            rec["finish_reason"] = ch["finish_reason"]
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
        return rec
    rec["e2e_ms"] = (time.perf_counter() - t0) * 1000.0
    rec["token_times_ms"] = times
    rec["chunk_ntok"] = ntoks
    rec["n_output_tokens"] = len(out_ids)
    rec["output_token_ids_sha256"] = C.sha256_ids(out_ids)
    rec["first_output_token_id"] = out_ids[0] if out_ids else None
    rec["n_multi_token_chunks"] = sum(1 for n in ntoks if n > 1)
    rec["ttft_ms"] = times[0] if times else None
    rec["itl_ms"] = [b - a for a, b in zip(times, times[1:])]
    rec["success"] = (rec["n_output_tokens"] == max_tokens and rec["error"] is None)
    if not rec["success"] and rec["error"] is None:
        rec["error"] = f"output_tokens {rec['n_output_tokens']} != {max_tokens}"
    if max_tokens <= 64:
        rec["output_token_ids"] = out_ids
    return rec


def _session():
    return aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0),
                                 timeout=aiohttp.ClientTimeout(total=None))


def _base_row(topo, exp, phase):
    return {"tag": topo.tag, "experiment": exp, "arm": topo.arm, "kind": topo.kind,
            "phase": phase, "plan_version": C.FROZEN["plan_version"],
            "instances": [{k: i[k] for k in ("name", "role", "gpu", "mbt")} for i in topo.instances]}


def _wait_metrics(topo, before, inst, key, min_delta, timeout=METRICS_SETTLE_TIMEOUT_S):
    """轮询直到 inst 的 key 增量 ≥ min_delta 且连续两次不变；超时返回 settled=False（不静默）。"""
    t0, last, stable = time.time(), None, 0
    snap = None
    while time.time() - t0 < timeout:
        snap = topo.snapshot()
        b, a = before[inst].get(key), snap[inst].get(key)
        d = None if (a is None or b is None) else a - b
        if d is not None and d >= min_delta:
            stable = stable + 1 if d == last else 0
            last = d
            if stable >= 1:
                return snap, True
        time.sleep(0.25)
    return snap or topo.snapshot(), False


def _deltas(before, after):
    return {n: C.snapshot_delta(before[n], after[n]) for n in after}


# --------------------------------------------------------------------------- E0
async def run_e0(topo, raw_path):
    """Gate 1：P 真的 prefill、KV 真的传输、D 不重算、代理 fail-closed。"""
    n_w = 3
    async with _session() as s:
        for i in range(n_w + E0_N):
            phase = "warmup" if i < n_w else "measure"
            topo.wait_idle()
            before = topo.snapshot()
            ids = prompt_ids(f"e0-{i}", E0_PROMPT)
            r = await stream_completion(s, topo.proxy_url, ids, E0_OUTPUT, f"e0-{i}")
            if topo.kind == "pd":
                d = next(x["name"] for x in topo.instances if x["role"] == "D")
                after, settled = _wait_metrics(topo, before, d, "prompt_external_kv_transfer", E0_PROMPT)
            else:
                topo.wait_idle()
                after, settled = topo.snapshot(), True
            row = _base_row(topo, "e0", phase)
            row.update(r, idx=i, request_key=f"e0-{i}", metrics_settled=settled,
                       metrics_delta=_deltas(before, after), **C.kv_theory(E0_PROMPT))
            C.append_jsonl(raw_path, row)
        if topo.kind == "pd" and topo.allow_fault:
            topo.wait_idle()
            before = topo.snapshot()
            ids = prompt_ids("e0-fault", E0_PROMPT)
            r = await stream_completion(s, topo.proxy_url, ids, E0_OUTPUT, "e0-fault",
                                        fault="drop_kv_params")
            time.sleep(2)
            topo.wait_idle()
            after = topo.snapshot()
            row = _base_row(topo, "e0", "fault_probe")
            row.update(r, request_key="e0-fault", fault="drop_kv_params",
                       metrics_delta=_deltas(before, after))
            row["success"] = r.get("http_status") == 502
            C.append_jsonl(raw_path, row)


# --------------------------------------------------------------------------- E1
async def run_e1(topo, raw_path, contexts, out_tokens, warmup, measure):
    assert topo.kind == "pd", "E1 只在 PD 拓扑上跑"
    d = next(x["name"] for x in topo.instances if x["role"] == "D")
    async with _session() as s:
        for L in contexts:
            for i in range(warmup + measure):
                phase = "warmup" if i < warmup else "measure"
                topo.wait_idle()
                before = topo.snapshot()
                key = f"e1-{L}-{i}"
                r = await stream_completion(s, topo.proxy_url, prompt_ids(key, L), out_tokens, key)
                after, settled = _wait_metrics(topo, before, d, "nixl_xfer_count", 1)
                row = _base_row(topo, "e1", phase)
                row.update(r, context=L, idx=i, request_key=key, metrics_settled=settled,
                           metrics_delta=_deltas(before, after), **C.kv_theory(L))
                C.append_jsonl(raw_path, row)
                print(f"[e1] L={L} {phase} {i} ok={r['success']} settled={settled}", flush=True)


# --------------------------------------------------------------------------- E2
async def run_e2(topo, raw_path, p_lens, warmup, measure):
    cfg = C.FROZEN["e2"]
    async with _session() as s:
        for p_len in p_lens:
            for i in range(warmup + measure):
                phase = "warmup" if i < warmup else "measure"
                topo.wait_idle()
                snap0 = topo.snapshot()
                ev = asyncio.Event()
                d_key, p_key = f"e2-d-{i}", f"e2-p-{p_len}-{i}"
                t_ref = time.perf_counter()

                def hit(n):
                    if n >= cfg["inject_after_token"] and not ev.is_set():
                        ev.set()

                d_task = asyncio.create_task(stream_completion(
                    s, topo.proxy_url, prompt_ids(d_key, cfg["d_prompt"]), cfg["d_output"],
                    d_key, on_tokens=hit))
                waiter = asyncio.create_task(ev.wait())
                await asyncio.wait({d_task, waiter}, return_when=asyncio.FIRST_COMPLETED)
                inj, snap1, t_inject_ms = None, None, None
                if ev.is_set():
                    snap1 = await asyncio.to_thread(topo.snapshot)
                    if p_len > 0:
                        t_inject_ms = (time.perf_counter() - t_ref) * 1000.0
                        inj = await stream_completion(s, topo.proxy_url, prompt_ids(p_key, p_len),
                                                      cfg["p_output"], p_key)
                waiter.cancel()
                d = await d_task
                topo.wait_idle()
                snap2 = topo.snapshot()
                row = _base_row(topo, "e2", phase)
                row.update(p_len=p_len, idx=i, d_request_key=d_key,
                           p_request_key=p_key if p_len else None, d=d, inj=inj,
                           t_inject_rel_d_send_ms=t_inject_ms,
                           injection_fired=bool(inj) or p_len == 0,
                           metrics_delta_d_phase=_deltas(snap0, snap1) if snap1 else None,
                           metrics_delta_inj_phase=_deltas(snap1, snap2) if snap1 else None,
                           **({"inj_kv_theory": C.kv_theory(p_len)} if p_len else {}))
                valid = d.get("success") and (p_len == 0 or (inj and inj.get("success")))
                if valid and p_len > 0:
                    inj_first_abs = t_inject_ms + inj["ttft_ms"]
                    after = [t for t in d["token_times_ms"] if t > inj_first_abs]
                    row["d_tokens_after_inj_first_token"] = len(after)
                    valid = len(after) >= E2_MIN_D_TOKENS_AFTER_INJECTION
                row["overlap_valid"] = bool(valid)
                row["success"] = bool(valid)
                C.append_jsonl(raw_path, row)
                print(f"[e2] {topo.arm} p={p_len} {phase} {i} valid={valid}", flush=True)


# --------------------------------------------------------------------------- E3/E4/E5
def build_schedule(shape: str, rate: float, duration_s: float, seed: int,
                   max_requests: int | None = None) -> list[dict]:
    """Poisson arrivals; optional first-N cap is exclusively for online warmup."""
    rng = random.Random(f"{shape}-{rate:.6f}-{seed}")
    shapes = C.FROZEN["e3"]["shapes"]
    mix = C.FROZEN["e3"]["mixture"]
    names, weights = list(mix), [mix[k] for k in mix]
    t, out, i = 0.0, [], 0
    while max_requests is None or i < max_requests:
        t += rng.expovariate(rate)
        if t >= duration_s:
            break
        sname = shape if shape != "mixture" else rng.choices(names, weights)[0]
        inl, outl = shapes[sname]
        out.append({"i": i, "t_offset_s": t, "shape": sname, "prompt_len": inl,
                    "max_tokens": outl, "key": f"e3-{shape}-{seed}-{rate:.6f}-{i}"})
        i += 1
    return out


def schedule_sha(sched) -> str:
    return hashlib.sha256(json.dumps(
        [(x["t_offset_s"], x["prompt_len"], x["max_tokens"]) for x in sched]).encode()).hexdigest()


async def _fire(session, base, sched, t0_perf):
    async def one(x):
        delay = t0_perf + x["t_offset_s"] - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        lag = (time.perf_counter() - t0_perf - x["t_offset_s"]) * 1000.0
        r = await stream_completion(session, base, prompt_ids(x["key"], x["prompt_len"]),
                                    x["max_tokens"], x["key"])
        r.update(sched_i=x["i"], shape=x["shape"], t_offset_s=x["t_offset_s"],
                 dispatch_lag_ms=lag)
        return r
    return await asyncio.gather(*(one(x) for x in sched))


async def _sample_metrics(topo, path, stop: asyncio.Event, t0_perf, run_id):
    """运行期间 1 Hz 采样各实例 running / waiting / KV usage / NIXL 累计量（只记录，不参与判定）。"""
    while not stop.is_set():
        try:
            snap = await asyncio.to_thread(topo.snapshot)
            C.append_jsonl(path, {"run_id": run_id, "t_rel_s": time.perf_counter() - t0_perf,
                                  "snapshot": snap})
        except Exception as e:
            C.append_jsonl(path, {"run_id": run_id, "t_rel_s": time.perf_counter() - t0_perf,
                                  "error": f"{type(e).__name__}: {e}"})
        try:
            await asyncio.wait_for(stop.wait(), timeout=METRICS_SAMPLE_PERIOD_S)
        except asyncio.TimeoutError:
            pass


async def run_online(topo, raw_path, exp, shape, rate, seed, load_frac, rates_sha,
                     duration_s, phase="measure"):
    sched = build_schedule(shape, rate, duration_s, seed)
    run_id = f"{phase}-{shape}-{seed}-{rate:.6f}"
    ts_path = os.path.join(C.METRICS, f"{topo.prefix}_timeseries.jsonl")
    topo.wait_idle()
    before = topo.snapshot()
    async with _session() as s:
        t0 = time.perf_counter()
        t0_unix = time.time()
        stop = asyncio.Event()
        sampler = asyncio.create_task(_sample_metrics(topo, ts_path, stop, t0, run_id))
        results = await _fire(s, topo.proxy_url, sched, t0)
        wall = time.perf_counter() - t0
        stop.set()
        await sampler
    topo.wait_idle(timeout=600)
    after = topo.snapshot()
    for r in results:
        row = _base_row(topo, exp, phase)
        row.update(r, workload_shape=shape, rate=rate, load_frac=load_frac, seed=seed,
                   schedule_sha256=schedule_sha(sched), rates_file_sha256=rates_sha)
        C.append_jsonl(raw_path, row)
    ok = [r for r in results if r["success"]]
    summ = _base_row(topo, exp, f"{phase}_run_summary")
    summ.update(workload_shape=shape, rate=rate, load_frac=load_frac, seed=seed,
                n_offered=len(sched), n_ok=len(ok), n_fail=len(results) - len(ok),
                offered_rate=len(sched) / duration_s, wall_s=wall, t0_unix=t0_unix,
                achieved_rate=len(ok) / wall if wall > 0 else None,
                schedule_sha256=schedule_sha(sched), rates_file_sha256=rates_sha,
                run_id=run_id, timeseries_file=os.path.relpath(ts_path, C.ROOT),
                ttft_p99_ms=pctl([r["ttft_ms"] for r in ok], 99),
                metrics_delta=_deltas(before, after), success=len(ok) == len(results))
    C.append_jsonl(raw_path, summ)
    print(f"[{exp}] {topo.arm} {shape} rate={rate:.3f} seed={seed} ok={len(ok)}/{len(results)}",
          flush=True)
    return summ


async def warmup_online(topo, raw_path, exp, shape):
    sched = build_schedule(shape, 2.0, 1e9, 9999, max_requests=E3_WARMUP_REQUESTS)
    for k, x in enumerate(sched):
        x["t_offset_s"] = 0.25 * k
    async with _session() as s:
        results = await _fire(s, topo.proxy_url, sched, time.perf_counter())
    for r in results:
        row = _base_row(topo, exp, "warmup")
        row.update(r, workload_shape=shape)
        C.append_jsonl(raw_path, row)


async def calibrate(topo, raw_path, exp, shapes, rates_path, warmup, measure):
    """在同 GPU 预算的 colocated 基线上找每个 shape 的 C_cal（规则见 M27_H20_PLAN §E3）。"""
    out = {"arm": topo.arm, "exp": exp, "rule": {
        "start_rate": CAL_START_RATE, "step": CAL_STEP, "horizon_s": CAL_HORIZON_S,
        "rate_denominator": "arrival_horizon_s for both delivered and offered; wall throughput recorded separately",
        "pass_if": f"delivered_per_arrival_window>={CAL_ACHIEVED_FRAC}*offered and n_fail==0 and "
                   f"ttft_p99<={CAL_TTFT_MULT}*unloaded_ttft_median"}, "shapes": {}}
    shapes_cfg = C.FROZEN["e3"]["shapes"]
    base_needed = sorted({b for sh in shapes for b in (shapes_cfg if sh == "mixture" else [sh])})
    unloaded = {}
    async with _session() as s:
        for base in base_needed:
            inl, outl = shapes_cfg[base]
            unl = []
            for i in range(warmup + measure):
                topo.wait_idle()
                key = f"{exp}-unloaded-{base}-{i}"
                r = await stream_completion(s, topo.proxy_url, prompt_ids(key, inl), outl, key)
                row = _base_row(topo, exp, "calibration_unloaded_warmup" if i < warmup
                                else "calibration_unloaded")
                row.update(r, workload_shape=base)
                C.append_jsonl(raw_path, row)
                if i >= warmup and r["success"]:
                    unl.append(r)
            itls = [x for r in unl for x in r["itl_ms"]]
            unloaded[base] = {
                "ttft_median_ms": statistics.median([r["ttft_ms"] for r in unl]) if unl else None,
                "itl_median_ms": statistics.median(itls) if itls else None,
                "n": len(unl)}
    out["unloaded"] = unloaded
    for shape in shapes:
        comps = list(shapes_cfg) if shape == "mixture" else [shape]
        meds = [unloaded[c]["ttft_median_ms"] for c in comps]
        ttft_med = None if any(m is None for m in meds) else max(meds)
        ladder, c_cal, rate = [], None, CAL_START_RATE
        for _ in range(CAL_MAX_STEPS):
            summ = await run_online(topo, raw_path, exp, shape, rate, 0, None, None,
                                    CAL_HORIZON_S, phase="calibration")
            # The Poisson arrival window ends at 30 s, but _fire drains every
            # in-flight request. Divide completed and offered counts by the
            # same frozen arrival horizon; never count normal drain as a miss.
            delivered = C.calibration_delivered_rate(summ["n_ok"], CAL_HORIZON_S)
            passed = (ttft_med is not None and summ["n_fail"] == 0 and
                      delivered >= CAL_ACHIEVED_FRAC * summ["offered_rate"] and
                      (summ["ttft_p99_ms"] or 1e18) <= CAL_TTFT_MULT * ttft_med)
            ladder.append({"rate": rate, "passed": passed, "achieved": delivered,
                           "achieved_wall": summ["achieved_rate"],
                           "offered": summ["offered_rate"], "ttft_p99_ms": summ["ttft_p99_ms"]})
            if not passed:
                break
            c_cal = rate
            rate *= CAL_STEP
        out["shapes"][shape] = {"ttft_threshold_basis_ms": ttft_med,
                                "ladder": ladder, "c_cal": c_cal,
                                "ladder_exhausted": bool(ladder and ladder[-1]["passed"])}
    C.save_json(rates_path, out)
    return out
