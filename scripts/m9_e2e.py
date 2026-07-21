"""M9 · E2E 单变量对照（compiled decode，M9 开 vs 关）。

- 校验开关是否真的生效（读 model_runner._m9_coalesce）。
- decode 单步 latency：batch∈{1,8,64}，warmup 后 ≥30 重复，cuda.sync 夹 perf_counter，报 median/p95/stdev。
- 机制确认：torch.profiler 跑若干 decode step，统计 cudaMemcpyAsync/step（M9 开应比关 −1）。
- 回归：prefill（len 512/2000）单步 latency。

用法（每次独立进程，flag 在引擎 init 时读取）：
  VLLM_LAB_M9_COALESCE=1 ... python scripts/m9_e2e.py --m9 on  --out outputs/m9/e2e_on.json
  VLLM_LAB_M9_COALESCE=0 ... python scripts/m9_e2e.py --m9 off --out outputs/m9/e2e_off.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m6_benchmark as m6
from vllm import SamplingParams


class A:
    pass


def _args_obj():
    a = A(); a.model = "Qwen/Qwen2.5-0.5B-Instruct"; a.max_model_len = 2048
    a.gpu_memory_utilization = 0.6; a.seed = 0; a.max_num_seqs = 64
    a.max_num_batched_tokens = 4096
    return a


def memcpy_per_step(engine, spy, tf, out_dir, tag):
    """torch.profiler 跑 20 个 decode step，数 cudaMemcpyAsync/step。"""
    ids = tf.ids(32); sched = m6.get_scheduler(engine)
    rids = [f"mc-{i}" for i in range(1)]
    for rid in rids:
        engine.add_request(rid, {"prompt_token_ids": ids},
                           SamplingParams(temperature=0.0, max_tokens=100000, ignore_eos=True, seed=0))
    g = 0
    while g < 30:
        so = spy.last
        if so is not None and len(getattr(sched, "running", [])) == 1 and int(so.total_num_scheduled_tokens) == 1:
            break
        engine.step(); g += 1
    for _ in range(10):
        engine.step()
    trace = os.path.join(out_dir, f"_m9_trace_{tag}.json")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(20):
            engine.step()
        torch.cuda.synchronize()
    prof.export_chrome_trace(trace)
    ev = json.load(open(trace))["traceEvents"]
    n_memcpy = sum(1 for e in ev if e.get("name") == "cudaMemcpyAsync")
    engine.abort_request(rids); m6.drain(engine)
    try:
        os.remove(trace)
    except Exception:
        pass
    return round(n_memcpy / 20, 2)


def decode_latency(engine, spy, tf, batch, warmup=10, reps=40):
    ids = tf.ids(32); sched = m6.get_scheduler(engine)
    rids = [f"d-{batch}-{i}" for i in range(batch)]
    for rid in rids:
        engine.add_request(rid, {"prompt_token_ids": ids},
                           SamplingParams(temperature=0.0, max_tokens=100000, ignore_eos=True, seed=0))
    g = 0
    while g < 30:
        so = spy.last
        if so is not None and len(getattr(sched, "running", [])) == batch and int(so.total_num_scheduled_tokens) == batch:
            break
        engine.step(); g += 1
    for _ in range(warmup):
        engine.step()
    walls = []
    for _ in range(reps):
        dt, _ = m6.timed_step(engine); walls.append(dt * 1e3)
    engine.abort_request(rids); m6.drain(engine)
    return {"batch": batch, "median_ms": round(statistics.median(walls), 4),
            "p95_ms": round(sorted(walls)[int(len(walls) * 0.95)], 4),
            "stdev_ms": round(statistics.pstdev(walls), 4),
            "min_ms": round(min(walls), 4), "n": reps}


def prefill_latency(engine, tf, length, warmup=5, reps=15):
    ids = tf.ids(length)
    def one():
        rid = f"pf-{length}-{time.perf_counter_ns()}"
        engine.add_request(rid, {"prompt_token_ids": ids}, SamplingParams(temperature=0.0, max_tokens=1, seed=0))
        dt, _ = m6.timed_step(engine); m6.drain(engine); return dt * 1e3
    for _ in range(warmup):
        one()
    walls = [one() for _ in range(reps)]
    return {"length": length, "median_ms": round(statistics.median(walls), 4),
            "stdev_ms": round(statistics.pstdev(walls), 4), "n": reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m9", choices=["on", "off"], required=True)
    ap.add_argument("--out", default="outputs/m9/e2e.json")
    args = ap.parse_args()
    os.environ["VLLM_LAB_M9_COALESCE"] = "1" if args.m9 == "on" else "0"

    engine = m6.build_engine("compiled", "off", _args_obj())
    spy = m6.ScheduleSpy(m6.get_scheduler(engine))
    tf = m6.TokenFactory(engine)

    # 校验开关生效
    try:
        flag = engine.model_executor.collective_rpc(
            lambda w: bool(getattr(w.model_runner, "_m9_coalesce", False)))
        flag = flag[0] if isinstance(flag, list) else flag
    except Exception as e:  # noqa
        flag = f"probe_err:{e!r}"

    # 正确性：单请求输出 token
    engine.add_request("corr", {"prompt_token_ids": tf.ids(11)},
                       SamplingParams(temperature=0.0, max_tokens=16, seed=0))
    out_ids = []
    while engine.has_unfinished_requests():
        for o in engine.step():
            if getattr(o, "finished", False):
                out_ids = list(o.outputs[0].token_ids)

    mc = memcpy_per_step(engine, spy, tf, os.path.dirname(args.out) or ".", args.m9)
    decode = [decode_latency(engine, spy, tf, b) for b in (1, 8, 64)]
    prefill = [prefill_latency(engine, tf, L) for L in (512, 2000)]
    spy.restore()

    res = {"m9": args.m9, "flag_effective": flag, "output_token_ids": out_ids,
           "cudaMemcpyAsync_per_step": mc, "decode": decode, "prefill": prefill}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"[m9_e2e {args.m9}] flag_effective={flag} memcpy/step={mc}")
    for d in decode:
        print(f"  decode b={d['batch']}: median={d['median_ms']}ms p95={d['p95_ms']} stdev={d['stdev_ms']}")
    print(f"  out_ids[:6]={out_ids[:6]}")
    print(f"[m9_e2e] -> {args.out}")


if __name__ == "__main__":
    main()
