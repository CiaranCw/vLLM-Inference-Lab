"""M7 · Profiler 定位瓶颈 —— 用 PyTorch Profiler 拆解单个 engine step 的 CPU / GPU 构成。

目的：验证 M6 留下的两个假设——
  (1) decode / 短 prefill 的“~7ms 固定开销地板”到底是 CPU/launch 开销，还是 GPU 计算？
  (2) 长 prefill 是否真的由 GPU 计算（GEMM/Attention kernel）主导？
并解释 CUDA Graph（compiled 的 FULL/PIECEWISE）为什么能削掉这个地板。

方法（每个 workload×mode）：
  - 复用 M6 的引擎构建 / 精确 token / step 循环基础设施。
  - **干净计时相**（不开 profiler，profiler 有开销）：cuda.synchronize 夹 perf_counter 测 step wall。
  - **profiler 相**：torch.profiler 包住若干个目标 step，取 GPU kernel 总时间、CPU 总时间、
    CUDA kernel 启动次数、按 CUDA/CPU 排序的 top 项，并导出 chrome trace。
  - 关键推断：cpu_gap_ms = 干净 step_wall − gpu_busy_ms ≈ GPU 空等 CPU 下发 kernel 的开销
    （地板假设成立则该 gap 在 eager 下很大、compiled(图回放) 下很小）。

工具说明（实测）：本机 WSL2 下 nsys 无法采集 GPU kernel（CUPTI device trace 不可用），
故以 torch.profiler 为主力（其 CUPTI 通道在本机可正常取 CPU+CUDA），chrome trace 提供时间线/launch gap。

用法（每种 mode 独立进程，见 scripts/m7_run_all.py）：
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python scripts/m7_profile.py --mode eager   --out-dir outputs/m7
  ... --mode compiled ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m6_benchmark as m6  # 复用 build_engine / TokenFactory / ScheduleSpy / timed_step / drain / get_scheduler

from vllm import SamplingParams


def _dev_us(e):
    """取事件的 GPU(device) 时间（μs），兼容不同 torch 版本字段名。"""
    return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0) or 0


def _extract(prof, n_steps: int) -> dict:
    """从 profiler 结果提取每 step 的 CPU/GPU/launch 摘要与 top 项。"""
    ka = prof.key_averages()
    gpu_us = sum(_dev_us(e) for e in ka)
    cpu_us = sum(e.self_cpu_time_total for e in ka)
    # CUDA kernel 启动次数：CPU 侧 cudaLaunchKernel 运行时调用
    launches = 0
    for e in prof.events():
        if e.name in ("cudaLaunchKernel", "cudaLaunchKernelExC"):
            launches += 1
    # top GPU kernels（按 device 时间）
    gpu_rows = sorted(ka, key=_dev_us, reverse=True)
    top_gpu = [{"name": e.key[:60], "cuda_ms_per_step": round(_dev_us(e) / 1000 / n_steps, 4),
                "calls": e.count} for e in gpu_rows if _dev_us(e) > 0][:8]
    # top CPU 算子（按 self cpu 时间）
    cpu_rows = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)
    top_cpu = [{"name": e.key[:60], "cpu_ms_per_step": round(e.self_cpu_time_total / 1000 / n_steps, 4),
                "calls": e.count} for e in cpu_rows][:8]
    return {
        "n_profiled_steps": n_steps,
        "gpu_busy_ms_per_step": round(gpu_us / 1000 / n_steps, 4),
        "cpu_time_ms_per_step": round(cpu_us / 1000 / n_steps, 4),
        "cuda_launches_per_step": round(launches / n_steps, 2),
        "top_gpu_kernels": top_gpu,
        "top_cpu_ops": top_cpu,
    }


def _clean_wall_ms(step_fn, n: int) -> dict:
    """干净计时（不开 profiler）：返回若干 step wall 的中位数等。"""
    import statistics
    walls = []
    for _ in range(n):
        walls.append(step_fn() * 1000)
    return {"median_ms": round(statistics.median(walls), 4),
            "mean_ms": round(statistics.mean(walls), 4),
            "min_ms": round(min(walls), 4), "n": n,
            "samples_ms": [round(w, 3) for w in walls]}


# ============================ decode workload ============================
def profile_decode(engine, spy, tf, mode, batch, out_dir, warmup, n_prof):
    sched = m6.get_scheduler(engine)
    ids = tf.ids(32)
    rids = [f"m7dec-{batch}-{b}" for b in range(batch)]
    for rid in rids:
        engine.add_request(rid, {"prompt_token_ids": ids},
                           SamplingParams(temperature=0.0, max_tokens=1000000, ignore_eos=True, seed=0))
    # 推进到全 decode
    guard = 0
    while guard < 30:
        so = spy.last
        if so is not None and len(getattr(sched, "running", [])) == batch \
                and int(so.total_num_scheduled_tokens) == batch:
            break
        engine.step()
        guard += 1
    # decode warmup
    for _ in range(warmup):
        engine.step()
    seq0 = None
    try:
        seq0 = int(getattr(sched.running[0], "num_computed_tokens", 0))
    except Exception:  # noqa
        pass
    # 干净计时相
    wall = _clean_wall_ms(lambda: m6.timed_step(engine)[0], n_prof)
    # profiler 相
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_prof):
            engine.step()
        torch.cuda.synchronize()
    prof_summary = _extract(prof, n_prof)
    trace_path = os.path.join(out_dir, "traces", f"decode_b{batch}_{mode}.json")
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    prof.export_chrome_trace(trace_path)
    engine.abort_request(rids)
    m6.drain(engine)
    return _record("decode", mode, batch, 32, seq0, wall, prof_summary, trace_path)


# ============================ prefill workload ============================
def profile_prefill(engine, spy, tf, mode, length, out_dir, warmup, n_prof):
    ids = tf.ids(length)

    def one_prefill_step():
        rid = f"m7pf-{length}-{time.perf_counter_ns()}"
        engine.add_request(rid, {"prompt_token_ids": ids},
                           SamplingParams(temperature=0.0, max_tokens=1, seed=0))
        dt, _ = m6.timed_step(engine)
        m6.drain(engine)
        return dt

    for _ in range(warmup):
        one_prefill_step()
    wall = _clean_wall_ms(one_prefill_step, n_prof)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(n_prof):
            rid = f"m7pfp-{length}-{time.perf_counter_ns()}"
            engine.add_request(rid, {"prompt_token_ids": ids},
                               SamplingParams(temperature=0.0, max_tokens=1, seed=0))
            engine.step()
            m6.drain(engine)
        torch.cuda.synchronize()
    prof_summary = _extract(prof, n_prof)
    trace_path = os.path.join(out_dir, "traces", f"prefill_len{length}_{mode}.json")
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    prof.export_chrome_trace(trace_path)
    return _record("prefill", mode, 1, length, length, wall, prof_summary, trace_path)


def _record(workload, mode, batch, prompt_len, seq_len, wall, prof_summary, trace_path):
    gpu = prof_summary["gpu_busy_ms_per_step"]
    rec = {
        "workload": workload, "mode": mode, "batch": batch,
        "prompt_len": prompt_len, "seq_len": seq_len,
        "clean_step_wall": wall,
        "profiler": prof_summary,
        # 关键推断：干净 wall 里 GPU 真正忙碌之外的部分（≈ CPU 下发 + GPU 空等）
        "inferred_cpu_gap_ms": round(wall["median_ms"] - gpu, 4),
        "gpu_busy_fraction": round(gpu / wall["median_ms"], 3) if wall["median_ms"] else None,
        "chrome_trace": trace_path,
    }
    return rec


def parse_args():
    p = argparse.ArgumentParser(description="vLLM M7 profiler (PyTorch Profiler)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--mode", choices=["eager", "compiled"], required=True)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-seqs", type=int, default=64)
    p.add_argument("--max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--n-prof", type=int, default=20)
    p.add_argument("--out-dir", default="outputs/m7")
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 80)
    print(f"[m7] mode={args.mode} vllm={os.path.realpath(m6.vllm.__file__)}")
    print("=" * 80)

    engine = m6.build_engine(args.mode, "off", args)
    spy = m6.ScheduleSpy(m6.get_scheduler(engine))
    tf = m6.TokenFactory(engine)

    records = []
    # workload 1: decode-heavy 小 batch（batch=1），另加 batch=64 作摊销对照
    for B in (1, 64):
        rec = profile_decode(engine, spy, tf, args.mode, B, args.out_dir, args.warmup, args.n_prof)
        records.append(rec)
        print(f"[m7] decode b={B}: wall={rec['clean_step_wall']['median_ms']}ms "
              f"gpu={rec['profiler']['gpu_busy_ms_per_step']}ms "
              f"launches={rec['profiler']['cuda_launches_per_step']} "
              f"gap={rec['inferred_cpu_gap_ms']}ms gpu_frac={rec['gpu_busy_fraction']}")
    # workload 2: prefill-heavy 长输入（len=2000），另加 len=32 作短输入对照
    for L in (32, 2000):
        rec = profile_prefill(engine, spy, tf, args.mode, L, args.out_dir, args.warmup, args.n_prof)
        records.append(rec)
        print(f"[m7] prefill len={L}: wall={rec['clean_step_wall']['median_ms']}ms "
              f"gpu={rec['profiler']['gpu_busy_ms_per_step']}ms "
              f"launches={rec['profiler']['cuda_launches_per_step']} "
              f"gap={rec['inferred_cpu_gap_ms']}ms gpu_frac={rec['gpu_busy_fraction']}")

    spy.restore()
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, f"profile_{args.mode}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"mode": args.mode, "records": records}, f, ensure_ascii=False, indent=2)
    print(f"[m7] wrote -> {out}")


if __name__ == "__main__":
    main()
