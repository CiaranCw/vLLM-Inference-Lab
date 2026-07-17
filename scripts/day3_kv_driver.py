"""Day 3 · 手动驱动 V1 LLMEngine 观察 KV Cache / Block 管理。

复用 Day2 的手动驱动思路，新增 KV 相关旋钮：
  --num-gpu-blocks-override   把 GPU KV 块数强行设小，逼出 preemption
  --enable-prefix-caching     开/关 prefix caching（--no-enable-prefix-caching 关）

配合 vllm-src 中 component=kv 的被动 trace 钩子（get_computed_blocks / allocate_slots /
free / get_new_blocks / free_blocks / evict）与 scheduler 的 preempt 事件观察：
  K1 块分配 + block_table + 跨块边界
  K2 prefix 命中（hash/ref_cnt/块对齐） vs 关闭
  K3 释放 → cached-free → 复用
  K4 KV 不足 → 抢占 + 重算（prefix on/off 对照）

用法示例：
  VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_COMPONENTS=kv,scheduler VLLM_LAB_TRACE_MAX_STEPS=300 \
  VLLM_LAB_TRACE_JSONL=outputs/day3/K1.jsonl VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/day3_kv_driver.py --scenario K1
"""

from __future__ import annotations

import argparse
import json
import sys

from vllm import EngineArgs, SamplingParams
from vllm import _lab_trace
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.request import RequestStatus

_SENT = "大模型推理会把已经算过的 Key 和 Value 缓存起来，避免每一步都重算历史。"


def make_prompt(repeat: int = 1, uniq: str = "") -> str:
    """uniq 非空时前置一段独有内容，破坏跨请求前缀共享（K4 需要各请求互不共享）。"""
    p = _SENT * repeat
    return (uniq + p) if uniq else p


def build_scenario(name: str):
    """返回 (cfg, arrivals)；arrivals=[(arrive_at_step, req_id, prompt, max_tokens)]。"""
    if name == "K1":
        # 单请求：prompt 跨 2 块，decode 越过 32/48 边界 → 观察新块分配
        cfg = dict(max_num_seqs=2, num_gpu_blocks_override=None,
                   enable_prefix_caching=False)
        arrivals = [(0, "R", make_prompt(1), 30)]
    elif name == "K2":
        # prefix 命中：A、B 相同 prompt，B 晚到命中 A 缓存的整块
        cfg = dict(max_num_seqs=2, num_gpu_blocks_override=None,
                   enable_prefix_caching=True)
        arrivals = [
            (0, "A", make_prompt(2), 4),
            (2, "B", make_prompt(2), 4),
        ]
    elif name == "K3":
        # 释放→复用：A 先跑完释放（块带 hash 保留），B 用相同 prompt 命中复用
        cfg = dict(max_num_seqs=2, num_gpu_blocks_override=None,
                   enable_prefix_caching=True)
        arrivals = [
            (0, "A", make_prompt(2), 4),
            (10, "B", make_prompt(2), 4),  # A 跑完后再到
        ]
    elif name == "K4":
        # KV 不足抢占：块数设小 + max_model_len 收小(否则 init 校验要求单请求能放下 max_model_len)
        # + 4 个各自独立的中等上下文请求，decode 增长到组合占用超过总块数 → 抢占
        cfg = dict(max_num_seqs=4, num_gpu_blocks_override=48,
                   enable_prefix_caching=False, max_model_len=512)
        arrivals = [
            (0, "A", make_prompt(6, "甲甲甲甲甲甲甲甲"), 60),
            (0, "B", make_prompt(6, "乙乙乙乙乙乙乙乙"), 60),
            (0, "C", make_prompt(6, "丙丙丙丙丙丙丙丙"), 60),
            (0, "D", make_prompt(6, "丁丁丁丁丁丁丁丁"), 60),
        ]
    elif name == "K5":
        # 由 run_k5 专用控制流驱动；此处仅提供 engine 配置，arrivals 不使用
        cfg = dict(max_num_seqs=4, num_gpu_blocks_override=40,
                   enable_prefix_caching=True, max_model_len=512,
                   scheduling_policy="priority")
        arrivals = []
    else:
        raise ValueError(f"unknown scenario {name}")
    return cfg, arrivals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM Day3 KV/block driver")
    p.add_argument("--scenario", required=True, choices=["K1", "K2", "K3", "K4", "K5"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--num-gpu-blocks-override", type=int, default=None)
    p.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction,
                   default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=400)
    return p.parse_args()


def run_k5(engine, args) -> None:
    """K5：priority 调度使持块多的 D 成为抢占 victim；抢占后 block D 使其块保持
    cached-free；持续加压逼 get_new_blocks 驱逐 D 的块；再放行 D 恢复。全部证据看 trace。"""
    scheduler = engine.engine_core.engine_core.scheduler

    def sp(mt):
        return SamplingParams(temperature=0.0, max_tokens=mt, seed=args.seed)

    # D：低优先级（priority 数字大=最先被抢占）、长 prompt（≈12 整块）
    d_id = engine.add_request("D", prompt=make_prompt(8, "顶顶顶顶"),
                              params=sp(60), priority=10)
    print(f"[k5] added D (priority=10, ~12 blocks) id={d_id}")

    pressure_added = blocked = unblocked = False
    block_step = None
    wave = 0
    step = 0
    while step < args.max_steps:
        # 阶段A：D 建块若干步后加压力 P1,P2（高优先级），逼 KV 耗尽 → D 被抢占
        if not pressure_added and step >= 10:
            engine.add_request("P1", prompt=make_prompt(6, "甲甲甲甲"),
                               params=sp(50), priority=0)
            engine.add_request("P2", prompt=make_prompt(6, "乙乙乙乙"),
                               params=sp(50), priority=0)
            pressure_added = True
            print(f"[k5] step {step}: add pressure P1,P2 (priority=0)")
        # 阶段C：D 被 block 后，渐进加小压力波逼系统取新块 → 驱逐 D 的 cached 块
        if blocked and not unblocked:
            if step % 4 == 0 and wave < 10:
                wave += 1
                engine.add_request(f"W{wave}", prompt=make_prompt(1, f"波{wave}佐"),
                                   params=sp(20), priority=0)
            if block_step is not None and step >= block_step + 14:
                _lab_trace.clear_blocked()
                unblocked = True
                print(f"[k5] step {step}: clear_blocked → 放行 D 恢复")

        if not engine.has_unfinished_requests() and pressure_added and unblocked:
            break
        engine.step()

        # 阶段B：抢占检测（step 之后）→ block D
        if not blocked:
            rd = scheduler.requests.get(d_id)
            if rd is not None and rd.status == RequestStatus.PREEMPTED:
                _lab_trace.set_blocked({d_id})
                blocked = True
                block_step = step
                print(f"[k5] step {step}: D PREEMPTED → set_blocked({{D}})")
        step += 1

    _lab_trace.clear_blocked()
    rd = scheduler.requests.get(d_id)
    print(f"[k5] done step={step} preempt_triggered={blocked} unblocked={unblocked} "
          f"D_final_status={getattr(rd, 'status', None)}")


def main() -> None:
    args = parse_args()
    cfg, arrivals = build_scenario(args.scenario)

    max_num_seqs = args.max_num_seqs or cfg.get("max_num_seqs")
    ngbo = (args.num_gpu_blocks_override
            if args.num_gpu_blocks_override is not None
            else cfg.get("num_gpu_blocks_override"))
    epc = (args.enable_prefix_caching
           if args.enable_prefix_caching is not None
           else cfg.get("enable_prefix_caching"))
    max_model_len = cfg.get("max_model_len") or args.max_model_len

    engine_args = EngineArgs(
        model=args.model,
        dtype="auto",
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,
        seed=args.seed,
        max_num_seqs=max_num_seqs,
        num_gpu_blocks_override=ngbo,
        enable_prefix_caching=epc,
        scheduling_policy=cfg.get("scheduling_policy") or "fcfs",
        async_scheduling=False,
        disable_log_stats=True,
    )

    print("=" * 80)
    print(f"[day3] scenario={args.scenario} max_num_seqs={max_num_seqs} "
          f"num_gpu_blocks_override={ngbo} enable_prefix_caching={epc}")
    for a in arrivals:
        print(f"        step {a[0]:>2}  {a[1]:<4}  prompt_chars={len(a[2]):<4}  "
              f"max_tokens={a[3]}")
    print("=" * 80)

    engine = LLMEngine.from_engine_args(
        engine_args, usage_context=UsageContext.ENGINE_CONTEXT,
        enable_multiprocessing=False,
    )

    if args.scenario == "K5":
        run_k5(engine, args)
        return

    by_step: dict[int, list] = {}
    for arrive_at, rid, prompt, max_toks in arrivals:
        by_step.setdefault(arrive_at, []).append((rid, prompt, max_toks))

    finished_ids, step_idx, driver_log = [], 0, []
    while step_idx < args.max_steps:
        injected = []
        for rid, prompt, max_toks in by_step.get(step_idx, []):
            engine.add_request(
                request_id=rid, prompt=prompt,
                params=SamplingParams(temperature=0.0, max_tokens=max_toks,
                                      seed=args.seed),
            )
            injected.append(rid)
        if injected:
            print(f"[day3] >>> step {step_idx}: add_request {injected}")

        pending_future = any(s > step_idx for s in by_step)
        if not engine.has_unfinished_requests() and not injected and not pending_future:
            break

        outputs = engine.step()
        step_finished = [o.request_id for o in outputs
                         if getattr(o, "finished", False)]
        finished_ids.extend(step_finished)
        if step_finished:
            print(f"[day3] step {step_idx}: finished={step_finished} "
                  f"unfinished={engine.get_num_unfinished_requests()}")
        driver_log.append({"step": step_idx, "injected": injected,
                           "finished": step_finished})

        if not engine.has_unfinished_requests() and not pending_future:
            break
        step_idx += 1

    print("=" * 80)
    print(f"[day3] done. finished={sorted(set(finished_ids))} steps={len(driver_log)}")
    print("[day3] driver_summary=" + json.dumps(driver_log, ensure_ascii=False),
          file=sys.stderr)
    print("=" * 80)


if __name__ == "__main__":
    main()
