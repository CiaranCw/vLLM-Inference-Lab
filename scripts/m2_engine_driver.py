"""M2 · 手动驱动 V1 LLMEngine 观察 Continuous Batching。

用 `LLMEngine.from_engine_args` 在进程内构建引擎，按「到达表」在指定 step
调用 `add_request` 注入请求，逐 `step()` 驱动，直到全部完成。调度细节由
M1 的 `_lab_trace.py` 钩子（scheduler/schedule）输出到 stderr + JSONL。

场景：
  E1  单请求（基线，验证手动驱动与 M1 一致）
  E2  错峰 2 请求、长度不同（观察 A decode 与 B prefill 同批交错）
  E3  4 请求 + max_num_seqs=2（观察 C/D waiting→running 排队）
  E4  chunked prefill（小 max_num_batched_tokens + 长 prompt，观察 prefill 分步）

用法（采样器在本机 sm_120 必须用原生，见 nvcc_flashinfer_check）：
  VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_COMPONENTS=scheduler VLLM_LAB_TRACE_MAX_STEPS=200 \
  VLLM_LAB_TRACE_JSONL=outputs/m2/E2.jsonl VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m2_engine_driver.py --scenario E2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from vllm import EngineArgs, SamplingParams
from vllm import _lab_trace
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.llm_engine import LLMEngine

# 基础文本块（重复以构造不同长度 prompt；实际 token 数以 trace 的 num_prompt 为准）
_SENT = "大模型推理会把已经算过的 Key 和 Value 缓存起来，避免每一步都重算历史。"


def make_prompt(repeat: int) -> str:
    return (_SENT * repeat).strip()


# 每个场景：engine 配置覆盖 + 到达表 [(arrive_at_step, req_id, prompt, max_tokens)]
#          + block 计划 [(block_at_step, req_key)]（实验性，仅 E5/E5b 用）
def build_scenario(name: str):
    block_plan = []
    if name == "E1":
        cfg = dict(max_num_seqs=2, max_num_batched_tokens=None)
        arrivals = [(0, "R0", make_prompt(1), 8)]
    elif name == "E2":
        cfg = dict(max_num_seqs=2, max_num_batched_tokens=None)
        arrivals = [
            (0, "A_long", make_prompt(4), 24),   # 早到、prompt 较长、多生成
            (3, "B_short", make_prompt(1), 12),  # 晚 3 步到达、prompt 较短
        ]
    elif name == "E3":
        cfg = dict(max_num_seqs=2, max_num_batched_tokens=None)
        arrivals = [
            (0, "A", make_prompt(1), 12),
            (0, "B", make_prompt(1), 12),
            (2, "C", make_prompt(1), 12),  # running 已满(2)，必须排队
            (2, "D", make_prompt(1), 12),
        ]
    elif name == "E4":
        # 小 token 预算 + 长 prompt → chunked prefill 分步
        cfg = dict(max_num_seqs=2, max_num_batched_tokens=64)
        arrivals = [(0, "LONG", make_prompt(12), 4)]
    elif name in ("E5", "E5b"):
        # blocked decode + chunked prefill。E5: 2 槽位（预算争用）；E5b: 1 槽位（槽位争用）。
        # 关闭 prefix caching，避免 D/P 前缀共享干扰 prefill token 数。
        cfg = dict(
            max_num_seqs=(2 if name == "E5" else 1),
            max_num_batched_tokens=64,
            enable_prefix_caching=False,
        )
        arrivals = [
            (0, "D", make_prompt(1), 20),    # 先 prefill 再 decode，max_tokens 足够长
            (2, "P", make_prompt(12), 4),    # 长 prompt，chunked prefill
        ]
        block_plan = [(2, "D")]  # 在 driver step 2（P 加入那步）阻塞 D，仅该步
    else:
        raise ValueError(f"unknown scenario {name}")
    return cfg, arrivals, block_plan


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="vLLM M2 continuous-batching driver")
    p.add_argument("--scenario", required=True,
                   choices=["E1", "E2", "E3", "E4", "E5", "E5b"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.50)
    p.add_argument("--max-num-seqs", type=int, default=None)
    p.add_argument("--max-num-batched-tokens", type=int, default=None)
    p.add_argument("--async-scheduling", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg, arrivals, block_plan = build_scenario(args.scenario)

    # 命令行显式值可覆盖场景默认
    max_num_seqs = args.max_num_seqs or cfg.get("max_num_seqs")
    max_num_batched_tokens = (
        args.max_num_batched_tokens
        if args.max_num_batched_tokens is not None
        else cfg.get("max_num_batched_tokens")
    )

    engine_args = EngineArgs(
        model=args.model,
        dtype="auto",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=1,
        enforce_eager=True,
        seed=args.seed,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        enable_prefix_caching=cfg.get("enable_prefix_caching"),
        async_scheduling=args.async_scheduling,
        disable_log_stats=True,
    )

    print("=" * 80)
    print(f"[m2] scenario={args.scenario} "
          f"max_num_seqs={max_num_seqs} "
          f"max_num_batched_tokens={max_num_batched_tokens} "
          f"async_scheduling={args.async_scheduling}")
    print(f"[m2] arrivals (arrive_at_step, req_id, prompt_len_chars, max_tokens):")
    for a in arrivals:
        print(f"        step {a[0]:>2}  {a[1]:<8}  chars={len(a[2]):<4}  max_tokens={a[3]}")
    print("=" * 80)

    engine = LLMEngine.from_engine_args(
        engine_args,
        usage_context=UsageContext.ENGINE_CONTEXT,
        enable_multiprocessing=False,
    )

    # 按 step 分组的到达表
    by_step: dict[int, list] = {}
    for arrive_at, rid, prompt, max_toks in arrivals:
        by_step.setdefault(arrive_at, []).append((rid, prompt, max_toks))

    total_reqs = len(arrivals)
    finished_ids: list[str] = []
    step_idx = 0
    driver_log = []
    key_to_id: dict[str, str] = {}  # 外部 req_key -> 引擎内部后缀 id（用于精确 block）
    block_by_step: dict[int, list] = {}
    for bstep, bkey in block_plan:
        block_by_step.setdefault(bstep, []).append(bkey)

    while step_idx < args.max_steps:
        # 到点注入（在本 step 的 schedule 之前）
        injected = []
        for rid, prompt, max_toks in by_step.get(step_idx, []):
            internal = engine.add_request(
                request_id=rid,
                prompt=prompt,
                params=SamplingParams(temperature=0.0, max_tokens=max_toks,
                                      seed=args.seed),
            )
            key_to_id[rid] = internal  # add_request 返回带后缀的内部 id
            injected.append(rid)
        if injected:
            print(f"[m2] >>> step {step_idx}: add_request {injected}")

        # 实验性：本 step 需要阻塞哪些请求（在 schedule 前设置，step 后清除）
        blocked_now = set()
        for bkey in block_by_step.get(step_idx, []):
            if bkey in key_to_id:
                blocked_now.add(key_to_id[bkey])
        if blocked_now:
            _lab_trace.set_blocked(blocked_now)
            print(f"[m2] >>> step {step_idx}: BLOCK {sorted(blocked_now)} "
                  f"(需 VLLM_LAB_BLOCK=1 生效)")
        else:
            _lab_trace.clear_blocked()

        # 还有未来到达 或 仍有未完成请求 才继续
        pending_future = any(s > step_idx for s in by_step)
        if not engine.has_unfinished_requests() and not injected and not pending_future:
            _lab_trace.clear_blocked()
            break

        outputs = engine.step()
        _lab_trace.clear_blocked()  # 阻塞仅作用于本 step
        step_finished = [o.request_id for o in outputs if getattr(o, "finished", False)]
        finished_ids.extend(step_finished)
        n_unfinished = engine.get_num_unfinished_requests()

        rec = {
            "step": step_idx,
            "injected": injected,
            "finished_this_step": step_finished,
            "num_unfinished": n_unfinished,
        }
        driver_log.append(rec)
        if injected or step_finished:
            print(f"[m2] step {step_idx}: injected={injected} "
                  f"finished={step_finished} unfinished={n_unfinished}")

        if not engine.has_unfinished_requests() and not pending_future:
            break
        step_idx += 1

    print("=" * 80)
    print(f"[m2] done. total_reqs={total_reqs} finished={sorted(set(finished_ids))} "
          f"driver_steps={len(driver_log)}")
    # 把 driver 侧交叉校验写到 stderr 尾部（便于与 trace JSONL 对照）
    print("[m2] driver_summary=" + json.dumps(driver_log, ensure_ascii=False),
          file=sys.stderr)
    print("=" * 80)


if __name__ == "__main__":
    main()
