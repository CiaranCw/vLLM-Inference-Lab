#!/usr/bin/env python3
"""M24 P2 · Prefill CP 负例。预期建引擎失败（本版本无 supports_pcp=True）。

成功建起来 = 版本漂移，记入 json 供调查，不当作性能成功。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model", required=True)
    a = p.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    out = {
        "tag": a.tag,
        "expected": "engine_init_fails_supports_pcp",
        "model": os.path.abspath(a.model),
        "ok_as_negative": False,
        "built_engine": False,
        "error_type": None,
        "error": None,
        "traceback": None,
    }
    t0 = time.perf_counter()
    try:
        from vllm import LLMEngine
        from vllm.engine.arg_utils import EngineArgs
        from vllm.usage.usage_lib import UsageContext

        args = EngineArgs(
            model=a.model,
            tensor_parallel_size=4,
            prefill_context_parallel_size=2,
            decode_context_parallel_size=1,
            max_model_len=2048,
            gpu_memory_utilization=0.4,
            enforce_eager=True,
            async_scheduling=False,
            enable_prefix_caching=False,
            seed=0,
        )
        LLMEngine.from_engine_args(
            args,
            usage_context=UsageContext.ENGINE_CONTEXT,
            enable_multiprocessing=False,
        )
        out["built_engine"] = True
        out["error"] = "UNEXPECTED_SUCCESS: PCP=2 建引擎成功，需要调查版本漂移"
    except Exception as e:  # noqa: BLE001
        out["error_type"] = type(e).__name__
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()
        text = (out["error"] or "") + (out["traceback"] or "")
        # 本树 PCP 乘进 world_size：4 卡 TP=4 × PCP=2 → world_size=8，
        # 会在 ParallelConfig 校验处先被拒，到不了 supports_pcp 断言。
        # 两种失败都是有效负例（引擎未建成）；成功建起来才是版本漂移。
        out["ok_as_negative"] = (
            "supports_pcp" in text
            or "PCP" in text
            or "prefill context" in text.lower()
            or "World size" in text
            or "world_size" in text.lower()
        )
    out["elapsed_s"] = round(time.perf_counter() - t0, 3)
    dest = os.path.join(ROOT, "outputs", "m24", "raw", f"{a.tag}_pcp_negative.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: out[k] for k in out if k != "traceback"}, ensure_ascii=False, indent=2))
    print(f"wrote {dest}")
    # 负例「符合预期失败」用退出码 0，方便编排器继续
    return 0 if (out["ok_as_negative"] or not out["built_engine"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
