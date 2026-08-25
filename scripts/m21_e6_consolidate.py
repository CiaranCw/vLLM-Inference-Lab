#!/usr/bin/env python3
"""M21 · E6 演练结果合并（纯离线 derived 生成，不重跑任何故障注入）。

为什么单独成脚本：`outputs/m21/e6_drills.json` 是 derived artifact，早期是手工拼的，
无法复现。本脚本从 append-only 的 raw jsonl 重建它，并施加两处确定性修正：

1. 单位修正（2026-08-12 audit）：`m21_fault_drills.py` 首版把
   `histogram_quantile(... vllm:time_to_first_token_seconds_bucket ...)` 的返回值
   （单位=秒）直接写进了 `server_hist_ttft_p99_ms_during_probe`。
   raw 里的 19.5675 实际是 19.5675 s ≈ 19567.5 ms，不是 19.57ms。
   这里按 ×1000 换算补出 `_ms`，并保留 `_s` 原值与 provenance 说明。
   raw 文件一个字节都不改。
2. 术语修正：`oom` 场景实际注入的是启动期显存预算校验失败
   （EngineCore 拒绝初始化），不是运行中的 CUDA allocator OOM。
   场景 key 保留 `oom` 以兼容既有 artifact，另加 `display_name=startup_memory_rejection`。

用法：python scripts/m21_e6_consolidate.py
"""

import json
import os
import time

import m21_common as C

RAW_FILES = ["e6_drills_attempt1.jsonl", "e6_drills.jsonl"]
ORDER = ["worker_exit", "oom", "request_timeout", "preemption"]

DISPLAY_NAME = {
    "worker_exit": "worker_exit",
    "oom": "startup_memory_rejection",
    "request_timeout": "request_timeout",
    "preemption": "preemption",
}

SCOPE_NOTE = {
    "oom": ("实际注入 = 启动期显存预算校验失败（vLLM 主动拒绝初始化，报 "
            "`ValueError: Free memory on device cuda:0 ... is less than desired "
            "gpu_memory_utilization`）。**本模块没有注入 runtime CUDA allocator OOM**"
            "（`torch.cuda.OutOfMemoryError`），那属于 scope boundary / optional extension。"),
}


def load_raw():
    recs = []
    for f in RAW_FILES:
        p = os.path.join(C.RAW, f)
        if not os.path.exists(p):
            continue
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                r["_source_file"] = f
                recs.append(r)
    return recs


def fix_units(rec):
    """把误存为秒的 *_ms 字段修正过来；raw 不动，只改 derived 副本。"""
    key_ms = "server_hist_ttft_p99_ms_during_probe"
    if key_ms in rec and "server_hist_ttft_p99_s_during_probe" not in rec:
        seconds = rec[key_ms]
        rec["server_hist_ttft_p99_s_during_probe"] = seconds
        rec[key_ms] = None if seconds is None else seconds * 1000.0
        rec["_unit_correction"] = (
            f"raw 中 {key_ms}={seconds} 实为**秒**（PromQL 源指标 "
            "vllm:time_to_first_token_seconds，脚本首版漏了 ×1000）；"
            f"此处 derived 值已换算为 {None if seconds is None else seconds * 1000.0} ms。"
            "raw 文件未修改。")
    return rec


def main():
    recs = load_raw()
    if not recs:
        raise SystemExit("没有找到 E6 raw 记录")

    final = {}
    for r in recs:
        d = r["drill"]
        if d not in final:
            final[d] = r
        elif r.get("PASS") or not final[d].get("PASS"):
            final[d] = r  # 取最新的成功记录；都失败时取最新一次

    drills = []
    for k in ORDER:
        if k not in final:
            continue
        rec = fix_units(dict(final[k]))
        rec["display_name"] = DISPLAY_NAME[k]
        if k in SCOPE_NOTE:
            rec["scope_note"] = SCOPE_NOTE[k]
        drills.append(rec)

    out = {
        "generated_at": time.strftime("%F %T%z"),
        "generated_by": "scripts/m21_e6_consolidate.py（离线，从 raw jsonl 重建）",
        "note": ("worker_exit / startup_memory_rejection 为 2026-08-12 设备重启后重跑"
                 "（含子进程 PID 记录与孤儿清理）；request_timeout / preemption 为 2026-08-11 首轮。"
                 "早期未通过的尝试保留在 outputs/m21/e6_drills_attempt*.json 与 "
                 "raw/e6_drills_attempt1.jsonl。"),
        "corrections_applied": [
            "单位：server_hist_ttft_p99_ms_during_probe 由秒换算为毫秒（见各记录的 _unit_correction）",
            "术语：oom → display_name=startup_memory_rejection（非 runtime CUDA allocator OOM）",
        ],
        "drills": drills,
    }
    path = C.save_json(os.path.join(C.OUT, "e6_drills.json"), out)
    for d in drills:
        extra = ""
        if "server_hist_ttft_p99_ms_during_probe" in d:
            extra = (f" server_hist_ttft_p99 = {d['server_hist_ttft_p99_s_during_probe']:.2f}s "
                     f"= {d['server_hist_ttft_p99_ms_during_probe']:.0f}ms")
        print(f"  {d['display_name']:<28} PASS={d.get('PASS')}{extra}")
    print("->", path)


if __name__ == "__main__":
    raise SystemExit(main())
