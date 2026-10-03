#!/usr/bin/env python3
"""M27 · 日志 / 指标证据汇总（离线，不需 GPU、不 import vllm）。

    python scripts/m27_trace_analyze.py --tag h20_formal

扫描 logs/m27/<tag>/*.log 与 outputs/m27/metrics/<tag>_*_startup.prom，输出
outputs/m27/<tag>_log_evidence.json —— 作为「传输成功 / 无失败 / 无隐式重算」的
**第二证据源**（第一证据源是 D 侧 Prometheus 计数增量）：
  - KV layout（应恒为 HND）/ attention backend / GPU KV cache tokens
  - NIXL compatibility check passed / hash mismatch
  - "Failing N request(s) due to KV load failure" / "Recovered from KV load failure"
  - Traceback 计数
  - vllm:cache_config_info 的 num_gpu_blocks / block_size（启动快照）
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402
from m27_launch import parse_log_evidence  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    a = p.parse_args()
    logs = sorted(glob.glob(os.path.join(C.LOGS, a.tag, "*.log")))
    if not logs:
        print(f"[trace] PENDING：logs/m27/{a.tag}/ 无日志（实验未执行）")
        return 0
    out = {"tag": a.tag, "instances": {}, "startup_metrics": {}}
    totals = {"kv_load_fail_events": 0, "kv_load_recovered_events": 0, "nixl_compat_mismatch": 0,
              "tracebacks": 0, "non_hnd_layout_instances": []}
    for lp in logs:
        name = os.path.basename(lp)[:-4]
        if name.endswith("_proxy"):
            continue
        ev = parse_log_evidence(lp)
        out["instances"][name] = ev
        for k in ("kv_load_fail_events", "kv_load_recovered_events", "nixl_compat_mismatch",
                  "tracebacks"):
            totals[k] += ev.get(k) or 0
        if ev.get("kv_layout_log") != ["HND"]:
            totals["non_hnd_layout_instances"].append(name)
    for pp in sorted(glob.glob(os.path.join(C.METRICS, f"{a.tag}_*_startup.prom"))):
        rows = C.parse_prom(open(pp).read())
        info = [lab for n, lab, _ in rows if n == "vllm:cache_config_info"]
        out["startup_metrics"][os.path.basename(pp)] = {
            "num_gpu_blocks": (info[0].get("num_gpu_blocks") if info else None),
            "block_size": (info[0].get("block_size") if info else None)}
    out["totals"] = totals
    path = C.save_json(os.path.join(C.OUT, f"{a.tag}_log_evidence.json"), out)
    print(f"[trace] {len(out['instances'])} instance logs -> {os.path.relpath(path, C.ROOT)}")
    print(f"[trace] totals: {totals}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
