"""Day 2 · 读取 scheduler_steps JSONL，生成每场景的 Markdown 时间线表。

输入：由 `_lab_trace.py` 的 VLLM_LAB_TRACE_JSONL 写出的 scheduler 事件
（component=scheduler，event=schedule / update_from_output）。

输出（stdout）：一张 step × request 的相位甘特表 + 每请求生命周期摘要。

用法：
  python scripts/day2_build_timeline.py outputs/day2/E2.jsonl --title "E2 错峰2请求" \
      > outputs/day2/E2_timeline.md
"""

from __future__ import annotations

import argparse
import json


def load_events(path: str):
    schedule_by_step = {}
    update_by_step = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("component") != "scheduler":
                continue
            step = rec["step"]
            if rec["event"] == "schedule":
                schedule_by_step[step] = rec["fields"]
            elif rec["event"] == "update_from_output":
                update_by_step[step] = rec["fields"]
    return schedule_by_step, update_by_step


def phase_char(phase: str) -> str:
    return {"prefill": "P", "decode": "D"}.get(phase, "?")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    sched, upd = load_events(args.jsonl)
    steps = sorted(sched)

    # 按首次出现顺序收集 request 列
    req_order = []
    first_step, last_step, num_prompt = {}, {}, {}
    for s in steps:
        for rid, info in sched[s].get("per_req", {}).items():
            if rid not in first_step:
                first_step[rid] = s
                req_order.append(rid)
            last_step[rid] = s
            num_prompt[rid] = info.get("num_prompt")

    def short(rid: str) -> str:
        return rid.split("-")[0]

    title = args.title or args.jsonl
    print(f"### 时间线 · {title}\n")
    print(f"数据：`{args.jsonl}`；单元格 `P{{n}}`=prefill 调度 n token，"
          f"`D{{n}}`=decode 调度 n token，`·`=该请求本步未被调度。\n")

    # 表头
    cols = ["step"] + [short(r) for r in req_order] + [
        "total_sched", "running", "waiting", "事件"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")

    for s in steps:
        f = sched[s]
        per = f.get("per_req", {})
        row = [str(s)]
        for rid in req_order:
            if rid in per:
                info = per[rid]
                row.append(f"{phase_char(info.get('phase'))}{info.get('scheduled')}")
            else:
                row.append("·")
        row.append(str(f.get("total_scheduled")))
        row.append(str(f.get("running")))
        row.append(str(f.get("waiting")))
        # 事件：进入 / 退出
        ev = []
        for rid in req_order:
            if first_step.get(rid) == s:
                ev.append(f"＋{short(rid)}")
            if last_step.get(rid) == s:
                ev.append(f"✓{short(rid)}")
        row.append(" ".join(ev))
        print("| " + " | ".join(row) + " |")

    # 每请求生命周期摘要
    print("\n#### 请求生命周期摘要\n")
    print("| request | num_prompt | 首次调度 step | 末次调度 step | 跨 step 数 |")
    print("|---|---|---|---|---|")
    for rid in req_order:
        span = last_step[rid] - first_step[rid] + 1
        print(f"| {short(rid)} | {num_prompt.get(rid)} | {first_step[rid]} "
              f"| {last_step[rid]} | {span} |")


if __name__ == "__main__":
    main()
