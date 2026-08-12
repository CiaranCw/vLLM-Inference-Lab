#!/usr/bin/env python3
"""M21 · E5 「一张卡能承载多少用户」—— 纯离线推导（不占 GPU、不 import vllm）。

方法：Little's law。系统里的并发用户数
    N = λ × R
其中 λ = 每用户请求速率之和 = QPS，R = 一次「用户可感知的往返周期」= 服务时间 + 思考时间。
反解得：**可承载用户数 N_users = QPS_stable × (E2EL + think_time)**。

**必须显式声明的假设**（否则这个数字毫无意义）：
  A1 每个用户串行发请求：拿到上一条完整回答后思考 think_time 秒再发下一条。
  A2 请求形状固定为本实验的 Shape（in256/out128 或 in1024/out32），真实用户不会这么整齐。
  A3 SLO 采用 M12 冻结口径（P99 TTFT ≤ 300ms），容量取「3 seed worst-case 仍达标」的最大档。
  A4 结论仅限：单卡 RTX 5080 / WSL2 / Qwen2.5-0.5B-Instruct / enforce_eager / 单副本。
     换模型、换 shape、开 CUDA Graph 都会变，**不外推**。

输出区间而非单点：think_time 取 5 / 15 / 30 s 三档做敏感性。

用法：python scripts/m21_capacity_users.py
"""

import json
import os
import statistics

import m21_common as C

THINK_TIMES_S = [5, 15, 30]
LADDER_FILE = os.path.join(C.RAW, "e3_shapeA_60s_runs.jsonl")


def load_runs(path):
    if not os.path.exists(path):
        raise SystemExit(f"缺少 {path}，请先跑 m21_capacity_sweep.py")
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def by_qps(runs):
    d = {}
    for r in runs:
        d.setdefault(r["offered_qps"], []).append(r)
    return dict(sorted(d.items()))


def main():
    runs = load_runs(LADDER_FILE)
    grouped = by_qps(runs)

    table = []
    for qps, rs in grouped.items():
        p99s = [r["client_p99_ttft_ms"] for r in rs if r.get("client_p99_ttft_ms") is not None]
        e2els = [r["client_median_e2el_ms"] for r in rs
                 if r.get("client_median_e2el_ms") is not None]
        succ = [r.get("success_rate", 0) for r in rs]
        row = {
            "offered_qps": qps,
            "seeds": len(rs),
            "worst_seed_p99_ttft_ms": max(p99s) if p99s else None,
            "median_seed_p99_ttft_ms": statistics.median(p99s) if p99s else None,
            "median_e2el_ms": statistics.median(e2els) if e2els else None,
            "achieved_qps_mean": statistics.mean(
                [r["client_request_throughput"] for r in rs
                 if r.get("client_request_throughput") is not None]) if rs else None,
            "min_success_rate": min(succ) if succ else None,
            "waiting_max": max([r.get("waiting_max") or 0 for r in rs]),
        }
        row["slo_pass_all_seeds"] = bool(
            row["worst_seed_p99_ttft_ms"] is not None
            and row["worst_seed_p99_ttft_ms"] <= C.SLO_TTFT_P99_MS
            and (row["min_success_rate"] or 0) >= 0.99)
        table.append(row)

    stable = [r for r in table if r["slo_pass_all_seeds"]]
    qps_stable = max([r["offered_qps"] for r in stable]) if stable else None
    over = [r["offered_qps"] for r in table if not r["slo_pass_all_seeds"]]
    first_over = min(over) if over else None

    out = {
        "method": "Little's law: N_users = QPS_stable × (E2EL + think_time)",
        "assumptions": {
            "A1": "每用户串行：收到完整回答后思考 think_time 秒再发下一条",
            "A2": "请求形状固定 Shape A (in256/out128)，真实流量不会这么整齐",
            "A3": "SLO = M12 冻结口径 P99 TTFT ≤ 300ms 且 success ≥ 0.99；容量取 3-seed worst 仍达标的最大档",
            "A4": "边界：单卡 RTX 5080 / WSL2 / Qwen2.5-0.5B-Instruct / enforce_eager / 单副本 / 监控栈同机运行",
        },
        "ladder": table,
        "qps_stable_req_s": qps_stable,
        "first_qps_over_slo": first_over,
        "capacity_interval_note": (
            f"容量落在 ({qps_stable}, {first_over}] 之间，网格步长决定分辨率；"
            "不是系统精确极限" if qps_stable and first_over else "网格未同时给出上下界"),
    }

    if qps_stable is not None:
        e2el_ms = next(r["median_e2el_ms"] for r in table if r["offered_qps"] == qps_stable)
        out["e2el_at_qps_stable_ms"] = e2el_ms
        est = []
        for t in THINK_TIMES_S:
            cycle_s = (e2el_ms or 0) / 1000.0 + t
            est.append({"think_time_s": t,
                        "cycle_s": round(cycle_s, 3),
                        "estimated_concurrent_users": round(qps_stable * cycle_s, 1)})
        out["user_capacity_sensitivity"] = est
        out["headline"] = (
            f"在 P99 TTFT ≤ 300ms 的 SLO 下，本机本配置最大稳定 offered QPS = {qps_stable} req/s；"
            f"按思考时间 {THINK_TIMES_S[0]}–{THINK_TIMES_S[-1]}s 折算，可承载并发用户约 "
            f"{est[0]['estimated_concurrent_users']}–{est[-1]['estimated_concurrent_users']} 人")

    path = C.save_json(os.path.join(C.OUT, "e5_user_capacity.json"), out)
    print(json.dumps({k: out[k] for k in ("qps_stable_req_s", "first_qps_over_slo",
                                          "e2el_at_qps_stable_ms", "user_capacity_sensitivity",
                                          "headline") if k in out},
                     ensure_ascii=False, indent=2))
    print("->", path)


if __name__ == "__main__":
    raise SystemExit(main())
