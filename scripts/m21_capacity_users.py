#!/usr/bin/env python3
"""M21 · E5 「一张卡能承载多少用户」—— 纯离线推导（不占 GPU、不 import vllm）。

方法：**Interactive Response Time Law**（闭环交互式系统的 population 估计），
可由 Little's law 推出，但**必须和「系统内并发请求数」区分开**：

    closed_loop_user_population  N = X × (R + Z)
    average_in_system_requests   L = X × R          ← 这才是「同时在服务器里的请求数」
    estimated_thinking_users     T = X × Z          ← 正在思考、**不在**服务系统内

其中 X = 稳定吞吐（本模块取 SLO 达标的最大 offered QPS），R = 服务响应时间（E2EL），
Z = 用户思考时间。

**语义红线**：`X × (R + Z)` 得到的是**闭环用户总数**（含正在等响应的 + 正在思考的），
**不能**读作「同时有这么多请求在服务器里执行」——后者是 `X × R`，量级完全不同。
（本脚本首版把前者命名为 `estimated_concurrent_users`，属于误导性命名，已修正。）

**必须显式声明的假设**（否则这个数字毫无意义）：
  A1 每个用户串行发请求：拿到上一条完整回答后思考 think_time 秒再发下一条。
  A2 请求形状固定为本实验的 Shape（in256/out128），真实用户不会这么整齐。
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
        "method": ("Interactive Response Time Law（闭环 population）：N = X × (R + Z)；"
                   "由 Little's law 推出。X=稳定吞吐, R=E2EL 服务响应时间, Z=用户思考时间"),
        "quantity_definitions": {
            "average_in_system_requests": "L = X × R —— 平均**在服务系统内**的请求数",
            "estimated_thinking_users": "T = X × Z —— 正在思考、**不在**服务系统内的用户数",
            "estimated_closed_loop_users": "N = X × (R + Z) = L + T —— 闭环**用户总数**",
            "WARNING": ("N 是用户 population，**不能**读作「同时有 N 个请求在服务器里执行」；"
                        "后者是 L。首版把 N 命名为 estimated_concurrent_users，属误导，已修正"),
        },
        "assumptions": {
            "A1": "每用户串行：收到完整回答后思考 think_time 秒再发下一条",
            "A2": "请求形状固定 Shape A (in256/out128)，真实流量不会这么整齐",
            "A3": "SLO = M12 冻结口径 P99 TTFT ≤ 300ms 且 success ≥ 0.99；容量取 3-seed worst 仍达标的最大档",
            "A4": "边界：单卡 RTX 5080 / WSL2 / Qwen2.5-0.5B-Instruct / enforce_eager / 单副本 / 监控栈同机运行",
        },
        "ladder": table,
        "qps_stable_req_s": qps_stable,
        "first_qps_over_slo": first_over,
        "capacity_notation": (
            {
                "last_measured_passing_point_req_s": qps_stable,
                "first_measured_failing_point_req_s": first_over,
                "knee_localized_to": f"({qps_stable}, {first_over}]",
                "knee_meaning": "在测试网格上，拐点（从达标翻转为不达标）落在这个半开区间内",
                "unknown_true_capacity_C": f"{qps_stable} <= C < {first_over}  即 C ∈ [{qps_stable}, {first_over})",
                "C_meaning": ("C = 未知的真实最大 SLO 达标容量。已知 24 达标故 C ≥ 24；已知 32 不达标故 C < 32。"
                              "两种记号含义不同，不可互换"),
                "grid_step_req_s": 8,
            } if qps_stable and first_over else {"note": "网格未同时给出上下界"}),
    }

    if qps_stable is not None:
        e2el_ms = next(r["median_e2el_ms"] for r in table if r["offered_qps"] == qps_stable)
        out["e2el_at_qps_stable_ms"] = e2el_ms
        r_s = (e2el_ms or 0) / 1000.0
        out["average_in_system_requests"] = round(qps_stable * r_s, 1)
        est = []
        for t in THINK_TIMES_S:
            cycle_s = r_s + t
            est.append({"think_time_s": t,
                        "cycle_s_R_plus_Z": round(cycle_s, 3),
                        "estimated_thinking_users": round(qps_stable * t, 1),
                        "estimated_closed_loop_users": round(qps_stable * cycle_s, 1)})
        out["user_population_sensitivity"] = est
        out["headline"] = (
            f"在 P99 TTFT ≤ 300ms 的 SLO 下，本机本配置最大稳定 offered QPS = {qps_stable} req/s，"
            f"平均**在系统内的请求数** ≈ {out['average_in_system_requests']}；"
            f"若按闭环交互模型（思考时间 {THINK_TIMES_S[0]}–{THINK_TIMES_S[-1]}s）折算，"
            f"可支撑的**闭环用户总数**约 {est[0]['estimated_closed_loop_users']}–"
            f"{est[-1]['estimated_closed_loop_users']} 人（含正在思考、不在系统内的用户）")

    path = C.save_json(os.path.join(C.OUT, "e5_user_capacity.json"), out)
    print(json.dumps({k: out[k] for k in ("qps_stable_req_s", "first_qps_over_slo",
                                          "e2el_at_qps_stable_ms", "average_in_system_requests",
                                          "user_population_sensitivity", "capacity_notation",
                                          "headline") if k in out},
                     ensure_ascii=False, indent=2))
    print("->", path)


if __name__ == "__main__":
    raise SystemExit(main())
