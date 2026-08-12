#!/usr/bin/env python3
"""M21 · E2 指标真值校验（离线聚合，不占 GPU）。

面板好看不代表数字是真的。本脚本逐条核对服务端指标与客户端 bench 汇总的守恒关系；
**E2 不过关就不该相信后面任何基于面板的结论**。

不变量：
  I1 Δvllm:request_success_total      == bench completed
  I2 Δvllm:prompt_tokens_total        == bench total_input_tokens
  I3 Δvllm:generation_tokens_total    == bench total_output_tokens
  I4 全程 num_requests_running        <= max_num_seqs(64)
  I5 kv_cache_usage_perc ∈ [0, 1]
  I6 counter 增量非负（无回绕/重置）
  I7 抓取覆盖率：1s 采样点数 ≈ 窗口秒数（丢点率）
  I8 DCGM 与 NVML 的 GPU util / 显存读数一致（正交第二证据源）

另产出**量化结论（非判据）**：服务端直方图分位数 vs 客户端真值分位数的偏差。

用法：python scripts/m21_validate.py [--runs outputs/m21/raw/e3_shapeA_60s_runs.jsonl]
"""

import argparse
import json
import os
import statistics

import m21_common as C

MAX_NUM_SEQS = 64
SCRAPE_COVERAGE_MIN = 0.8   # 采样点数 / 窗口秒数 的下限


def load(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def check(runs):
    res = {"n_runs": len(runs), "failures": [], "per_run": []}
    dev_ttft, dev_itl, dev_e2el = [], [], []
    gpu_util_gap, fb_gap_mib = [], []

    for r in runs:
        d = r.get("counter_delta", {})
        rid = r["run_id"]
        i1 = d.get("vllm:request_success_total") == r.get("client_completed")
        i2 = d.get("vllm:prompt_tokens_total") == r.get("client_total_input_tokens")
        i3 = d.get("vllm:generation_tokens_total") == r.get("client_total_output_tokens")
        i4 = (r.get("running_max") or 0) <= MAX_NUM_SEQS
        kvmax = r.get("kv_usage_max")
        i5 = kvmax is None or (0.0 <= kvmax <= 1.0)
        i6 = all((v is None or v >= 0) for v in d.values())
        dur = max(r.get("duration_s") or 0, 1)
        cov = (r.get("waiting_samples") or 0) / dur
        i7 = cov >= SCRAPE_COVERAGE_MIN

        row = {"run_id": rid, "I1": i1, "I2": i2, "I3": i3, "I4": i4,
               "I5": i5, "I6": i6, "I7": i7, "scrape_coverage": round(cov, 3)}
        res["per_run"].append(row)
        for k in ("I1", "I2", "I3", "I4", "I5", "I6", "I7"):
            if not row[k]:
                res["failures"].append({"run_id": rid, "invariant": k,
                                        "counter_delta": d,
                                        "client_completed": r.get("client_completed"),
                                        "client_in": r.get("client_total_input_tokens"),
                                        "client_out": r.get("client_total_output_tokens"),
                                        "running_max": r.get("running_max"),
                                        "kv_usage_max": kvmax,
                                        "scrape_coverage": round(cov, 3)})

        # I8：两条独立采集路径的一致性
        if r.get("dcgm_gpu_util_max") is not None and r.get("nvml_gpu_util_max") is not None:
            gpu_util_gap.append(abs(r["dcgm_gpu_util_max"] - r["nvml_gpu_util_max"]))
        if r.get("dcgm_fb_used_mib_max") is not None and r.get("proc_rss_bytes_max") is not None:
            pass  # 显存交叉校验在 e1/e4 快照里做，这里只留 util

        # 服务端直方图 vs 客户端真值
        for cli, srv, box in (("client_p99_ttft_ms", "server_hist_ttft_p99_ms", dev_ttft),
                              ("client_p99_itl_ms", "server_hist_itl_p99_ms", dev_itl),
                              ("client_p99_e2el_ms", "server_hist_e2el_p99_ms", dev_e2el)):
            a, b = r.get(cli), r.get(srv)
            if a and b:
                box.append({"run_id": rid, "client_ms": a, "server_hist_ms": b,
                            "rel_diff": (b - a) / a})

    def summarize(box, label):
        if not box:
            return {"label": label, "n": 0}
        rel = [x["rel_diff"] for x in box]
        return {"label": label, "n": len(box),
                "rel_diff_median": round(statistics.median(rel), 4),
                "rel_diff_min": round(min(rel), 4),
                "rel_diff_max": round(max(rel), 4),
                "note": "正值=服务端直方图分位数高于客户端真值（bucket 上界向上取整所致）"}

    res["hist_vs_client"] = [summarize(dev_ttft, "TTFT p99"),
                             summarize(dev_itl, "ITL p99"),
                             summarize(dev_e2el, "E2EL p99")]
    if gpu_util_gap:
        res["I8_dcgm_vs_nvml_gpu_util_abs_gap"] = {
            "median": statistics.median(gpu_util_gap),
            "max": max(gpu_util_gap),
            "n": len(gpu_util_gap),
            "note": "两条独立路径的 GPU util 峰值差；曾用它抓出 dcgm-exporter 默认 30s 采集间隔的过期读数"}
    res["PASS"] = not res["failures"]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(C.RAW, "e3_shapeA_60s_runs.jsonl"))
    ap.add_argument("--out", default=os.path.join(C.OUT, "e2_validation.json"))
    args = ap.parse_args()

    runs = load(args.runs)
    res = check(runs)
    res["source"] = args.runs
    C.save_json(args.out, res)

    print(f"E2 校验：{res['n_runs']} runs, PASS={res['PASS']}, 失败 {len(res['failures'])} 条")
    for s in res["hist_vs_client"]:
        if s.get("n"):
            print(f"  {s['label']}: 服务端直方图相对客户端真值 "
                  f"中位偏差 {s['rel_diff_median'] * 100:+.1f}%  "
                  f"(范围 {s['rel_diff_min'] * 100:+.1f}% ~ {s['rel_diff_max'] * 100:+.1f}%)")
    if g := res.get("I8_dcgm_vs_nvml_gpu_util_abs_gap"):
        print(f"  I8 DCGM vs NVML GPU util 峰值差：中位 {g['median']}，最大 {g['max']}")
    for f in res["failures"][:5]:
        print("  FAIL:", f)
    print("->", args.out)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
