#!/usr/bin/env python3
"""M27 · 离线汇总：raw JSONL → 派生 CSV（不需 GPU、不 import vllm）。

    python scripts/m27_summarize.py --tag h20_formal

口径（冻结于 notes/m27/M27_H20_PLAN.md §统计）：
  - 只用 phase=measure 且 success=True；warmup / calibration / fault_probe 一律排除。
  - E1/E2 小样本：median difference 的 Bootstrap 95% CI（random.Random(0)，10000 次，与 M17 同算法）。
  - E3/E4 尾部：以**请求为簇**的 Poisson bootstrap（numpy default_rng(0)，10000 次）求
    p99 差值的 95% CI —— ITL 在请求内相关，不能把 chunk 当独立样本。
  - 缺 raw 的实验打印 PENDING 并跳过，不写空 CSV、不写占位数字。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import random
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402

BOOT = C.FROZEN["bootstrap"]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def raws(tag, exp):
    out = {}
    for p in sorted(glob.glob(os.path.join(C.RAW, f"{tag}_{exp}_*.jsonl"))):
        if p.endswith("_proxy.jsonl"):
            continue
        arm = os.path.basename(p)[len(f"{tag}_{exp}_"):-len(".jsonl")]
        out[arm] = load_jsonl(p)
    return out


def proxy_index(tag, exp, arm):
    p = os.path.join(C.RAW, f"{tag}_{exp}_{arm}_proxy.jsonl")
    if not os.path.exists(p):
        return {}
    return {r["proxy_request_id"]: r for r in load_jsonl(p) if r.get("proxy_request_id")}


def measure(rows):
    return [r for r in rows if r.get("phase") == "measure" and r.get("success") is True]


def pctl(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * q / 100.0
    f, c = math.floor(k), math.ceil(k)
    return s[f] + (s[c] - s[f]) * (k - f)


def stats(xs, prefix):
    xs = [x for x in xs if x is not None]
    return {f"{prefix}_n": len(xs),
            f"{prefix}_median": statistics.median(xs) if xs else None,
            f"{prefix}_stdev": statistics.stdev(xs) if len(xs) > 1 else None,
            f"{prefix}_p95": pctl(xs, 95), f"{prefix}_p99": pctl(xs, 99),
            f"{prefix}_max": max(xs) if xs else None}


def boot_median_diff(a, b, iters=BOOT["iters"], seed=BOOT["seed"]):
    """与 check_invariants.boot_ci_median_diff / M17 同算法同种子。"""
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.median(ra) - statistics.median(rb))
    diffs.sort()
    return (statistics.median(a) - statistics.median(b),
            diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1])


def cluster_boot_quantile_diff(clusters_a, clusters_b, q, iters=BOOT["iters"], seed=BOOT["seed"]):
    """请求为簇的 Poisson bootstrap：每次给每个请求一个 Poisson(1) 权重，求加权分位数差。"""
    import numpy as np

    rng = np.random.default_rng(seed)

    def prep(clusters):
        vals = np.concatenate([np.asarray(c, float) for c in clusters if len(c)])
        rid = np.concatenate([np.full(len(c), i) for i, c in enumerate(clusters) if len(c)])
        order = np.argsort(vals)
        return vals[order], rid[order], len(clusters)

    def wq(vals, rid, w, q):
        ww = w[:, rid]
        cs = np.cumsum(ww, axis=1)
        tot = cs[:, -1:]
        idx = (cs < q / 100.0 * tot).sum(axis=1)
        return vals[np.minimum(idx, len(vals) - 1)]

    va, ra, na = prep(clusters_a)
    vb, rb, nb = prep(clusters_b)
    point = float(np.quantile(va, q / 100.0) - np.quantile(vb, q / 100.0))
    diffs = []
    batch = max(1, min(200, int(2e7 // max(len(va), len(vb), 1))))
    done = 0
    while done < iters:
        k = min(batch, iters - done)
        wa = rng.poisson(1.0, size=(k, na)).astype(float)
        wb = rng.poisson(1.0, size=(k, nb)).astype(float)
        diffs.append(wq(va, ra, wa, q) - wq(vb, rb, wb, q))
        done += k
    d = np.sort(np.concatenate(diffs))
    return point, float(d[int(0.025 * iters)]), float(d[int(0.975 * iters) - 1])


def write_csv(path, rows):
    if not rows:
        return None
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"[summarize] wrote {os.path.relpath(path, C.ROOT)} ({len(rows)} rows)")
    return path


def _inst(row, role):
    return next((i["name"] for i in row.get("instances", []) if i["role"] == role), None)


# --------------------------------------------------------------------------- E0
def summarize_e0(tag):
    R = raws(tag, "e0")
    if not R:
        print("[summarize] PENDING e0：无 raw")
        return
    pd, colo = R.get("pd_1p1d", []), R.get("colo1_mbt8192", [])
    out = {"tag": tag, "pd_measure": len(measure(pd)), "colo_measure": len(measure(colo))}
    pm = measure(pd)
    if pm:
        d, p = _inst(pm[0], "D"), _inst(pm[0], "P")
        dm = [r["metrics_delta"] for r in pm]
        out["d_external_kv_tokens"] = sum((x[d]["prompt_external_kv_transfer"] or 0) for x in dm)
        out["d_local_compute_tokens"] = sum((x[d]["prompt_local_compute"] or 0) for x in dm)
        out["p_local_compute_tokens"] = sum((x[p]["prompt_local_compute"] or 0) for x in dm)
        out["prompt_tokens_sent"] = sum(r["prompt_len"] for r in pm)
        out["d_nixl_transfers"] = sum((x[d]["nixl_xfer_count"] or 0) for x in dm)
        out["d_nixl_failed"] = sum((x[d]["nixl_failed_transfers"] or 0) for x in dm)
        out["metrics_settled_all"] = all(r.get("metrics_settled") for r in pm)
    fp = [r for r in pd if r.get("phase") == "fault_probe"]
    if fp:
        f = fp[-1]
        d = _inst(f, "D")
        out["fault_probe_http_status"] = f.get("http_status")
        out["fault_probe_d_prompt_tokens_delta"] = f["metrics_delta"][d]["prompt_tokens_total"]
    cm = {r["request_key"]: r for r in measure(colo)}
    pair = [(r, cm[r["request_key"]]) for r in pm if r["request_key"] in cm]
    out["n_pairs"] = len(pair)
    out["first_token_match"] = sum(a["first_output_token_id"] == b["first_output_token_id"]
                                   for a, b in pair)
    out["full_output_match"] = sum(a["output_token_ids_sha256"] == b["output_token_ids_sha256"]
                                   for a, b in pair)
    C.save_json(os.path.join(C.OUT, f"{tag}_e0_gate.json"), out)
    print(f"[summarize] e0 gate -> outputs/m27/{tag}_e0_gate.json")


# --------------------------------------------------------------------------- E1
def summarize_e1(tag):
    R = raws(tag, "e1").get("pd_1p1d")
    if not R:
        print("[summarize] PENDING e1：无 raw")
        return
    px = proxy_index(tag, "e1", "pd_1p1d")
    rows = []
    by_ctx = {}
    for r in measure(R):
        by_ctx.setdefault(r["context"], []).append(r)
    for L in sorted(by_ctx):
        xs = by_ctx[L]
        d = _inst(xs[0], "D")
        md = [x["metrics_delta"][d] for x in xs]
        nbytes = [m["nixl_bytes_sum"] for m in md]
        xfer = [m["nixl_xfer_s_sum"] * 1000 if m["nixl_xfer_s_sum"] is not None else None for m in md]
        post = [m["nixl_post_s_sum"] * 1000 if m["nixl_post_s_sum"] is not None else None for m in md]
        bw = [b / (t / 1000) / 1e9 for b, t in zip(nbytes, xfer) if b and t]
        ratio = [b / x["theory_bytes_blocks"] for b, x in zip(nbytes, xs) if b]
        p_ms, d_ms = [], []
        for x in xs:
            pr = px.get(x.get("proxy_request_id")) or {}
            if pr.get("t_p_done") and pr.get("t_p_send"):
                p_ms.append((pr["t_p_done"] - pr["t_p_send"]) * 1000)
            if pr.get("t_stream_first_byte") and pr.get("t_d_send"):
                d_ms.append((pr["t_stream_first_byte"] - pr["t_d_send"]) * 1000)
        row = {"tag": tag, "context": L, "n": len(xs), **C.kv_theory(L)}
        row.update(stats(nbytes, "nixl_bytes"))
        row.update(stats(xfer, "nixl_xfer_ms"))
        row.update(stats(post, "nixl_post_ms"))
        row.update(stats([m["nixl_xfer_count"] for m in md], "nixl_handles"))
        row.update(stats([m["nixl_desc_sum"] for m in md], "nixl_desc"))
        row.update(stats(bw, "bw_GBps"))
        row.update(stats(ratio, "measured_over_theory_blocks"))
        row.update(stats([x["ttft_ms"] for x in xs], "client_ttft_ms"))
        row.update(stats(p_ms, "proxy_p_phase_ms"))
        row.update(stats(d_ms, "proxy_d_first_byte_ms"))
        rows.append(row)
    write_csv(os.path.join(C.OUT, f"{tag}_e1_transfer.csv"), rows)


# --------------------------------------------------------------------------- E2
def e2_sample_metrics(r):
    d = r["d"]
    itl = d.get("itl_ms") or []
    times = d.get("token_times_ms") or []
    out = {"d_itl_p50": pctl(itl, 50), "d_itl_p95": pctl(itl, 95), "d_itl_p99": pctl(itl, 99),
           "d_itl_max": max(itl) if itl else None}
    if r.get("inj") and r.get("t_inject_rel_d_send_ms") is not None:
        t_inj = r["t_inject_rel_d_send_ms"]
        t_first = t_inj + r["inj"]["ttft_ms"]
        after = [b - a for a, b in zip(times, times[1:]) if b > t_inj]
        win = [b - a for a, b in zip(times, times[1:]) if t_inj < b <= t_first + 500]
        out.update(max_stall_ms=max(after) if after else None,
                   win_itl_p95=pctl(win, 95), win_itl_p99=pctl(win, 99),
                   inj_ttft_ms=r["inj"]["ttft_ms"])
    else:
        k = C.FROZEN["e2"]["inject_after_token"] - 1
        out.update(max_stall_ms=max(itl[k:]) if len(itl) > k else None,
                   win_itl_p95=None, win_itl_p99=None, inj_ttft_ms=None)
    return out


def summarize_e2(tag):
    R = raws(tag, "e2")
    if not R:
        print("[summarize] PENDING e2：无 raw")
        return
    rows, per = [], {}
    for arm, rr in R.items():
        for r in measure(rr):
            m = e2_sample_metrics(r)
            if r["p_len"] and r.get("metrics_delta_inj_phase") and r["kind"] == "pd":
                d = _inst(r, "D")
                md = r["metrics_delta_inj_phase"][d]
                m["inj_nixl_xfer_ms"] = md["nixl_xfer_s_sum"] * 1000 if md["nixl_xfer_s_sum"] else None
                m["inj_nixl_bytes"] = md["nixl_bytes_sum"]
            per.setdefault((arm, r["p_len"]), []).append(m)
    for (arm, p_len), ms in sorted(per.items()):
        row = {"tag": tag, "arm": arm, "p_len": p_len, "n": len(ms)}
        for k in ("max_stall_ms", "d_itl_p50", "d_itl_p95", "d_itl_p99", "d_itl_max",
                  "win_itl_p99", "inj_ttft_ms", "inj_nixl_xfer_ms", "inj_nixl_bytes"):
            vals = [m.get(k) for m in ms if m.get(k) is not None]
            row[f"{k}_median"] = statistics.median(vals) if vals else None
            row[f"{k}_p95"] = pctl(vals, 95)
        rows.append(row)
    write_csv(os.path.join(C.OUT, f"{tag}_e2_interference.csv"), rows)
    eff = []
    for (arm, p_len), ms in sorted(per.items()):
        if arm == "pd_1p1d" or ("pd_1p1d", p_len) not in per:
            continue
        pdm = per[("pd_1p1d", p_len)]
        for k in ("max_stall_ms", "d_itl_p99", "inj_ttft_ms"):
            a = [m[k] for m in ms if m.get(k) is not None]
            b = [m[k] for m in pdm if m.get(k) is not None]
            if len(a) < 2 or len(b) < 2:
                continue
            d, lo, hi = boot_median_diff(a, b)
            eff.append({"tag": tag, "p_len": p_len, "metric": k, "colo_arm": arm,
                        "pd_arm": "pd_1p1d", "n_colo": len(a), "n_pd": len(b),
                        "median_diff_colo_minus_pd": d, "ci95_lo": lo, "ci95_hi": hi,
                        "ci_excludes_0": lo > 0 or hi < 0})
    write_csv(os.path.join(C.OUT, f"{tag}_e2_effect.csv"), eff)


# --------------------------------------------------------------------------- E3/E4/E5
def summarize_online(tag, exp, ref_arm_prefix):
    R = raws(tag, exp)
    if not R:
        print(f"[summarize] PENDING {exp}：无 raw")
        return
    rates = C.load_json(os.path.join(C.CONFIGS, f"{tag}_{exp}cal_rates.json")) or {}
    unl = rates.get("unloaded", {})
    rows, groups = [], {}
    for arm, rr in R.items():
        summ = {(s["workload_shape"], s["load_frac"], s["seed"]): s
                for s in rr if s.get("phase") == "measure_run_summary"}
        for r in measure(rr):
            groups.setdefault((arm, r["workload_shape"], r["load_frac"]), []).append(r)
        for (shape, frac, seed), s in sorted(summ.items(), key=lambda x: str(x[0])):
            reqs = [r for r in measure(rr) if r["workload_shape"] == shape and
                    r["load_frac"] == frac and r["seed"] == seed]
            row = {"tag": tag, "exp": exp, "arm": arm, "shape": shape, "load_frac": frac,
                   "seed": seed, "rate": s["rate"], "n_offered": s["n_offered"], "n_ok": s["n_ok"],
                   "n_fail": s["n_fail"], "achieved_rate": s["achieved_rate"]}
            row.update(_online_stats(reqs, unl, s.get("wall_s")))
            rows.append(row)
    write_csv(os.path.join(C.OUT, f"{tag}_{exp}_online.csv"), rows)
    eff = []
    for (arm, shape, frac), reqs in sorted(groups.items(), key=lambda x: str(x[0])):
        if not arm.startswith("pd_"):
            continue
        for (carm, cshape, cfrac), creqs in sorted(groups.items(), key=lambda x: str(x[0])):
            if cshape != shape or cfrac != frac or not carm.startswith(ref_arm_prefix):
                continue
            for metric, q, key in (("ttft_p99", 99, "ttft"), ("itl_p99", 99, "itl"),
                                   ("ttft_p50", 50, "ttft")):
                ca = [[r["ttft_ms"]] if key == "ttft" else r["itl_ms"] for r in creqs]
                cb = [[r["ttft_ms"]] if key == "ttft" else r["itl_ms"] for r in reqs]
                d, lo, hi = cluster_boot_quantile_diff(ca, cb, q)
                eff.append({"tag": tag, "exp": exp, "shape": shape, "load_frac": frac,
                            "metric": metric, "colo_arm": carm, "pd_arm": arm,
                            "n_colo_req": len(creqs), "n_pd_req": len(reqs),
                            "diff_colo_minus_pd_ms": d, "ci95_lo": lo, "ci95_hi": hi,
                            "ci_excludes_0": lo > 0 or hi < 0,
                            "method": "request-cluster Poisson bootstrap, numpy default_rng(0), 10000"})
    write_csv(os.path.join(C.OUT, f"{tag}_{exp}_effect.csv"), eff)


def _online_stats(reqs, unl, wall_s):
    ttft = [r["ttft_ms"] for r in reqs]
    itl = [x for r in reqs for x in r["itl_ms"]]
    tpot = [(r["e2e_ms"] - r["ttft_ms"]) / (r["n_output_tokens"] - 1)
            for r in reqs if r["n_output_tokens"] > 1]
    out = {"ttft_p50": pctl(ttft, 50), "ttft_p95": pctl(ttft, 95), "ttft_p99": pctl(ttft, 99),
           "itl_p50": pctl(itl, 50), "itl_p95": pctl(itl, 95), "itl_p99": pctl(itl, 99),
           "itl_max": max(itl) if itl else None, "tpot_p50": pctl(tpot, 50),
           "e2e_p50": pctl([r["e2e_ms"] for r in reqs], 50),
           "e2e_p99": pctl([r["e2e_ms"] for r in reqs], 99),
           "output_tok_per_s": (sum(r["n_output_tokens"] for r in reqs) / wall_s) if wall_s else None}
    for m in C.FROZEN["e3"]["slo_multipliers"]:
        good = 0
        for r in reqs:
            u = unl.get(r["shape"]) or {}
            t0, i0 = u.get("ttft_median_ms"), u.get("itl_median_ms")
            if t0 is None or i0 is None:
                good = None
                break
            if r["ttft_ms"] <= m * t0 and (pctl(r["itl_ms"], 95) or 0) <= m * i0:
                good += 1
        out[f"goodput_x{m}_req_per_s"] = (good / wall_s) if (good is not None and wall_s) else None
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--only", default="", help="e0,e1,e2,e3,e4,e5")
    a = p.parse_args()
    want = set(a.only.split(",")) if a.only else {"e0", "e1", "e2", "e3", "e4", "e5"}
    if "e0" in want:
        summarize_e0(a.tag)
    if "e1" in want:
        summarize_e1(a.tag)
    if "e2" in want:
        summarize_e2(a.tag)
    if "e3" in want:
        summarize_online(a.tag, "e3", "colo2_")
    if "e4" in want:
        summarize_online(a.tag, "e4", "colo4_")
    if "e5" in want:
        summarize_online(a.tag, "e5", "colo8_")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
