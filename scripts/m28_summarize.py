#!/usr/bin/env python3
"""M28 汇总：raw JSONL → 派生 CSV（离线；不需 GPU、不 import vllm）。

统计口径（notes/m28/M28_PLAN.md §5）：只用 phase=measure ∧ success=True；
失败样本数单列 n_fail，不静默剔除。差值 CI = median difference 的 bootstrap 95%（固定 seed）。

用法：python scripts/m28_summarize.py [--raw-dir outputs/m28/raw] [--out-dir outputs/m28]
"""

import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m28_common as C  # noqa: E402

A = C.FROZEN["A"]
B = C.FROZEN["B"]

PAIRED = [  # (名字, arm, path_a, path_b)：同一 arm、同一 (P, idx) 配对
    ("offload_cpu_hit_minus_cold", "offload", "cpu_hit", "cold"),
    ("offload_cpu_hit_minus_gpu_hot", "offload", "cpu_hit", "gpu_hot"),
    ("base_recompute_minus_cold", "base", "recompute_evicted", "cold"),
]
UNPAIRED = [  # (名字, arm_a, path_a, arm_b, path_b)：跨进程，非配对 bootstrap
    ("offload_cpu_hit_vs_base_recompute", "offload", "cpu_hit", "base", "recompute_evicted"),
    ("offload_cold_vs_base_cold", "offload", "cold", "base", "cold"),
    ("offload_gpu_hot_vs_base_gpu_hot", "offload", "gpu_hot", "base", "gpu_hot"),
]


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    return path


def fmt(v, nd=3):
    return None if v is None else round(v, nd)


def load_a(raw_dir):
    out = {}
    for arm in A["arms"]:
        rows = C.load_jsonl(os.path.join(raw_dir, f"m28_a_{arm}.jsonl"))
        if rows is not None:
            out[arm] = rows
    return out


def measure(rows, p_len, path):
    sel = [r for r in rows if r["phase"] == "measure" and r["p_len"] == p_len and r["path"] == path]
    return [r for r in sel if r["success"]], [r for r in sel if not r["success"]]


def summarize_a(R, out_dir):
    lat, paths, contrasts, corr = [], [], [], []
    for arm, rows in R.items():
        lens = A["evict_prefix_lens"] if arm == "evict" else A["prefix_lens"]
        for p_len in lens:
            for path in A["arms"][arm]["paths"]:
                ok, bad = measure(rows, p_len, path)
                t = C.describe([r["online_ttft_ms"] for r in ok])
                e = C.describe([r["e2e_ms"] for r in ok])
                lat.append({"arm": arm, "p_len": p_len, "path": path, "n": t["n"], "n_fail": len(bad),
                            "ttft_median_ms": fmt(t["median"]), "ttft_stdev_ms": fmt(t["stdev"]),
                            "ttft_p95_ms": fmt(t["p95"]), "e2e_median_ms": fmt(e["median"]),
                            "e2e_p95_ms": fmt(e["p95"])})
                d = [r["delta"] for r in ok]
                lb = [x["load_bytes"] for x in d if x.get("load_bytes")]
                lt = [x["load_time_s"] for x in d if x.get("load_bytes")]
                eh = [x["ext_hits"] for x in d if x.get("load_bytes")]
                sb = [x["store_bytes"] for x in d if x.get("store_bytes") and x.get("store_time_s")]
                stt = [x["store_time_s"] for x in d if x.get("store_bytes") and x.get("store_time_s")]
                paths.append({
                    "arm": arm, "p_len": p_len, "path": path, "n_measure": len(ok) + len(bad), "n_path_ok": len(ok),
                    "expected_hit_tokens": C.expected_hit_tokens(p_len + A["suffix_len"]),
                    "local_hits_median": fmt(statistics.median([x["local_hits"] or 0 for x in d])) if d else None,
                    "ext_hits_median": fmt(statistics.median([x["ext_hits"] or 0 for x in d])) if d else None,
                    "cached_tokens_median": fmt(statistics.median([r["cached_tokens"] for r in ok])) if ok else None,
                    "load_bytes_median": fmt(statistics.median(lb)) if lb else None,
                    "load_bytes_per_hit_token": fmt(statistics.median([b / h for b, h in zip(lb, eh)])) if lb else None,
                    "load_time_ms_median": fmt(statistics.median(lt) * 1e3) if lt else None,
                    "load_gbps_median": fmt(statistics.median([b / t / 1e9 for b, t in zip(lb, lt) if t]), 2)
                    if lb else None,
                    "n_with_store": len(sb),
                    "store_bytes_median": fmt(statistics.median(sb)) if sb else None,
                    "store_time_ms_median": fmt(statistics.median(stt) * 1e3) if sb else None,
                    "store_gbps_median": fmt(statistics.median([b / t / 1e9 for b, t in zip(sb, stt)]), 2)
                    if sb else None,
                })
    for name, arm, pa, pb in PAIRED:
        if arm not in R:
            continue
        for p_len in A["prefix_lens"]:
            a = {r["idx"]: r["online_ttft_ms"] for r in measure(R[arm], p_len, pa)[0]}
            b = {r["idx"]: r["online_ttft_ms"] for r in measure(R[arm], p_len, pb)[0]}
            keys = sorted(set(a) & set(b))
            diffs = [a[k] - b[k] for k in keys]
            med, lo, hi = C.boot_ci_median_paired(diffs)
            contrasts.append({"contrast": name, "kind": "paired", "p_len": p_len, "n": len(keys),
                              "median_diff_ms": fmt(med), "ci95_lo_ms": fmt(lo), "ci95_hi_ms": fmt(hi),
                              "median_a_ms": fmt(statistics.median([a[k] for k in keys])) if keys else None,
                              "median_b_ms": fmt(statistics.median([b[k] for k in keys])) if keys else None})
    for name, aa, pa, ab, pb in UNPAIRED:
        if aa not in R or ab not in R:
            continue
        for p_len in A["prefix_lens"]:
            a = [r["online_ttft_ms"] for r in measure(R[aa], p_len, pa)[0]]
            b = [r["online_ttft_ms"] for r in measure(R[ab], p_len, pb)[0]]
            med, lo, hi = C.boot_ci_median_diff(a, b)
            contrasts.append({"contrast": name, "kind": "unpaired", "p_len": p_len, "n": min(len(a), len(b)),
                              "median_diff_ms": fmt(med), "ci95_lo_ms": fmt(lo), "ci95_hi_ms": fmt(hi),
                              "median_a_ms": fmt(statistics.median(a)) if a else None,
                              "median_b_ms": fmt(statistics.median(b)) if b else None})
    corr = correctness_rows(R)
    write_csv(os.path.join(out_dir, "m28_a_latency.csv"), lat, list(lat[0]) if lat else ["arm"])
    write_csv(os.path.join(out_dir, "m28_a_paths.csv"), paths, list(paths[0]) if paths else ["arm"])
    write_csv(os.path.join(out_dir, "m28_a_contrasts.csv"), contrasts,
              list(contrasts[0]) if contrasts else ["contrast"])
    write_csv(os.path.join(out_dir, "m28_a_correctness.csv"), corr, list(corr[0]) if corr else ["check"])
    return lat, paths, contrasts, corr


def _tok(rows, path):
    return {(r["p_len"], r["idx"]): r for r in rows if r["phase"] == "measure" and r["path"] == path
            and r["request_ok"]}


CORR = [  # (检查名, (arm_a, path_a), (arm_b, path_b))
    ("offload_cpu_hit_eq_gpu_hot", ("offload", "cpu_hit"), ("offload", "gpu_hot")),
    ("ref_a_eq_ref_b", ("ref", "ref_a"), ("ref", "ref_b")),
    ("offload_cold_eq_ref", ("offload", "cold"), ("ref", "ref_a")),
    ("base_cold_eq_ref", ("base", "cold"), ("ref", "ref_a")),
    ("offload_cpu_hit_eq_ref", ("offload", "cpu_hit"), ("ref", "ref_a")),
    ("offload_gpu_hot_eq_ref", ("offload", "gpu_hot"), ("ref", "ref_a")),
    ("base_gpu_hot_eq_ref", ("base", "gpu_hot"), ("ref", "ref_a")),
]


def compare(ra, rb):
    ta, tb = ra["token_ids"], rb["token_ids"]
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    la, lb = ra.get("token_logprobs") or [], rb.get("token_logprobs") or []
    n = first if first is not None else min(len(la), len(lb))
    dl = max((abs(x - y) for x, y in zip(la[:n], lb[:n]) if x is not None and y is not None), default=0.0)
    return first, dl


def correctness_rows(R):
    rows = []
    for name, (aa, pa), (ab, pb) in CORR:
        if aa not in R or ab not in R:
            continue
        a, b = _tok(R[aa], pa), _tok(R[ab], pb)
        for p_len in A["prefix_lens"]:
            keys = sorted(k for k in set(a) & set(b) if k[0] == p_len)
            mism, maxdl = [], 0.0
            for k in keys:
                first, dl = compare(a[k], b[k])
                maxdl = max(maxdl, dl)
                if first is not None:
                    mism.append(f"{k[1]}@{first}")
            rows.append({"check": name, "p_len": p_len, "n_pairs": len(keys), "n_match": len(keys) - len(mism),
                         "max_abs_logprob_diff_before_divergence": fmt(maxdl, 6),
                         "mismatches": ";".join(mism)})
    return rows


# --------------------------------------------------------------------------- B
def first_after(tl, t0, pred):
    for s in tl:
        if s["ts"] >= t0 and pred(s):
            return round(s["ts"] - t0, 3)
    return None


def outcome(r):
    if r.get("missing"):
        return "missing"
    if r.get("terminal") is False:
        return "pending"
    if r.get("exception"):
        return "exception:" + r["exception"].split(":")[0]
    if r.get("error_event"):
        code = (r["error_event"].get("error") or {}).get("code") if isinstance(r["error_event"], dict) else None
        return f"error_event:{code}" if r.get("http_status") == 200 else f"http_{r.get('http_status')}"
    if r.get("http_status") != 200:
        return f"http_{r.get('http_status')}"
    if r.get("stream") and not r.get("done_marker") and not r.get("finish_reason"):
        return "eof_without_done"
    return f"finish:{r.get('finish_reason')}"


def summarize_b(rows, out_dir):
    out = []
    for d in rows:
        tl, tf = d.get("timeline") or [], d.get("t_fault")
        reqs = d.get("requests") or []
        oc, oc_after = {}, {}
        for r in reqs:
            k = ("s:" if r.get("stream") else "ns:") + outcome(r)
            oc[k] = oc.get(k, 0) + 1
        for r, a in zip(reqs, d.get("requests_after_cleanup") or []):
            if r.get("terminal") is False:
                k = ("s:" if a.get("stream") else "ns:") + outcome(a)
                oc_after[k] = oc_after.get(k, 0) + 1
        s_tok = [r["n_tokens"] for r in reqs if r.get("stream") and r.get("n_tokens") is not None]
        shm_new = [x for x in d.get("dev_shm_after") or [] if x not in set((d.get("pre") or {}).get("dev_shm") or [])]
        pf = {}
        for p in d.get("probes") or []:
            k = outcome(p)
            pf[k] = pf.get(k, 0) + 1
        t_send = [r["t_send"] for r in reqs if r.get("t_send")]
        cal = (d.get("calibration") or {}).get("duration_s")
        complete = sum(1 for r in reqs if r.get("finish_reason") == "length" and r.get("n_tokens") == r.get("max_tokens"))
        mk = d.get("log_markers") or {}
        out.append({
            "label": d["label"], "drill": d["drill"], "shutdown_timeout": d.get("shutdown_timeout"), "rep": d["rep"],
            "error": d.get("error"), "n_inflight": len(reqs), "n_completed_full": complete,
            "outcomes": ";".join(f"{k}={v}" for k, v in sorted(oc.items())),
            "n_pending_at_observe_end": sum(1 for r in reqs if r.get("terminal") is False),
            "outcomes_after_cleanup": ";".join(f"{k}={v}" for k, v in sorted(oc_after.items())),
            "tokens_stream_median": statistics.median(s_tok) if s_tok else None,
            "t_health_non200_s": first_after(tl, tf, lambda s: s["health"] != 200) if tf else None,
            "t_health_503_s": first_after(tl, tf, lambda s: s["health"] == 503) if tf else None,
            "t_engine_exit_s": first_after(tl, tf, lambda s: not s["engine_alive"]) if tf else None,
            "t_api_exit_s": first_after(tl, tf, lambda s: not s["api_alive"]) if tf else None,
            "t_port_closed_s": first_after(tl, tf, lambda s: not s["port_listening"]) if tf else None,
            "probe_forms": ";".join(f"{k}={v}" for k, v in sorted(pf.items())),
            "calibration_s": fmt(cal), "remaining_work_s": fmt(cal - (tf - min(t_send))) if cal and tf and t_send else None,
            "api_returncode": d.get("api_returncode_before_cleanup"),
            "cleanup_needed": d.get("cleanup_needed"),
            "api_returncode_final": d.get("api_returncode"),
            "dev_shm_new": ";".join(shm_new),
            "gpu_reclaim_ok": (d.get("gpu_reclaim") or {}).get("ok"),
            "gpu_reclaim_s": (d.get("gpu_reclaim") or {}).get("waited_s"),
            "restart_ok": (d.get("restart") or {}).get("ok"),
            "restart_ready_s": (d.get("restart") or {}).get("ready_s"),
            "log_aborting": mk.get("engine_aborting"), "log_draining": mk.get("engine_draining"),
            "log_force_killing": mk.get("force_killing"), "log_engine_exited": mk.get("engine_exited_unexpectedly"),
            "log_engine_dead_error": mk.get("engine_dead_error"),
        })
    if out:
        write_csv(os.path.join(out_dir, "m28_b_drills.csv"), out, list(out[0]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=C.RAW)
    ap.add_argument("--out-dir", default=C.OUT)
    a = ap.parse_args()
    R = load_a(a.raw_dir)
    if R:
        lat, paths, con, corr = summarize_a(R, a.out_dir)
        print(f"[m28-summarize] A: {len(lat)} latency 行 / {len(con)} contrast 行 / {len(corr)} correctness 行")
    else:
        print("[m28-summarize] A raw 不存在（未执行），不生成 A CSV")
    b = C.load_jsonl(os.path.join(a.raw_dir, "m28_b_drills.jsonl"))
    if b:
        print(f"[m28-summarize] B: {len(summarize_b(b, a.out_dir))} 行")
    else:
        print("[m28-summarize] B raw 不存在（未执行），不生成 B CSV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
