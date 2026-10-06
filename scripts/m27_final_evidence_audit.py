#!/usr/bin/env python3
"""M27 · final evidence audit 的离线生成器（不需 GPU、不 import vllm、不启动任何 benchmark）。

    python scripts/m27_final_evidence_audit.py            # 写 outputs/m27/final_evidence_audit_regenerated.json
    python scripts/m27_final_evidence_audit.py --check    # 只计算并与已落盘文件比对，不写

H20 上的 outputs/m27/final_evidence_audit.json 是收口时手工汇总的 73 个布尔值，没有入库生成器。
本脚本只读已落盘的 raw / configs / lifecycle / 派生 CSV / 图，按同名 73 项逐条重算并附证据；
原文件保持不动（它是 H20 交付的历史产物）。

单文件 >100MB 的 formal raw 不在普通 Git 历史中（见 notes/m27/H20_RAW_ARCHIVE_MANIFEST.tsv）。
它们不在本地时，依赖它们的检查记为 null（不可推导），整体 status=INCOMPLETE，不算通过。
输出确定性：不含当前时间，键排序固定。figures_current 依赖 matplotlib 渲染，跨版本可能不逐像素一致。
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402

TAG = "h20_formal"
ORIGINAL = "outputs/m27/final_evidence_audit.json"
OUTPUT = "outputs/m27/final_evidence_audit_regenerated.json"
RAW_MANIFEST = "notes/m27/H20_RAW_ARCHIVE_MANIFEST.tsv"
LEDGER = "outputs/m27/wall_clock_ledger.csv"
ARCHIVE_SHA256 = "fb1664ca9506efb75bb784ab19a8b65186c6eabedf69127805e8e72b29633ad6"
MODEL_DIR_NAME = "Qwen2.5-7B-Instruct"
ONLINE_METRICS = ("ttft_p99", "itl_p99", "ttft_p50")
FIGURES = (("e1_transfer", "fig16_m27_e1_transfer", "fig_e1"),
           ("e2_interference", "fig17_m27_e2_interference", "fig_e2"),
           ("e3_online", "fig18_m27_e3_online", "fig_e3"))
TS_RE = re.compile(rb'"unix_ts": ([0-9.]+)')


def _p(*parts):
    return os.path.join(C.ROOT, *parts)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _csv(rel):
    path = _p(rel)
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def task_start():
    """TASK_START 取自 wall-clock ledger 首行 note（H20 记录的计费起点）。"""
    rows = _csv(LEDGER) or []
    m = re.search(r"TASK_START=(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)", rows[0]["note"]) if rows else None
    if not m:
        raise SystemExit(f"{LEDGER} 首行缺 TASK_START")
    t = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    return m.group(1), t.timestamp()


def external_raw():
    """manifest 中每个外部 raw：match / absent / mismatch。"""
    out = {}
    if not os.path.exists(_p(RAW_MANIFEST)):
        return {"<manifest>": "absent"}
    with open(_p(RAW_MANIFEST), newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        path = _p(r["relative_path"])
        if not os.path.exists(path):
            out[r["relative_path"]] = "absent"
        elif os.path.getsize(path) != int(r["size_bytes"]) or _sha256(path) != r["sha256"]:
            out[r["relative_path"]] = "mismatch"
        else:
            out[r["relative_path"]] = "match"
    return out


def scan_raw(rel, parse_all=False):
    """流式读取 raw：最早 unix_ts + 需要的行（大文件只解析 run summary / arm_done）。"""
    path = _p(rel)
    if not os.path.exists(path):
        return None
    min_ts, rows = None, []
    with open(path, "rb") as f:
        for line in f:
            m = TS_RE.search(line)
            if m:
                ts = float(m.group(1))
                min_ts = ts if min_ts is None else min(min_ts, ts)
            if parse_all or b'"measure_run_summary"' in line or b'"arm_done"' in line:
                rows.append(json.loads(line))
    return {"min_unix_ts": min_ts, "rows": rows}


def _formal_arms(F):
    return [("e0", "pd_1p1d"), ("e0", "colo1_mbt8192"), ("e1", "pd_1p1d")] + \
        [("e2", a) for a in ("pd_1p1d", "colo1_mbt8192", "colo1_mbt2048", "colo1_mbt4096")] + \
        [("e3", a) for a in ("pd_1p1d", "colo2_mbt8192", "colo2_mbt4096", "colo2_mbt2048", "colo1_mbt8192")] + \
        [("e4", a) for a in ("colo4_mbt8192", "pd_1p3d", "pd_2p2d", "pd_3p1d")]


def _workloads(F, exp):
    return list(F["e3"]["shapes"]) + ["mixture"] if exp == "e3" else list(F["e4"]["shapes"])


def _config_ok(exp, arm, F):
    ef = C.load_json(_p("outputs/m27/configs", f"{TAG}_{exp}_{arm}_effective.json"))
    if not ef:
        return False, {"effective": "missing"}
    t = C.parse_arm(arm)
    insts = ef.get("instances") or {}
    roles = sorted(v["instance"]["role"] for v in insts.values())
    want_roles = sorted(["P"] * t["n_p"] + ["D"] * t["n_d"] + ["colo"] * t["n_colo"])
    e = F["engine"]
    bad = []
    for name, v in insts.items():
        x = v.get("effective") or {}
        want = {"tensor_parallel_size": 1, "dtype": e["dtype"], "enforce_eager": True,
                "enable_prefix_caching": False, "async_scheduling": False, "enable_chunked_prefill": True,
                "max_model_len": e["max_model_len"], "max_num_batched_tokens": t["mbt"],
                "attention_backend": "FLASH_ATTN", "vllm_env_kv_layout": "HND",
                "quantization": None, "speculative_config": None,
                "kv_role": {"P": "kv_producer", "D": "kv_consumer", "colo": None}[v["instance"]["role"]]}
        bad += [(name, k, x.get(k)) for k, w in want.items() if x.get(k) != w]
    ok = (ef.get("effective_matches_requested") is True and ef.get("mismatches") == []
          and roles == want_roles and len(insts) == t["n_gpus"]
          and os.path.basename(ef.get("model_dir", "")) == MODEL_DIR_NAME and not bad)
    return ok, {"n_instances": len(insts), "roles": roles, "violations": bad[:5],
                "effective_matches_requested": ef.get("effective_matches_requested")}


def _render_figures():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import m27_figures as FG
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except ImportError as e:
        return None, {"error": f"ImportError: {e}"}
    ev, ok = {"matplotlib": matplotlib.__version__}, True
    with tempfile.TemporaryDirectory(prefix="m27_fig_") as tmp:
        for key, fname, fn in FIGURES:
            rows = _csv(f"outputs/m27/{TAG}_{key}.csv")
            ref = _p("figures", f"{fname}_{TAG}.png")
            if not rows or not os.path.exists(ref):
                ev[fname] = "missing"
                ok = False
                continue
            fig = getattr(FG, fn)(TAG, rows, plt)
            out = os.path.join(tmp, fname + ".png")
            fig.savefig(out, dpi=130)
            plt.close(fig)
            a, b = mpimg.imread(out), mpimg.imread(ref)
            same = a.shape == b.shape and bool((a == b).all())
            ev[fname] = {"shape": list(b.shape), "pixel_identical": same}
            ok = ok and same
    return ok, ev


def build():
    F = C.FROZEN
    ts_iso, ts0 = task_start()
    ext = external_raw()
    checks, ev = {}, {}

    def put(name, value, evidence):
        checks[name] = value
        ev[name] = evidence

    pf = C.load_json(_p("outputs/m27/configs", f"preflight_{TAG}.json")) or {}
    pf_t = dt.datetime.strptime(pf["time"], "%Y-%m-%dT%H:%M:%S%z").timestamp() if pf.get("time") else None
    put("fresh_preflight", pf.get("result") == "PASS" and pf.get("fails") == [] and pf_t is not None and pf_t >= ts0,
        {"result": pf.get("result"), "time": pf.get("time"), "task_start": ts_iso})

    gm = C.load_json(_p("outputs/m27/configs", f"gpu_map_{TAG}.json")) or {}
    mat = (gm.get("topo_parsed") or {}).get("matrix") or {}
    links = [x for g, row in mat.items() for j, x in enumerate(row) if f"GPU{j}" != g]
    put("gpu_map_8_nv18", gm.get("gpu_ids") == list(range(8)) and gm.get("nv_clique") == list(range(8))
        and len(links) == 56 and set(links) == {"NV18"},
        {"gpu_ids": gm.get("gpu_ids"), "pairs": len(links), "link_types": sorted(set(links))})

    effs = sorted(glob.glob(_p("outputs/m27/configs", f"{TAG}_*_effective.json")))
    dirs = {os.path.basename((C.load_json(p) or {}).get("model_dir", "")) for p in effs}
    le = C.load_json(_p("outputs/m27", f"{TAG}_log_evidence.json")) or {}
    eng = [v.get("engine_config_log") or {} for v in (le.get("instances") or {}).values()]
    put("no_model_fallback", gm.get("max_model_len_fallback") is False and dirs == {MODEL_DIR_NAME}
        and bool(eng) and all(x.get("dtype") == "torch.bfloat16" and x.get("max_seq_len") == "32768" for x in eng),
        {"effective_files": len(effs), "model_dirs": sorted(dirs), "log_instances": len(eng),
         "max_model_len_fallback": gm.get("max_model_len_fallback")})

    g = C.load_json(_p("outputs/m27", f"{TAG}_e0_gate.json")) or {}
    put("e0_true_kv_no_recompute", g.get("d_local_compute_tokens") == 0 and g.get("d_nixl_failed") == 0
        and g.get("d_external_kv_tokens") == g.get("prompt_tokens_sent") == g.get("p_local_compute_tokens")
        and g.get("fault_probe_http_status") == 502 and g.get("fault_probe_d_prompt_tokens_delta") == 0,
        {k: g.get(k) for k in ("d_external_kv_tokens", "d_local_compute_tokens", "p_local_compute_tokens",
                               "prompt_tokens_sent", "d_nixl_failed", "fault_probe_http_status")})

    e1 = _csv(f"outputs/m27/{TAG}_e1_transfer.csv") or []
    got = {int(r["context"]): int(r["n"]) for r in e1}
    put("e1_grid", got == {c: F["measure"] for c in F["e1"]["contexts"]}, {"context_n": got})

    e2 = _csv(f"outputs/m27/{TAG}_e2_interference.csv") or []
    cells = {(r["arm"], int(r["p_len"])): int(r["n"]) for r in e2}
    want = {(a, p): F["measure"] for a in F["e2"]["arms"] for p in F["e2"]["p_lens"]}
    put("e2_4arms_4injections", cells == want, {"cells": len(cells), "expected": len(want)})

    with open(_p("outputs/m27/raw", f"{TAG}_lifecycle.jsonl"), encoding="utf-8") as f:
        life = [json.loads(line) for line in f if line.strip()]
    raw_cache, unavailable = {}, set()
    for exp, arm in _formal_arms(F) + [("e3cal", F["e3"]["calibration_arm"]), ("e4cal", F["e4"]["calibration_arm"])]:
        rel = f"outputs/m27/raw/{TAG}_{exp}_{arm}.jsonl"
        raw_cache[(exp, arm)] = scan_raw(rel, parse_all=exp in ("e0", "e1", "e2"))
        if raw_cache[(exp, arm)] is None:
            unavailable.add((exp, arm))

    for exp, arm in _formal_arms(F):
        key = f"{exp}_{arm}"
        sc = raw_cache[(exp, arm)]
        if sc is None:
            for k in ("fresh", "done") + (("overlap",) if exp == "e2" else ()) + (("online",) if exp in ("e3", "e4") else ()):
                put(f"{key}_{k}", None, {"raw": "absent (external evidence, see manifest)"})
            ok, cev = _config_ok(exp, arm, F)
            put(f"{key}_config", ok, cev)
            continue
        starts = [x["unix_ts"] for x in life if x.get("event") == "start" and x.get("exp") == exp and x.get("arm") == arm]
        stops = [x for x in life if x.get("event") == "stop" and x.get("exp") == exp and x.get("arm") == arm
                 and x["unix_ts"] >= sc["min_unix_ts"]]
        put(f"{key}_fresh", sc["min_unix_ts"] >= ts0 and bool(starts) and max(starts) >= ts0,
            {"raw_min_unix_ts": sc["min_unix_ts"], "last_lifecycle_start": max(starts) if starts else None,
             "task_start_epoch": ts0})
        done = [r for r in sc["rows"] if r.get("phase") == "arm_done"]
        put(f"{key}_done", bool(done) and all(r.get("success") is True for r in done)
            and len(stops) == 1 and stops[0].get("reason") == "normal",
            {"arm_done_rows": len(done), "formal_stops": [s.get("reason") for s in stops]})
        ok, cev = _config_ok(exp, arm, F)
        put(f"{key}_config", ok, cev)
        if exp == "e2":
            rows = [r for r in sc["rows"] if r.get("phase") in ("warmup", "measure")]
            put(f"{key}_overlap", bool(rows) and all(r.get("overlap_valid") is True for r in rows),
                {"rows": len(rows), "overlap_valid": sum(r.get("overlap_valid") is True for r in rows)})
        if exp in ("e3", "e4"):
            summ = [r for r in sc["rows"] if r.get("phase") == "measure_run_summary"]
            n_exp = len(_workloads(F, exp)) * len(F["e3"]["load_fractions"]) * len(F["e3"]["seeds"])
            csv_rows = [r for r in (_csv(f"outputs/m27/{TAG}_{exp}_online.csv") or []) if r["arm"] == arm]
            put(f"{key}_online", len(summ) == len(csv_rows) == n_exp
                and all(r["n_fail"] == 0 and r["n_ok"] == r["n_offered"] for r in summ)
                and all(int(r["n_fail"]) == 0 for r in csv_rows),
                {"run_summaries": len(summ), "csv_rows": len(csv_rows), "expected": n_exp,
                 "requests_offered": sum(r["n_offered"] for r in summ)})

    e2_raw = [raw_cache[("e2", a)] for a in F["e2"]["arms"]]
    if any(x is None for x in e2_raw):
        put("e2_240_valid_measure", None, {"raw": "absent"})
    else:
        n = sum(1 for x in e2_raw for r in x["rows"]
                if r.get("phase") == "measure" and r.get("success") is True and r.get("overlap_valid") is True)
        want_n = len(F["e2"]["arms"]) * len(F["e2"]["p_lens"]) * F["measure"]
        put("e2_240_valid_measure", n == want_n == 240, {"valid_measure": n, "expected": want_n})

    sched_ev, sched_ok = {}, True
    for exp in ("e3", "e4"):
        arms = [a for x, a in _formal_arms(F) if x == exp]
        if any((exp, a) in unavailable for a in arms):
            sched_ok = None
            sched_ev[exp] = "raw absent"
            continue
        rates_sha = C.sha256_file(_p("outputs/m27/configs", f"{TAG}_{exp}cal_rates.json"))
        by_key, bad = {}, 0
        for a in arms:
            for r in raw_cache[(exp, a)]["rows"]:
                if r.get("phase") == "measure_run_summary":
                    by_key.setdefault((r["workload_shape"], r["seed"], r["load_frac"]), {})[a] = r["schedule_sha256"]
                    bad += r.get("rates_file_sha256") != rates_sha
        split = [k for k, v in by_key.items() if set(v) != set(arms) or len(set(v.values())) != 1]
        sched_ok = sched_ok and not split and not bad and bool(by_key) if sched_ok is not None else None
        sched_ev[exp] = {"keys": len(by_key), "arms": len(arms), "split": len(split), "rates_sha_mismatch": bad,
                         "rates_sha256": rates_sha}
    put("e3_e4_shared_schedule", sched_ok, sched_ev)

    cal_ev, cal_ok = {}, True
    for exp in ("e3cal", "e4cal"):
        arm = F[exp[:2]]["calibration_arm"]
        sc = raw_cache[(exp, arm)]
        rates = C.load_json(_p("outputs/m27/configs", f"{TAG}_{exp}_rates.json"))
        starts = [x["unix_ts"] for x in life if x.get("event") == "start" and x.get("exp") == exp and x.get("arm") == arm]
        if sc is None:
            cal_ok, cal_ev[exp] = None, "raw absent"
            continue
        ok = sc["min_unix_ts"] >= ts0 and bool(starts) and max(starts) >= ts0 and bool(rates) and \
            all(not s.get("ladder_exhausted") for s in rates.get("shapes", {}).values())
        cal_ok = cal_ok and ok if cal_ok is not None else None
        cal_ev[exp] = {"raw_min_unix_ts": sc["min_unix_ts"], "last_lifecycle_start": max(starts) if starts else None,
                       "shapes": sorted(rates.get("shapes", {})) if rates else None}
    put("e3_e4_cal_fresh", cal_ok, cal_ev)

    od_ev, od_ok = {}, True
    for exp, pd_arms, colo_prefix in (("e3", ["pd_1p1d"], "colo2_"), ("e4", ["pd_1p3d", "pd_2p2d", "pd_3p1d"], "colo4_")):
        arms = [a for x, a in _formal_arms(F) if x == exp]
        online = _csv(f"outputs/m27/{TAG}_{exp}_online.csv") or []
        eff = _csv(f"outputs/m27/{TAG}_{exp}_effect.csv") or []
        n_runs = len(arms) * len(_workloads(F, exp)) * len(F["e3"]["load_fractions"]) * len(F["e3"]["seeds"])
        colo = [a for a in arms if a.startswith(colo_prefix)]
        want = {(s, str(f), m, c, p) for s in _workloads(F, exp) for f in F["e3"]["load_fractions"]
                for m in ONLINE_METRICS for c in colo for p in pd_arms}
        got = {(r["shape"], r["load_frac"], r["metric"], r["colo_arm"], r["pd_arm"]) for r in eff}
        numeric = all(r["ci95_lo"] not in ("", None) and r["ci95_hi"] not in ("", None) for r in eff)
        ok = len(online) == n_runs and len(eff) == len(want) and got == want and numeric
        od_ok = od_ok and ok
        od_ev[exp] = {"online_rows": len(online), "expected_runs": n_runs, "effect_rows": len(eff),
                      "expected_effect": len(want)}
    put("online_derived_complete", od_ok, od_ev)

    pooled = []
    for exp in ("e0", "e1", "e2", "e3", "e4"):
        pooled += sorted(glob.glob(os.path.join(C.RAW, f"{TAG}_{exp}_*.jsonl")))
    pooled = [p for p in pooled if not p.endswith("_proxy.jsonl")]
    early = []
    for p in pooled:
        key = tuple(os.path.basename(p)[len(TAG) + 1:-len(".jsonl")].split("_", 1))
        sc = raw_cache.get(key) or scan_raw(os.path.relpath(p, C.ROOT))
        if sc["min_unix_ts"] is None or sc["min_unix_ts"] < ts0:
            early.append(os.path.basename(p))
    absent = sorted(f"{e}_{a}" for e, a in unavailable if e in ("e0", "e1", "e2", "e3", "e4"))
    put("no_previous_host_raw_pooled",
        None if absent else (bool(pooled) and not early and not any("backup_" in p for p in pooled)),
        {"summarizer_glob_files": len(pooled), "before_task_start": early, "absent": absent,
         "backup_dirs": sorted(os.path.basename(d) for d in glob.glob(os.path.join(C.RAW, "backup_*")))})

    ok, fev = _render_figures()
    put("figures_current", ok, fev)

    vals = list(checks.values())
    tampered = any(v == "mismatch" for v in ext.values())
    status = ("FAIL" if tampered or any(v is False for v in vals)
              else "INCOMPLETE" if any(v is None for v in vals) else "PASS")
    orig = C.load_json(_p(ORIGINAL)) or {}
    oc = orig.get("checks") or {}
    return {
        "generator": "scripts/m27_final_evidence_audit.py",
        "tag": TAG,
        "task_start_utc": ts_iso,
        "status": status,
        "n_checks": len(checks),
        "n_pass": sum(v is True for v in vals),
        "n_unavailable": sum(v is None for v in vals),
        "checks": checks,
        "evidence": ev,
        "external_raw": {"manifest": RAW_MANIFEST, "archive_sha256": ARCHIVE_SHA256, "files": ext},
        "original": {"path": ORIGINAL, "sha256": _sha256(_p(ORIGINAL)) if os.path.exists(_p(ORIGINAL)) else None,
                     "check_names_equal": list(oc) == list(checks),
                     "all_original_true": bool(oc) and all(v is True for v in oc.values())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只比对已落盘文件，不写")
    a = ap.parse_args()
    out = build()
    text = json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path = _p(OUTPUT)
    if a.check:
        same = False
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                same = f.read() == text
        print(f"[final_evidence_audit] status={out['status']} {out['n_pass']}/{out['n_checks']} PASS；"
              f"与 {OUTPUT} {'一致' if same else '不一致'}")
        return 0 if same else 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[final_evidence_audit] status={out['status']} {out['n_pass']}/{out['n_checks']} PASS "
          f"(unavailable {out['n_unavailable']}) -> {OUTPUT}")
    return 0 if out["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
