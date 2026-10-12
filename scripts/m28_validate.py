#!/usr/bin/env python3
"""M28 不变量检查（离线；不需 GPU、不 import vllm）。

原则：
  - 路径判据 / token 比对 / 计数都在这里**独立重算**，不复用 runner 的 classify；
    prompt 由冻结生成器重新生成后比对 sha（证明 raw 里的 prompt 没被改过）。
  - raw 不存在 → PENDING（未执行），**不得记为通过**。
  - --selftest：把一份基准数据集复制到临时目录，逐条注入反例，要求目标规则 PASS→FAIL；
    反例只存在于临时目录，绝不写入正式 raw。

用法：
  python scripts/m28_validate.py                                  # 正式数据
  python scripts/m28_validate.py --raw-dir /tmp/x --derived-dir /tmp/x --allow-mock
  python scripts/m28_validate.py --selftest --raw-dir /tmp/x --derived-dir /tmp/x
"""

import argparse
import copy
import csv
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m28_common as C  # noqa: E402

A = C.FROZEN["A"]
B = C.FROZEN["B"]
FREEZE_RE = re.compile(r"<!-- M28_FROZEN_CONFIG:BEGIN -->\s*```json\s*(\{.*?\})\s*```\s*<!-- M28_FROZEN_CONFIG:END -->",
                       re.S)
SCRIPTS = ["m28_common.py", "m28_offload.py", "m28_faults.py", "m28_mock_server.py", "m28_summarize.py",
           "m28_validate.py", "m28_figures.py"]
REQUIRED_A_ARMS = ["offload", "base", "ref"]


class R:
    def __init__(self):
        self.rows = []

    def add(self, status, rule, detail=""):
        self.rows.append((status, rule, detail))

    def check(self, rule, cond, fail, ok=""):
        self.add("PASS" if cond else "FAIL", rule, ok if cond else fail)
        return cond

    def status(self, rule):
        return next((s for s, r, _ in self.rows if r == rule), None)


def _load(raw_dir, name):
    return C.load_jsonl(os.path.join(raw_dir, name))


def _csv(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


# --------------------------------------------------------------------------- 静态
def check_static(rep):
    import py_compile
    missing, broken = [], []
    with tempfile.TemporaryDirectory(prefix="m28_pyc_") as td:
        for n in SCRIPTS:
            p = os.path.join(C.ROOT, "scripts", n)
            if not os.path.exists(p):
                missing.append(n)
                continue
            try:
                py_compile.compile(p, cfile=os.path.join(td, n + "c"), doraise=True)
            except py_compile.PyCompileError as e:
                broken.append(f"{n}: {str(e).strip().splitlines()[-1][:120]}")
    rep.check("x_scripts_present", not missing and not broken, f"缺失 {missing} 编译失败 {broken}",
              f"{len(SCRIPTS)} 个脚本齐全且 py_compile 通过")
    plan = open(C.PLAN, encoding="utf-8").read() if os.path.exists(C.PLAN) else ""
    m = FREEZE_RE.search(plan)
    block = None
    if m:
        try:
            block = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            block = f"JSON 解析失败 {e}"
    rep.check("x_plan_frozen_matches_code", block == json.loads(json.dumps(C.FROZEN)),
              "M28_PLAN.md 冻结 JSON 块缺失或与 m28_common.FROZEN 不一致",
              f"计划冻结块 == FROZEN（plan_version={C.FROZEN['plan_version']}）")


# --------------------------------------------------------------------------- A
def recheck_path(r):
    """独立实现的路径判据（与 runner 的 classify 分开写）。"""
    n = r["n_tokens"]
    H = ((n - 1) // 16) * 16
    d, cached, path = r["delta"], r["cached_tokens"], r["path"]
    z = lambda v: v is None or v == 0  # noqa: E731
    if d.get("request_success") != 1 or d.get("prompt_tokens") != n:
        return False
    if r["expected_hit_tokens"] != H:
        return False
    if path in ("cold", "recompute_evicted", "ref_a", "ref_b"):
        return z(d.get("local_hits")) and z(d.get("ext_hits")) and z(d.get("load_bytes")) and cached == 0
    if path == "gpu_hot":
        return d.get("local_hits") == H and z(d.get("ext_hits")) and z(d.get("load_bytes")) and cached == H
    if path == "cpu_hit":
        return z(d.get("local_hits")) and d.get("ext_hits") == H and (d.get("load_bytes") or 0) > 0 and cached == H
    if path == "cpu_evicted":
        return z(d.get("local_hits")) and cached == (d.get("ext_hits") or 0)
    return False


def req_ok(r):
    return (r.get("http_status") == 200 and not r.get("exception") and not r.get("error_event")
            and len(r.get("token_ids") or []) == A["max_tokens"] and r.get("finish_reason") == "length")


def check_a(rep, raw_dir, derived_dir, allow_mock):
    Rr = {arm: _load(raw_dir, f"m28_a_{arm}.jsonl") for arm in A["arms"]}
    missing = [a for a in REQUIRED_A_ARMS if Rr[a] is None]
    if missing:
        rep.add("PENDING", "a_raw_present", f"A arm {missing} 未执行（raw 不存在）；A 未完成，不得记为通过")
        return
    rep.add("PASS", "a_raw_present", "offload/base/ref raw 齐全" + ("；evict 也在" if Rr["evict"] else "；evict（可选）未执行"))
    Rr = {k: v for k, v in Rr.items() if v is not None}
    allrows = [r for rows in Rr.values() for r in rows]
    mocks = sum(1 for r in allrows if r.get("mock"))
    rep.check("a_no_mock", allow_mock or mocks == 0, f"正式 raw 中有 {mocks} 条 mock 记录",
              "无 mock 记录" if not mocks else f"allow_mock：{mocks} 条 mock（仅自检）")
    bad_cnt, n_fail = [], []
    for arm, rows in Rr.items():
        lens = A["evict_prefix_lens"] if arm == "evict" else A["prefix_lens"]
        for p_len in lens:
            for path in A["arms"][arm]["paths"]:
                w = [r for r in rows if r["phase"] == "warmup" and r["p_len"] == p_len and r["path"] == path]
                m = [r for r in rows if r["phase"] == "measure" and r["p_len"] == p_len and r["path"] == path]
                if len(w) != A["warmup_trials"] or len(m) != A["measure_trials"] or \
                        sorted(r["idx"] for r in m) != list(range(A["warmup_trials"], A["warmup_trials"] + A["measure_trials"])):
                    bad_cnt.append((arm, p_len, path, len(w), len(m)))
                f = [r["idx"] for r in m if not r["success"]]
                if f:
                    n_fail.append((arm, p_len, path, f))
    rep.check("a_counts", not bad_cnt, f"warmup/measure 计数不符冻结值：{bad_cnt[:5]}",
              f"每 (arm, P, path) warmup={A['warmup_trials']} measure={A['measure_trials']}，idx 连续")
    rep.check("a_measure_all_success", not n_fail, f"measure 有失败样本（必须在结果中说明，不得剔除）：{n_fail[:5]}",
              "全部 measure 样本 success=True")
    mism = []
    for r in allrows:
        if r["path"] == "filler":
            continue
        indep = recheck_path(r) and req_ok(r)
        if indep != bool(r["success"]):
            mism.append((r["arm"], r["p_len"], r["idx"], r["path"], r["success"], indep))
    rep.check("a_path_recheck", not mism, f"runner 的 success 与独立重算不一致：{mism[:5]}",
              "全部非 filler 样本的路径判据独立重算与 runner 一致")
    cm = [(r["arm"], r["p_len"], r["idx"], r["path"]) for r in allrows if r["path"] != "filler" and r["phase"] != "server_warmup"
          and r["cached_tokens"] != (r["delta"].get("local_hits") or 0) + (r["delta"].get("ext_hits") or 0)]
    rep.check("a_cached_eq_metrics", not cm, f"usage.cached_tokens ≠ Δlocal_hits+Δext_hits：{cm[:5]}",
              "请求级 cached_tokens 与 Prometheus 增量（两个独立来源）逐条相等")
    hit = [r for r in Rr.get("offload", []) if r["path"] == "cpu_hit" and r["phase"] == "measure" and r["success"]]
    per_len = {p: sum(1 for r in hit if r["p_len"] == p) for p in A["prefix_lens"]}
    rep.check("a_gate_external_hit", all(v == A["measure_trials"] for v in per_len.values()),
              f"CPU 命中（external hit）未在每个 P 上被 15/15 证实：{per_len}",
              f"每个 P 的 cpu_hit 均 15/15 由指标证实：{per_len}")
    ratios = sorted({round(r["delta"]["load_bytes"] / r["delta"]["ext_hits"], 3) for r in hit if r["delta"].get("ext_hits")})
    others = [(r["arm"], r["path"]) for r in allrows if r["path"] not in ("cpu_hit", "cpu_evicted", "filler")
              and (r["delta"].get("load_bytes") or 0) != 0]
    rep.check("a_load_bytes_consistent", len(ratios) == 1 and not others,
              f"load_bytes/ext_hit_token 不恒定 {ratios[:5]} 或非 CPU 命中路径有 load：{others[:5]}",
              f"load_bytes/命中 token 恒为 {ratios[0] if ratios else None} B（理论 {C.FROZEN['kv_bytes_per_token_theory']}）")
    tamper = []
    for r in allrows:
        if r["phase"] in ("warmup", "measure") and r["path"] != "filler":
            if r["prompt_sha256"] != C.sha_tokens(C.trial_prompt(r["p_len"], r["idx"])):
                tamper.append((r["arm"], r["p_len"], r["idx"], r["path"]))
    rep.check("a_prompt_pairing", not tamper, f"prompt sha 与冻结生成器不符：{tamper[:5]}",
              "全部样本 prompt sha == 冻结生成器重算值（跨 arm 同一 (P, idx) 同一 prompt）")
    tok = {}
    for r in allrows:
        if r["phase"] == "measure" and req_ok(r):
            tok[(r["arm"], r["path"], r["p_len"], r["idx"])] = r
    def eq(a, pa, b, pb, logprob_exact=False):
        bad, n = [], 0
        for p in A["prefix_lens"]:
            for i in range(A["warmup_trials"], A["warmup_trials"] + A["measure_trials"]):
                x, y = tok.get((a, pa, p, i)), tok.get((b, pb, p, i))
                if x is None or y is None:
                    bad.append((p, i, "缺样本"))
                    continue
                n += 1
                if x["token_ids"] != y["token_ids"]:
                    bad.append((p, i, "token"))
                elif logprob_exact and x["token_logprobs"] != y["token_logprobs"]:
                    bad.append((p, i, "logprob"))
        return bad, n
    for rule, a, pa, b, pb, exact in [
            ("a_corr_offload_roundtrip", "offload", "cpu_hit", "offload", "gpu_hot", True),
            ("a_corr_ref_determinism", "ref", "ref_a", "ref", "ref_b", False),
            ("a_corr_offload_cold_vs_ref", "offload", "cold", "ref", "ref_a", False),
            ("a_corr_base_cold_vs_ref", "base", "cold", "ref", "ref_a", False)]:
        bad, n = eq(a, pa, b, pb, exact)
        rep.check(rule, not bad, f"{a}.{pa} vs {b}.{pb} 不一致 {len(bad)}/{n}：{bad[:5]}",
                  f"{a}.{pa} == {b}.{pb}：{n}/{n}" + ("（token 与 logprob 逐位相等）" if exact else "（token 相等）"))
    sets = {}
    for arm, path in (("offload", "cpu_hit"), ("offload", "gpu_hot")):
        sets[path] = {(p, i) for p in A["prefix_lens"] for i in range(A["warmup_trials"], A["warmup_trials"] + A["measure_trials"])
                      if (arm, path, p, i) in tok and ("ref", "ref_a", p, i) in tok
                      and tok[(arm, path, p, i)]["token_ids"] != tok[("ref", "ref_a", p, i)]["token_ids"]}
    rep.check("a_corr_offload_adds_no_divergence", sets["cpu_hit"] == sets["gpu_hot"],
              f"cpu_hit 相对 ref 的分歧集合 ≠ gpu_hot 的：cpu_hit 独有 {sorted(sets['cpu_hit'] - sets['gpu_hot'])[:5]}",
              f"cpu_hit 与 gpu_hot 相对独立 recompute 的分歧集合相同（各 {len(sets['cpu_hit'])} 个）")
    check_a_csv(rep, Rr, derived_dir)


def check_a_csv(rep, Rr, derived_dir):
    lat = _csv(os.path.join(derived_dir, "m28_a_latency.csv"))
    if lat is None:
        rep.add("PENDING", "a_csv_matches_raw", "m28_a_latency.csv 未生成（先跑 m28_summarize.py）")
        return
    bad = []
    for row in lat:
        rows = Rr.get(row["arm"], [])
        xs = [r["online_ttft_ms"] for r in rows if r["phase"] == "measure" and r["success"]
              and r["p_len"] == int(row["p_len"]) and r["path"] == row["path"]]
        med = round(statistics.median(xs), 3) if xs else None
        got = float(row["ttft_median_ms"]) if row["ttft_median_ms"] not in ("", None) else None
        if int(row["n"]) != len(xs) or got != med:
            bad.append((row["arm"], row["p_len"], row["path"], row["n"], len(xs), got, med))
    rep.check("a_csv_matches_raw", not bad, f"latency CSV 与 raw 重算不符：{bad[:4]}",
              f"latency CSV {len(lat)} 行的 n / median 与 raw 重算相等")


# --------------------------------------------------------------------------- B
def check_b(rep, raw_dir, derived_dir, allow_mock):
    rows = _load(raw_dir, "m28_b_drills.jsonl")
    if rows is None:
        rep.add("PENDING", "b_raw_present", "B 演练未执行（raw 不存在）；B1/B3 未完成，不得记为通过")
        return
    rep.add("PASS", "b_raw_present", f"{len(rows)} 条演练记录")
    mocks = sum(1 for r in rows if r.get("mock"))
    rep.check("b_no_mock", allow_mock or mocks == 0, f"正式 raw 中有 {mocks} 条 mock 记录",
              "无 mock 记录" if not mocks else f"allow_mock：{mocks} 条 mock（仅自检）")
    errs = [(r["label"], r["error"]) for r in rows if r.get("error")]
    rep.check("b_no_drill_error", not errs, f"演练出错/被阻塞（须如实报告）：{errs[:3]}", "全部演练无 error")
    cnt = {}
    for r in rows:
        k = (r["drill"], r.get("shutdown_timeout"))
        cnt[k] = cnt.get(k, 0) + 1
    want = {("b1", None): B["B1"]["repeats"]}
    want.update({("b3", t): B["B3"]["repeats"] for t in B["B3"]["shutdown_timeouts"]})
    rep.check("b_counts", all(cnt.get(k) == v for k, v in want.items()),
              f"演练次数不符冻结值：实际 {cnt} 期望 {want}", f"B1×{B['B1']['repeats']}，B3 每档×{B['B3']['repeats']}"
              + (f"，B2×{cnt.get(('b2', None))}" if ("b2", None) in cnt else "，B2 未执行"))
    ident, infl, term, clean, b3pre, b2res, rst = [], [], [], [], [], [], []
    for r in rows:
        if r.get("error"):
            continue
        f, idt = r["fault"], r["fault"].get("identity") or {}
        if r["drill"] in ("b1", "b2"):
            ok = (f["verified"] and f["sent"] and C.ENGINE_CORE_TITLE in idt.get("cmdline", "")
                  and idt.get("ppid") == r["api_pid"] and idt.get("sid") == r["api_pid"]
                  and f["pid"] == r["engine_pid"] and [c["pid"] for c in r["engine_cores_ready"]] == [r["engine_pid"]])
        else:
            ok = f["verified"] and f["sent"] and f["pid"] == r["api_pid"] and idt.get("sid") == r["api_pid"]
        if not ok:
            ident.append(r["label"])
        ev = r["inflight_evidence"]
        if not (ev["ok"] and ev["running_gauge"] == B["n_stream"] + B["n_nonstream"]
                and ev["stream_first_token_events"] == B["n_stream"]):
            infl.append(r["label"])
        reqs = r["requests"]
        if len(reqs) != B["n_stream"] + B["n_nonstream"] or any(q.get("missing") or not q.get("t_end") for q in reqs):
            term.append(r["label"])
        gr = r.get("gpu_reclaim") or {}
        shm_new = [x for x in r["dev_shm_after"] if x not in r["pre"]["dev_shm"]]
        if r["alive_known_after_cleanup"] or not r["port_free_after"] or shm_new or \
                not (gr.get("ok") or (allow_mock and gr.get("skipped") == "mock")):
            clean.append((r["label"], r["alive_known_after_cleanup"], r["port_free_after"], shm_new, gr))
        if r["drill"] == "b3":
            cal = (r.get("calibration") or {}).get("duration_s")
            t0 = min(q["t_send"] for q in reqs if q.get("t_send"))
            remaining = cal - (r["t_fault"] - t0) if cal else None
            t = r["shutdown_timeout"]
            if t > 0 and (remaining is None or (t < 10 and remaining <= t + 1) or (t >= 10 and cal >= t - 5)):
                b3pre.append((r["label"], round(remaining, 2) if remaining else None, cal))
        if r["drill"] == "b2" and not (r.get("resume") or {}).get("sent"):
            b2res.append(r["label"])
        if r["drill"] == "b1" and not (r.get("restart") or {}).get("ok"):
            rst.append(r["label"])
    rep.check("b_target_identity", not ident, f"注入目标身份核对不完整：{ident}",
              "全部注入目标为本会话进程且角色/ppid/sid 核对通过（B1/B2 = 唯一 VLLM::EngineCore 子进程；B3 = API 进程）")
    rep.check("b_inflight_before_fault", not infl, f"注入前在途证据不足：{infl}",
              f"注入前 {B['n_stream']} 条 streaming 均已收到首 token，且 running 指标 == {B['n_stream'] + B['n_nonstream']}")
    rep.check("b_all_requests_terminal", not term, f"有在途请求缺少终态记录：{term}", "每条在途请求都有终态记录")
    rep.check("b_cleanup_clean", not clean, f"演练后资源未清理干净：{clean[:3]}",
              "每次演练后：无存活的已知 PID、端口释放、/dev/shm 无新增、显存回到基线容差内")
    rep.check("b_b3_timeout_precondition", not b3pre,
              f"B3 有限超时档的前提不成立（不足档剩余工作须 > t+1s；充足档校准时长须 < t-5s）：{b3pre}",
              "B3 不足档剩余工作 > timeout+1s、充足档校准时长 < timeout-5s（由同进程校准批次实测）")
    rep.check("b_b2_resumed", not b2res, f"B2 未确认 SIGCONT：{b2res}", "B2（若执行）均已 SIGCONT")
    rep.check("b_b1_restart_ok", not rst, f"B1 重启失败：{rst}", "每次 B1 后同配置重启并成功完成 1 个请求")
    bcsv = _csv(os.path.join(derived_dir, "m28_b_drills.csv"))
    if bcsv is None:
        rep.add("PENDING", "b_csv_matches_raw", "m28_b_drills.csv 未生成")
    else:
        got = sorted(x["label"] for x in bcsv)
        rep.check("b_csv_matches_raw", got == sorted(r["label"] for r in rows), "B CSV 行与 raw 演练不一一对应",
                  f"B CSV {len(bcsv)} 行与 raw 一一对应")


def check_lifecycle(rep, raw_dir, allow_mock):
    life = _load(raw_dir, "m28_lifecycle.jsonl")
    if life is None:
        rep.add("PENDING", "x_lifecycle", "无 lifecycle 记录（GPU 实验未执行）")
        return
    iv = sorted((d["t_start"], d["t_end"], d.get("arm") or d.get("label")) for d in life)
    ov = [(a[2], b[2]) for a, b in zip(iv, iv[1:]) if b[0] < a[1]]
    rep.check("x_no_engine_overlap", not ov, f"引擎生命周期重叠（A/B 不得同时运行）：{ov[:3]}",
              f"{len(iv)} 个 server 生命周期互不重叠")
    a_life = [d for d in life if d.get("exp") == "A"]
    bad, env_bad, mk = [], [], []
    for d in a_life:
        gr = d.get("gpu_reclaim") or {}
        if d.get("error") or d.get("alive_known_after_cleanup") or not d.get("port_free_after") or \
                not (gr.get("ok") or (allow_mock and gr.get("skipped") == "mock")):
            bad.append((d["arm"], d.get("error"), d.get("alive_known_after_cleanup"), gr))
        fp = d.get("env_fingerprint") or {}
        if not (all((fp.get("server_env_unset_ok") or {}).values())
                and fp.get("vllm_src_dirty_files") == C.FROZEN["vllm_src_dirty_files_expected"]
                and str(fp.get("vllm_src_head", "")).startswith(C.FROZEN["vllm_source_ref"])
                and (fp.get("model_identity") or {}).get("blob_matches_frozen")):
            env_bad.append(d["arm"])
        if not d.get("mock") and A["arms"][d["arm"]]["connector"]:
            m = d.get("log_markers") or {}
            if not (m.get("offloading_connector", 0) > 0 and m.get("mmap_created", 0) > 0):
                mk.append((d["arm"], m))
    rep.check("a_lifecycle_clean", not bad, f"A server 生命周期有错误/残留：{bad[:3]}",
              f"{len(a_life)} 个 A server：无 error、无残留 PID、端口释放、显存回收")
    rep.check("a_env_fingerprint", not env_bad, f"环境指纹不符（LAB 钩子变量 / vllm-src HEAD 或脏文件数 / 模型 blob）：{env_bad}",
              "VLLM_LAB_* 均未设置；vllm-src HEAD=ee0da84、脏文件数 22；模型 blob sha 符合冻结值")
    rep.check("a_connector_log_evidence", not mk, f"connector arm 日志缺少 OffloadingConnector/mmap 证据：{mk}",
              "connector arm 日志含 OffloadingConnector 与 mmap 创建记录（cudaHostRegister 失败次数见 lifecycle）")


def validate(raw_dir, derived_dir, allow_mock=False, static=True):
    rep = R()
    if static:
        check_static(rep)
    check_a(rep, raw_dir, derived_dir, allow_mock)
    check_b(rep, raw_dir, derived_dir, allow_mock)
    check_lifecycle(rep, raw_dir, allow_mock)
    return rep


def print_rep(rep, verbose=True):
    for s, rule, detail in rep.rows:
        if s == "PASS" and not verbose:
            continue
        print(f"[{s:^7}] {rule:<36} {detail}")
    c = {k: sum(1 for s, _, _ in rep.rows if s == k) for k in ("PASS", "FAIL", "PENDING")}
    print(f"\n合计：{c['PASS']} PASS / {c['FAIL']} FAIL / {c['PENDING']} PENDING")
    return c


# --------------------------------------------------------------------------- selftest
def _rewrite(path, fn):
    rows = C.load_jsonl(path)
    rows = fn(rows)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _first(rows, pred):
    return next(i for i, r in enumerate(rows) if pred(r))


def _inj_ext_zero(rows):
    i = _first(rows, lambda r: r["path"] == "cpu_hit" and r["phase"] == "measure")
    rows[i]["delta"]["ext_hits"] = 0
    return rows


def _inj_token_flip(rows):
    i = _first(rows, lambda r: r["path"] == "cpu_hit" and r["phase"] == "measure")
    rows[i]["token_ids"][5] += 1
    return rows


def _inj_drop(rows):
    i = _first(rows, lambda r: r["path"] == "gpu_hot" and r["phase"] == "measure")
    return rows[:i] + rows[i + 1:]


def _inj_cached(rows):
    i = _first(rows, lambda r: r["path"] == "gpu_hot" and r["phase"] == "measure")
    rows[i]["cached_tokens"] += 16
    return rows


def _inj_prompt(rows):
    i = _first(rows, lambda r: r["path"] == "cold" and r["phase"] == "measure")
    rows[i]["prompt_sha256"] = "0" * 64
    return rows


def _inj_b1_target(rows):
    i = _first(rows, lambda r: r["drill"] == "b1")
    rows[i]["fault"]["identity"]["cmdline"] = "/usr/bin/python some_other_job.py"
    return rows


def _inj_leftover(rows):
    rows[0]["alive_known_after_cleanup"] = [424242]
    return rows


def _inj_b3_pre(rows):
    i = _first(rows, lambda r: r["drill"] == "b3" and r["shutdown_timeout"] == 2)
    rows[i]["calibration"]["duration_s"] = 1.5
    return rows


def _inj_missing_req(rows):
    rows[0]["requests"][0] = {"missing": True}
    return rows


def _inj_overlap(rows):
    rows[1]["t_start"] = rows[0]["t_start"] + 1
    return rows


def _inj_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    rows[0]["ttft_median_ms"] = str(float(rows[0]["ttft_median_ms"]) + 1.0)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


INJECTIONS = [  # (名字, 目标规则, 文件, 变换, 是否关 allow_mock)
    ("cpu_hit 的 Δext_hits 改 0（success 保持 True）", "a_path_recheck", "m28_a_offload.jsonl", _inj_ext_zero, False),
    ("cpu_hit 的一个输出 token +1", "a_corr_offload_roundtrip", "m28_a_offload.jsonl", _inj_token_flip, False),
    ("删掉一条 gpu_hot measure 样本", "a_counts", "m28_a_base.jsonl", _inj_drop, False),
    ("gpu_hot 的 cached_tokens +16", "a_cached_eq_metrics", "m28_a_offload.jsonl", _inj_cached, False),
    ("篡改一条 cold 的 prompt sha", "a_prompt_pairing", "m28_a_base.jsonl", _inj_prompt, False),
    ("latency CSV 首行 median +1ms", "a_csv_matches_raw", "m28_a_latency.csv", None, False),
    ("mock 数据当正式数据校验", "a_no_mock", None, None, True),
    ("B1 目标 cmdline 换成别的进程", "b_target_identity", "m28_b_drills.jsonl", _inj_b1_target, False),
    ("演练后残留一个已知 PID", "b_cleanup_clean", "m28_b_drills.jsonl", _inj_leftover, False),
    ("B3 不足档校准时长改成 1.5s", "b_b3_timeout_precondition", "m28_b_drills.jsonl", _inj_b3_pre, False),
    ("一条在途请求终态缺失", "b_all_requests_terminal", "m28_b_drills.jsonl", _inj_missing_req, False),
    ("两个 server 生命周期重叠", "x_no_engine_overlap", "m28_lifecycle.jsonl", _inj_overlap, False),
]


def selftest(raw_dir, derived_dir, report_path=None):
    base = validate(raw_dir, derived_dir, allow_mock=True, static=False)
    print("== 基准数据集（allow_mock）==")
    print_rep(base, verbose=False)
    results = []
    for name, rule, fname, fn, strict in INJECTIONS:
        with tempfile.TemporaryDirectory(prefix="m28_inject_") as td:
            for n in os.listdir(raw_dir):
                p = os.path.join(raw_dir, n)
                if os.path.isfile(p) and (n.endswith(".jsonl") or n.endswith(".csv")):
                    shutil.copy(p, td)
            if os.path.realpath(derived_dir) != os.path.realpath(raw_dir):
                for n in os.listdir(derived_dir):
                    if n.endswith(".csv"):
                        shutil.copy(os.path.join(derived_dir, n), td)
            if fname and fname.endswith(".csv"):
                _inj_csv(os.path.join(td, fname))
            elif fname:
                _rewrite(os.path.join(td, fname), fn)
            rep = validate(td, td, allow_mock=not strict, static=False)
            before, after = base.status(rule), rep.status(rule)
            detected = before == "PASS" and after == "FAIL"
            results.append({"injection": name, "rule": rule, "baseline": before, "injected": after,
                            "detected": detected})
            print(f"  [{'检出' if detected else '未检出'}] {name:<36} → {rule}: {before} → {after}")
    ok = all(r["detected"] for r in results)
    print(f"\n反例检出 {sum(r['detected'] for r in results)}/{len(results)}")
    if report_path:
        C.save_json(report_path, {"base_raw_dir": raw_dir, "base_is_mock": True,
                                  "note": "反例只在临时目录注入；基准数据为 mock 演练产物，不是实验数据",
                                  "baseline": base.rows, "injections": results, "all_detected": ok,
                                  "lab_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=C.ROOT,
                                                             capture_output=True, text=True).stdout.strip()})
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default=C.RAW)
    ap.add_argument("--derived-dir", default=C.OUT)
    ap.add_argument("--allow-mock", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", default=None, help="selftest 报告 JSON 路径")
    a = ap.parse_args()
    if a.selftest:
        return 0 if selftest(a.raw_dir, a.derived_dir, a.report) else 1
    c = print_rep(validate(a.raw_dir, a.derived_dir, a.allow_mock))
    return 1 if c["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
