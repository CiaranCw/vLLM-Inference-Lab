#!/usr/bin/env python3
"""M27 · 编排器：Gate → E0 → E1 → E2 → E3 → E4 →（E5 可选）→ 打包。

    python scripts/m27_run_all.py --tag h20_formal --gpu-map --model $M27_MODEL   # preflight 末尾自动调用
    python scripts/m27_run_all.py --tag h20_formal --transfer-test
    python scripts/m27_run_all.py --tag h20_formal --stage e0 --model $M27_MODEL
    python scripts/m27_run_all.py --tag h20_formal --stage e1 --model $M27_MODEL
    python scripts/m27_run_all.py --tag h20_formal --stage e2 --model $M27_MODEL [--arms pd_1p1d]
    python scripts/m27_run_all.py --tag h20_formal --stage e3cal --model $M27_MODEL
    python scripts/m27_run_all.py --tag h20_formal --stage e3 --model $M27_MODEL [--arms ...] [--shapes ...]
    python scripts/m27_run_all.py --tag h20_formal --stage e4cal / e4 / e5cal / e5 ...
    python scripts/m27_run_all.py --tag h20_formal --package

一个 arm 一次 Topology（每实例一个 vllm serve 进程 = 一个引擎）；本进程只跑客户端，不建引擎。
断点续跑：raw/<tag>_<exp>_<arm>.jsonl 末尾有 phase=arm_done 才算完成；
未完成或 --force 时，旧文件**移动备份**到 raw/backup_<stamp>/，绝不删除。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m27_common as C  # noqa: E402

STAGES = ("e0", "e1", "e2", "e3cal", "e3", "e4cal", "e4", "e5cal", "e5")


# --------------------------------------------------------------------------- Gate 0: GPU map
def parse_topo(text: str) -> dict:
    """解析 `nvidia-smi topo -m` 的 GPU×GPU 子矩阵。"""
    lines = [ln.rstrip("\n") for ln in text.splitlines() if ln.strip()]
    header = None
    rows = {}
    for ln in lines:
        clean = re.sub(r"\x1b\[[0-9;]*m", "", ln)
        toks = clean.split()
        if header is None and toks and toks[0].startswith("GPU"):
            header = [t for t in toks if re.fullmatch(r"GPU\d+", t)]
            continue
        if header and toks and re.fullmatch(r"GPU\d+", toks[0]):
            rows[toks[0]] = toks[1:1 + len(header)]
    return {"gpus": header or [], "matrix": rows}


def nv_clique(topo: dict) -> list[int]:
    """最大的两两 NV# 互联 GPU 集合（贪心 + 穷举小规模；GPU ≤ 8 足够）。"""
    from itertools import combinations
    gpus = [int(g[3:]) for g in topo["gpus"]]
    idx = {g: i for i, g in enumerate(gpus)}

    def nv(a, b):
        row = topo["matrix"].get(f"GPU{a}") or []
        v = row[idx[b]] if idx[b] < len(row) else ""
        return v.startswith("NV")

    for k in range(len(gpus), 1, -1):
        for combo in combinations(gpus, k):
            if all(nv(a, b) for a, b in combinations(combo, 2)):
                return list(combo)
    return []


def write_gpu_map(tag: str, model_dir: str) -> int:
    topo_txt = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
    topo = parse_topo(topo_txt)
    clique = nv_clique(topo)
    cfg = C.load_json(os.path.join(model_dir, "config.json")) if model_dir else None
    mpe = (cfg or {}).get("max_position_embeddings")
    fallback = mpe is not None and mpe < C.FROZEN["engine"]["max_model_len"]
    out = {
        "tag": tag, "hostname": C.env_fingerprint()["hostname"],
        "visible_gpus": [int(g[3:]) for g in topo["gpus"]],
        "nv_clique": clique, "gpu_ids": clique,
        "e0_e3_allowed": len(clique) >= 2, "e4_allowed": len(clique) >= 4,
        "e5_allowed": len(clique) >= 8,
        "topo_raw": topo_txt, "topo_parsed": topo,
        "model_max_position_embeddings": mpe, "max_model_len_fallback": bool(fallback),
        "rule": "gpu_ids = 最大两两 NV# 互联集合；不假设 GPU0–7 同一 NVLink 域",
    }
    path = C.save_json(os.path.join(C.CONFIGS, f"gpu_map_{tag}.json"), out)
    print(f"[gpu-map] clique={clique} e4={out['e4_allowed']} e5={out['e5_allowed']} "
          f"mpe={mpe} fallback={fallback} -> {os.path.relpath(path, C.ROOT)}")
    return 0 if out["e0_e3_allowed"] and mpe is not None else 1


# --------------------------------------------------------------------------- raw bookkeeping
def raw_path(tag, exp, arm):
    return os.path.join(C.RAW, f"{tag}_{exp}_{arm}.jsonl")


def arm_done(path) -> bool:
    if not os.path.exists(path):
        return False
    with open(path) as f:
        for line in f:
            if '"phase": "arm_done"' in line:
                return True
    return False


def backup_existing(paths):
    stamp = time.strftime("%Y%m%dT%H%M%S")
    bdir = os.path.join(C.RAW, f"backup_{stamp}")
    for p in paths:
        if os.path.exists(p):
            sub = os.path.join(bdir, os.path.basename(os.path.dirname(p)))
            os.makedirs(sub, exist_ok=True)
            shutil.move(p, os.path.join(sub, os.path.basename(p)))
            print(f"[run_all] 备份（不删除）{os.path.relpath(p, C.ROOT)} -> {os.path.relpath(sub, C.ROOT)}/")


def prepare(tag, exp, arm, force) -> str | None:
    rp = raw_path(tag, exp, arm)
    if arm_done(rp) and not force:
        print(f"[run_all] SKIP {exp} {arm}（已完成；--force 重跑会先备份）")
        return None
    prefix = f"{tag}_{exp}_{arm}"
    stale = [rp, os.path.join(C.RAW, f"{prefix}_proxy.jsonl")]
    for d in (C.CONFIGS, C.METRICS):
        if os.path.isdir(d):
            stale += sorted(os.path.join(d, n) for n in os.listdir(d) if n.startswith(prefix + "_"))
    if exp.endswith("cal"):
        stale.append(os.path.join(C.CONFIGS, f"{tag}_{exp}_rates.json"))
    backup_existing(stale)
    return rp


def mark_done(rp, tag, exp, arm, extra=None):
    C.append_jsonl(rp, {"tag": tag, "experiment": exp, "arm": arm, "phase": "arm_done",
                        "success": True, **(extra or {})})


# --------------------------------------------------------------------------- stages
def _contexts(tag, values):
    mp = C.max_prompt_len(tag)
    mml = C.effective_engine(tag)["max_model_len"]
    keep = [v for v in values if v <= mp and v < mml]
    dropped = [v for v in values if v not in keep]
    if dropped:
        print(f"[run_all] 合法回退：删除超出 max_prompt/max_model_len 的点 {dropped}")
    return keep, dropped


def run_stage(a) -> int:
    import m27_workload as W
    from m27_launch import Topology

    tag, exp = a.tag, a.stage
    gm = C.load_json(os.path.join(C.CONFIGS, f"gpu_map_{tag}.json")) or {}
    gpu_ids = C.gpu_map(tag)
    wu, ms = C.FROZEN["warmup"], C.FROZEN["measure"]

    def arms_for(default):
        return [x for x in (a.arms.split(",") if a.arms else default) if x]

    if exp in ("e4cal", "e4") and not gm.get("e4_allowed"):
        raise SystemExit("[run_all] Gate：NV 互联 GPU < 4，E4 不允许（见 gpu_map）")
    if exp in ("e5cal", "e5") and not gm.get("e5_allowed"):
        raise SystemExit("[run_all] Gate 8：8 卡未两两 NV 互联，E5 不执行（不假设拓扑）")

    if exp == "e0":
        for arm, fault in (("pd_1p1d", True), ("colo1_mbt8192", False)):
            rp = prepare(tag, exp, arm, a.force)
            if rp is None:
                continue
            with Topology(tag, exp, arm, gpu_ids, a.model, allow_fault=fault) as topo:
                asyncio.run(W.run_e0(topo, rp))
            mark_done(rp, tag, exp, arm)
        return 0

    if exp == "e1":
        ctx, dropped = _contexts(tag, C.FROZEN["e1"]["contexts"])
        arm = "pd_1p1d"
        rp = prepare(tag, exp, arm, a.force)
        if rp:
            with Topology(tag, exp, arm, gpu_ids, a.model) as topo:
                asyncio.run(W.run_e1(topo, rp, ctx, C.FROZEN["e1"]["output_tokens"], wu, ms))
            mark_done(rp, tag, exp, arm, {"contexts": ctx, "dropped_contexts": dropped})
        return 0

    if exp == "e2":
        p_lens, dropped = _contexts(tag, C.FROZEN["e2"]["p_lens"])
        for arm in arms_for(C.FROZEN["e2"]["arms"]):
            rp = prepare(tag, exp, arm, a.force)
            if rp is None:
                continue
            with Topology(tag, exp, arm, gpu_ids, a.model) as topo:
                asyncio.run(W.run_e2(topo, rp, p_lens, wu, ms))
            mark_done(rp, tag, exp, arm, {"p_lens": p_lens, "dropped_p_lens": dropped})
        return 0

    online = {"e3": ("e3", C.FROZEN["e3"]["arms"], list(C.FROZEN["e3"]["shapes"]) + ["mixture"],
                     C.FROZEN["e3"]["calibration_arm"]),
              "e4": ("e4", C.FROZEN["e4"]["arms"], C.FROZEN["e4"]["shapes"],
                     C.FROZEN["e4"]["calibration_arm"]),
              "e5": ("e5", C.FROZEN["e5_optional"]["arms"], C.FROZEN["e4"]["shapes"],
                     C.FROZEN["e5_optional"]["arms"][0])}
    base = exp.replace("cal", "")
    _, arms, shapes, cal_arm = online[base]
    shapes = a.shapes.split(",") if a.shapes else shapes
    rates_path = os.path.join(C.CONFIGS, f"{tag}_{base}cal_rates.json")

    if exp.endswith("cal"):
        rp = prepare(tag, exp, cal_arm, a.force)
        if rp:
            with Topology(tag, exp, cal_arm, gpu_ids, a.model) as topo:
                res = asyncio.run(W.calibrate(topo, rp, exp, shapes, rates_path, wu, ms))
            mark_done(rp, tag, exp, cal_arm, {"rates_file": os.path.relpath(rates_path, C.ROOT)})
            bad = [s for s, v in res["shapes"].items() if v["c_cal"] is None or v["ladder_exhausted"]]
            if bad:
                print(f"[run_all] STOP：校准未得到有效 C_cal 的 shape：{bad}（见 {rates_path}）")
                return 3
        return 0

    rates = C.load_json(rates_path)
    if not rates:
        raise SystemExit(f"[run_all] 缺 {os.path.relpath(rates_path, C.ROOT)}：先跑 --stage {base}cal")
    rates_sha = C.sha256_file(rates_path)
    for arm in arms_for(arms):
        rp = prepare(tag, exp, arm, a.force)
        if rp is None:
            continue
        with Topology(tag, exp, arm, gpu_ids, a.model) as topo:
            for shape in shapes:
                c_cal = rates["shapes"][shape]["c_cal"]
                if c_cal is None:
                    raise SystemExit(f"[run_all] shape={shape} 无 C_cal")
                asyncio.run(W.warmup_online(topo, rp, exp, shape))
                for seed in C.FROZEN["e3"]["seeds"]:
                    for frac in C.FROZEN["e3"]["load_fractions"]:
                        asyncio.run(W.run_online(topo, rp, exp, shape, frac * c_cal, seed, frac,
                                                 rates_sha, C.FROZEN["e3"]["duration_s"]))
        mark_done(rp, tag, exp, arm, {"shapes": shapes, "rates_file_sha256": rates_sha})
    return 0


# --------------------------------------------------------------------------- transfer / package
def transfer_test(tag: str) -> str:
    os.makedirs(C.OUT, exist_ok=True)
    path = os.path.join(C.OUT, f"transfer_test_{tag}.txt")
    payload = (f"m27 transfer test\ntag={tag}\ncreated={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
               f"host={os.uname().nodename}\n" + ("x" * 4096) + "\n")
    with open(path, "w") as f:
        f.write(payload)
    sha = C.sha256_file(path)
    with open(path + ".sha256", "w") as f:
        f.write(f"{sha}  {os.path.basename(path)}\n")
    print(f"[run_all] transfer test: {path}\n[run_all] sha256 = {sha}")
    print(f"本地主动 pull：rsync -avP <USER>@<H20_HOST>:{path}{{,.sha256}} ./m27_transfer_test/")
    return path


def package(tag: str) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S")
    members = []
    for base in (C.OUT, C.LOGS):
        for dp, _, files in os.walk(base):
            for fn in files:
                members.append(os.path.join(dp, fn))
    members.sort()
    sums_path = os.path.join(C.OUT, f"SHA256SUMS_{tag}_{stamp}.txt")
    with open(sums_path, "w") as f:
        for m in members:
            f.write(f"{C.sha256_file(m)}  {os.path.relpath(m, C.ROOT)}\n")
    members.append(sums_path)
    tar_path = os.path.join(C.ROOT, f"m27_{tag}_{stamp}.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tf:
        for m in members:
            tf.add(m, arcname=os.path.relpath(m, C.ROOT))
    sha = C.sha256_file(tar_path)
    with open(tar_path + ".sha256", "w") as f:
        f.write(f"{sha}  {os.path.basename(tar_path)}\n")
    C.save_json(os.path.join(C.OUT, f"package_{tag}.json"), {
        "tag": tag, "created": stamp, "n_files": len(members), "tar": os.path.basename(tar_path),
        "sha256": sha, "size_bytes": os.path.getsize(tar_path),
        "per_file_sums": os.path.relpath(sums_path, C.ROOT)})
    print(f"[run_all] packaged {len(members)} files -> {tar_path}\n[run_all] sha256 = {sha}")
    return tar_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True, choices=C.ALLOWED_TAGS)
    p.add_argument("--model", default=os.environ.get("M27_MODEL", ""))
    p.add_argument("--stage", choices=STAGES)
    p.add_argument("--arms", default="")
    p.add_argument("--shapes", default="")
    p.add_argument("--force", action="store_true")
    p.add_argument("--gpu-map", action="store_true")
    p.add_argument("--transfer-test", action="store_true")
    p.add_argument("--package", action="store_true")
    a = p.parse_args()
    if a.transfer_test:
        transfer_test(a.tag)
        return 0
    if a.package:
        package(a.tag)
        return 0
    if a.gpu_map:
        return write_gpu_map(a.tag, a.model)
    if not a.stage:
        p.error("需要 --stage / --gpu-map / --transfer-test / --package 之一")
    if not a.model or not os.path.isdir(a.model):
        raise SystemExit("[run_all] --model / M27_MODEL 必须是本地模型目录（不联网下载）")
    C.formal_guard(a.tag, 2)
    t0 = time.time()
    rc = run_stage(a)
    print(f"[run_all] stage={a.stage} rc={rc} elapsed={time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
