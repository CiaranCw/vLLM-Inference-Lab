#!/usr/bin/env python3
"""M24 · 编排器：每个 DCP 档独立子进程，断点续跑，打包回传。

    python scripts/m24_run_all.py --tag h20_formal --model $M24_MODEL --dcp 1,2
    python scripts/m24_run_all.py --tag h20_formal --model $M24_MODEL --dcp 2 --profile
    python scripts/m24_run_all.py --tag h20_formal --model $M24_MODEL --pcp-negative
    python scripts/m24_run_all.py --tag h20_formal --package
    python scripts/m24_run_all.py --tag h20_formal --transfer-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "outputs", "m24", "raw")
LOGS = os.path.join(ROOT, "logs", "m24")

ENV_EXTRA = {
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def shard_done(tag: str, dcp: int, shard: str) -> bool:
    p = os.path.join(RAW, f"{tag}_dcp{dcp}_{shard}.jsonl")
    return os.path.exists(p) and os.path.getsize(p) > 0


def run_one(tag, dcp, shard, model, quick, force) -> int:
    if shard_done(tag, dcp, shard) and not force:
        print(f"[run_all] SKIP {tag} dcp={dcp} {shard}（已有数据；--force 可强制重跑）")
        return 0
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"{tag}_dcp{dcp}_{shard}.log")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "m24_dcp.py"),
        "--dcp",
        str(dcp),
        "--shard",
        shard,
        "--tag",
        tag,
        "--model",
        model,
    ]
    if quick:
        cmd.append("--quick")
    env = dict(os.environ)
    env.update(ENV_EXTRA)
    print(f"[run_all] === tag={tag} dcp={dcp} shard={shard} === -> {log}", flush=True)
    t0 = time.perf_counter()
    with open(log, "w") as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n# env: {ENV_EXTRA}\n")
        lf.flush()
        rc = subprocess.call(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT
        )
    print(f"[run_all] dcp={dcp} {shard} rc={rc} elapsed={time.perf_counter() - t0:.1f}s")
    return rc


def transfer_test(tag: str) -> str:
    d = os.path.join(ROOT, "outputs", "m24")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"transfer_test_{tag}.txt")
    payload = (
        f"m24 transfer test\ntag={tag}\n"
        f"created={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"host={os.uname().nodename}\n" + ("x" * 4096) + "\n"
    )
    with open(path, "w") as f:
        f.write(payload)
    h = hashlib.sha256(payload.encode()).hexdigest()
    with open(path + ".sha256", "w") as f:
        f.write(f"{h}  {os.path.basename(path)}\n")
    print(f"[run_all] transfer test file: {path}")
    print(f"[run_all] sha256 = {h}")
    print("\n[run_all] 现在到**本地** pull（H20 未必能反连）：")
    print(f"  rsync -avP <USER>@<H20_HOST>:{path}{{,.sha256}} ./m24_transfer_test/")
    return path


def package(tag: str) -> str:
    import tarfile

    out_dir = os.path.join(ROOT, "outputs", "m24")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    tar_path = os.path.join(ROOT, f"m24_{tag}_{stamp}.tar.gz")
    members = []
    for base in (
        os.path.join(ROOT, "outputs", "m24"),
        os.path.join(ROOT, "logs", "m24"),
    ):
        for dirpath, _, files in os.walk(base):
            for fn in files:
                members.append(os.path.join(dirpath, fn))
    with tarfile.open(tar_path, "w:gz") as tf:
        for m in sorted(members):
            tf.add(m, arcname=os.path.relpath(m, ROOT))
    h = hashlib.sha256()
    with open(tar_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    with open(tar_path + ".sha256", "w") as f:
        f.write(f"{sha}  {os.path.basename(tar_path)}\n")
    manifest = {
        "tag": tag,
        "created": stamp,
        "n_files": len(members),
        "tar": os.path.basename(tar_path),
        "sha256": sha,
        "size_bytes": os.path.getsize(tar_path),
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"package_{tag}.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[run_all] packaged {len(members)} files -> {tar_path}")
    print(f"[run_all] sha256 = {sha}")
    print("\n[run_all] 也可不打包、整目录拷回 outputs/m24 与 logs/m24。")
    return tar_path


def pcp_negative(tag: str, model: str) -> int:
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RAW, exist_ok=True)
    log = os.path.join(LOGS, f"{tag}_pcp_negative.log")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "m24_pcp_negative.py"),
        "--tag",
        tag,
        "--model",
        model,
    ]
    env = dict(os.environ)
    env.update(ENV_EXTRA)
    with open(log, "w") as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n")
        lf.flush()
        rc = subprocess.call(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT
        )
    print(f"[run_all] pcp-negative rc={rc} log={log}")
    return 0  # 预期非零；不阻断后续 package


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--dcp", default="1,2")
    p.add_argument("--model", default=os.environ.get("M24_MODEL", ""))
    p.add_argument("--quick", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--package", action="store_true")
    p.add_argument("--transfer-test", action="store_true")
    p.add_argument("--pcp-negative", action="store_true")
    p.add_argument("--no-summarize", action="store_true")
    a = p.parse_args()

    if a.transfer_test:
        transfer_test(a.tag)
        return 0
    if a.package:
        package(a.tag)
        return 0
    if a.pcp_negative:
        if not a.model:
            raise SystemExit("需要 --model")
        return pcp_negative(a.tag, a.model)

    if not a.model or not os.path.isdir(a.model):
        raise SystemExit("[run_all] --model 必须是本地模型目录")

    dcps = [int(x) for x in a.dcp.split(",") if x.strip()]
    shard = "profile" if a.profile else "run"
    for dcp in dcps:
        rc = run_one(a.tag, dcp, shard, a.model, a.quick, a.force)
        if rc != 0:
            print(
                f"[run_all] ABORT: dcp={dcp} {shard} rc={rc}，"
                f"见 logs/m24/{a.tag}_dcp{dcp}_{shard}.log"
            )
            return rc

    if not a.no_summarize and not a.profile:
        rc = subprocess.call(
            [
                sys.executable,
                os.path.join(ROOT, "scripts", "m24_dcp.py"),
                "--shard",
                "summarize",
                "--tag",
                a.tag,
            ],
            cwd=ROOT,
        )
        if rc != 0:
            print("[run_all] summarize 失败")
            return rc
    print("[run_all] all done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
