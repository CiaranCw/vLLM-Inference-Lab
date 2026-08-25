"""M22 · 编排器：每个 TP 档一个独立子进程，支持断点续跑与打包回传。

    python scripts/m22_run_all.py --tag h20_formal --tp 1,2,4
    python scripts/m22_run_all.py --tag h20_formal --tp 2,4 --profile   # P1 通信证据
    python scripts/m22_run_all.py --tag h20_formal --package            # 只打包

为什么要有断点续跑：H20 是 SSH 临时机器，连接可能中断。
已完成的 shard（`outputs/m22/raw/<tag>_tp<N>_<shard>.jsonl` 存在且非空）默认跳过，
所以重跑本命令等于「从中断处继续」，不会覆盖已采到的数据，也不浪费机时。
要强制重跑必须显式 `--force`。

为什么每档一个子进程：vLLM 的 KV 显存 `del engine` 归还不掉（M6 实证），
且 TP>1 会拉起 worker 进程组，进程内切换 TP 档不可靠。
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
RAW = os.path.join(ROOT, "outputs", "m22", "raw")
LOGS = os.path.join(ROOT, "logs", "m22")

ENV_EXTRA = {
    # llm_engine.py:224-226 —— 不设为 0 会强制走多进程 EngineCore，内省全部失效
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


def shard_done(tag: str, tp: int, shard: str) -> bool:
    p = os.path.join(RAW, f"{tag}_tp{tp}_{shard}.jsonl")
    return os.path.exists(p) and os.path.getsize(p) > 0


def run_one(tag, tp, shard, model, quick, extra_env, force) -> int:
    if shard_done(tag, tp, shard) and not force:
        print(f"[run_all] SKIP {tag} tp={tp} {shard}（已有数据；--force 可强制重跑）")
        return 0
    os.makedirs(LOGS, exist_ok=True)
    log = os.path.join(LOGS, f"{tag}_tp{tp}_{shard}.log")
    cmd = [
        sys.executable,
        os.path.join(ROOT, "scripts", "m22_tensor_parallel.py"),
        "--tp",
        str(tp),
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
    env.update(extra_env)
    print(f"[run_all] === tag={tag} tp={tp} shard={shard} === -> {log}", flush=True)
    t0 = time.perf_counter()
    # 日志实时落盘：子进程直接写文件，SSH 断线也不丢
    with open(log, "w") as lf:
        lf.write(f"# cmd: {' '.join(cmd)}\n# env: {ENV_EXTRA}\n")
        lf.flush()
        rc = subprocess.call(
            cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT
        )
    print(f"[run_all] tp={tp} {shard} rc={rc} elapsed={time.perf_counter() - t0:.1f}s")
    return rc


def transfer_test(tag: str) -> str:
    """在远端生成一个几 KB 的测试文件 + sha256，供本地尽早做一次 pull 演练。

    为什么必须早做：本地能 SSH 到 H20，不代表 H20 能反向连到本地，
    所以回传只能由本地主动 pull。而 rsync 是否存在、路径/权限/key 是否对，
    这些错误必须在 T+0:30 就暴露，绝不能等到 T+3:25 打包时才第一次发现。
    """
    d = os.path.join(ROOT, "outputs", "m22")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"transfer_test_{tag}.txt")
    payload = (
        f"m22 transfer test\ntag={tag}\n"
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
    print("\n[run_all] 现在到**本地**执行 pull 演练（二选一）：")
    print(f"  rsync -avP <USER>@<H20_HOST>:{path}{{,.sha256}} ./m22_transfer_test/")
    print(f"  scp <USER>@<H20_HOST>:'{path} {path}.sha256' ./m22_transfer_test/")
    print("  cd m22_transfer_test && sha256sum -c *.sha256   # 必须 OK 才算链路通")
    return path


def package(tag: str) -> str:
    """打包 + checksum。数据没带走 == 实验失败，所以这一步必须能独立执行。"""
    import tarfile

    out_dir = os.path.join(ROOT, "outputs", "m22")
    stamp = time.strftime("%Y%m%dT%H%M%S")
    tar_path = os.path.join(ROOT, f"m22_{tag}_{stamp}.tar.gz")
    members = []
    for base in (
        os.path.join(ROOT, "outputs", "m22"),
        os.path.join(ROOT, "logs", "m22"),
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
    with open(os.path.join(out_dir, f"package_{tag}.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[run_all] packaged {len(members)} files -> {tar_path}")
    print(f"[run_all] sha256 = {sha}")
    print("\n[run_all] ⚠ 回传必须由**本地主动 pull**（H20 未必能反向连到本地）：")
    print(f"  rsync -avP <USER>@<H20_HOST>:{tar_path}{{,.sha256}} <LOCAL_DEST>/")
    print(
        f"  # 无 rsync 时 fallback: scp <USER>@<H20_HOST>:'{tar_path} {tar_path}.sha256' <LOCAL_DEST>/"
    )
    print(f"  cd <LOCAL_DEST> && sha256sum -c {os.path.basename(tar_path)}.sha256")
    print("  # checksum 一致才算回传成功。没带回的数据 = 没有实验。")
    return tar_path


def main():
    p = argparse.ArgumentParser(description="M22 编排器（断点续跑 + 打包）")
    p.add_argument("--tag", required=True)
    p.add_argument("--tp", default="1,2,4", help="逗号分隔，如 1,2,4")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--profile", action="store_true", help="跑 P1 通信证据 shard")
    p.add_argument("--force", action="store_true", help="重跑已完成的 shard（会覆盖）")
    p.add_argument("--package", action="store_true", help="只打包，不跑实验")
    p.add_argument(
        "--transfer-test",
        action="store_true",
        help="只在远端生成小测试文件+sha256，供本地尽早演练 pull（不跑实验）",
    )
    p.add_argument("--no-summarize", action="store_true")
    a = p.parse_args()

    if a.transfer_test:
        transfer_test(a.tag)
        return 0
    if a.package:
        package(a.tag)
        return 0

    tps = [int(x) for x in a.tp.split(",") if x.strip()]
    shard = "profile" if a.profile else "run"
    for tp in tps:
        rc = run_one(a.tag, tp, shard, a.model, a.quick, {}, a.force)
        if rc != 0:
            print(
                f"[run_all] ABORT: tp={tp} {shard} 失败（rc={rc}），见 "
                f"logs/m22/{a.tag}_tp{tp}_{shard}.log\n"
                "[run_all] 已完成的 shard 已落盘；修复后重跑本命令会自动从中断处继续。"
            )
            return rc

    if not a.no_summarize:
        rc = subprocess.call(
            [
                sys.executable,
                os.path.join(ROOT, "scripts", "m22_tensor_parallel.py"),
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
    sys.exit(main())
