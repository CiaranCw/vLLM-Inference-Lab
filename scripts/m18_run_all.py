#!/usr/bin/env python3
"""M18 · 独立进程编排器：每个 (模型 × 量化配置) 在独立子进程跑 m18_quant.py。

一进程一引擎（AGENTS.md §4）。串行执行并持有 lab 级互斥锁 —— 教训来自 M21：
并发跑两个实验会轮流抢同一资源，写出交错且标签错位的记录，整批数据作废。

用法：
    python scripts/m18_run_all.py                       # 全矩阵 2 模型 × 3 配置
    python scripts/m18_run_all.py --models 0.5b --configs Q0,Q1
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
import time

ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
LOCK_FILE = "outputs/.gpu_experiment.lock"


def acquire_lock(owner: str):
    """lab 级 GPU 实验互斥锁；进程死掉后自动接管（防残留锁把自己锁死）。"""
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            info = json.load(open(LOCK_FILE))
            alive = os.path.exists(f"/proc/{int(info['pid'])}")
        except Exception:
            info, alive = {}, False
        if alive:
            raise SystemExit(
                f"已有 GPU 实验在跑（pid={info.get('pid')} owner={info.get('owner')} "
                f"since={info.get('since')}）。并发会污染数据，拒绝启动。"
            )
        print(f"[run_all] 发现残留锁（pid={info.get('pid')} 已退出），接管", flush=True)
    json.dump({"pid": os.getpid(), "owner": owner, "since": time.strftime("%F %T%z")},
              open(LOCK_FILE, "w"))
    atexit.register(release_lock)


def release_lock():
    try:
        if json.load(open(LOCK_FILE)).get("pid") == os.getpid():
            os.remove(LOCK_FILE)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.5b,1.5b")
    ap.add_argument("--configs", default="Q0,Q1,Q2")
    ap.add_argument("--out-dir", default="outputs/m18")
    args = ap.parse_args()

    acquire_lock("m18_run_all")
    os.makedirs("logs/m18", exist_ok=True)
    env = dict(os.environ)
    env.update(ENV)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]

    for model in models:
        for cfg in configs:
            tag = f"{model}_{cfg}"
            log = f"logs/m18/{tag}.log"
            cmd = [sys.executable, "scripts/m18_quant.py", "--model", model,
                   "--config", cfg, "--out-dir", args.out_dir]
            print(f"[run_all] === {tag} === -> {log}", flush=True)
            t0 = time.perf_counter()
            with open(log, "w", encoding="utf-8") as lf:
                rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
            print(f"[run_all] {tag} rc={rc} elapsed={time.perf_counter() - t0:.1f}s", flush=True)
            if rc != 0:
                print(f"[run_all] ABORT: {tag} 失败，见 {log}", flush=True)
                sys.exit(rc)
    print("[run_all] done.", flush=True)


if __name__ == "__main__":
    main()
