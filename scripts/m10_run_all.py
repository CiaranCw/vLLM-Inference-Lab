"""M10 · 独立进程编排器：每个量化配置在独立子进程跑 m10_quant.py。"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

ENV = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
DEFAULT = ["C0", "C1", "C0T", "C2", "C2dyn"]  # 核心 C0/C1/C0T/C2 + 可选 C2dyn(动态scale对照)；C3 可加


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(DEFAULT))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default="outputs/m10")
    args = ap.parse_args()
    os.makedirs("logs/m10", exist_ok=True)
    env = dict(os.environ); env.update(ENV)
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        log = f"logs/m10/{cfg}{'_quick' if args.quick else ''}.log"
        cmd = [sys.executable, "scripts/m10_quant.py", "--config", cfg, "--out-dir", args.out_dir]
        if args.quick:
            cmd.append("--quick")
        print(f"[run_all] === {cfg} === -> {log}")
        t0 = time.perf_counter()
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        print(f"[run_all] {cfg} rc={rc} elapsed={time.perf_counter()-t0:.1f}s")
        if rc != 0:
            print(f"[run_all] ABORT: {cfg} 失败，见 {log}"); sys.exit(rc)
    print("[run_all] done.")


if __name__ == "__main__":
    main()
