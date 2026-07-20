"""M7 · 独立进程编排器 —— eager / compiled 各在独立子进程里跑 m7_profile.py。
理由同 M6：CUDA context / CUDA Graph 内存池 / torch.compile 缓存不跨配置污染。

用法：python scripts/m7_run_all.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

MODES = ["eager", "compiled"]
ENV_EXTRA = {"VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}


def main():
    os.makedirs("logs/m7", exist_ok=True)
    env = dict(os.environ)
    env.update(ENV_EXTRA)
    for mode in MODES:
        log = f"logs/m7/{mode}.log"
        cmd = [sys.executable, "scripts/m7_profile.py", "--mode", mode,
               "--warmup", "10", "--n-prof", "20", "--out-dir", "outputs/m7"]
        print(f"[m7_run_all] === {mode} === -> {log}")
        t0 = time.perf_counter()
        with open(log, "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        print(f"[m7_run_all] {mode} rc={rc} elapsed={time.perf_counter()-t0:.1f}s")
        if rc != 0:
            print(f"[m7_run_all] ABORT: {mode} 失败，见 {log}")
            sys.exit(rc)
    print("[m7_run_all] done.")


if __name__ == "__main__":
    main()
