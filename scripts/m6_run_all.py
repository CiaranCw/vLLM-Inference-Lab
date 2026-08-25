"""M6 · 独立进程编排器 —— 每种 (mode, prefix) 配置在独立子进程里跑 m6_benchmark.py，
避免 CUDA context / caching allocator / CUDA Graph 内存池 / torch.compile 缓存跨配置污染显存快照。

用法：
  python scripts/m6_run_all.py [--quick]

正式配置顺序：
  ① eager   + prefix off  -> mem,prefill,decode
  ② compiled + prefix off  -> mem,prefill,decode
  ③ eager   + prefix on   -> mem,prefix
  （④ compiled + prefix on 视价值可选，默认不跑）
每个子进程日志写入 logs/m6/<tag>.log；退出码非 0 则中止并报告。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

CONFIGS = [
    # eager + prefix off：主基线；顺带跑 prefix 的 OFF 对照 + 逐长度/逐 batch 执行模式诊断
    ("eager", "off", "mem,prefill,decode,prefix,diag"),
    # compiled + prefix off：编译/图对照；diag 给出逐 batch decode=FULL 与逐长度 prefill 模式证据
    ("compiled", "off", "mem,prefill,decode,diag"),
    # eager + prefix on：prefix cold/hot 命中实验
    ("eager", "on", "mem,prefix"),
]

ENV_EXTRA = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default="outputs/m6")
    ap.add_argument("--log-dir", default="logs/m6")
    args = ap.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    env = dict(os.environ)
    env.update(ENV_EXTRA)

    for mode, prefix, exps in CONFIGS:
        tag = f"{mode}_{prefix}"
        log_path = os.path.join(args.log_dir, f"{tag}{'_quick' if args.quick else ''}.log")
        cmd = [
            sys.executable, "scripts/m6_benchmark.py",
            "--mode", mode, "--prefix", prefix, "--experiments", exps,
            "--out-dir", args.out_dir,
        ]
        if args.quick:
            cmd.append("--quick")
        print(f"[run_all] === {tag} ({exps}) === -> {log_path}")
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        dt = time.perf_counter() - t0
        print(f"[run_all] {tag} rc={rc} elapsed={dt:.1f}s")
        if rc != 0:
            print(f"[run_all] ABORT: {tag} 失败，见 {log_path}")
            sys.exit(rc)

    print("[run_all] all configs done.")


if __name__ == "__main__":
    main()
