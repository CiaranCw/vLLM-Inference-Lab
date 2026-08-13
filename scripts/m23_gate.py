#!/usr/bin/env python3
"""M23 · Gate 0 / 1 / 2 执行器（容器内跑 TRT-P，host 侧只做编排）。

设计冻结见 `notes/m23/cross_engine_plan.md`（DESIGN FROZEN v3）。本脚本严格实现其中的约束：

  - **container-only**：绝不在 host 安装 TRT-LLM（`pip install tensorrt_llm` 有替换 torch 的记录，
    会摧毁 lab venv 与 M6–M22 的可复现性）。
  - **唯一镜像**：`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24`（TRT-P 路线）。
    **不自动换 tag** —— 失败必须保存证据后回报所有者。
  - **Gate 1 红线**：import 成功 ≠ PASS，model load 成功 ≠ PASS，GPU 可见 ≠ PASS；
    **必须真实 generate 出文本才算 PASS**（sm_120 的 kernel/runtime 问题可能只在真正
    执行 forward/attention/decode 时才暴露）。
  - **默认 dry-run**：不传 `--run` 就只打印将要执行的命令，**不拉镜像、不启动容器、不碰 GPU**。

用法：
    python scripts/m23_gate.py --gate 0            # dry-run，只打印命令
    python scripts/m23_gate.py --gate 0 --run      # 真正执行（需所有者授权 + GPU 空闲）
    python scripts/m23_gate.py --gate 1 --run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time

# ---- 冻结常量（改这里等于改设计，需重新走审查）----
IMAGE = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24"
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"        # 本地已缓存，无需下载
DOCKER = os.path.expanduser("~/bin/docker")  # WSL session 常缺 docker 组，走 wrapper
OUT, LOGD = "outputs/m23", "logs/m23"
HF_CACHE = os.path.expanduser("~/.cache/huggingface")

# GPU 占用超过该阈值就拒绝启动（本机 16GB；他人任务在跑时不得抢占）
GPU_BUSY_MIB = 4000

GATE0_PAYLOAD = r"""
import json, sys
rec = {"gate": 0}
try:
    import tensorrt_llm as t
    rec["tensorrt_llm_version"] = getattr(t, "__version__", "unknown")
    rec["import_ok"] = True
except Exception as e:
    rec["import_ok"] = False
    rec["import_error"] = f"{type(e).__name__}: {e}"
try:
    import torch
    rec["torch_version"] = torch.__version__
    rec["torch_cuda"] = torch.version.cuda
    rec["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        rec["device_name"] = torch.cuda.get_device_name(0)
        rec["capability_int"] = cap[0] * 10 + cap[1]
        rec["arch_list"] = torch.cuda.get_arch_list()
except Exception as e:
    rec["torch_error"] = f"{type(e).__name__}: {e}"
# Gate 0 判据：import 成功 且 能看到 sm_120 设备
rec["PASS"] = bool(rec.get("import_ok") and rec.get("cuda_available")
                   and rec.get("capability_int") == 120)
print("M23_RESULT " + json.dumps(rec, ensure_ascii=False))
"""

GATE1_PAYLOAD = r"""
import json, time, traceback
rec = {"gate": 1, "model": "%s"}
try:
    from tensorrt_llm import LLM, SamplingParams
    t0 = time.perf_counter()
    # TRT-P：HF checkpoint 直接加载，**没有 engine build / checkpoint 转换**（迁移指南 U3）
    llm = LLM(model="%s")
    rec["load_s"] = round(time.perf_counter() - t0, 2)
    rec["load_ok"] = True
    outs = llm.generate(["The capital of France is"],
                        SamplingParams(temperature=0, max_tokens=16))
    text = outs[0].outputs[0].text
    rec["text"] = text
    rec["n_chars"] = len(text.strip())
    # ⭐ 红线：load 成功不算 PASS，必须真的生成出非空文本
    rec["PASS"] = bool(text and text.strip())
except Exception as e:
    rec["PASS"] = False
    rec["error"] = f"{type(e).__name__}: {e}"
    rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
print("M23_RESULT " + json.dumps(rec, ensure_ascii=False))
""" % (MODEL, MODEL)

PAYLOADS = {0: GATE0_PAYLOAD, 1: GATE1_PAYLOAD}


def docker_cmd(payload: str) -> list[str]:
    """构造容器命令。注意：不用 -it（无 TTY），HF 缓存只读挂载以复用已下载模型。"""
    return [
        DOCKER, "run", "--rm",
        "--gpus", "all",
        "--ipc", "host",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface:ro",
        IMAGE,
        "python3", "-c", payload,
    ]


def gpu_snapshot() -> dict:
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        used, total, util = [x.strip() for x in q.split(",")]
        return {"mem_used_mib": int(used), "mem_total_mib": int(total), "gpu_util_pct": int(util)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def image_present() -> bool:
    if not os.path.exists(DOCKER):
        return False
    p = subprocess.run([DOCKER, "images", "-q", IMAGE], capture_output=True, text=True)
    return bool(p.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=int, required=True, choices=[0, 1])
    ap.add_argument("--run", action="store_true",
                    help="真正执行；不加则只 dry-run 打印命令（默认 dry-run）")
    ap.add_argument("--allow-busy-gpu", action="store_true",
                    help="显式覆盖 GPU 占用检查（默认拒绝在他人任务运行时启动）")
    args = ap.parse_args()

    cmd = docker_cmd(PAYLOADS[args.gate])
    printable = " ".join(x if "\n" not in x else "<PAYLOAD>" for x in cmd)

    print(f"[m23] Gate {args.gate}")
    print(f"[m23] image  : {IMAGE}")
    print(f"[m23] docker : {printable}")
    print(f"[m23] payload: {len(PAYLOADS[args.gate])} chars（容器内 python3 -c 执行）")

    if not args.run:
        print("\n[m23] **DRY-RUN**：未执行任何命令。加 --run 才会真正拉起容器。")
        print("[m23] 前置条件：① 所有者已授权；② GPU 空闲；③ 镜像已拉取或允许拉取。")
        print(f"[m23] 镜像当前是否在本地：{image_present()}")
        return 0

    # ---- 真正执行前的安全闸 ----
    gpu = gpu_snapshot()
    print(f"[m23] GPU: {gpu}")
    if "error" in gpu:
        raise SystemExit("[m23] 读不到 nvidia-smi，拒绝启动")
    if gpu["mem_used_mib"] > GPU_BUSY_MIB and not args.allow_busy_gpu:
        raise SystemExit(
            f"[m23] GPU 已占用 {gpu['mem_used_mib']} MiB（阈值 {GPU_BUSY_MIB}）——"
            "可能有他人任务在跑，拒绝启动。确认空闲后再跑，或显式 --allow-busy-gpu。")
    if not shutil.which("nvidia-smi") or not os.path.exists(DOCKER):
        raise SystemExit("[m23] 缺少 nvidia-smi 或 docker wrapper，拒绝启动")

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)
    log_path = os.path.join(LOGD, f"gate{args.gate}.log")

    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        lf.write(proc.stdout or "")
    elapsed = round(time.time() - t0, 1)

    # 从容器 stdout 里捞结构化结果
    result = None
    for line in (proc.stdout or "").splitlines():
        if line.startswith("M23_RESULT "):
            result = json.loads(line[len("M23_RESULT "):])

    rec = {
        "gate": args.gate,
        "image": IMAGE,
        "ran_at": time.strftime("%F %T%z"),
        "elapsed_s": elapsed,
        "docker_returncode": proc.returncode,
        "gpu_before": gpu,
        "result": result,
        "log_path": log_path,
        "PASS": bool(result and result.get("PASS")),
        "evidence_note": ("失败时本文件 + log_path 即为回报所有者的完整证据；"
                          "按冻结稿 §8，不得自行换 tag 或进入其他补救路线"),
    }
    out_path = os.path.join(OUT, f"gate{args.gate}.json")
    with open(out_path, "w") as f:
        json.dump(rec, f, ensure_ascii=False, indent=2)

    print(f"\n[m23] PASS={rec['PASS']} rc={proc.returncode} elapsed={elapsed}s")
    if result:
        print("[m23] result:", json.dumps(result, ensure_ascii=False)[:400])
    print(f"[m23] -> {out_path} / {log_path}")
    if not rec["PASS"]:
        print("[m23] ⚠️ 未通过：保存证据并**停止**，回报所有者（冻结稿 §8）。")
    return 0 if rec["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
