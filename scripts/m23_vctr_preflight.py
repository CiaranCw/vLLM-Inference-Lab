#!/usr/bin/env python3
"""M23 · V-ctr correctness preflight（正式 benchmark 前的轻量门）。

所有者要求（2026-08-13）：host 上的 V-O smoke 不能替代本检查 ——
V-host 与 V-ctr 的 vLLM / software stack 不同（host pinned `ee0da84` vs 容器内 `0.24.0+092c4842`），
在容器里没验证过就直接跑正式性能，等于把未验证的前提当结论基础。

做什么：
  - 复用 `outputs/m23/preflight_inputs.json` 里同一份 6 条 token ids（与 host smoke 同源）；
  - 容器内分别验 V-E（Graph/optimization OFF）与 V-O（optimized ON）；
  - 检查：生成非空 / 长度精确 / logprob 无 NaN·Inf / 无 runtime error；
  - 回读并保存实际 compilation / CUDA Graph 配置（不信入参，只信回读）；
  - 每条 lane 跑两次独立容器进程（across-process）+ 每次进程内两遍生成（within-process），
    记录 within-lane repeatability。

判据：
  - 硬失败 → 停止 ON lane 正式性能实验并回报（不临时切回 host 凑矩阵）。
  - 跨 lane 的 exact-match 不作为阻塞条件，但作为 limitation 保留。

用法：python scripts/m23_vctr_preflight.py
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import time

OUT, LOGD = "outputs/m23", "logs/m23"
DOCKER = os.path.expanduser("~/bin/docker")
VLLM_IMAGE = "nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3"   # M20-C 已验证
HF_CACHE = os.path.expanduser("~/.cache/huggingface")
INPUTS = os.path.join(OUT, "preflight_inputs.json")
# 直接指向容器内的本地 snapshot 路径，绕过 hub 解析：
#   容器里较新的 huggingface_hub 在 offline 模式下会做「完整快照校验」，而本地缓存缺 3 个
#   纯文档文件（.gitattributes / LICENSE / README.md）→ 报 IncompleteSnapshotError。
#   那是 harness 与 hub 版本的交互问题，不是模型或引擎问题（权重与 tokenizer 齐全）。
#   用本地路径既不联网（守住「不自动下载模型」），又保证 V/T 两侧加载的是同一份字节。
MODEL_PATH_IN_CTR = "/root/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
MAX_TOKENS = 64
REPEATS_IN_PROC = 2       # 进程内重复
RUNS_PER_LANE = 2         # 独立进程重复

PAYLOAD = r'''
import json, math, time, traceback
LANE = "__LANE__"
IDS = json.loads(r"""__IDS__""")
MAX_TOKENS = __MAXTOK__
REPEATS = __REPEATS__
rec = {"lane": LANE}
try:
    from vllm import LLM, SamplingParams
    import vllm
    rec["vllm_version"] = vllm.__version__
    t0 = time.perf_counter()
    llm = LLM(model="__MODELPATH__",
              enforce_eager=(LANE == "E"),
              dtype="bfloat16", max_model_len=2048,
              gpu_memory_utilization=0.6, max_num_seqs=64,
              max_num_batched_tokens=4096, enable_prefix_caching=False, seed=0)
    rec["build_s"] = round(time.perf_counter() - t0, 2)

    # 回读**生效**的 execution / graph 配置
    vc = llm.llm_engine.vllm_config
    cc = vc.compilation_config
    rec["effective"] = {
        "compilation_mode": str(getattr(cc, "mode", None)),
        "cudagraph_mode": str(getattr(cc, "cudagraph_mode", None)),
        "cudagraph_capture_sizes": list(getattr(cc, "cudagraph_capture_sizes", []) or [])[:12],
        "max_cudagraph_capture_size": getattr(cc, "max_cudagraph_capture_size", None),
        "dtype": str(vc.model_config.dtype),
        "kv_cache_dtype": str(vc.cache_config.cache_dtype),
        "kv_cache_size_tokens": getattr(vc.cache_config, "kv_cache_size_tokens", None),
        "enforce_eager_requested": (LANE == "E"),
    }

    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS, seed=0,
                        ignore_eos=True, logprobs=1)
    passes = []
    for r in range(REPEATS):
        outs = llm.generate([{"prompt_token_ids": x} for x in IDS], sp, use_tqdm=False)
        one = []
        for o in outs:
            out0 = o.outputs[0]
            # NaN / Inf 检查：翻每个位置的 logprob
            bad = 0
            for d in (out0.logprobs or []):
                for lp in d.values():
                    v = getattr(lp, "logprob", None)
                    if v is None or math.isnan(v) or math.isinf(v):
                        bad += 1
            one.append({"text": out0.text, "token_ids": list(out0.token_ids),
                        "n_tokens": len(out0.token_ids), "bad_logprobs": bad})
        passes.append(one)
    rec["passes"] = passes
except Exception as e:
    rec["error"] = f"{type(e).__name__}: {e}"
    rec["traceback_tail"] = traceback.format_exc().strip().splitlines()[-8:]
print("M23_VCTR " + json.dumps(rec, ensure_ascii=False))
'''


def run_container(lane: str, ids, run_idx: int) -> dict:
    payload = (PAYLOAD.replace("__LANE__", lane)
               .replace("__IDS__", json.dumps(ids))
               .replace("__MAXTOK__", str(MAX_TOKENS))
               .replace("__REPEATS__", str(REPEATS_IN_PROC))
               .replace("__MODELPATH__", MODEL_PATH_IN_CTR))
    cmd = [
        DOCKER, "run", "--rm", "--gpus", "all", "--ipc", "host",
        "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
        "-e", "HF_HUB_OFFLINE=1", "-e", "HF_HOME=/root/.cache/huggingface",
        "-e", "TQDM_DISABLE=1", "-e", "VLLM_USE_FLASHINFER_SAMPLER=0", "-e", "VLLM_ENABLE_V1_MULTIPROCESSING=0",
        "-v", f"{HF_CACHE}:/root/.cache/huggingface",
        VLLM_IMAGE, "python3", "-c", payload,
    ]
    log = os.path.join(LOGD, f"vctr_preflight_lane{lane}_run{run_idx}.log")
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    with open(log, "w") as f:
        f.write(proc.stdout or "")
    rec = None
    dec = json.JSONDecoder()
    for line in (proc.stdout or "").splitlines():
        i = line.find("M23_VCTR ")
        if i >= 0:
            # raw_decode：只取紧跟其后的第一个 JSON 对象。
            # 必须这么做——vLLM 的 tqdm 进度条用 \r 刷新，捕获后会和本行粘连，
            # 直接 json.loads 会报 "Extra data"。
            try:
                rec, _ = dec.raw_decode(line[i + len("M23_VCTR "):])
            except ValueError:
                pass
    return {"lane": lane, "run": run_idx, "returncode": proc.returncode,
            "elapsed_s": round(time.time() - t0, 1), "record": rec, "log": log}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(LOGD, exist_ok=True)
    ids = json.load(open(INPUTS))["token_ids"]

    runs = []
    for lane in ("E", "O"):
        for i in range(1, RUNS_PER_LANE + 1):
            print(f"[m23] === V-ctr lane {lane} run {i}/{RUNS_PER_LANE} ...", flush=True)
            r = run_container(lane, ids, i)
            ok = r["record"] is not None and "error" not in (r["record"] or {})
            print(f"[m23]     rc={r['returncode']} elapsed={r['elapsed_s']}s ok={ok}", flush=True)
            runs.append(r)

    report = {"generated_at": time.strftime("%F %T%z"), "image": VLLM_IMAGE,
              "inputs_file": INPUTS, "max_tokens": MAX_TOKENS,
              "runs_per_lane": RUNS_PER_LANE, "repeats_in_proc": REPEATS_IN_PROC,
              "runs": runs}

    # ---------- 硬失败判据 ----------
    hard = []
    for r in runs:
        tag = f"lane{r['lane']}run{r['run']}"
        rec = r["record"]
        if r["returncode"] != 0 or rec is None:
            hard.append(f"{tag}: 容器进程失败 rc={r['returncode']}")
            continue
        if "error" in rec:
            hard.append(f"{tag}: {rec['error']}")
            continue
        for pi, one in enumerate(rec["passes"]):
            for gi, g in enumerate(one):
                if not g["text"].strip():
                    hard.append(f"{tag} pass{pi} prompt{gi}: 空输出")
                if g["n_tokens"] != MAX_TOKENS:
                    hard.append(f"{tag} pass{pi} prompt{gi}: 长度 {g['n_tokens']} != {MAX_TOKENS}")
                if g["bad_logprobs"]:
                    hard.append(f"{tag} pass{pi} prompt{gi}: {g['bad_logprobs']} 个 NaN/Inf logprob")

    # ---------- within-lane repeatability ----------
    def toks(rec, pass_idx):
        return [g["token_ids"] for g in rec["passes"][pass_idx]]

    rep = {}
    for lane in ("E", "O"):
        lane_runs = [r for r in runs if r["lane"] == lane and r["record"]
                     and "error" not in r["record"]]
        entry = {}
        if lane_runs:
            r0 = lane_runs[0]["record"]
            a, b = toks(r0, 0), toks(r0, 1)
            entry["within_process"] = {
                "matched": sum(1 for x, y in zip(a, b) if x == y), "total": len(a)}
        if len(lane_runs) >= 2:
            a, b = toks(lane_runs[0]["record"], 0), toks(lane_runs[1]["record"], 0)
            entry["across_process"] = {
                "matched": sum(1 for x, y in zip(a, b) if x == y), "total": len(a)}
        rep[lane] = entry
    report["within_lane_repeatability"] = rep

    # ---------- 跨 lane exact-match（诊断项，不阻塞）----------
    e_runs = [r for r in runs if r["lane"] == "E" and r["record"] and "error" not in r["record"]]
    o_runs = [r for r in runs if r["lane"] == "O" and r["record"] and "error" not in r["record"]]
    if e_runs and o_runs:
        a, b = toks(e_runs[0]["record"], 0), toks(o_runs[0]["record"], 0)
        m = sum(1 for x, y in zip(a, b) if x == y)
        report["cross_lane_exact_match"] = {
            "matched": m, "total": len(a), "rate": round(m / len(a), 4),
            "blocking": False,
            "limitation": ("**不作为性能实验阻塞条件，但必须随所有 V-O 结论保留**。"
                           "当前证据只能说明「**未观察到明显 functional degradation**」，"
                           "**不得**升级成「两条 execution path 数值等价」。"),
        }

    report["hard_fail_reasons"] = hard
    report["PASS"] = not hard
    report["verdict"] = ("V-ctr preflight 通过，可实现并运行 m23_bench.py"
                         if not hard else
                         "**硬失败 → 停止 ON lane 正式性能实验并回报；不切回 host 凑矩阵**")

    path = os.path.join(OUT, "vctr_preflight.json")
    json.dump(report, open(path, "w"), ensure_ascii=False, indent=2)

    print(f"\n[m23] V-ctr preflight PASS={report['PASS']}")
    for lane in ("E", "O"):
        lr = [r for r in runs if r["lane"] == lane and r["record"]
              and "error" not in r["record"]]
        if lr:
            eff = lr[0]["record"]["effective"]
            print(f"[m23] V-{lane} 生效: cudagraph_mode={eff['cudagraph_mode']} "
                  f"compilation_mode={eff['compilation_mode']} "
                  f"vllm={lr[0]['record'].get('vllm_version')}")
        print(f"[m23] V-{lane} repeatability: {rep.get(lane)}")
    if cl := report.get("cross_lane_exact_match"):
        print(f"[m23] 跨 lane exact-match（诊断、不阻塞）: "
              f"{cl['matched']}/{cl['total']} = {cl['rate']}")
    for h in hard:
        print("[m23] ⚠️ 硬失败:", h)
    print("[m23] ->", path)
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
