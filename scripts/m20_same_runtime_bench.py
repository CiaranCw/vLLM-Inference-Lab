"""M20-C · Same-runtime 正式对照：container 内 vllm serve (C0) vs triton+vLLM backend (C1)。

约束（见 notes/m20/production_serving_and_triton.md）：
  - 同一镜像 / GPU / 模型 / EngineArgs；16GB 上 C0/C1 **串行**，禁止双引擎。
  - Shape A：in256 / out128；Poisson 到达；主点 QPS 32/40（可选 20）。
  - 禁止用 host-M12 数字减 container 称 Triton overhead。
  - 指标为**真在线 streaming** 客户端口径（TTFT/ITL/E2E），两边用同一负载器。

用法：
  export PATH="$HOME/bin:$PATH"   # WSL docker 组 wrapper
  python scripts/m20_same_runtime_bench.py --phase all
  python scripts/m20_same_runtime_bench.py --phase summarize
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "m20"
RAW = OUT / "raw"
LOG = ROOT / "logs" / "m20"
SERVING = ROOT / "serving" / "triton_model_repository"

IMAGE = "nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3"
HF_HUB = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct"
SNAP = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_IN_CTR = f"/models/qwen05b/snapshots/{SNAP}"

# 与 serving/.../1/model.json 对齐（C0 CLI 必须逐字对应）
ENGINE = dict(
    gpu_memory_utilization=0.45,
    max_model_len=2048,
    max_num_seqs=64,
    max_num_batched_tokens=4096,
    enforce_eager=True,
    enable_chunked_prefill=True,
    dtype="auto",
)
IN_LEN, OUT_LEN = 256, 128
HOST = "127.0.0.1"
C0_NAME, C1_NAME = "m20c-vllm-serve", "m20c-triton"
C0_PORT, C1_HTTP, C1_METRICS = 8021, 8000, 8002
DEFAULT_RATES = [20, 32, 40]
WARMUP_N = 30
MEASURE_N = 200
SEED = 0


def _env_docker() -> dict:
    env = dict(os.environ)
    # prefer ~/bin/docker wrapper
    home_bin = str(Path.home() / "bin")
    env["PATH"] = home_bin + os.pathsep + env.get("PATH", "")
    return env


def run(cmd, **kw):
    return subprocess.run(cmd, env=_env_docker(), text=True, **kw)


def docker(*args, check=True, capture=False):
    kw = {"check": check}
    if capture:
        kw["capture_output"] = True
    return run(["docker", *args], **kw)


def http_ok(url: str, timeout=3) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:
        return False


def wait_http(url: str, timeout_s: float, label: str) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if http_ok(url):
            print(f"[m20c] {label} ready ({time.time() - t0:.1f}s): {url}")
            return
        time.sleep(1.0)
    raise RuntimeError(f"{label} not ready within {timeout_s}s: {url}")


def stop_rm(name: str) -> None:
    docker("rm", "-f", name, check=False, capture=True)


def ensure_model_json() -> None:
    p = SERVING / "qwen25_05b" / "1" / "model.json"
    want = {
        "model": MODEL_IN_CTR,
        "disable_log_stats": False,
        "gpu_memory_utilization": ENGINE["gpu_memory_utilization"],
        "max_model_len": ENGINE["max_model_len"],
        "max_num_seqs": ENGINE["max_num_seqs"],
        "max_num_batched_tokens": ENGINE["max_num_batched_tokens"],
        "enforce_eager": True,
        "enable_chunked_prefill": True,
        "dtype": "auto",
    }
    p.parent.mkdir(parents=True, exist_ok=True)
    cur = json.loads(p.read_text()) if p.exists() else {}
    if cur != want:
        p.write_text(json.dumps(want, indent=2) + "\n")
        print(f"[m20c] synced {p}")


def common_docker_args() -> list[str]:
    if not HF_HUB.is_dir():
        raise FileNotFoundError(f"HF hub missing: {HF_HUB}")
    return [
        "--gpus",
        "all",
        "--shm-size",
        "1g",
        "-e",
        "VLLM_USE_FLASHINFER_SAMPLER=0",
        "-e",
        "HF_HUB_OFFLINE=1",
        "-e",
        "TRANSFORMERS_OFFLINE=1",
        "-v",
        f"{HF_HUB}:/models/qwen05b:ro",
        "-v",
        f"{SERVING}:/models_repo:ro",
    ]


def start_c0() -> None:
    stop_rm(C0_NAME)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        C0_NAME,
        *common_docker_args(),
        "-p",
        f"{C0_PORT}:8021",
        "--entrypoint",
        "vllm",
        IMAGE,
        "serve",
        MODEL_IN_CTR,
        "--host",
        "0.0.0.0",
        "--port",
        "8021",
        "--seed",
        "0",
        "--enforce-eager",
        "--dtype",
        "auto",
        "--max-model-len",
        str(ENGINE["max_model_len"]),
        "--gpu-memory-utilization",
        str(ENGINE["gpu_memory_utilization"]),
        "--max-num-seqs",
        str(ENGINE["max_num_seqs"]),
        "--max-num-batched-tokens",
        str(ENGINE["max_num_batched_tokens"]),
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--enable-chunked-prefill",
        "--disable-uvicorn-access-log",
    ]
    log = LOG / "c0_docker_run.log"
    with open(log, "w") as f:
        r = run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"C0 docker run failed; see {log}")
    wait_http(f"http://{HOST}:{C0_PORT}/health", 300, "C0 vllm serve")


def start_c1() -> None:
    stop_rm(C1_NAME)
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        C1_NAME,
        *common_docker_args(),
        "-p",
        f"{C1_HTTP}:8000",
        "-p",
        f"{C1_METRICS}:8002",
        IMAGE,
        "tritonserver",
        "--model-repository=/models_repo",
        "--model-control-mode=explicit",
        "--load-model=qwen25_05b",
        "--http-port=8000",
        "--grpc-port=8001",
        "--metrics-port=8002",
    ]
    log = LOG / "c1_docker_run.log"
    with open(log, "w") as f:
        r = run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"C1 docker run failed; see {log}")
    wait_http(f"http://{HOST}:{C1_HTTP}/v2/health/ready", 300, "C1 triton")


def build_prompts(n: int, seed: int) -> list[str]:
    """构造约 IN_LEN token 的可复现 prompt（用本机 tokenizer；与引擎同模型）。"""
    from transformers import AutoTokenizer

    tok_path = HF_HUB / "snapshots" / SNAP
    tok = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)
    rng = random.Random(seed)
    prompts = []
    # 用稳定词表片段拼到目标长度
    vocab = list(range(100, min(8000, len(tok))))
    for i in range(n):
        ids = [vocab[rng.randrange(len(vocab))] for _ in range(IN_LEN)]
        text = tok.decode(ids, skip_special_tokens=True)
        # 再编码一次修正到精确长度
        ids2 = tok.encode(text, add_special_tokens=False)[:IN_LEN]
        while len(ids2) < IN_LEN:
            ids2.append(vocab[rng.randrange(len(vocab))])
        prompts.append(tok.decode(ids2, skip_special_tokens=True))
    return prompts


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def bootstrap_median_diff(a: list[float], b: list[float], n=10000, seed=0):
    """median(a)-median(b) 的 Bootstrap 95% CI（与 M17 同口径）。"""
    rng = random.Random(seed)
    if not a or not b:
        return None
    diffs = []
    for _ in range(n):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.median(sa) - statistics.median(sb))
    diffs.sort()
    lo = diffs[int(0.025 * n)]
    hi = diffs[int(0.975 * n) - 1]
    point = statistics.median(a) - statistics.median(b)
    return {
        "median_diff": point,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci_excludes_zero": not (lo <= 0 <= hi),
    }


async def _one_c0(session, prompt: str, sem: asyncio.Semaphore) -> dict:
    import aiohttp

    url = f"http://{HOST}:{C0_PORT}/v1/completions"
    body = {
        "model": MODEL_IN_CTR,
        "prompt": prompt,
        "max_tokens": OUT_LEN,
        "temperature": 0,
        "top_p": 1,
        "stream": True,
        "ignore_eos": True,
    }
    t_arrive = time.perf_counter()
    async with sem:
        t_start = time.perf_counter()
        try:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {txt[:200]}",
                        "queue_delay_ms": (t_start - t_arrive) * 1000,
                    }
                ttft = None
                itls = []
                last = None
                n_chunks = 0
                async for raw in resp.content:
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = (now - t_start) * 1000
                    elif last is not None:
                        itls.append((now - last) * 1000)
                    last = now
                    n_chunks += 1
                e2e = (time.perf_counter() - t_start) * 1000
                return {
                    "success": True,
                    "ttft_ms": ttft,
                    "e2e_ms": e2e,
                    "itl_ms_mean": statistics.mean(itls) if itls else float("nan"),
                    "n_stream_chunks": n_chunks,
                    "queue_delay_ms": (t_start - t_arrive) * 1000,
                }
        except Exception as e:
            return {
                "success": False,
                "error": repr(e),
                "queue_delay_ms": (t_start - t_arrive) * 1000,
            }


async def _one_c1(session, prompt: str, sem: asyncio.Semaphore) -> dict:
    import aiohttp

    url = f"http://{HOST}:{C1_HTTP}/v2/models/qwen25_05b/generate_stream"
    body = {
        "text_input": prompt,
        "parameters": {
            "stream": True,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": OUT_LEN,
            "ignore_eos": True,
        },
    }
    t_arrive = time.perf_counter()
    async with sem:
        t_start = time.perf_counter()
        try:
            async with session.post(url, json=body, timeout=aiohttp.ClientTimeout(total=300)) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    return {
                        "success": False,
                        "error": f"HTTP {resp.status}: {txt[:200]}",
                        "queue_delay_ms": (t_start - t_arrive) * 1000,
                    }
                ttft = None
                itls = []
                last = None
                n_chunks = 0
                buf = b""
                async for raw in resp.content.iter_any():
                    buf += raw
                    while b"\n" in buf:
                        line_b, buf = buf.split(b"\n", 1)
                        line = line_b.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if not payload:
                            continue
                        now = time.perf_counter()
                        if ttft is None:
                            ttft = (now - t_start) * 1000
                        elif last is not None:
                            itls.append((now - last) * 1000)
                        last = now
                        n_chunks += 1
                e2e = (time.perf_counter() - t_start) * 1000
                return {
                    "success": True,
                    "ttft_ms": ttft,
                    "e2e_ms": e2e,
                    "itl_ms_mean": statistics.mean(itls) if itls else float("nan"),
                    "n_stream_chunks": n_chunks,
                    "queue_delay_ms": (t_start - t_arrive) * 1000,
                }
        except Exception as e:
            return {
                "success": False,
                "error": repr(e),
                "queue_delay_ms": (t_start - t_arrive) * 1000,
            }


async def run_load(
    backend: str,
    rate: float,
    prompts: list[str],
    phase: str,
    concurrency_cap: int = 256,
) -> list[dict]:
    import aiohttp

    handler = _one_c0 if backend == "c0" else _one_c1
    sem = asyncio.Semaphore(concurrency_cap)
    connector = aiohttp.TCPConnector(limit=concurrency_cap)
    results: list[dict] = []
    rng = random.Random(
        SEED
        + (0 if rate == float("inf") else int(rate * 1000))
        + (0 if backend == "c0" else 17)
    )

    arrivals = [0.0]
    for _ in range(1, len(prompts)):
        gap = 0.0 if rate == float("inf") else rng.expovariate(rate)
        arrivals.append(arrivals[-1] + gap)

    async with aiohttp.ClientSession(connector=connector) as session:

        async def launch(i, prompt, delay):
            await asyncio.sleep(delay)
            r = await handler(session, prompt, sem)
            r.update(
                {
                    "backend": backend,
                    "phase": phase,
                    "offered_qps": rate,
                    "req_index": i,
                    "metric_scope": "online_streaming_client",
                }
            )
            results.append(r)

        await asyncio.gather(
            *[
                asyncio.create_task(launch(i, p, arrivals[i]))
                for i, p in enumerate(prompts)
            ]
        )
    return results


def summarize_rows(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("success") and r.get("phase") == "measure"]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms") is not None]
    e2e = [r["e2e_ms"] for r in ok if r.get("e2e_ms") is not None]
    itl = [
        r["itl_ms_mean"]
        for r in ok
        if r.get("itl_ms_mean") is not None and not math.isnan(r["itl_ms_mean"])
    ]
    n = len(rows)
    n_ok = len(ok)
    wall = None
    return {
        "n_measure": n_ok,
        "n_total_phase_rows": n,
        "success_rate": n_ok / n if n else 0.0,
        "ttft_p50": _pct(ttft, 50),
        "ttft_p95": _pct(ttft, 95),
        "ttft_p99": _pct(ttft, 99),
        "e2e_p50": _pct(e2e, 50),
        "e2e_p95": _pct(e2e, 95),
        "e2e_p99": _pct(e2e, 99),
        "itl_p50": _pct(itl, 50),
        "itl_p95": _pct(itl, 95),
        "itl_p99": _pct(itl, 99),
        "ttft_median": statistics.median(ttft) if ttft else float("nan"),
        "e2e_median": statistics.median(e2e) if e2e else float("nan"),
        "itl_median": statistics.median(itl) if itl else float("nan"),
        "output_tps_approx": (
            (n_ok * OUT_LEN) / (sum(e2e) / 1000.0) if e2e else float("nan")
        ),
    }


def phase_bench(
    backend: str,
    rates: list[float],
    warmup_n: int = WARMUP_N,
    measure_n: int = MEASURE_N,
) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    LOG.mkdir(parents=True, exist_ok=True)
    ensure_model_json()

    print(f"[m20c] building prompts…")
    prompts_w = build_prompts(warmup_n, SEED + 7)
    prompts_m = build_prompts(measure_n, SEED)

    if backend == "c0":
        start_c0()
    else:
        start_c1()

    try:
        print(f"[m20c] {backend} warmup n={warmup_n}")
        warm = asyncio.run(run_load(backend, float("inf"), prompts_w, "warmup"))
        warm_path = RAW / f"c_{backend}_warmup.jsonl"
        with open(warm_path, "w") as f:
            for r in warm:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[m20c] warmup success={sum(1 for r in warm if r.get('success'))}/{len(warm)}")

        for rate in rates:
            print(f"[m20c] {backend} measure rate={rate} n={measure_n}")
            t0 = time.time()
            rows = asyncio.run(run_load(backend, float(rate), prompts_m, "measure"))
            elapsed = time.time() - t0
            path = RAW / f"c_{backend}_qps{int(rate)}.jsonl"
            with open(path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summ = summarize_rows(rows)
            summ.update(
                {
                    "backend": backend,
                    "offered_qps": rate,
                    "wall_s": round(elapsed, 2),
                    "in_len": IN_LEN,
                    "out_len": OUT_LEN,
                    "image": IMAGE,
                    "engine": ENGINE,
                    "warmup_n": warmup_n,
                    "measure_n": measure_n,
                }
            )
            sp = RAW / f"c_{backend}_qps{int(rate)}_summary.json"
            sp.write_text(json.dumps(summ, indent=2) + "\n")
            print(
                f"[m20c] {backend} qps={rate} success={summ['success_rate']:.3f} "
                f"ttft_p50={summ['ttft_p50']:.1f} e2e_p50={summ['e2e_p50']:.1f} "
                f"wall={elapsed:.1f}s -> {path.name}"
            )
    finally:
        name = C0_NAME if backend == "c0" else C1_NAME
        logf = LOG / f"{backend}_container.log"
        with open(logf, "w") as f:
            run(["docker", "logs", name], stdout=f, stderr=subprocess.STDOUT, check=False)
        stop_rm(name)
        print(f"[m20c] stopped {name}")


def phase_summarize(rates: list[float]) -> None:
    import csv

    OUT.mkdir(parents=True, exist_ok=True)
    rows_out = []
    effect_rows = []
    for rate in rates:
        s0 = RAW / f"c_c0_qps{int(rate)}_summary.json"
        s1 = RAW / f"c_c1_qps{int(rate)}_summary.json"
        if not s0.exists() or not s1.exists():
            print(f"[m20c] skip qps={rate}: missing summary")
            continue
        j0, j1 = json.loads(s0.read_text()), json.loads(s1.read_text())
        for tag, j in [("c0", j0), ("c1", j1)]:
            rows_out.append(
                {
                    "backend": tag,
                    "offered_qps": rate,
                    "n_measure": j["n_measure"],
                    "success_rate": j["success_rate"],
                    "ttft_p50_ms": j["ttft_p50"],
                    "ttft_p95_ms": j["ttft_p95"],
                    "ttft_p99_ms": j["ttft_p99"],
                    "e2e_p50_ms": j["e2e_p50"],
                    "e2e_p95_ms": j["e2e_p95"],
                    "e2e_p99_ms": j["e2e_p99"],
                    "itl_p50_ms": j["itl_p50"],
                    "itl_p95_ms": j["itl_p95"],
                    "itl_p99_ms": j["itl_p99"],
                    "wall_s": j.get("wall_s"),
                    "metric_scope": "online_streaming_client",
                }
            )
        # per-request bootstrap on TTFT / E2E
        def load_metric(backend, key):
            p = RAW / f"c_{backend}_qps{int(rate)}.jsonl"
            vals = []
            for line in p.read_text().splitlines():
                r = json.loads(line)
                if r.get("phase") == "measure" and r.get("success") and r.get(key) is not None:
                    vals.append(float(r[key]))
            return vals

        for key, name in [("ttft_ms", "ttft"), ("e2e_ms", "e2e")]:
            a, b = load_metric("c0", key), load_metric("c1", key)
            eff = bootstrap_median_diff(a, b, seed=SEED)
            if eff:
                effect_rows.append(
                    {
                        "offered_qps": rate,
                        "metric": name,
                        "c0_median": statistics.median(a),
                        "c1_median": statistics.median(b),
                        "median_diff_c0_minus_c1": eff["median_diff"],
                        "ci95_lo": eff["ci95_lo"],
                        "ci95_hi": eff["ci95_hi"],
                        "ci_excludes_zero": "yes" if eff["ci_excludes_zero"] else "no",
                        "interpretation": "negative => C1 slower than C0 on this metric",
                    }
                )

    sweep = OUT / "c_same_runtime_sweep.csv"
    with open(sweep, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else ["backend"])
        w.writeheader()
        w.writerows(rows_out)
    effect = OUT / "c_same_runtime_effect.csv"
    with open(effect, "w", newline="") as f:
        fields = list(effect_rows[0].keys()) if effect_rows else ["offered_qps"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(effect_rows)
    meta = {
        "image": IMAGE,
        "engine": ENGINE,
        "shape": {"in": IN_LEN, "out": OUT_LEN},
        "warmup_n": WARMUP_N,
        "measure_n": MEASURE_N,
        "rates": rates,
        "seed": SEED,
        "note": "C0=vllm serve in container; C1=tritonserver+vLLM backend; same image/model/EngineArgs",
    }
    (OUT / "c_same_runtime_config.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[m20c] wrote {sweep}")
    print(f"[m20c] wrote {effect}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["c0", "c1", "all", "summarize"], default="all")
    ap.add_argument("--rates", default=",".join(str(r) for r in DEFAULT_RATES))
    ap.add_argument("--warmup-n", type=int, default=30)
    ap.add_argument("--measure-n", type=int, default=200)
    args = ap.parse_args()
    rates = [float(x) for x in args.rates.split(",") if x.strip()]

    # dependency check
    try:
        import aiohttp  # noqa: F401
    except ImportError:
        print("need aiohttp in .venv: pip install aiohttp", file=sys.stderr)
        sys.exit(2)

    if args.phase in ("c0", "all"):
        stop_rm(C0_NAME)
        stop_rm(C1_NAME)
        phase_bench("c0", rates, warmup_n=args.warmup_n, measure_n=args.measure_n)
    if args.phase in ("c1", "all"):
        stop_rm(C0_NAME)
        stop_rm(C1_NAME)
        phase_bench("c1", rates, warmup_n=args.warmup_n, measure_n=args.measure_n)
    if args.phase in ("summarize", "all"):
        phase_summarize(rates)


if __name__ == "__main__":
    main()
