#!/usr/bin/env python3
"""M27 公共设施：冻结配置 / 拓扑 / 端口 / KV 理论值 / Prometheus 解析 / 环境指纹 / formal 门禁。

只用标准库 —— scripts/check_invariants.py 会 import 本文件的冻结常量，
因此这里禁止 import vllm / torch / aiohttp。

冻结配置的唯一来源是本文件的 FROZEN；notes/m27/M27_H20_PLAN.md 里的
`M27_FROZEN_CONFIG` JSON 块必须与之逐键相等（check_invariants m27_config_freeze）。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "m27")
RAW = os.path.join(OUT, "raw")
CONFIGS = os.path.join(OUT, "configs")
METRICS = os.path.join(OUT, "metrics")
TRACES = os.path.join(OUT, "traces")
LOGS = os.path.join(ROOT, "logs", "m27")

SCHEMA = "m27.v1"
FORMAL_TAG = "h20_formal"
SMOKE_TAG = "h20_smoke"
LOCAL_TAG = "local_preflight"
ALLOWED_TAGS = (FORMAL_TAG, SMOKE_TAG, LOCAL_TAG)

SERVED_MODEL_NAME = "m27-model"
HOST = "127.0.0.1"
PROXY_PORT = 8300
HTTP_PORT_BASE = 8400
SIDE_CHANNEL_PORT_BASE = 5600

FROZEN = {
    "plan_version": "v1",
    "software": {
        "python_minor": "3.11",
        "torch": "2.11.0+cu130",
        "cuda": "13.0",
        "vllm_wheel": "0.24.0",
        "vllm_source_ref": "ee0da84",
        "nixl": "1.2.0",
        "nixl_backend_wheel": "nixl-cu13==1.2.0",
    },
    "model": {
        "name": "Qwen2.5-7B-Instruct",
        "artifact": "M22 offline copy (no re-download)",
        "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
        "hf_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "identity_manifest": "notes/m27/model_artifact/qwen2.5-7b-instruct.sha256",
        "identity_manifest_sha256": "157f08c69340e16b5d0358a31a4d20716f73a3b90ce6538ed557cff5fbe49638",
        "file_count": 15,
        "total_size_bytes": 15242807272,
        "torch_dtype": "bfloat16",
        "max_position_embeddings": 32768,
        "num_hidden_layers": 28,
        "num_attention_heads": 28,
        "num_key_value_heads": 4,
        "hidden_size": 3584,
        "head_dim": 128,
        "kv_dtype_bytes": 2,
    },
    "engine": {
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "seed": 0,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 256,
        "max_num_batched_tokens": 8192,
        "enforce_eager": True,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "enable_chunked_prefill": True,
        "block_size": 16,
        "max_model_len": 32768,
        "attention_backend": "FLASH_ATTN",
        "stream_interval": 1,
        "kv_cache_dtype": "auto",
        "quantization": None,
        "speculative_config": None,
    },
    "max_prompt_len": 16384,
    "legal_fallback": {
        "max_model_len": 16384,
        "max_prompt_len": 8192,
        "rule": "only if config.json max_position_embeddings < 32768; drop context points >= 16384",
    },
    "kv_cache_layout": "HND",
    "kv_transfer": {
        "kv_connector": "NixlConnector",
        "producer_role": "kv_producer",
        "consumer_role": "kv_consumer",
        "kv_load_failure_policy": "fail",
        "kv_buffer_device": "cuda",
        "backends": ["UCX"],
    },
    "kv_bytes_per_token": 57344,
    "warmup": 5,
    "measure": 15,
    "e1": {"contexts": [512, 2048, 4096, 8192, 16384], "output_tokens": 8},
    "e2": {
        "d_prompt": 128,
        "d_output": 512,
        "inject_after_token": 64,
        "p_lens": [0, 2048, 8192, 16384],
        "p_output": 1,
        "arms": ["colo1_mbt2048", "colo1_mbt4096", "colo1_mbt8192", "pd_1p1d"],
    },
    "e3": {
        "shapes": {
            "d_heavy": [256, 256],
            "balanced": [2048, 256],
            "p_heavy": [8192, 64],
        },
        "mixture": {"d_heavy": 0.5, "balanced": 0.3, "p_heavy": 0.2},
        "seeds": [0, 1, 2],
        "burstiness": 1.0,
        "duration_s": 60,
        "load_fractions": [0.5, 0.75, 0.9],
        "arms": ["colo1_mbt8192", "colo2_mbt2048", "colo2_mbt4096", "colo2_mbt8192", "pd_1p1d"],
        "calibration_arm": "colo2_mbt8192",
        "slo_multipliers": [2, 5, 10],
    },
    "e4": {
        "shapes": ["d_heavy", "p_heavy", "mixture"],
        "arms": ["colo4_mbt8192", "pd_1p3d", "pd_2p2d", "pd_3p1d"],
        "calibration_arm": "colo4_mbt8192",
    },
    "e5_optional": {
        "arms": ["colo8_mbt8192", "pd_2p6d", "pd_4p4d", "pd_6p2d"],
        "gate": "all 8 GPUs pairwise NV# in nvidia-smi topo -m",
    },
    "bootstrap": {"iters": 10000, "seed": 0},
}

E2_PREFIX = "e2"
REQUIRED_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_KV_CACHE_LAYOUT": "HND",
    "VLLM_SERVER_DEV_MODE": "1",
    "PYTHONHASHSEED": "0",
}


# --------------------------------------------------------------------------- KV 理论值
def kv_bytes_per_token(model=None) -> int:
    m = model or FROZEN["model"]
    return 2 * m["num_hidden_layers"] * m["num_key_value_heads"] * m["head_dim"] * m["kv_dtype_bytes"]


def kv_theory(prompt_len: int, block_size: int | None = None) -> dict:
    """理论 KV 字节：token 口径与整 block 口径（NIXL 以 block 为单位搬运）。≠ 实测 NIXL bytes。"""
    bs = block_size or FROZEN["engine"]["block_size"]
    per_tok = kv_bytes_per_token()
    n_blocks = math.ceil(prompt_len / bs)
    return {
        "kv_bytes_per_token": per_tok,
        "theory_bytes_tokens": prompt_len * per_tok,
        "theory_blocks": n_blocks,
        "theory_bytes_blocks": n_blocks * bs * per_tok,
    }


# --------------------------------------------------------------------------- 拓扑
def parse_arm(arm: str) -> dict:
    """arm 名 → 拓扑描述。

    colo{N}_mbt{X}: N 个独立 colocated 副本（无 KV connector），proxy round-robin。
    pd_{x}p{y}d:    x 个 P(kv_producer) + y 个 D(kv_consumer)，mbt 固定为冻结值。
    """
    m = re.fullmatch(r"colo(\d+)_mbt(\d+)", arm)
    if m:
        n, mbt = int(m.group(1)), int(m.group(2))
        return {"arm": arm, "kind": "colo", "n_p": 0, "n_d": 0, "n_colo": n,
                "mbt": mbt, "n_gpus": n}
    m = re.fullmatch(r"pd_(\d+)p(\d+)d", arm)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        return {"arm": arm, "kind": "pd", "n_p": x, "n_d": y, "n_colo": 0,
                "mbt": FROZEN["engine"]["max_num_batched_tokens"], "n_gpus": x + y}
    raise ValueError(f"unknown arm {arm!r}")


def build_instances(arm: str, gpu_ids: list[int]) -> list[dict]:
    """按 arm 分配 GPU / 端口。gpu_ids 由 Gate 0 拓扑检查后冻结（configs/gpu_map_<tag>.json）。"""
    t = parse_arm(arm)
    if len(gpu_ids) < t["n_gpus"]:
        raise ValueError(f"{arm} 需要 {t['n_gpus']} 张 GPU，只给了 {gpu_ids}")
    roles = (["colo"] * t["n_colo"]) if t["kind"] == "colo" else (["P"] * t["n_p"] + ["D"] * t["n_d"])
    out = []
    for i, role in enumerate(roles):
        out.append({
            "name": f"{role}{i}",
            "role": role,
            "gpu": gpu_ids[i],
            "http_port": HTTP_PORT_BASE + i,
            "side_channel_port": SIDE_CHANNEL_PORT_BASE + i,
            "mbt": t["mbt"],
        })
    if len({x["gpu"] for x in out}) != len(out):
        raise ValueError(f"GPU 映射重复：{out}")
    return out


def kv_transfer_config(role: str) -> dict | None:
    if role == "colo":
        return None
    kt = FROZEN["kv_transfer"]
    return {
        "kv_connector": kt["kv_connector"],
        "kv_role": kt["producer_role"] if role == "P" else kt["consumer_role"],
        "kv_load_failure_policy": kt["kv_load_failure_policy"],
        "kv_buffer_device": kt["kv_buffer_device"],
        "kv_connector_extra_config": {"backends": list(kt["backends"])},
    }


def serve_cmd(model_dir: str, inst: dict, engine: dict | None = None) -> list[str]:
    e = dict(engine or FROZEN["engine"])
    cmd = [
        "vllm", "serve", model_dir,
        "--host", HOST, "--port", str(inst["http_port"]),
        "--served-model-name", SERVED_MODEL_NAME,
        "--tensor-parallel-size", str(e["tensor_parallel_size"]),
        "--dtype", e["dtype"],
        "--seed", str(e["seed"]),
        "--gpu-memory-utilization", str(e["gpu_memory_utilization"]),
        "--max-num-seqs", str(e["max_num_seqs"]),
        "--max-num-batched-tokens", str(inst["mbt"]),
        "--block-size", str(e["block_size"]),
        "--max-model-len", str(e["max_model_len"]),
        "--attention-backend", e["attention_backend"],
        "--stream-interval", str(e["stream_interval"]),
        "--kv-cache-dtype", e["kv_cache_dtype"],
        "--no-enable-prefix-caching",
        "--no-async-scheduling",
        "--enable-chunked-prefill",
        "--disable-uvicorn-access-log",
    ]
    if e["enforce_eager"]:
        cmd.append("--enforce-eager")
    ktc = kv_transfer_config(inst["role"])
    if ktc is not None:
        cmd += ["--kv-transfer-config", json.dumps(ktc, separators=(",", ":"))]
    return cmd


def instance_env(inst: dict, base: dict | None = None) -> dict:
    env = dict(base if base is not None else os.environ)
    env.update(REQUIRED_ENV)
    env["CUDA_VISIBLE_DEVICES"] = str(inst["gpu"])
    env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = HOST
    env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(inst["side_channel_port"])
    return env


# --------------------------------------------------------------------------- Prometheus
_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eEinfINFNaN]+)")
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom(text: str) -> list[tuple[str, dict, float]]:
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        labels = dict(_LABEL.findall(m.group(2) or ""))
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        rows.append((m.group(1), labels, v))
    return rows


def prom_sum(rows, name: str, **label_eq) -> float | None:
    """同名样本求和（多 engine label 时合计）；一个都没有返回 None（≠0，不得混淆）。"""
    vals = [v for n, lab, v in rows if n == name and all(lab.get(k) == x for k, x in label_eq.items())]
    return sum(vals) if vals else None


# D 侧（READ 发起方）记录 NIXL telemetry：base_worker.py:2003-2020 / stats.py:45-50
NIXL_SERIES = {
    "nixl_xfer_s_sum": "vllm:nixl_xfer_time_seconds_sum",
    "nixl_xfer_count": "vllm:nixl_xfer_time_seconds_count",
    "nixl_post_s_sum": "vllm:nixl_post_time_seconds_sum",
    "nixl_post_count": "vllm:nixl_post_time_seconds_count",
    "nixl_bytes_sum": "vllm:nixl_bytes_transferred_sum",
    "nixl_bytes_count": "vllm:nixl_bytes_transferred_count",
    "nixl_desc_sum": "vllm:nixl_num_descriptors_sum",
    "nixl_failed_transfers": "vllm:nixl_num_failed_transfers_total",
    "nixl_failed_notifications": "vllm:nixl_num_failed_notifications_total",
}
PROMPT_SOURCE_SERIES = "vllm:prompt_tokens_by_source_total"
PROMPT_SOURCES = ("local_compute", "local_cache_hit", "external_kv_transfer")


def metrics_snapshot(rows) -> dict:
    snap = {k: prom_sum(rows, v) for k, v in NIXL_SERIES.items()}
    for s in PROMPT_SOURCES:
        snap[f"prompt_{s}"] = prom_sum(rows, PROMPT_SOURCE_SERIES, source=s)
    snap["prompt_tokens_total"] = prom_sum(rows, "vllm:prompt_tokens_total")
    snap["generation_tokens_total"] = prom_sum(rows, "vllm:generation_tokens_total")
    snap["num_requests_running"] = prom_sum(rows, "vllm:num_requests_running")
    snap["num_requests_waiting"] = prom_sum(rows, "vllm:num_requests_waiting")
    snap["kv_cache_usage_perc"] = prom_sum(rows, "vllm:kv_cache_usage_perc")
    snap["request_success_total"] = prom_sum(rows, "vllm:request_success_total")
    snap["num_preemptions_total"] = prom_sum(rows, "vllm:num_preemptions_total")
    return snap


def snapshot_delta(before: dict, after: dict) -> dict:
    """after − before；任一侧缺失则为 None（缺失 ≠ 0）。"""
    out = {}
    for k in after:
        a, b = after.get(k), before.get(k)
        out[k] = None if (a is None or b is None) else a - b
    return out


# --------------------------------------------------------------------------- IO
def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = dict(row)
    row.setdefault("schema", SCHEMA)
    row.setdefault("unix_ts", time.time())
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: str, obj) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def sha256_ids(ids) -> str:
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def port_free(port: int, host: str = HOST) -> bool:
    s = socket.socket()
    try:
        return s.connect_ex((host, port)) != 0
    finally:
        s.close()


# --------------------------------------------------------------------------- 环境 / GPU
def nvidia_smi_gpus() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version",
             "--format=csv,noheader,nounits"], text=True, timeout=20)
    except Exception as e:
        return [{"error": f"{type(e).__name__}: {e}"}]
    gpus = []
    for line in out.strip().splitlines():
        idx, name, tot, used, drv = [x.strip() for x in line.split(",")]
        gpus.append({"index": int(idx), "name": name, "memory_total_mib": int(tot),
                     "memory_used_mib": int(used), "driver": drv})
    return gpus


def gpu_compute_apps() -> list[str]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"], text=True, timeout=20)
        return [x for x in out.strip().splitlines() if x.strip()]
    except Exception as e:
        return [f"error: {type(e).__name__}: {e}"]


def pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def env_fingerprint() -> dict:
    keys = sorted(set(REQUIRED_ENV) | {
        "CUDA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "VLLM_NIXL_SIDE_CHANNEL_HOST",
        "VLLM_NIXL_SIDE_CHANNEL_PORT", "M27_MODEL", "PYTHONPATH",
    } | {k for k in os.environ if k.startswith("UCX_") or k.startswith("NIXL_")})
    return {
        "hostname": socket.gethostname(),
        "uname": " ".join(platform.uname()),
        "python": sys.version.split()[0],
        "packages": {p: pkg_version(p) for p in
                     ("vllm", "torch", "nixl", "nixl-cu13", "nixl-cu12", "aiohttp", "transformers")},
        "env": {k: os.environ.get(k) for k in keys},
        "gpus": nvidia_smi_gpus(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def formal_guard(tag: str, need_gpus: int) -> None:
    """formal/h20 tag 的硬门禁：没有本机 preflight PASS 记录就拒绝启动。

    local_preflight 不受此限，但其产物全部带 tag=local_preflight，永不进入正式 CSV。
    """
    if tag not in ALLOWED_TAGS:
        raise SystemExit(f"[m27] tag 必须是 {ALLOWED_TAGS}，收到 {tag!r}")
    if tag == LOCAL_TAG:
        return
    pf = load_json(os.path.join(CONFIGS, f"preflight_{tag}.json"))
    if not pf or pf.get("result") != "PASS":
        raise SystemExit(f"[m27] 缺 outputs/m27/configs/preflight_{tag}.json 或 result!=PASS；"
                         "先跑 bash scripts/m27_preflight.sh 并通过")
    if pf.get("hostname") != socket.gethostname():
        raise SystemExit(f"[m27] preflight 记录的主机 {pf.get('hostname')} ≠ 当前 {socket.gethostname()}")
    gpus = [g for g in nvidia_smi_gpus() if "name" in g]
    if not gpus or not all("H20" in g["name"] for g in gpus):
        raise SystemExit(f"[m27] tag={tag} 只能在 H20 上跑；当前 GPU：{[g.get('name') for g in gpus]}")
    if len(gpus) < need_gpus:
        raise SystemExit(f"[m27] 需要 {need_gpus} 张 GPU，只看到 {len(gpus)}")
    apps = [a for a in gpu_compute_apps() if not a.startswith("error")]
    if apps:
        raise SystemExit(f"[m27] GPU 上已有计算进程（不 kill，停下问所有者）：{apps}")


def gpu_map(tag: str) -> list[int]:
    """Gate 0 冻结的 GPU 列表；没有就拒绝（不假设 GPU0–7 同一 NVLink 域）。"""
    gm = load_json(os.path.join(CONFIGS, f"gpu_map_{tag}.json"))
    if not gm or not gm.get("gpu_ids"):
        raise SystemExit(f"[m27] 缺 outputs/m27/configs/gpu_map_{tag}.json（Gate 0 产出）")
    return [int(x) for x in gm["gpu_ids"]]


def effective_engine(tag: str) -> dict:
    """Gate 0 的合法回退只允许改 max_model_len（及 max_prompt_len / 删 ≥16384 的点）。"""
    e = dict(FROZEN["engine"])
    gm = load_json(os.path.join(CONFIGS, f"gpu_map_{tag}.json")) or {}
    if gm.get("max_model_len_fallback"):
        e["max_model_len"] = FROZEN["legal_fallback"]["max_model_len"]
    return e


def max_prompt_len(tag: str) -> int:
    gm = load_json(os.path.join(CONFIGS, f"gpu_map_{tag}.json")) or {}
    if gm.get("max_model_len_fallback"):
        return FROZEN["legal_fallback"]["max_prompt_len"]
    return FROZEN["max_prompt_len"]
