#!/usr/bin/env python3
"""M28 公共设施：冻结配置 / server 生命周期与进程归属 / GPU 门禁 / 指标与流式客户端。

只用标准库 + psutil，不 import vllm（validator / figures 也会 import 本模块）。

设计约束（notes/m28/M28_PLAN.md）：
  - 冻结配置的唯一来源是本文件的 FROZEN；计划文档里的 JSON 块必须与它逐键相等。
  - server 一律 `start_new_session=True` 启动：API 进程 = 会话首进程，pid == sid == pgid。
    「本轮脚本亲自启动」的判据 = 进程 sid == 我们 Popen 的 pid 且 create_time 不早于启动时刻。
  - 只对上述判据通过、且角色（API / VLLM::EngineCore）再次核对过的 PID 发信号；
    禁止 pkill / killall / 按名字模糊匹配。
  - 任何失败都原样落盘（异常类型 + 原文），不吞。
"""

import hashlib
import http.client
import json
import os
import random
import re
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "m28")
RAW = os.path.join(OUT, "raw")
LOGD = os.path.join(ROOT, "logs", "m28")
PLAN = os.path.join(ROOT, "notes", "m28", "M28_PLAN.md")
VLLM_SRC = os.path.join(os.path.dirname(ROOT), "vllm-src")
HOST = "127.0.0.1"
GIB = 1 << 30

ENGINE_CORE_TITLE = "VLLM::EngineCore"

FROZEN = {
    "plan_version": "v1",
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "model_snapshot_commit": "7ae557604adf67be50417f59c2c2f167def9a775",
    "model_safetensors_blob_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
    "host": HOST,
    "port": 8028,
    "vllm_source_ref": "ee0da84",
    "vllm_src_dirty_files_expected": 22,
    "kv_bytes_per_token_theory": 12288,
    "block_size": 16,
    "server_common_args": [
        "--seed", "0", "--enforce-eager", "--dtype", "auto",
        "--gpu-memory-utilization", "0.45", "--max-num-seqs", "16",
        "--max-num-batched-tokens", "8192", "--no-async-scheduling",
        "--enable-chunked-prefill", "--disable-uvicorn-access-log",
        "--enable-prompt-tokens-details",
    ],
    "server_env_set": {
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
        "HF_HUB_OFFLINE": "1",
    },
    "server_env_must_unset": [
        "VLLM_LAB_TRACE", "VLLM_LAB_BLOCK", "VLLM_LAB_M9_COALESCE",
        "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH",
    ],
    "A": {
        "max_model_len": 16640,
        "num_gpu_blocks_override": 1280,
        "prefix_lens": [1024, 4096, 8192, 16384],
        "suffix_len": 20,
        "max_tokens": 16,
        "filler_len": 12288,
        "n_fillers": 2,
        "warmup_trials": 5,
        "measure_trials": 15,
        "server_warmup_requests": 2,
        "token_id_range": [1000, 150000],
        "metrics_settle_s": 0.3,
        "store_quiesce_s": 0.5,
        "store_quiesce_timeout_s": 20,
        "request_timeout_s": 120,
        "cpu_bytes_to_use": 4 * GIB,
        "evict_cpu_bytes_to_use": GIB // 4,
        "evict_prefix_lens": [4096],
        "arms": {
            "offload": {"connector": True, "prefix_caching": True,
                        "paths": ["cold", "gpu_hot", "cpu_hit"]},
            "base": {"connector": False, "prefix_caching": True,
                     "paths": ["cold", "gpu_hot", "recompute_evicted"]},
            "ref": {"connector": False, "prefix_caching": False,
                    "paths": ["ref_a", "ref_b"]},
            "evict": {"connector": True, "prefix_caching": True,
                      "paths": ["cold", "gpu_hot", "cpu_evicted"], "optional": True},
        },
        "arm_order": ["offload", "base", "ref", "evict"],
    },
    "B": {
        "max_model_len": 4096,
        "num_gpu_blocks_override": 2048,
        "prompt_len": 256,
        "n_stream": 4,
        "n_nonstream": 4,
        "health_poll_s": 0.1,
        "gpu_poll_s": 1.0,
        "server_ready_timeout_s": 300,
        "B1": {"max_tokens": 2048, "fault_after_first_token_s": 3.0,
               "observe_s": 60, "probe_new_request_at_s": [1.0, 8.0], "repeats": 3},
        "B3": {"shutdown_timeouts": [0, 2, 60], "max_tokens": 1024,
               "fault_after_first_token_s": 1.0, "observe_s": 90,
               "probe_interval_s": 0.25, "repeats": 3},
        "B2": {"stop_s": 10, "max_tokens": 2048, "fault_after_first_token_s": 3.0,
               "observe_s": 40, "repeats": 1, "requires_separate_authorization": True},
    },
    "stats": {"bootstrap_iters": 10000, "bootstrap_seed": 0, "ci": 0.95},
    "gpu_gate": {"max_baseline_used_mib": 6000, "reclaim_tol_mib": 800,
                 "reclaim_timeout_s": 240},
}


# --------------------------------------------------------------------------- 工具
def now():
    return time.time()


def sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_tokens(ids):
    return sha256_text(",".join(map(str, ids)))


def frozen_json():
    return json.dumps(FROZEN, ensure_ascii=False, indent=2, sort_keys=True)


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path


def load_jsonl(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


class JsonlWriter:
    """逐样本落盘；已存在的同名文件先整体搬到 raw_old_<ts>/（只备份不删除）。"""

    def __init__(self, path, backup_existing=False, append=False):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not append:
            if not backup_existing:
                raise SystemExit(f"{path} 已存在；如需重跑请加 --backup-existing（旧文件只搬走不删除）")
            bdir = os.path.join(os.path.dirname(path), "..",
                                "raw_old_" + time.strftime("%Y%m%dT%H%M%S"))
            os.makedirs(bdir, exist_ok=True)
            shutil.move(path, os.path.join(bdir, os.path.basename(path)))
        self._lock = threading.Lock()

    def write(self, rec):
        with self._lock, open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- prompts
def _rand_ids(key, n):
    lo, hi = FROZEN["A"]["token_id_range"]
    rng = random.Random(sha256_text(key))
    return [rng.randrange(lo, hi) for _ in range(n)]


def trial_prompt(p_len, idx, suffix_len=None, salt="a"):
    """(P, idx) → 确定性 token 序列；所有 arm 用同一序列（跨 arm 可配对）。"""
    s = FROZEN["A"]["suffix_len"] if suffix_len is None else suffix_len
    return _rand_ids(f"m28|{salt}|prefix|{p_len}|{idx}", p_len) + \
        _rand_ids(f"m28|{salt}|suffix|{p_len}|{idx}", s)


def filler_prompt(p_len, idx, j):
    return _rand_ids(f"m28|filler|{p_len}|{idx}|{j}", FROZEN["A"]["filler_len"])


def expected_hit_tokens(n_tokens, block=None):
    """前缀缓存最多命中 num_tokens-1 内的完整 block（最后一个 token 必须重算）。"""
    b = block or FROZEN["block_size"]
    return ((n_tokens - 1) // b) * b


def trial_order(idx, lens):
    order = list(lens)
    random.Random(f"m28|order|{idx}").shuffle(order)
    return order


# --------------------------------------------------------------------------- server 命令
def kv_transfer_config(cpu_bytes):
    return json.dumps({"kv_connector": "OffloadingConnector", "kv_role": "kv_both",
                       "kv_connector_extra_config": {"cpu_bytes_to_use": int(cpu_bytes)}},
                      separators=(",", ":"))


def a_server_args(arm):
    A = FROZEN["A"]
    cfg = A["arms"][arm]
    args = list(FROZEN["server_common_args"]) + [
        "--max-model-len", str(A["max_model_len"]),
        "--num-gpu-blocks-override", str(A["num_gpu_blocks_override"]),
    ]
    args.append("--enable-prefix-caching" if cfg["prefix_caching"] else "--no-enable-prefix-caching")
    if cfg["connector"]:
        cpu = A["evict_cpu_bytes_to_use"] if arm == "evict" else A["cpu_bytes_to_use"]
        args += ["--kv-transfer-config", kv_transfer_config(cpu)]
    return args


def b_server_args(shutdown_timeout=None):
    B = FROZEN["B"]
    args = list(FROZEN["server_common_args"]) + [
        "--max-model-len", str(B["max_model_len"]),
        "--num-gpu-blocks-override", str(B["num_gpu_blocks_override"]),
        "--no-enable-prefix-caching",
    ]
    if shutdown_timeout is not None:
        args += ["--shutdown-timeout", str(int(shutdown_timeout))]
    return args


def model_path():
    """本地 HF 快照目录：用路径启动 → vLLM 不走 hub 解析，杜绝自动下载；revision 由 commit 固定。"""
    org, name = FROZEN["model"].split("/")
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    return os.path.join(hub, f"models--{org}--{name}", "snapshots", FROZEN["model_snapshot_commit"])


def model_identity():
    p = model_path()
    st = os.path.join(p, "model.safetensors")
    blob = os.path.basename(os.path.realpath(st)) if os.path.islink(st) else None
    return {"path": p, "exists": os.path.isdir(p), "safetensors_blob": blob,
            "blob_matches_frozen": blob == FROZEN["model_safetensors_blob_sha256"]}


def vllm_serve_cmd(extra_args, port=None):
    vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
    return [vllm_bin, "serve", model_path(), "--served-model-name", FROZEN["model"], "--host", HOST,
            "--port", str(port or FROZEN["port"])] + list(extra_args)


def server_env():
    env = dict(os.environ)
    for k in FROZEN["server_env_must_unset"]:
        env.pop(k, None)
    env.update(FROZEN["server_env_set"])
    return env


# --------------------------------------------------------------------------- 主机/GPU 快照
def nvidia_smi():
    """整卡口径（WSL2 拿不到进程级显存）；compute_apps 有就记录，没有不代表没人用。"""
    out = {"ts": now()}
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        rows = [[x.strip() for x in line.split(",")] for line in q.splitlines()]
        out["gpus"] = [{"index": int(r[0]), "name": r[1], "mem_used_mib": int(r[2]),
                        "mem_total_mib": int(r[3]), "util_pct": int(r[4])} for r in rows]
        apps = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader,nounits"], text=True, timeout=20).strip()
        out["compute_apps"] = [line.strip() for line in apps.splitlines() if line.strip()]
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def gpu_used_mib(snap=None):
    snap = snap or nvidia_smi()
    try:
        return snap["gpus"][0]["mem_used_mib"]
    except (KeyError, IndexError):
        return None


def dev_shm_listing():
    try:
        return sorted(os.listdir("/dev/shm"))
    except OSError as e:
        return [f"<error {e}>"]


def host_mem():
    import psutil
    vm = psutil.virtual_memory()
    return {"total_gib": round(vm.total / GIB, 2), "available_gib": round(vm.available / GIB, 2)}


def gpu_gate(authorized, port=None):
    """GPU 授权门禁：未带授权标志 / 显存基线超冻结阈值 / 有可见计算进程 / 端口被占 → 拒绝。"""
    snap = nvidia_smi()
    reasons = []
    if not authorized:
        reasons.append("未提供 --i-have-gpu-authorization（需所有者本轮明确授权）")
    if "error" in snap:
        reasons.append(f"nvidia-smi 失败: {snap['error']}")
    else:
        used = gpu_used_mib(snap)
        lim = FROZEN["gpu_gate"]["max_baseline_used_mib"]
        if used is None or used > lim:
            reasons.append(f"GPU 已用 {used} MiB > 冻结阈值 {lim} MiB（可能有别人的任务）")
        if snap.get("compute_apps"):
            reasons.append(f"nvidia-smi 列出计算进程 {snap['compute_apps']}（停下来问所有者）")
    if not port_free(port or FROZEN["port"]):
        reasons.append(f"端口 {port or FROZEN['port']} 已被占用")
    mi = model_identity()
    if not (mi["exists"] and mi["blob_matches_frozen"]):
        reasons.append(f"本地模型快照不符合冻结身份：{mi}（不下载，停下来问所有者）")
    return (not reasons), reasons, snap


# --------------------------------------------------------------------------- 端口 / HTTP
def port_free(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((HOST, port))
            return True
        except OSError:
            return False


def port_listening(port, timeout=0.3):
    try:
        with socket.create_connection((HOST, port), timeout=timeout):
            return True
    except OSError:
        return False


def health_code(port, timeout=2.0):
    """200/503/...；连接失败返回 0，并保留异常类型。"""
    try:
        c = http.client.HTTPConnection(HOST, port, timeout=timeout)
        c.request("GET", "/health")
        r = c.getresponse()
        r.read()
        c.close()
        return r.status, None
    except Exception as e:
        return 0, type(e).__name__


def scrape_metrics(port, timeout=10):
    c = http.client.HTTPConnection(HOST, port, timeout=timeout)
    c.request("GET", "/metrics")
    r = c.getresponse()
    body = r.read().decode()
    c.close()
    if r.status != 200:
        raise RuntimeError(f"/metrics HTTP {r.status}")
    return body


_PROM_LINE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


def parse_prom(text):
    """sample 名 → 跨 label 求和（跳过 _created / _bucket）。"""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _PROM_LINE.match(line)
        if not m:
            continue
        name = m.group(1)
        if name.endswith("_created") or name.endswith("_bucket"):
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        out[name] = out.get(name, 0.0) + v
    return out


COUNTERS = {
    "local_queries": "vllm:prefix_cache_queries",
    "local_hits": "vllm:prefix_cache_hits",
    "ext_queries": "vllm:external_prefix_cache_queries",
    "ext_hits": "vllm:external_prefix_cache_hits",
    "load_bytes": "vllm:kv_offload_load_bytes",
    "load_time_s": "vllm:kv_offload_load_time",
    "store_bytes": "vllm:kv_offload_store_bytes",
    "store_time_s": "vllm:kv_offload_store_time",
    "prompt_tokens": "vllm:prompt_tokens",
    "generation_tokens": "vllm:generation_tokens",
    "request_success": "vllm:request_success",
}
GAUGES = {"running": "vllm:num_requests_running", "waiting": "vllm:num_requests_waiting"}


def counter(m, base):
    """返回 None 表示该指标在 /metrics 中不存在（与 0 区分）。"""
    for k in (base + "_total", base):
        if k in m:
            return m[k]
    return None


def metric_snapshot(port):
    m = parse_prom(scrape_metrics(port))
    snap = {k: counter(m, v) for k, v in COUNTERS.items()}
    snap.update({k: m.get(v) for k, v in GAUGES.items()})
    return snap


def metric_delta(before, after):
    d = {}
    for k in COUNTERS:
        b, a = before.get(k), after.get(k)
        d[k] = None if (a is None and b is None) else (a or 0.0) - (b or 0.0)
    return d


# --------------------------------------------------------------------------- completions 客户端
def _parse_token_ids(lp):
    ids = []
    for t in (lp or {}).get("tokens") or []:
        if isinstance(t, str) and t.startswith("token_id:"):
            ids.append(int(t.split(":", 1)[1]))
        else:
            ids.append(t)
    return ids


def completion(port, prompt_ids, max_tokens, stream=True, timeout=120, extra=None,
               on_first_token=None):
    """/v1/completions；返回逐请求记录（原始错误原样保留，不吞）。"""
    body = {"model": FROZEN["model"], "prompt": prompt_ids, "max_tokens": max_tokens,
            "temperature": 0.0, "top_p": 1.0, "ignore_eos": True, "logprobs": 1,
            "return_tokens_as_token_ids": True, "stream": stream}
    if stream:
        body["stream_options"] = {"include_usage": True}
    body.update(extra or {})
    rec = {"stream": stream, "n_prompt_tokens": len(prompt_ids), "prompt_sha256": sha_tokens(prompt_ids),
           "max_tokens": max_tokens, "t_send": None, "t_first_token": None, "t_end": None,
           "http_status": None, "token_ids": [], "token_logprobs": [], "finish_reason": None,
           "usage": None, "error_event": None, "exception": None, "n_sse_events": 0,
           "done_marker": False, "sse_tail": []}
    conn = None
    try:
        conn = http.client.HTTPConnection(HOST, port, timeout=timeout)
        payload = json.dumps(body)
        rec["t_send"] = now()
        t0 = time.perf_counter()
        conn.request("POST", "/v1/completions", body=payload,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        rec["http_status"] = resp.status
        if not stream or resp.status != 200:
            raw = resp.read().decode(errors="replace")
            rec["t_end"] = now()
            rec["e2e_ms"] = (time.perf_counter() - t0) * 1e3
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                rec["raw_body"] = raw[:2000]
                return rec
            if "error" in obj or resp.status != 200:
                rec["error_event"] = obj
                return rec
            ch = obj["choices"][0]
            rec["token_ids"] = _parse_token_ids(ch.get("logprobs"))
            rec["token_logprobs"] = (ch.get("logprobs") or {}).get("token_logprobs") or []
            rec["finish_reason"] = ch.get("finish_reason")
            rec["usage"] = obj.get("usage")
            return rec
        buf = b""
        while True:
            line = resp.readline()
            if not line:
                break
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip().decode(errors="replace")
            rec["n_sse_events"] += 1
            rec["sse_tail"] = (rec["sse_tail"] + [data[:400]])[-3:]
            if data == "[DONE]":
                rec["done_marker"] = True
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                buf += line
                continue
            if "error" in obj:
                rec["error_event"] = obj
                continue
            if obj.get("usage"):
                rec["usage"] = obj["usage"]
            for ch in obj.get("choices") or []:
                ids = _parse_token_ids(ch.get("logprobs"))
                if ids and rec["t_first_token"] is None:
                    rec["t_first_token"] = now()
                    rec["online_ttft_ms"] = (time.perf_counter() - t0) * 1e3
                    if on_first_token:
                        on_first_token()
                rec["token_ids"] += ids
                rec["token_logprobs"] += (ch.get("logprobs") or {}).get("token_logprobs") or []
                if ch.get("finish_reason"):
                    rec["finish_reason"] = ch["finish_reason"]
        rec["t_end"] = now()
        rec["e2e_ms"] = (time.perf_counter() - t0) * 1e3
        if buf:
            rec["unparsed_sse"] = buf[:400].decode(errors="replace")
    except Exception as e:
        rec["t_end"] = now()
        rec["exception"] = f"{type(e).__name__}: {e}"
    finally:
        if conn is not None:
            conn.close()
    return rec


def cached_tokens(rec):
    u = rec.get("usage") or {}
    d = u.get("prompt_tokens_details") or {}
    return d.get("cached_tokens")


def request_ok(rec):
    return (rec.get("http_status") == 200 and rec.get("exception") is None
            and rec.get("error_event") is None
            and len(rec.get("token_ids") or []) == rec.get("max_tokens")
            and rec.get("finish_reason") == "length")


# --------------------------------------------------------------------------- 进程归属
def proc_info(p):
    import psutil
    try:
        with p.oneshot():
            return {"pid": p.pid, "ppid": p.ppid(), "name": p.name(),
                    "cmdline": " ".join(p.cmdline())[:300], "sid": os.getsid(p.pid),
                    "pgid": os.getpgid(p.pid), "create_time": p.create_time(),
                    "status": p.status(), "rss_mib": round(p.memory_info().rss / 2**20, 1)}
    except (psutil.NoSuchProcess, ProcessLookupError):
        return {"pid": p.pid, "gone": True}
    except Exception as e:
        return {"pid": p.pid, "error": f"{type(e).__name__}: {e}"}


def pid_alive(pid):
    import psutil
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def is_engine_core(info):
    return ENGINE_CORE_TITLE in (info.get("cmdline") or "") or \
        (info.get("name") or "").startswith(ENGINE_CORE_TITLE[:15])


class Server:
    """只管理自己 Popen（新会话）出来的进程树。"""

    def __init__(self, cmd, env, log_path, port, label, active_file=None):
        self.cmd, self.env, self.log_path, self.port, self.label = cmd, env, log_path, port, label
        self.active_file = active_file
        self.proc = None
        self.lf = None
        self.t_start = None
        self.stopped_pids = set()  # 被 SIGSTOP 过、尚未确认 SIGCONT 的 PID
        self.known = {}  # pid -> 最近一次 info（只增不减，供事后清理核对）

    def _persist(self):
        if not self.active_file:
            return
        save_json(self.active_file, {"label": self.label, "api_pid": self.pid, "t_start": self.t_start,
                                     "port": self.port, "runner_pid": os.getpid(),
                                     "known": self.known, "stopped_pids": sorted(self.stopped_pids)})

    @property
    def pid(self):
        return self.proc.pid if self.proc else None

    def start(self, ready_timeout=300):
        if not port_free(self.port):
            raise RuntimeError(f"port {self.port} already in use")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.lf = open(self.log_path, "a")
        self.lf.write(f"\n===== {self.label} start {time.strftime('%F %T')} cmd={json.dumps(self.cmd)} =====\n")
        self.lf.flush()
        self.t_start = now()
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.lf, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        self._persist()
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < ready_timeout:
            code, _ = health_code(self.port)
            if code == 200:
                self.refresh_tree()
                return round(time.perf_counter() - t0, 2)
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited early rc={self.proc.returncode}; 见 {self.log_path}\n"
                                   + self.log_tail())
            self.refresh_tree()
            time.sleep(1.0)
        raise RuntimeError(f"server health timeout {ready_timeout}s; 见 {self.log_path}\n" + self.log_tail())

    def log_tail(self, n=40):
        try:
            with open(self.log_path, errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""

    def owned(self, info):
        """归属判据：sid == 我们的 API pid（新会话）且创建时间不早于启动时刻。"""
        return (self.pid is not None and info.get("sid") == self.pid
                and info.get("create_time", 0) >= self.t_start - 1.0)

    def refresh_tree(self):
        import psutil
        tree = []
        if self.pid is None:
            return tree
        try:
            root = psutil.Process(self.pid)
            procs = [root] + root.children(recursive=True)
        except psutil.NoSuchProcess:
            procs = []
        for p in procs:
            info = proc_info(p)
            if "gone" in info or "error" in info:
                continue
            if self.owned(info):
                self.known[info["pid"]] = info
                tree.append(info)
        self._persist()
        return tree

    def api_alive(self):
        return self.proc is not None and self.proc.poll() is None

    def engine_cores(self):
        return [i for i in self.refresh_tree() if is_engine_core(i) and i["pid"] != self.pid]

    def verify_target(self, pid, role):
        """发信号前的再次核对：必须是本会话、角色匹配、仍然存活。"""
        import psutil
        try:
            info = proc_info(psutil.Process(pid))
        except psutil.NoSuchProcess:
            return False, {"pid": pid, "reason": "不存在"}
        if "gone" in info or "error" in info:
            return False, info
        if not self.owned(info):
            return False, dict(info, reason="不属于本轮会话")
        if role == "engine_core":
            if not is_engine_core(info) or info["ppid"] != self.pid:
                return False, dict(info, reason="不是本 API 进程的 EngineCore 子进程")
        elif role == "api":
            if pid != self.pid or not self.api_alive():
                return False, dict(info, reason="不是本轮 Popen 的 API 进程")
        else:
            return False, dict(info, reason=f"未知角色 {role}")
        return True, info

    def signal(self, pid, sig, role):
        ok, info = self.verify_target(pid, role)
        rec = {"pid": pid, "role": role, "signal": int(sig), "verified": ok, "identity": info,
               "ts": now()}
        if ok:
            import signal as _sig
            if sig == _sig.SIGSTOP:
                self.stopped_pids.add(pid)
                self._persist()
            os.kill(pid, sig)
            rec["sent"] = True
            if sig == _sig.SIGCONT:
                self.stopped_pids.discard(pid)
                self._persist()
        else:
            rec["sent"] = False
        return rec

    def resume_stopped(self):
        """SIGSTOP 兜底：对所有记录过的被暂停 PID 发 SIGCONT（无论此前是否已发过）。"""
        import signal as _sig
        out = []
        for pid in sorted(self.stopped_pids):
            if pid_alive(pid):
                try:
                    os.kill(pid, _sig.SIGCONT)
                    out.append({"pid": pid, "signal": int(_sig.SIGCONT), "sent": True, "ts": now()})
                except ProcessLookupError:
                    out.append({"pid": pid, "sent": False, "reason": "已不存在"})
        self.stopped_pids.clear()
        self._persist()
        return out

    def cleanup(self, grace_s=30):
        """兜底清理：先 SIGTERM API（若还活），再对已知且仍属本会话的存活 PID 逐个 SIGKILL。"""
        import signal as _sig
        actions = self.resume_stopped()
        if self.api_alive():
            actions.append(self.signal(self.pid, _sig.SIGTERM, "api"))
            try:
                self.proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                actions.append({"pid": self.pid, "note": f"SIGTERM 后 {grace_s}s 未退出"})
        self.refresh_tree()
        for pid, info in sorted(self.known.items()):
            if not pid_alive(pid):
                continue
            import psutil
            try:
                cur = proc_info(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
            if self.owned(cur):
                os.kill(pid, _sig.SIGKILL)
                actions.append({"pid": pid, "signal": 9, "sent": True, "identity": cur, "ts": now()})
            else:
                actions.append({"pid": pid, "sent": False, "reason": "身份不再匹配，不发信号", "identity": cur})
        if self.proc is not None:
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                actions.append({"pid": self.pid, "note": "API 进程 SIGKILL 后仍未回收"})
        for _ in range(60):
            if port_free(self.port):
                break
            time.sleep(0.5)
        if self.lf:
            self.lf.close()
            self.lf = None
        if self.active_file and not self.alive_known():
            try:
                os.remove(self.active_file)
            except OSError:
                pass
        return actions

    def alive_known(self):
        return [pid for pid in self.known if pid_alive(pid)]

    def returncode(self):
        return None if self.proc is None else self.proc.poll()


def cleanup_from_record(active_file, dry_run=True):
    """人工停止路径：runner 异常死亡后，只按落盘记录清理（sid 与 create_time 都必须匹配）。"""
    import psutil
    import signal as _sig
    if not os.path.exists(active_file):
        return {"note": f"{active_file} 不存在，没有需要清理的记录"}
    with open(active_file) as f:
        rec = json.load(f)
    api_pid, t_start = rec["api_pid"], rec["t_start"]
    report = {"record": rec, "actions": []}
    if pid_alive(rec.get("runner_pid", -1)):
        report["refused"] = f"runner pid {rec['runner_pid']} 仍存活；请先让 runner 自己收尾（kill -INT）"
        return report
    cands = []
    for p in psutil.process_iter():
        info = proc_info(p)
        if info.get("sid") == api_pid and info.get("create_time", 0) >= t_start - 1.0:
            cands.append(info)
    report["candidates"] = cands
    for info in cands:
        if dry_run:
            report["actions"].append({"pid": info["pid"], "would": "SIGCONT+SIGKILL"})
            continue
        try:
            os.kill(info["pid"], _sig.SIGCONT)
            os.kill(info["pid"], _sig.SIGKILL)
            report["actions"].append({"pid": info["pid"], "sent": "SIGCONT+SIGKILL"})
        except ProcessLookupError:
            report["actions"].append({"pid": info["pid"], "sent": False, "reason": "已不存在"})
    if not dry_run and not cands:
        os.remove(active_file)
    return report


def wait_gpu_reclaim(baseline_mib, timeout_s=None, interval=2.0):
    """轮询整卡显存回到基线 + 容差；返回 (是否回收, 耗时秒, 最终读数)。"""
    tol = FROZEN["gpu_gate"]["reclaim_tol_mib"]
    timeout_s = timeout_s or FROZEN["gpu_gate"]["reclaim_timeout_s"]
    t0 = time.perf_counter()
    last = None
    while time.perf_counter() - t0 < timeout_s:
        last = gpu_used_mib()
        if last is not None and baseline_mib is not None and last - baseline_mib < tol:
            return True, round(time.perf_counter() - t0, 1), last
        time.sleep(interval)
    return False, round(time.perf_counter() - t0, 1), last


class GpuPoller(threading.Thread):
    """后台 nvidia-smi 轮询（整卡口径）。"""

    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            self.samples.append({"ts": now(), "mem_used_mib": gpu_used_mib()})
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=30)
        return self.samples


# --------------------------------------------------------------------------- 互斥锁 / 指纹
LOCK_FILE = os.path.join(RAW, ".m28_experiment.lock")


def acquire_lock(owner, raw_dir=RAW):
    lock = os.path.join(raw_dir, ".m28_experiment.lock")
    os.makedirs(raw_dir, exist_ok=True)
    if os.path.exists(lock):
        try:
            with open(lock) as f:
                info = json.load(f)
            alive = os.path.exists(f"/proc/{int(info['pid'])}")
        except Exception:
            info, alive = {}, False
        if alive:
            raise SystemExit(f"已有 M28 实验在跑（{info}），A/B 不得同时运行 GPU 引擎，拒绝启动。")
    with open(lock, "w") as f:
        json.dump({"pid": os.getpid(), "owner": owner, "since": time.strftime("%F %T%z")}, f)
    import atexit

    def _release():
        try:
            with open(lock) as f:
                if json.load(f).get("pid") == os.getpid():
                    os.remove(lock)
        except Exception:
            pass

    atexit.register(_release)
    return lock


def _git(args, cwd):
    try:
        return subprocess.check_output(["git"] + args, cwd=cwd, text=True, timeout=20).strip()
    except Exception as e:
        return f"<error {type(e).__name__}: {e}>"


def env_fingerprint(with_vllm_import=True):
    env = server_env()
    fp = {
        "ts": now(), "python": sys.version.split()[0], "executable": sys.executable,
        "lab_head": _git(["rev-parse", "HEAD"], ROOT),
        "lab_dirty_files": len([x for x in _git(["status", "--short"], ROOT).splitlines() if x.strip()]),
        "vllm_src_head": _git(["rev-parse", "HEAD"], VLLM_SRC),
        "vllm_src_dirty_files": len([x for x in _git(["status", "--short"], VLLM_SRC).splitlines() if x.strip()]),
        "server_env_set": {k: env.get(k) for k in FROZEN["server_env_set"]},
        "server_env_unset_ok": {k: (k not in env) for k in FROZEN["server_env_must_unset"]},
        "parent_env_lab_vars": {k: os.environ.get(k) for k in FROZEN["server_env_must_unset"]},
        "model_identity": model_identity(),
        "dev_shm_free_gib": round(shutil.disk_usage("/dev/shm").free / GIB, 2),
        "host_mem": host_mem(),
        "uname": " ".join(os.uname()),
    }
    if with_vllm_import:
        try:
            fp["vllm_version"] = subprocess.check_output(
                [sys.executable, "-c", "import vllm,os;print(vllm.__version__);print(os.path.dirname(vllm.__file__))"],
                env=env, text=True, timeout=180, stderr=subprocess.DEVNULL).strip().splitlines()
        except Exception as e:
            fp["vllm_version"] = f"<error {type(e).__name__}: {e}>"
    try:
        fp["nvidia_smi_driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
            text=True, timeout=20).strip()
    except Exception as e:
        fp["nvidia_smi_driver"] = f"<error {type(e).__name__}: {e}>"
    return fp


# --------------------------------------------------------------------------- 统计
def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def describe(xs):
    xs = list(xs)
    if not xs:
        return {"n": 0, "median": None, "stdev": None, "p95": None}
    return {"n": len(xs), "median": statistics.median(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0, "p95": pct(xs, 0.95)}


def boot_ci_median_paired(diffs, iters=None, seed=None):
    iters = iters or FROZEN["stats"]["bootstrap_iters"]
    seed = FROZEN["stats"]["bootstrap_seed"] if seed is None else seed
    if not diffs:
        return None, None, None
    rng = random.Random(seed)
    n = len(diffs)
    meds = sorted(statistics.median(rng.choices(diffs, k=n)) for _ in range(iters))
    return statistics.median(diffs), meds[int(0.025 * iters)], meds[int(0.975 * iters) - 1]


def boot_ci_median_diff(a, b, iters=None, seed=None):
    """median(a) - median(b) 的非配对 bootstrap 95% CI（固定 seed）。"""
    iters = iters or FROZEN["stats"]["bootstrap_iters"]
    seed = FROZEN["stats"]["bootstrap_seed"] if seed is None else seed
    if not a or not b:
        return None, None, None
    rng = random.Random(seed)
    ds = sorted(statistics.median(rng.choices(a, k=len(a))) - statistics.median(rng.choices(b, k=len(b)))
                for _ in range(iters))
    return statistics.median(a) - statistics.median(b), ds[int(0.025 * iters)], ds[int(0.975 * iters) - 1]
