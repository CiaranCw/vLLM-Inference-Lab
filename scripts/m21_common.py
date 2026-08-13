#!/usr/bin/env python3
"""M21 公共设施：server 生命周期 / Prometheus 查询 / bench 调用 / 指标快照。

设计约束（notes/m21/monitoring_and_capacity.md）：
  - 服务端配置**直接沿用 M12 冻结命令**（含端口 8014），保证结论可与 M12 交叉验证。
  - 进程管理只认**自己 Popen 出来的 PID**，并写 PID 文件；
    禁止 pkill/killall 之类模糊匹配（M12 曾因此 kill 掉执行 shell）。
"""

import json
import os
import re
import socket
import subprocess
import time
import urllib.parse
import urllib.request

HOST = "127.0.0.1"
PORT = 8014
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROM = "http://127.0.0.1:9090"

OUT = "outputs/m21"
LOGD = "logs/m21"
RAW = os.path.join(OUT, "raw")
PID_FILE = os.path.join(RAW, "server.pid")

ENV = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0")

# M12 冻结的 server 命令（唯一自变量在 bench 侧；E4/E6 需要改配置时用 extra_args 显式覆盖并记录）
SERVER_CMD = [
    "vllm", "serve", MODEL, "--host", HOST, "--port", str(PORT), "--seed", "0",
    "--enforce-eager", "--dtype", "auto", "--max-model-len", "2048",
    "--gpu-memory-utilization", "0.6", "--max-num-seqs", "64",
    "--max-num-batched-tokens", "4096", "--no-enable-prefix-caching",
    "--no-async-scheduling", "--enable-chunked-prefill", "--disable-uvicorn-access-log",
]

SHAPES = {"A": (256, 128), "B": (1024, 32),
          # C：长输出，用于从**需求侧**制造 KV 压力（压低显存预算到 0.10 仍只到 59%）
          "C": (128, 1900)}

# M12 冻结 SLO（本模块不新造标准，直接引用）
SLO_TTFT_P99_MS = 300.0


# --------------------------------------------------------------------------- HTTP
def http_get(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def health_code(port=PORT):
    try:
        with urllib.request.urlopen(f"http://{HOST}:{port}/health", timeout=3) as r:
            return r.status
    except Exception:
        return 0


def port_free(port=PORT):
    s = socket.socket()
    rc = s.connect_ex((HOST, port))
    s.close()
    return rc != 0


# --------------------------------------------------------------------------- vLLM /metrics
_COUNTER_RE = r"^{name}(?:\{{[^}}]*\}})?\s+([0-9.eE+-]+)"


def scrape_vllm_metrics():
    """抓一次 vLLM /metrics，返回 {gauge: 值, counter: 累计和}。

    counter 带 label（如 finished_reason / engine），故对同名样本**求和**；
    gauge 取最大值（单引擎下等价于取值）。
    """
    text = http_get(f"http://{HOST}:{PORT}/metrics", timeout=10)
    out = {}
    gauges = ("vllm:num_requests_running", "vllm:num_requests_waiting",
              "vllm:kv_cache_usage_perc")
    counters = ("vllm:num_preemptions_total", "vllm:prompt_tokens_total",
                "vllm:generation_tokens_total", "vllm:request_success_total",
                "vllm:num_requests_total")
    for name in gauges:
        vals = [float(m) for m in re.findall(_COUNTER_RE.format(name=re.escape(name)), text, re.M)]
        out[name] = max(vals) if vals else None
    for name in counters:
        vals = [float(m) for m in re.findall(_COUNTER_RE.format(name=re.escape(name)), text, re.M)]
        out[name] = sum(vals) if vals else 0.0
    out["_raw_len"] = len(text)
    return out


def wait_idle(timeout=40, need=3, interval=0.5):
    """轮询直到 running==0 且 waiting==0 连续 need 次。返回 (drained, elapsed_s)。"""
    t0 = time.perf_counter()
    ok = 0
    while time.perf_counter() - t0 < timeout:
        try:
            m = scrape_vllm_metrics()
            if (m.get("vllm:num_requests_running") or 0) == 0 and \
               (m.get("vllm:num_requests_waiting") or 0) == 0:
                ok += 1
                if ok >= need:
                    return True, round(time.perf_counter() - t0, 2)
            else:
                ok = 0
        except Exception:
            ok = 0
        time.sleep(interval)
    return False, round(time.perf_counter() - t0, 2)


def warmup(n=10):
    """打若干条极短请求，避开 **Triton GPU-kernel JIT** 冷启动（AGENTS.md §5）。

    这里的 Triton 指 **OpenAI Triton GPU 编程语言/编译器**的 kernel JIT，
    **不是** NVIDIA Triton Inference Server（M21 主实验没有用到推理服务器，
    Triton Inference Server 是 M20 的对象）。下文简称 Triton kernel JIT。
    """
    url = f"http://{HOST}:{PORT}/v1/completions"
    body = json.dumps({"model": MODEL, "prompt": "warmup", "max_tokens": 8,
                       "temperature": 0}).encode()
    for _ in range(n):
        try:
            urllib.request.urlopen(urllib.request.Request(
                url, data=body, headers={"Content-Type": "application/json"}), timeout=60).read()
        except Exception:
            pass
    wait_idle(60)


# --------------------------------------------------------------------------- Prometheus
def prom_query(expr, timeout=10):
    url = f"{PROM}/api/v1/query?{urllib.parse.urlencode({'query': expr})}"
    return json.loads(http_get(url, timeout=timeout))


def prom_scalar(expr, default=None):
    """取瞬时向量的第一个样本值；查不到返回 default。"""
    try:
        d = prom_query(expr)
        res = d["data"]["result"]
        if not res:
            return default
        return float(res[0]["value"][1])
    except Exception:
        return default


def prom_scalar_at(expr, ts, default=None):
    """在指定时刻求值（用于「对整个压测窗口做 increase() 后取分位数」这类查询）。

    **单位警告**：本函数原样返回 PromQL 的数值，**不做任何单位换算**。
    vLLM 的延迟直方图是 `*_seconds_bucket`，返回值单位是**秒**；
    存进名为 `*_ms` 的字段前必须自己 ×1000（E6 首版漏了这一步，19.57s 被记成 19.57ms）。
    """
    q = urllib.parse.urlencode({"query": expr, "time": ts})
    try:
        d = json.loads(http_get(f"{PROM}/api/v1/query?{q}", timeout=20))
        res = d["data"]["result"]
        if not res:
            return default
        v = float(res[0]["value"][1])
        return default if v != v else v  # NaN -> default
    except Exception:
        return default


def prom_range(expr, start_ts, end_ts, step="1s", timeout=30):
    """返回 [(ts, value), ...]（取第一条 series）。"""
    q = urllib.parse.urlencode({"query": expr, "start": start_ts, "end": end_ts, "step": step})
    d = json.loads(http_get(f"{PROM}/api/v1/query_range?{q}", timeout=timeout))
    res = d["data"]["result"]
    if not res:
        return []
    return [(float(t), float(v)) for t, v in res[0]["values"]]


def prom_range_agg(expr, start_ts, end_ts, step="1s"):
    """对 range 结果做 max/mean/last 汇总，缺数据返回 None。"""
    pts = [v for _, v in prom_range(expr, start_ts, end_ts, step) if v == v]
    if not pts:
        return {"max": None, "mean": None, "last": None, "n": 0}
    return {"max": max(pts), "mean": sum(pts) / len(pts), "last": pts[-1], "n": len(pts)}


def prom_targets():
    d = json.loads(http_get(f"{PROM}/api/v1/targets?state=any", timeout=10))
    return {t["labels"]["job"]: {"health": t["health"], "lastError": t.get("lastError", "")}
            for t in d["data"]["activeTargets"]}


def prom_alerts():
    d = json.loads(http_get(f"{PROM}/api/v1/alerts", timeout=10))
    return d["data"]["alerts"]


# --------------------------------------------------------------------------- Server
class Server:
    """只管理自己 Popen 出来的进程；PID 落盘供 exporter 与故障演练使用。"""

    def __init__(self, extra_args=None, port=PORT, pid_file=PID_FILE):
        self.extra_args = list(extra_args or [])
        self.port = port
        self.pid_file = pid_file
        self.proc = None
        self.lf = None

    @property
    def cmd(self):
        cmd = list(SERVER_CMD)
        if self.port != PORT:
            cmd[cmd.index("--port") + 1] = str(self.port)
        return cmd + self.extra_args

    def start(self, log_path, health_timeout=180, expect_failure=False):
        """启动并等 /health。expect_failure=True 时（OOM 演练）允许早退并返回 None。"""
        if not port_free(self.port):
            raise RuntimeError(f"port {self.port} already in use")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.pid_file), exist_ok=True)
        self.lf = open(log_path, "a")
        self.lf.write(f"\n===== start {time.strftime('%F %T')} cmd={' '.join(self.cmd)} =====\n")
        self.lf.flush()
        self.proc = subprocess.Popen(self.cmd, env=ENV, stdout=self.lf,
                                     stderr=subprocess.STDOUT)
        with open(self.pid_file, "w") as f:
            f.write(str(self.proc.pid))
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < health_timeout:
            if health_code(self.port) == 200:
                return round(time.perf_counter() - t0, 2)
            if self.proc.poll() is not None:
                rc = self.proc.returncode
                self._clear_pid()
                if expect_failure:
                    return None
                raise RuntimeError(f"server exited early rc={rc}; 见 {log_path}")
            time.sleep(2)
        if expect_failure:
            return None
        raise RuntimeError(f"server health timeout {health_timeout}s")

    def pid(self):
        return self.proc.pid if self.proc else None

    def child_pids(self):
        """列出 server 的子进程（vLLM V1 的 EngineCore 跑在子进程里，显存由它持有）。

        必须在 kill 父进程**之前**调用并落盘：SIGKILL 父进程不会连带杀死 EngineCore，
        它会被 init 收养并继续占着显存（实测 9.8GB 不释放，新实例随后起不来）。
        记录下来才能在演练后按「只操作已记录 PID」的规则安全清理。
        """
        import psutil
        try:
            p = psutil.Process(self.pid())
            return [{"pid": c.pid, "name": c.name(),
                     "cmdline": " ".join(c.cmdline())[:200],
                     "cwd": (c.cwd() if c.is_running() else None)}
                    for c in p.children(recursive=True)]
        except Exception as e:
            return [{"error": f"{type(e).__name__}: {e}"}]

    @staticmethod
    def kill_recorded_pids(records, expect_name="VLLM::EngineCore", sig=9):
        """清理先前**已记录**的子进程 PID；逐个校验身份后才发信号。"""
        import psutil
        out = []
        for r in records:
            pid = r.get("pid")
            if pid is None:
                continue
            try:
                p = psutil.Process(pid)
                name = p.name()
                if expect_name not in name and expect_name not in " ".join(p.cmdline()):
                    out.append({"pid": pid, "killed": False, "reason": f"身份不符: {name}"})
                    continue
                p.send_signal(sig)
                p.wait(timeout=30)
                out.append({"pid": pid, "killed": True, "name": name})
            except psutil.NoSuchProcess:
                out.append({"pid": pid, "killed": False, "reason": "已不存在"})
            except Exception as e:
                out.append({"pid": pid, "killed": False, "reason": f"{type(e).__name__}: {e}"})
        return out

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def cmdline_of_pid(self, pid):
        """读 /proc 校验 PID 身份 —— kill 前的安全检查。"""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                return f.read().decode(errors="replace").replace("\x00", " ").strip()
        except Exception:
            return ""

    def kill_self_pid(self, sig=9):
        """E6 worker-exit 演练专用：只 kill 自己启动且 cmdline 校验通过的 PID。

        返回 (killed, pid, cmdline)。任何校验不过一律不发信号。
        """
        pid = self.pid()
        if pid is None or not self.alive():
            return False, pid, ""
        cmdline = self.cmdline_of_pid(pid)
        if "vllm" not in cmdline or "serve" not in cmdline:
            return False, pid, cmdline
        os.kill(pid, sig)
        try:
            self.proc.wait(timeout=30)
        except Exception:
            pass
        self._clear_pid()
        return True, pid, cmdline

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except Exception:
                self.proc.kill()
                self.proc.wait(timeout=15)
        self.proc = None
        self._clear_pid()
        for _ in range(40):
            if port_free(self.port):
                break
            time.sleep(0.5)
        if self.lf:
            self.lf.close()
            self.lf = None

    def _clear_pid(self):
        try:
            os.remove(self.pid_file)
        except OSError:
            pass


# --------------------------------------------------------------------------- bench
def run_bench(shape, rate, num_prompts, run_id, seed=0, timeout_s=1800,
              result_dir=RAW, extra=None):
    """调用官方 vllm bench serve（开环 Poisson），参数与 M12 一致。"""
    inl, outl = SHAPES[shape]
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(os.path.join(LOGD, "per_run"), exist_ok=True)
    result_file = f"{run_id}.json"
    cmd = ["vllm", "bench", "serve",
           "--backend", "openai", "--model", MODEL,
           "--host", HOST, "--port", str(PORT), "--endpoint", "/v1/completions",
           "--dataset-name", "random",
           "--random-input-len", str(inl), "--random-output-len", str(outl),
           "--random-range-ratio", "0",
           "--num-prompts", str(num_prompts),
           "--request-rate", str(rate), "--burstiness", "1.0",
           "--ignore-eos", "--temperature", "0", "--top-p", "1",
           "--seed", str(seed), "--num-warmups", "0",
           "--percentile-metrics", "ttft,tpot,itl,e2el",
           "--metric-percentiles", "50,95,99",
           "--save-result", "--result-dir", result_dir, "--result-filename", result_file]
    cmd += list(extra or [])

    log_path = os.path.join(LOGD, "per_run", f"{run_id}.log")
    t0 = time.time()
    with open(log_path, "w") as lg:
        try:
            rc = subprocess.call(cmd, env=ENV, stdout=lg, stderr=subprocess.STDOUT,
                                 timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            rc = -1
            timed_out = True
    t1 = time.time()

    result_path = os.path.join(result_dir, result_file)
    result = None
    if os.path.exists(result_path):
        with open(result_path) as f:
            result = json.load(f)
    return {"run_id": run_id, "cmd": cmd, "exit_code": rc, "timeout": timed_out,
            "wall_start_ts": t0, "wall_end_ts": t1, "duration_s": round(t1 - t0, 2),
            "result_path": result_path, "result": result, "log_path": log_path}


def gpu_snapshot():
    """nvidia-smi 快照（整卡口径；WSL2 下拿不到进程级显存）。"""
    try:
        q = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        used, total, util = [x.strip() for x in q.split(",")]
        return {"mem_used_mib": int(used), "mem_total_mib": int(total), "gpu_util_pct": int(util)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


LOCK_FILE = os.path.join(RAW, ".m21_experiment.lock")


def acquire_lock(owner):
    """全模块互斥锁：同一时刻只允许一个 M21 实验进程跑。

    教训（2026-08-11）：B3 阶梯曾因我并发启动了第二个实验进程而数据交错污染
    —— 两个进程轮流抢占 8014 端口，写出「同一测量值被标成两个不同 util」的记录。
    锁文件里记 PID，进程死掉后自动接管（防止残留锁把自己锁死）。
    """
    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                info = json.load(f)
            pid = int(info["pid"])
            alive = os.path.exists(f"/proc/{pid}")
        except Exception:
            info, alive = {}, False
        if alive:
            raise SystemExit(
                f"已有 M21 实验在跑（pid={info.get('pid')} owner={info.get('owner')} "
                f"since={info.get('since')}）。并发跑会污染数据，拒绝启动。")
        print(f"[m21] 发现残留锁（pid={info.get('pid')} 已退出），接管", flush=True)
    with open(LOCK_FILE, "w") as f:
        json.dump({"pid": os.getpid(), "owner": owner,
                   "since": time.strftime("%F %T%z")}, f)
    import atexit
    atexit.register(release_lock)


def release_lock():
    try:
        with open(LOCK_FILE) as f:
            if json.load(f).get("pid") == os.getpid():
                os.remove(LOCK_FILE)
    except Exception:
        pass


def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path
