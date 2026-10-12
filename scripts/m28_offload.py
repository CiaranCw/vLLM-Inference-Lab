#!/usr/bin/env python3
"""M28-A · 原生 OffloadingConnector（GPU→CPU）三路径对照 + 前缀长度阶梯。

每个 arm = 一个独立 `vllm serve` 子进程（一个进程一个引擎）；请求严格串行，
每个被测请求前后各抓一次 /metrics，用计数器增量独立证明路径：
  cold      : local_hits == 0 且 ext_hits == 0 且 load_bytes == 0 且 cached_tokens == 0
  gpu_hot   : local_hits == H 且 ext_hits == 0 且 load_bytes == 0 且 cached_tokens == H
  cpu_hit   : local_hits == 0 且 ext_hits == H 且 load_bytes > 0  且 cached_tokens == H
  （H = floor((N-1)/16)*16；另需 Δrequest_success == 1、Δprompt_tokens == N 证明这次抓取覆盖到了本请求）

用法：
  python scripts/m28_offload.py --dry-run                      # 打印矩阵/命令/资源预估，不碰 GPU
  python scripts/m28_offload.py --mock --out-dir /tmp/x        # CPU 演练（mock server，产物只进临时目录）
  python scripts/m28_offload.py --smoke --i-have-gpu-authorization
  python scripts/m28_offload.py --arm offload --i-have-gpu-authorization
"""

import argparse
import json
import os
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m28_common as C  # noqa: E402

A = C.FROZEN["A"]


def classify(path, n_tokens, delta, cached, connector):
    """返回 (path_ok, reasons)。reasons 为空才算路径被证实。"""
    H = C.expected_hit_tokens(n_tokens)
    lh, eh, lb = delta.get("local_hits"), delta.get("ext_hits"), delta.get("load_bytes")
    zero = lambda v: v in (None, 0, 0.0)  # noqa: E731
    r = []
    if delta.get("request_success") != 1:
        r.append(f"Δrequest_success={delta.get('request_success')}≠1（抓取未覆盖本请求）")
    if delta.get("prompt_tokens") != n_tokens:
        r.append(f"Δprompt_tokens={delta.get('prompt_tokens')}≠N={n_tokens}")
    if connector and (eh is None or delta.get("ext_queries") is None):
        r.append("connector arm 缺少 external_prefix_cache_* 指标")
    if path in ("cold", "recompute_evicted", "ref_a", "ref_b"):
        if not zero(lh):
            r.append(f"local_hits={lh}≠0")
        if not zero(eh):
            r.append(f"ext_hits={eh}≠0")
        if not zero(lb):
            r.append(f"load_bytes={lb}≠0")
        if cached != 0:
            r.append(f"cached_tokens={cached}≠0")
    elif path == "gpu_hot":
        if lh != H:
            r.append(f"local_hits={lh}≠H={H}")
        if not zero(eh):
            r.append(f"ext_hits={eh}≠0")
        if not zero(lb):
            r.append(f"load_bytes={lb}≠0")
        if cached != H:
            r.append(f"cached_tokens={cached}≠H={H}")
    elif path == "cpu_hit":
        if not zero(lh):
            r.append(f"local_hits={lh}≠0（GPU 未被挤出）")
        if eh != H:
            r.append(f"ext_hits={eh}≠H={H}")
        if not (lb or 0) > 0:
            r.append(f"load_bytes={lb} 不>0")
        if cached != H:
            r.append(f"cached_tokens={cached}≠H={H}")
    elif path == "cpu_evicted":
        # 只要求路径记账自洽；淘汰程度（ext_hits 相对 H）是观测量，不预设
        if not zero(lh):
            r.append(f"local_hits={lh}≠0（GPU 未被挤出）")
        if cached != (eh or 0):
            r.append(f"cached_tokens={cached}≠ext_hits={eh}")
    else:
        r.append(f"未知路径 {path}")
    return (not r), r


class Runner:
    def __init__(self, arm, port, out_dir, log_dir, mock, writer, life):
        self.arm, self.port, self.mock = arm, port, mock
        self.cfg = A["arms"][arm]
        self.writer, self.life = writer, life
        self.out_dir, self.log_dir = out_dir, log_dir

    def base(self, **kw):
        d = {"module": "m28", "exp": "A", "arm": self.arm, "mock": self.mock,
             "plan_version": C.FROZEN["plan_version"], "ts": C.now()}
        d.update(kw)
        return d

    def snap(self):
        return C.metric_snapshot(self.port)

    def quiesce(self):
        """等 GPU→CPU store 计数器连续两次不变（connector arm）；非 connector arm 等同样时长。"""
        t0 = time.perf_counter()
        if not self.cfg["connector"]:
            time.sleep(A["store_quiesce_s"])
            return {"waited_s": round(time.perf_counter() - t0, 3), "stable": True}
        prev = self.snap().get("store_bytes")
        while time.perf_counter() - t0 < A["store_quiesce_timeout_s"]:
            time.sleep(A["store_quiesce_s"])
            cur = self.snap().get("store_bytes")
            if cur == prev:
                return {"waited_s": round(time.perf_counter() - t0, 3), "stable": True,
                        "store_bytes": cur}
            prev = cur
        return {"waited_s": round(time.perf_counter() - t0, 3), "stable": False, "store_bytes": prev}

    def measured(self, phase, p_len, idx, path, prompt, max_tokens=None):
        q = self.quiesce()
        before = self.snap()
        rec = C.completion(self.port, prompt, max_tokens or A["max_tokens"], stream=True,
                           timeout=A["request_timeout_s"])
        time.sleep(A["metrics_settle_s"])
        after = self.snap()
        delta = C.metric_delta(before, after)
        retries = 0
        while delta.get("request_success") != 1 and retries < 5:
            time.sleep(0.2)
            retries += 1
            after = self.snap()
            delta = C.metric_delta(before, after)
        cached = C.cached_tokens(rec)
        ok_req = C.request_ok(rec)
        if path == "filler":
            path_ok, reasons = (delta.get("request_success") == 1), []
        else:
            path_ok, reasons = classify(path, len(prompt), delta, cached, self.cfg["connector"])
        out = self.base(phase=phase, p_len=p_len, idx=idx, path=path, n_tokens=len(prompt),
                        expected_hit_tokens=C.expected_hit_tokens(len(prompt)),
                        prompt_sha256=rec["prompt_sha256"], token_ids=rec["token_ids"],
                        token_logprobs=rec["token_logprobs"], finish_reason=rec["finish_reason"],
                        http_status=rec["http_status"], exception=rec["exception"],
                        error_event=rec["error_event"], online_ttft_ms=rec.get("online_ttft_ms"),
                        e2e_ms=rec.get("e2e_ms"), cached_tokens=cached, usage=rec["usage"],
                        metrics_before=before, metrics_after=after, delta=delta,
                        settle_retries=retries, quiesce=q, request_ok=ok_req, path_ok=path_ok,
                        path_reasons=reasons, success=bool(ok_req and path_ok),
                        metric_scope="online_http_stream_serial")
        self.writer.write(out)
        return out

    def trial(self, phase, p_len, idx):
        prompt = C.trial_prompt(p_len, idx)
        res = []
        if self.arm == "ref":
            for path in self.cfg["paths"]:
                res.append(self.measured(phase, p_len, idx, path, prompt))
            return res
        p_cold, p_hot, p_after = self.cfg["paths"]
        res.append(self.measured(phase, p_len, idx, p_cold, prompt))
        res.append(self.measured(phase, p_len, idx, p_hot, prompt))
        for j in range(A["n_fillers"]):
            self.measured(phase, p_len, idx, "filler", C.filler_prompt(p_len, idx, j), max_tokens=1)
        res.append(self.measured(phase, p_len, idx, p_after, prompt))
        return res

    def server_warmup(self, lens):
        for p_len in lens:
            for k in range(A["server_warmup_requests"]):
                self.measured("server_warmup", p_len, -1 - k, "filler",
                              C.trial_prompt(p_len, -1 - k, salt="warm"))
        self.measured("server_warmup", 0, -1, "filler", C.filler_prompt(0, -1, 0), max_tokens=1)


def run_arm(arm, args, out_dir, log_dir, smoke=False):
    lens = A["evict_prefix_lens"] if arm == "evict" else A["prefix_lens"]
    raw_name = "m28_a_smoke.jsonl" if smoke else f"m28_a_{arm}.jsonl"
    writer = C.JsonlWriter(os.path.join(out_dir, raw_name), backup_existing=args.backup_existing)
    life = C.JsonlWriter(os.path.join(out_dir, "m28_lifecycle.jsonl"), append=True)
    if args.mock:
        cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "m28_mock_server.py"),
               "--port", str(args.port), "--arm", arm]
    else:
        cmd = C.vllm_serve_cmd(C.a_server_args(arm), port=args.port)
    env = C.server_env()
    log_path = os.path.join(log_dir, f"m28_a_{'smoke' if smoke else arm}_server.log")
    srv = C.Server(cmd, env, log_path, args.port, f"A/{arm}",
                   active_file=os.path.join(out_dir, "m28_active_server.json"))
    lc = {"module": "m28", "exp": "A", "arm": arm, "mock": args.mock, "smoke": smoke,
          "cmd": cmd, "log_path": os.path.relpath(log_path, C.ROOT),
          "env_fingerprint": C.env_fingerprint(with_vllm_import=not args.mock),
          "gpu_before": C.nvidia_smi(), "dev_shm_before": C.dev_shm_listing(),
          "host_mem_before": C.host_mem(), "t_start": C.now()}
    runner = Runner(arm, args.port, out_dir, log_dir, args.mock, writer, life)
    err = None
    try:
        lc["ready_s"] = srv.start(ready_timeout=C.FROZEN["B"]["server_ready_timeout_s"])
        lc["tree_ready"] = srv.refresh_tree()
        lc["api_pid"] = srv.pid
        lc["gpu_ready"] = C.nvidia_smi()
        lc["dev_shm_ready"] = C.dev_shm_listing()
        lc["host_mem_ready"] = C.host_mem()
        runner.server_warmup(lens)
        if smoke:
            runner.trial("smoke", lens[0], -100)
        else:
            n = A["warmup_trials"] + A["measure_trials"]
            for idx in range(n):
                phase = "warmup" if idx < A["warmup_trials"] else "measure"
                for p_len in C.trial_order(idx, lens):
                    runner.trial(phase, p_len, idx)
                print(f"[m28-A/{arm}] trial {idx + 1}/{n} done", flush=True)
        lc["gpu_peak_end"] = C.nvidia_smi()
    except Exception as e:
        err = e
        lc["error"] = f"{type(e).__name__}: {e}"
        lc["traceback"] = traceback.format_exc()
    finally:
        lc["log_markers"] = scan_log(log_path)
        lc["t_stop_begin"] = C.now()
        lc["cleanup_actions"] = srv.cleanup()
        lc["api_returncode"] = srv.returncode()
        lc["alive_known_after_cleanup"] = srv.alive_known()
        lc["port_free_after"] = C.port_free(args.port)
        base_mib = C.gpu_used_mib(lc["gpu_before"])
        if args.mock:
            lc["gpu_reclaim"] = {"skipped": "mock"}
        else:
            ok, secs, last = C.wait_gpu_reclaim(base_mib)
            lc["gpu_reclaim"] = {"ok": ok, "waited_s": secs, "mem_used_mib": last,
                                 "baseline_mib": base_mib}
        lc["dev_shm_after"] = C.dev_shm_listing()
        lc["t_end"] = C.now()
        life.write(dict(lc, phase="lifecycle"))
    if err is not None:
        raise SystemExit(f"[m28-A/{arm}] 失败：{lc['error']}（详见 {log_path} 与 lifecycle 记录）")
    return lc


LOG_MARKERS = {
    "offloading_connector": "OffloadingConnector",
    "mmap_created": "Created mmap file",
    "cuda_host_register_failed": "cudaHostRegister failed",
    "traceback": "Traceback (most recent call last)",
    "engine_dead": "EngineDeadError",
}


def scan_log(path):
    out = {k: 0 for k in LOG_MARKERS}
    lines = {}
    try:
        with open(path, errors="replace") as f:
            for line in f:
                for k, pat in LOG_MARKERS.items():
                    if pat in line:
                        out[k] += 1
                        lines.setdefault(k, line.strip()[:300])
    except OSError as e:
        out["error"] = str(e)
    out["first_lines"] = lines
    return out


def dry_run(arms):
    total_req = 0
    print("M28-A 冻结矩阵（plan_version=%s）" % C.FROZEN["plan_version"])
    for arm in arms:
        lens = A["evict_prefix_lens"] if arm == "evict" else A["prefix_lens"]
        cfg = A["arms"][arm]
        per_trial = len(cfg["paths"]) + (0 if arm == "ref" else A["n_fillers"])
        n = (A["warmup_trials"] + A["measure_trials"]) * len(lens) * per_trial + \
            len(lens) * A["server_warmup_requests"] + 1
        total_req += n
        print(f"\n[{arm}] paths={cfg['paths']} lens={lens} requests≈{n}")
        print("  cmd:", " ".join(C.vllm_serve_cmd(C.a_server_args(arm))))
    kvb = C.FROZEN["kv_bytes_per_token_theory"]
    gpu_kv = A["num_gpu_blocks_override"] * C.FROZEN["block_size"] * kvb
    print(f"\n总请求数≈{total_req}")
    print(f"GPU KV 池 = {A['num_gpu_blocks_override']} blocks × 16 × {kvb} B = {gpu_kv / 2**20:.0f} MiB；"
          f"filler 总长 {A['n_fillers'] * A['filler_len']} tokens > GPU 池 "
          f"{A['num_gpu_blocks_override'] * 16} tokens（保证挤出）")
    print(f"CPU 层 = {A['cpu_bytes_to_use'] / C.GIB:.2f} GiB（evict arm {A['evict_cpu_bytes_to_use'] / 2**20:.0f} MiB）"
          "，位于 /dev/shm（mmap）")
    for p in A["prefix_lens"]:
        n = p + A["suffix_len"]
        print(f"  P={p}: N={n} H={C.expected_hit_tokens(n)} 理论 load_bytes={C.expected_hit_tokens(n) * kvb}")
    print("\n服务端环境：", C.FROZEN["server_env_set"], "强制 unset：", C.FROZEN["server_env_must_unset"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", choices=list(A["arms"]), help="可重复；默认 offload/base/ref")
    ap.add_argument("--smoke", action="store_true", help="offload arm 单 trial 冒烟（phase=smoke）")
    ap.add_argument("--dry-run", action="store_true", help="只打印矩阵/命令/资源预估")
    ap.add_argument("--mock", action="store_true", help="对 mock server 跑（CPU 演练；必须配 --out-dir 临时目录）")
    ap.add_argument("--mock-fast", action="store_true", help="仅 mock：把 quiesce/settle 等待缩到 0.05s")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--port", type=int, default=C.FROZEN["port"])
    ap.add_argument("--backup-existing", action="store_true", help="已有 raw 先搬到 raw_old_<ts>/（不删除）")
    ap.add_argument("--i-have-gpu-authorization", action="store_true",
                    help="所有者已明确授权本轮 GPU 实验（否则拒绝启动）")
    a = ap.parse_args()
    arms = a.arm or ["offload", "base", "ref"]
    if a.smoke:
        arms = ["offload"]
    if a.dry_run:
        dry_run(arms)
        return 0
    if a.mock_fast and not a.mock:
        raise SystemExit("--mock-fast 只能与 --mock 一起用")
    if a.mock_fast:
        A["store_quiesce_s"] = A["metrics_settle_s"] = 0.05
    if a.mock:
        out_dir = a.out_dir or tempfile.mkdtemp(prefix="m28_mock_")
        if os.path.realpath(out_dir).startswith(os.path.realpath(C.OUT)):
            raise SystemExit("mock 产物不得写入 outputs/m28")
        log_dir = os.path.join(out_dir, "logs")
    else:
        out_dir, log_dir = a.out_dir or C.RAW, C.LOGD
        ok, reasons, snap = C.gpu_gate(a.i_have_gpu_authorization, a.port)
        print("[m28-A] nvidia-smi:", json.dumps(snap, ensure_ascii=False))
        if not ok:
            print("[m28-A] GPU 门禁未通过，不启动：", *reasons, sep="\n  - ")
            return 2
    C.acquire_lock(f"m28_offload {' '.join(arms)}", raw_dir=out_dir)
    for arm in arms:
        print(f"[m28-A] arm={arm} start", flush=True)
        lc = run_arm(arm, a, out_dir, log_dir, smoke=a.smoke)
        print(f"[m28-A] arm={arm} done ready_s={lc.get('ready_s')} markers={lc['log_markers']}", flush=True)
    print("[m28-A] 产物目录:", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
