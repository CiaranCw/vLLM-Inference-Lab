#!/usr/bin/env python3
"""M21 host exporter —— 补 dcgm-exporter 拿不到的两类指标。

职责（见 notes/m21/monitoring_and_capacity.md §3）：
  1. 进程级 CPU% / RSS / 线程数 —— 回答「瓶颈是不是 CPU」（Q3 的 CPU 分支）。
     M7 已证 eager decode 是 CPU 提交受限，本指标是它在面板上的可观测投影。
  2. NVML 显存 / util 读数 —— 作为 dcgm-exporter 的**正交第二证据源**（交叉校验不变量 I8）。

不做的事：不采集 SM 占用率（本环境 DCGM_FI_PROF_* 全 N/A，见 Gate 0 结论）。

用法：
    python scripts/m21_host_exporter.py --port 9836 --pid-file outputs/m21/raw/server.pid
    python scripts/m21_host_exporter.py --port 9836 --discover "vllm serve"

进程发现是**只读**的（读 /proc cmdline），不涉及任何 kill/信号操作。
"""

import argparse
import os
import time

import psutil
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import GaugeMetricFamily

try:
    import pynvml
except ImportError:  # pragma: no cover - 环境已验证存在
    pynvml = None


class HostCollector:
    """每次被 scrape 时现采，保证样本时间戳与 Prometheus 抓取时刻对齐。"""

    def __init__(self, pid_file=None, discover=None, gpu_index=0):
        self.pid_file = pid_file
        self.discover = discover
        self.gpu_index = gpu_index
        self._proc = None
        # psutil 的 cpu_percent(interval=None) 返回「距上次调用」的均值，
        # 首次调用恒为 0.0 —— 这里预热一次，避免第一个样本被误读成 CPU 空闲。
        psutil.cpu_percent(interval=None)
        # 子进程 Process 对象必须**跨 scrape 复用**：每次 children() 重新构造对象的话，
        # 每个对象的 cpu_percent() 都是首次调用、恒返回 0，会把 EngineCore 子进程的
        # CPU 全部漏掉（vLLM V1 的引擎跑在子进程里，正是我们最想看的那个）。
        self._children = {}
        self._nvml_handle = None
        if pynvml is not None:
            try:
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            except Exception:
                self._nvml_handle = None

    # ---------- 目标进程 ----------
    def _read_pid_file(self):
        try:
            with open(self.pid_file) as f:
                return int(f.read().strip())
        except Exception:
            return None

    def _discover_pid(self):
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                cmdline = " ".join(p.info["cmdline"] or [])
            except Exception:
                continue
            if self.discover in cmdline and "m21_host_exporter" not in cmdline:
                return p.info["pid"]
        return None

    def _target(self):
        """返回存活的目标 psutil.Process，或 None。"""
        if self._proc is not None and self._proc.is_running():
            return self._proc
        pid = self._read_pid_file() if self.pid_file else None
        if pid is None and self.discover:
            pid = self._discover_pid()
        if pid is None:
            self._proc = None
            return None
        try:
            proc = psutil.Process(pid)
            proc.cpu_percent(interval=None)  # 预热该进程的 CPU 采样基准
            self._proc = proc
            return proc
        except Exception:
            self._proc = None
            return None

    def _live_children(self, proc):
        """返回复用的子进程对象列表；新出现的子进程先预热一次 CPU 采样基准。"""
        try:
            current = {c.pid: c for c in proc.children(recursive=True)}
        except psutil.Error:
            return []
        for pid in list(self._children):
            if pid not in current:
                del self._children[pid]
        for pid, c in current.items():
            if pid not in self._children:
                try:
                    c.cpu_percent(interval=None)
                except psutil.Error:
                    continue
                self._children[pid] = c
        return list(self._children.values())

    # ---------- collector 接口 ----------
    def collect(self):
        t0 = time.perf_counter()

        up = GaugeMetricFamily("m21_target_up", "目标服务进程是否存活(1/0)")
        cpu = GaugeMetricFamily("m21_target_cpu_percent", "目标进程(含子进程)CPU 占用百分比, 100=一个核跑满")
        rss = GaugeMetricFamily("m21_target_rss_bytes", "目标进程(含子进程)RSS 常驻内存字节")
        thr = GaugeMetricFamily("m21_target_num_threads", "目标进程线程数")

        proc = self._target()
        if proc is None:
            up.add_metric([], 0.0)
        else:
            try:
                total_cpu = proc.cpu_percent(interval=None)
                total_rss = proc.memory_info().rss
                total_thr = proc.num_threads()
                for child in self._live_children(proc):
                    try:
                        total_cpu += child.cpu_percent(interval=None)
                        total_rss += child.memory_info().rss
                        total_thr += child.num_threads()
                    except psutil.Error:
                        continue
                up.add_metric([], 1.0)
                cpu.add_metric([], total_cpu)
                rss.add_metric([], float(total_rss))
                thr.add_metric([], float(total_thr))
            except psutil.Error:
                up.add_metric([], 0.0)
        yield from (up, cpu, rss, thr)

        host_cpu = GaugeMetricFamily("m21_host_cpu_percent", "整机 CPU 占用百分比")
        host_cpu.add_metric([], psutil.cpu_percent(interval=None))
        yield host_cpu

        load1 = GaugeMetricFamily("m21_host_load1", "1 分钟负载")
        load1.add_metric([], os.getloadavg()[0])
        yield load1

        ncpu = GaugeMetricFamily("m21_host_cpu_count", "逻辑核数")
        ncpu.add_metric([], float(psutil.cpu_count() or 0))
        yield ncpu

        yield from self._nvml_metrics()

        dur = GaugeMetricFamily("m21_exporter_scrape_duration_seconds", "本 exporter 单次采集耗时")
        dur.add_metric([], time.perf_counter() - t0)
        yield dur

    def _nvml_metrics(self):
        """NVML 读数 —— 与 DCGM 独立的第二证据源，用于交叉校验(I8)。"""
        ok = GaugeMetricFamily("m21_nvml_ok", "NVML 是否可读(1/0)")
        if self._nvml_handle is None:
            ok.add_metric([], 0.0)
            yield ok
            return
        try:
            h = self._nvml_handle
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            temp = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            sm_clock = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
        except Exception:
            ok.add_metric([], 0.0)
            yield ok
            return

        ok.add_metric([], 1.0)
        yield ok
        for name, doc, value in [
            ("m21_nvml_gpu_util_percent", "NVML GPU 时间占比(非 SM 占用率)", float(util.gpu)),
            ("m21_nvml_mem_util_percent", "NVML 显存控制器活动时间占比", float(util.memory)),
            ("m21_nvml_fb_used_bytes", "NVML 已用显存字节", float(mem.used)),
            ("m21_nvml_fb_free_bytes", "NVML 空闲显存字节", float(mem.free)),
            ("m21_nvml_fb_total_bytes", "NVML 总显存字节", float(mem.total)),
            ("m21_nvml_power_watts", "NVML 功耗(W)", power),
            ("m21_nvml_temp_celsius", "NVML 温度(C)", float(temp)),
            ("m21_nvml_sm_clock_mhz", "NVML SM 时钟(MHz)", float(sm_clock)),
        ]:
            g = GaugeMetricFamily(name, doc)
            g.add_metric([], value)
            yield g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9836)
    ap.add_argument("--pid-file", default=None, help="目标服务 PID 文件(优先)")
    ap.add_argument("--discover", default=None, help="按 cmdline 子串只读发现目标进程")
    ap.add_argument("--gpu-index", type=int, default=0)
    args = ap.parse_args()

    if not args.pid_file and not args.discover:
        ap.error("必须给 --pid-file 或 --discover 之一")

    REGISTRY.register(HostCollector(args.pid_file, args.discover, args.gpu_index))
    start_http_server(args.port)
    print(f"[m21_host_exporter] listening on :{args.port} "
          f"(pid_file={args.pid_file}, discover={args.discover})", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
