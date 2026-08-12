#!/usr/bin/env python3
"""M21 · 环境指纹落盘。

动机（诚实记录）：M21 复核 M12 容量时发现同配置下单步解码变慢，但 **M12 当年没有落盘任何
环境指纹**（驱动/torch/triton 版本、CPU、内核），导致今天**无法归因**这次回归。
→ 从本模块起，每次性能实验都落一份指纹，未来才能做跨会话对比。

用法：python scripts/m21_env_fingerprint.py [--out outputs/m21/env_fingerprint.json]
"""

import argparse
import json
import os
import platform
import subprocess
import time

import m21_common as C


def sh(cmd, timeout=30):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.STDOUT, timeout=timeout).strip()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(C.OUT, "env_fingerprint.json"))
    args = ap.parse_args()

    fp = {
        "captured_at": time.strftime("%F %T%z"),
        "host": {
            "uname": platform.uname()._asdict(),
            "python": platform.python_version(),
            "cpu_model": sh("rg -m1 'model name' /proc/cpuinfo | cut -d: -f2"),
            "cpu_logical_cores": os.cpu_count(),
            "mem_total_kb": sh("rg -m1 MemTotal /proc/meminfo"),
            "wsl": "microsoft" in platform.uname().release.lower(),
        },
        "gpu": {
            "nvidia_smi_header": sh("nvidia-smi | head -4"),
            "driver_version": sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader"),
            "name": sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
            "memory_total_mib": sh("nvidia-smi --query-gpu=memory.total --format=csv,noheader"),
            "clocks_max_sm_mhz": sh("nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader"),
            "power_limit_w": sh("nvidia-smi --query-gpu=power.limit --format=csv,noheader"),
            "snapshot_at_capture": C.gpu_snapshot(),
        },
        "python_packages": sh("pip freeze | rg '^(vllm|torch|torchvision|torchaudio|triton|"
                              "transformers|numpy|flashinfer|xformers|prometheus-client|psutil)'"),
        "vllm_src_commit": sh("git -C ../vllm-src log -1 --format='%h %cd %s'"),
        "monitoring_stack": {
            "containers": sh(f"{os.path.expanduser('~/bin/docker')} ps --filter name=m21- "
                             "--format '{{.Names}} {{.Image}} {{.Status}}'"),
            "dcgm_exporter_digest": sh(
                f"{os.path.expanduser('~/bin/docker')} image inspect "
                "nvcr.io/nvidia/k8s/dcgm-exporter:latest --format '{{index .RepoDigests 0}}'"),
            "prometheus_digest": sh(
                f"{os.path.expanduser('~/bin/docker')} image inspect prom/prometheus:latest "
                "--format '{{index .RepoDigests 0}}'"),
            "grafana_digest": sh(
                f"{os.path.expanduser('~/bin/docker')} image inspect grafana/grafana:latest "
                "--format '{{index .RepoDigests 0}}'"),
        },
        "note": "M12(2026-07-24) 未留指纹，故 M12→M21 的解码变慢无法归因；本文件是为后续模块留的基线。",
    }
    C.save_json(args.out, fp)
    print(json.dumps({"driver": fp["gpu"]["driver_version"],
                      "gpu": fp["gpu"]["name"],
                      "cpu_cores": fp["host"]["cpu_logical_cores"],
                      "kernel": fp["host"]["uname"]["release"]}, ensure_ascii=False, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
