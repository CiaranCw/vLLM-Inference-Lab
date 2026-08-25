#!/usr/bin/env python3
"""M21 · 环境指纹落盘。

动机（诚实记录）：M21 复核 M12 容量时发现同配置下单步解码变慢，但 M12 当年没有落盘任何
环境指纹（驱动/torch/triton 版本、CPU、内核），导致今天无法归因这次回归。
→ 从本模块起，每次性能实验都落一份指纹，未来才能做跨会话对比。

用法：python scripts/m21_env_fingerprint.py [--out outputs/m21/env_fingerprint.json]
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import m21_common as C


UNAVAILABLE = "unavailable"


def sh(cmd, timeout=30):
    """跑一条 shell 命令；失败一律记 unavailable + 原始报错，绝不猜。"""
    try:
        out = subprocess.check_output(cmd, shell=True, text=True,
                                      stderr=subprocess.STDOUT, timeout=timeout).strip()
        return out if out else UNAVAILABLE
    except Exception as e:
        return f"{UNAVAILABLE} <{type(e).__name__}: {e}>"


def python_stack_versions():
    """在子进程里取运行时版本，避免在指纹进程内 import 重型库。

    只 import、不建引擎、不碰 GPU。
    """
    probe = (
        "import json,sys\n"
        "d={'python_executable': sys.executable, 'python_version': sys.version.split()[0]}\n"
        "for name, attrs in (('vllm', ('__version__',)), ('torch', ('__version__',)),\n"
        "                    ('triton', ('__version__',)), ('transformers', ('__version__',))):\n"
        "    try:\n"
        "        m=__import__(name)\n"
        "        d[name]=getattr(m, attrs[0], 'unknown')\n"
        "    except Exception as e:\n"
        "        d[name]=f'unavailable <{type(e).__name__}: {e}>'\n"
        "try:\n"
        "    import torch; d['torch_version_cuda']=torch.version.cuda\n"
        "    d['torch_cuda_is_available_flag']=torch.cuda.is_available()\n"
        "except Exception as e:\n"
        "    d['torch_version_cuda']=f'unavailable <{type(e).__name__}: {e}>'\n"
        "print(json.dumps(d))\n"
    )
    try:
        out = subprocess.check_output([sys.executable, "-c", probe], text=True, timeout=300)
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        return {"error": f"{UNAVAILABLE} <{type(e).__name__}: {e}>"}


# AGENTS.md §5 规定本机跑 vLLM 必须设置的变量；指纹要能看出「采集时的 shell 是否等价于实验 shell」
REQUIRED_ENV = {
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
}


def relevant_env_vars():
    """记录会影响 vLLM 行为的环境变量；未设置就写 unset（不臆造默认值）。"""
    explicit = ["CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "HF_HOME",
                "HUGGING_FACE_HUB_CACHE", "OMP_NUM_THREADS", "TOKENIZERS_PARALLELISM"]
    out = {k: os.environ.get(k, "<unset>") for k in explicit}
    out["VLLM_*"] = ({k: v for k, v in sorted(os.environ.items()) if k.startswith("VLLM_")}
                     or "<none set>")
    observed = {k: os.environ.get(k, "<unset>") for k in REQUIRED_ENV}
    out["required_by_agents_md"] = {
        "expected": REQUIRED_ENV,
        "observed_at_capture": observed,
        "captured_in_experiment_equivalent_shell": all(
            observed[k] == v for k, v in REQUIRED_ENV.items()),
        "note": ("若为 false，说明指纹是在**未导出必需变量**的 shell 里采的 —— "
                 "指纹本身仍有效，但跨会话比对时要注意实验 shell 与此不同。"),
    }
    return out


def cpu_power_state():
    """CPU governor / 能耗模式；WSL2 下多半读不到 —— 读不到就如实记 unavailable。"""
    gov = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null")
    driver = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver 2>/dev/null")
    avail = sh("cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_available_governors 2>/dev/null")
    return {"scaling_governor": gov, "scaling_driver": driver,
            "available_governors": avail,
            "note": "WSL2 通常不暴露 cpufreq sysfs；显示 unavailable 即为读不到，不做推测"}


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
        "runtime_versions": python_stack_versions(),
        "env_vars": relevant_env_vars(),
        "cpu_power_state": cpu_power_state(),
        "python_packages": sh("pip freeze | rg '^(vllm|torch|torchvision|torchaudio|triton|"
                              "transformers|numpy|flashinfer|xformers|prometheus-client|psutil)'"),
        "vllm_src": {
            "head_sha_full": sh("git -C ../vllm-src rev-parse HEAD"),
            "head_short": sh("git -C ../vllm-src log -1 --format='%h %cd %s'"),
            "dirty_file_count": sh("git -C ../vllm-src status --porcelain | wc -l"),
            "is_dirty": sh("test -n \"$(git -C ../vllm-src status --porcelain)\" "
                           "&& echo true || echo false"),
            "porcelain": sh("git -C ../vllm-src status --porcelain"),
            "note": "本仓基线是 22 个脏文件（所有者的中文学习注释），不等于产品源码被改",
        },
        "lab_repo": {
            "head_sha_full": sh("git rev-parse HEAD"),
            "is_dirty": sh("test -n \"$(git status --porcelain)\" && echo true || echo false"),
            "dirty_file_count": sh("git status --porcelain | wc -l"),
        },
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
            "note": ("M12(2026-07-24) 未留指纹，故 M12→M21 的高并发 TPOT 回归无法归因；本文件是为后续模块留的基线。"
                 "2026-08-12 audit 扩充：补 vLLM/torch/triton/transformers 运行时版本、vllm-src 完整 SHA 与 dirty flag、"
                 "VLLM_* 环境变量、CPU 能耗状态（读不到即记 unavailable，不猜）。"),
    }
    C.save_json(args.out, fp)
    rv = fp["runtime_versions"]
    print(json.dumps({"driver": fp["gpu"]["driver_version"],
                      "gpu": fp["gpu"]["name"],
                      "cpu_cores": fp["host"]["cpu_logical_cores"],
                      "kernel": fp["host"]["uname"]["release"],
                      "vllm": rv.get("vllm"), "torch": rv.get("torch"),
                      "torch_cuda": rv.get("torch_version_cuda"),
                      "triton": rv.get("triton"), "transformers": rv.get("transformers"),
                      "vllm_src_head": fp["vllm_src"]["head_sha_full"],
                      "vllm_src_dirty": fp["vllm_src"]["is_dirty"],
                      "cpu_governor": fp["cpu_power_state"]["scaling_governor"]},
                     ensure_ascii=False, indent=1))
    print("->", args.out)


if __name__ == "__main__":
    raise SystemExit(main())
