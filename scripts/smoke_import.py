"""最小环境验证脚本（smoke test）。

只做导入与设备可见性检查：
- 不加载模型
- 不下载模型
- 不创建 vLLM Engine
- 不执行推理或 benchmark

用法：
    cd /home/ciaran/projects/vllm-inference-lab
    source .venv/bin/activate
    python scripts/smoke_import.py
"""

from __future__ import annotations

import os
import sys

import torch
import vllm


def main() -> None:
    print("sys.executable        =", sys.executable)
    print("vllm.__file__         =", os.path.realpath(vllm.__file__))
    print("vllm.__version__      =", getattr(vllm, "__version__", "unknown"))
    print("torch.__version__     =", torch.__version__)
    print("torch.version.cuda    =", torch.version.cuda)
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("torch.cuda.device_count()   =", torch.cuda.device_count())
        print("torch.cuda.current_device() =", torch.cuda.current_device())
        print("torch.cuda.get_device_name(0) =", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()
