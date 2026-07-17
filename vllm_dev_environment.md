# vLLM 开发环境记录

记录时间：2026-07-14（本地时区 +08:00）
本轮目标：为 vLLM V1 Python 层源码学习建立独立 editable 开发环境，保留干净基线用于对照。

## 一、路径

| 名称 | 绝对路径 |
| --- | --- |
| Workspace | `/home/ciaran/projects` |
| vllm-inference-lab | `/home/ciaran/projects/vllm-inference-lab` |
| vllm-src（源码，唯一真源） | `/home/ciaran/projects/vllm-src` |
| vllm_playground | `/home/ciaran/projects/vllm_playground` |
| baseline venv（只读，未修改） | `/home/ciaran/projects/vllm_playground/.venv` |
| development venv（本轮新建） | `/home/ciaran/projects/vllm-inference-lab/.venv` |

## 二、Git 基线（vllm-src）

| 项 | 值 |
| --- | --- |
| remote | `https://github.com/vllm-project/vllm.git` |
| branch | detached HEAD（无分支） |
| commit | `ee0da84ab9e04ac7610e28580af62c365e898389`（tag `v0.24.0`，2026-06-28） |
| 初始工作区状态 | 21 个已修改文件（+1226/-98，多为中文学习注释），0 个未跟踪文件 |
| 最终工作区状态 | 与初始完全一致（探针文件已删除，无非预期改动） |

状态快照：`logs/vllm_git_status_before.txt`、`logs/vllm_git_status_after.txt`（`diff` 结果为空）。

## 三、Python 环境

| 项 | 值 |
| --- | --- |
| OS | Ubuntu 26.04 LTS（WSL2，kernel 6.18） |
| Python | CPython 3.12.13（uv 管理） |
| uv | 0.11.26 |
| pip | 26.1.2（venv 内，`--seed`） |
| sys.executable | `/home/ciaran/projects/vllm-inference-lab/.venv/bin/python` |
| sys.prefix | `/home/ciaran/projects/vllm-inference-lab/.venv` |

## 四、PyTorch 与 CUDA

| 项 | 值 |
| --- | --- |
| torch | `2.11.0+cu130` |
| torch.version.cuda | `13.0` |
| torch.cuda.is_available() | `True` |
| GPU | NVIDIA GeForce RTX 5080（16303 MiB，Blackwell sm_120） |
| 驱动 | 610.62（CUDA UMD 13.3） |
| triton | `3.6.0` |

## 五、vLLM

| 项 | 值 |
| --- | --- |
| version | `0.24.1.dev0+gee0da84ab.d20260714.precompiled` |
| vllm.__file__ | `/home/ciaran/projects/vllm-src/vllm/__init__.py` |
| pip show → Editable project location | `/home/ciaran/projects/vllm-src` |
| 安装命令 | `VLLM_USE_PRECOMPILED=1 uv pip install --python <lab .venv python> -U -e /home/ciaran/projects/vllm-src --torch-backend=auto` |
| 使用 VLLM_USE_PRECOMPILED=1 | 是（复用预编译 kernel，不本地编译 CUDA/C++/rust） |
| 安装日志 | `logs/editable_install.log`（exit code 0） |

## 六、验证结果

- editable 路径验证：`vllm.__file__` 与 importlib `submodule_search_locations` 均指向 `../vllm-src/vllm` ✅
- 临时模块探针：在 `vllm-src/vllm/` 创建唯一探针 → lab venv 立即 import 成功并打印 MARKER，来源路径为 `../vllm-src/vllm/` → 已删除 ✅
- smoke import：`scripts/smoke_import.py` 正常，检测到 RTX 5080，未加载模型 ✅
- 基线环境：`../vllm_playground/.venv` 全程只读，未执行任何写操作，保持 torch 2.11.0+cu130 / vllm 0.24.0（site-packages 普通安装） ✅

## 七、日常使用命令

```bash
# 激活并冒烟检查
cd /home/ciaran/projects/vllm-inference-lab
source .venv/bin/activate
python scripts/smoke_import.py

# 运行实验脚本
python scripts/<实验脚本>.py

# 退出环境
deactivate

# 确认当前加载的源码路径
python -c "import os, vllm; print(os.path.realpath(vllm.__file__))"
```

## 八、修改生效边界

- **Python 层**（`vllm-src/vllm/**.py`）：无需重装，重启 Python 进程即可生效；需确保没有旧 worker / server 进程驻留。
- **C++/CUDA/rust**（`vllm-src/csrc/**` 等原生扩展）：`VLLM_USE_PRECOMPILED=1` 下源码修改不自动生效，需重新构建对应原生扩展；本轮不涉及，进入 kernel 学习阶段再单独建立增量编译流程。

## 九、下一阶段基线模型（已定，未下载）

- 主力模型：`Qwen/Qwen2.5-0.5B-Instruct`（HF Transformers 格式，非 GGUF），初始 `max_model_len=2048`。
- 冒烟备用：`facebook/opt-125m`（仅极简冒烟，不作主力）。
- 第二阶段对比：`TinyLlama-1.1B`。
- 本轮**未下载任何模型**、未创建 Engine、未做推理 / benchmark。
