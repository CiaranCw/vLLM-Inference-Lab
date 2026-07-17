# vLLM V1 Inference Trace & Optimization Lab

本目录是围绕 vLLM V1 推理主链路进行**动态追踪、实验、性能分析与代码修改**的长期项目工作区。
学习路线见 [`study_path.md`](study_path.md)，进度流水见 [`study_process_record.md`](study_process_record.md)，环境细节见 [`vllm_dev_environment.md`](vllm_dev_environment.md)。

## 目录用途与三者关系

```text
/home/ciaran/projects/
├── vllm-src/           # vLLM 源码（唯一真源）。commit ee0da84 = v0.24.0，含中文学习注释
├── vllm_playground/    # 旧实验目录 + 基线环境（.venv，只读，不得修改）
└── vllm-inference-lab/ # 本工作区：实验脚本 / 笔记 / 日志 / 结果 / 开发环境
```

- **源码修改只发生在 `../vllm-src`**；本目录不复制、不改写 vLLM 源码。
- 本目录 `.venv` 以 **editable 模式**安装了 `../vllm-src`，因此对 `vllm-src` 的 Python 改动会直接反映到本环境的运行结果。
- `../vllm_playground/.venv` 是独立的干净基线（`site-packages` 普通安装），**只读对照，不得写入**。

## 目录结构

```text
vllm-inference-lab/
├── .venv/                    # 开发环境（Python 3.12，editable 安装 vllm-src）
├── .vscode/settings.json     # 指向本 .venv
├── scripts/
│   └── smoke_import.py       # 最小环境验证（不加载模型）
├── notes/                    # 源码学习笔记
├── logs/                     # 安装 / 实验日志、git 状态快照
├── outputs/                  # 实验输出
├── README.md
├── vllm_dev_environment.md   # 环境与版本基线记录
├── study_path.md             # 学习路线（原始计划，保持不改）
└── study_process_record.md   # 学习进度流水（逐 Day/场景追加）
```

## 环境使用

激活开发环境并做冒烟检查：

```bash
cd /home/ciaran/projects/vllm-inference-lab
source .venv/bin/activate
python scripts/smoke_import.py
```

运行实验脚本：

```bash
cd /home/ciaran/projects/vllm-inference-lab
source .venv/bin/activate
python scripts/<实验脚本>.py
```

退出环境：

```bash
deactivate
```

确认当前实际加载的 vLLM 源码路径：

```bash
python -c "import os, vllm; print(os.path.realpath(vllm.__file__))"
# 期望：/home/ciaran/projects/vllm-src/vllm/__init__.py
```

## 修改如何生效（重要）

| 修改位置 | 是否需要重装 | 生效方式 |
| --- | --- | --- |
| `vllm-src/vllm/**.py`（Python 层） | 否 | **重启 Python 进程**即可（editable 安装直接读源码）。注意杀掉旧的 worker / server 进程，已 import 的旧进程不会自动刷新。 |
| `vllm-src/csrc/**`、CUDA/C++ 扩展、kernel | 是 | 当前用 `VLLM_USE_PRECOMPILED=1` 复用预编译 kernel，源码级修改**不会**自动生效；需另建增量编译流程（进入 kernel 学习阶段再做）。 |

## 约束

- 不修改 `../vllm_playground/.venv`（只读基线）。
- 不对 `../vllm-src` 执行破坏性 git 操作（`reset --hard` / `clean -fd` / `checkout -- .` / `restore` / `stash`）。
- 本目录当前**不是** git 仓库（未 `git init`）。
- 环境记录中不写入任何密码 / Token / 密钥。
