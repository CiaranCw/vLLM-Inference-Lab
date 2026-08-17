# vLLM 源码改动 patch

本仓库的实验依赖对 vLLM 源码的少量改动（学习注释 + 运行时 Trace 钩子 + 一个实验性调度钩子）。
为保持可复现且不把整个 vLLM 塞进本仓库，这里以 **patch + 新增文件** 的形式保存，基准是固定的上游 commit。

## 基准
- 上游仓库：`https://github.com/vllm-project/vllm`
- 基准 commit：**`ee0da84ab9e04ac7610e28580af62c365e898389`**（tag `v0.24.0`）

## 内容
- `vllm-src.patch`：对 21 个已跟踪文件的改动（相对基准 commit 的 `git diff`），包含：
  - 中文学习注释；
  - M1 被动 Trace 钩子（`# VLLM_INFERENCE_LAB_TRACE`，默认关闭，只读不改逻辑）；
  - M2 E5 实验性调度钩子（`# VLLM_INFERENCE_LAB_EXPERIMENT`，`VLLM_LAB_BLOCK` 门控，默认关闭）。
- `new_files/vllm/_lab_trace.py`：新增的统一 Trace 工具（原始树未跟踪，故不在 diff 内，单独提供）。

## 复现步骤
```bash
# 1) 取到基准版本的 vLLM
git clone https://github.com/vllm-project/vllm vllm-src
cd vllm-src && git checkout ee0da84ab9e04ac7610e28580af62c365e898389

# 2) 应用改动
git apply /path/to/this-repo/patches/vllm-src.patch
cp /path/to/this-repo/patches/new_files/vllm/_lab_trace.py vllm/_lab_trace.py

# 3) editable 安装（复用预编译 kernel，免 nvcc 本地编译）
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
```

## 运行开关（默认全部关闭 → 零副作用）
- `VLLM_LAB_TRACE=1`：开启结构化 Trace；配合 `VLLM_LAB_TRACE_COMPONENTS`（如 `scheduler`）、`VLLM_LAB_TRACE_MAX_STEPS`、`VLLM_LAB_TRACE_LAYER`、`VLLM_LAB_TRACE_JSONL=<path>`。
- `VLLM_LAB_BLOCK=1`：开启实验性「按请求 block」调度钩子（仅 E5/E5b 用）。
- 本机（RTX 5080 / sm_120）还需 `VLLM_USE_FLASHINFER_SAMPLER=0` 规避 FlashInfer 采样器 JIT；单进程观察用 `VLLM_ENABLE_V1_MULTIPROCESSING=0`。

> 注：本机原始工作树的备份在 `../backups/`（未入库）；如需逐文件比对可参考。
