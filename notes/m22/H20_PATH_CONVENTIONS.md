# M22 · 路径约定（Portable Paths）

> 所有实验过程文档与 Agent 指令使用**仓库相对路径**或**环境变量**。  
> **禁止**在远端执行文档中写死操作者本机绝对路径（例如 `/home/<user>/...`）。

## 必须先 export 的变量

在 H20 的 tmux 会话中（每个新 window 都要设）：

```bash
# 项目根：git clone 或 bundle 解压后的目录（按实际修改）
export REPO_ROOT="$PWD"          # 建议：先 cd 到项目根再 export REPO_ROOT="$PWD"
export REMOTE_DIR="$REPO_ROOT"   # Runbook 里的 REMOTE_DIR 与此同义

# 模型目录（拖拽或 Path A 下载后）
export MODEL_PATH="$REPO_ROOT/models/Qwen2.5-7B-Instruct"
# fallback 时改为：
# export MODEL_PATH="$REPO_ROOT/models/Qwen2.5-1.5B-Instruct"

# 必需环境变量（脚本会断言）
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_LAB_TRACE VLLM_LAB_M9_COALESCE
```

操作者本机（仅用于**上传源**，不是 H20 上的路径）：

```bash
export OFFLINE="<操作者本机 Offline Pack 根目录>"
# Runbook 里写的 ~/m22_offline 仅表示「操作者 Offline Pack」，
# 请一律改写成 $OFFLINE，不要假设 H20 上存在该目录。
export H20="<USER>@<H20_HOST>"
export LOCAL_DEST="<操作者本机拉取目录>"   # 例：相对仓库的 _h20_pull
```

## Runbook 用词 → 可移植含义

| Runbook / 旧文档写法 | 正确解释 |
|---|---|
| `~/m22_offline` | **仅操作者本机** Offline Pack → `$OFFLINE` |
| `$REMOTE_DIR` | H20 上的项目根 → `$REPO_ROOT` |
| `$REMOTE_DIR/models/...` | `$REPO_ROOT/models/...` |
| `logs/m22/...`、`outputs/m22/...` | 相对 `$REPO_ROOT` |
| Freeze Manifest 中的 `/home/ciaran/m22_offline/...` | **冻结时操作者本机快照路径**，历史记录；H20 **不要**使用 |
| 本地 RTX 5080 / WSL2 | **不是** H20；绝对值不可横向比较 |

## 仓库内相对布局（H20 上应呈现）

```text
$REPO_ROOT/
  MULTI_GPU_EXPERIMENT_PLAN.md
  notes/m22/
    README.md
    REMOTE_AGENT_PROMPT.md
    H20_RUNBOOK.md
    ...
    frozen_runtime_snapshot/
  scripts/m22_*.py
  scripts/m22_preflight.sh
  config/                    # 正式跑前用 frozen snapshot 对齐
  models/                    # 拖拽或下载；不入 Git
    Qwen2.5-7B-Instruct/
    Qwen2.5-1.5B-Instruct/   # 可选 fallback
  wheelhouse/                # 可选；candidate wheel
  logs/m22/
  outputs/m22/
    raw/
    traces/
```

## 引用脚本时

```bash
cd "$REPO_ROOT"
bash scripts/m22_preflight.sh
python scripts/m22_nccl_smoke.py
python scripts/m22_run_all.py --help   # 以实际 CLI 为准
python scripts/m22_tensor_parallel.py --help
```

不要写 `python /home/.../scripts/m22_....py`。
