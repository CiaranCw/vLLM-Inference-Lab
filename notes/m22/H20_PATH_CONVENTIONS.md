# M22 · 路径约定（Portable Paths）

> 使用**仓库相对路径**或**环境变量**。禁止在远端文档/命令里写死操作者本机绝对路径。

## 必须先 export 的变量

```bash
cd <项目根>                         # git clone 后的目录
export REPO_ROOT="$PWD"
export REMOTE_DIR="$REPO_ROOT"

# 操作者已整目录拖入 m22_offline
export OFFLINE="$REPO_ROOT/m22_offline"
# 若拖到其他相对位置，只改这一行，例如：
# export OFFLINE="$REPO_ROOT/../m22_offline"

export MODEL_PATH="$OFFLINE/models/Qwen2.5-7B-Instruct"
# fallback：
# export MODEL_PATH="$OFFLINE/models/Qwen2.5-1.5B-Instruct"

export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_LAB_TRACE VLLM_LAB_M9_COALESCE
```

操作者本机（仅上传/回传用，不写进 H20 命令）：

```bash
export H20="<USER>@<H20_HOST>"
export LOCAL_DEST="<本机拉取目录>"    # 例：仓库下 _h20_pull
# 本机 Offline 源目录：自行定位后整目录拖到远端 $REPO_ROOT/m22_offline
```

## Runbook 用词 → 可移植含义

| Runbook / 旧文档写法 | 正确解释 |
|---|---|
| `~/m22_offline` | Offline Pack → **H20 上** `$OFFLINE`（=`$REPO_ROOT/m22_offline`） |
| `$REMOTE_DIR` | `$REPO_ROOT` |
| `$REMOTE_DIR/models/...` | 优先 `$OFFLINE/models/...`（整包拖入后不必再复制） |
| Freeze 中的 `/home/ciaran/m22_offline/...` | **冻结时本机快照**，历史记录；H20 不用 |
| 本地 RTX 5080 / WSL2 | 不是 H20；绝对值不可横向比较 |

## H20 上期望布局

```text
$REPO_ROOT/
  notes/m22/
  scripts/m22_*.py
  m22_offline/                 # ← 整目录拖入
    bundles/
    models/Qwen2.5-7B-Instruct/
    models/Qwen2.5-1.5B-Instruct/
    manifests/
    wheelhouse/
  logs/m22/
  outputs/m22/raw/
  outputs/m22/traces/
```

## 引用脚本

```bash
cd "$REPO_ROOT"
bash scripts/m22_preflight.sh
python scripts/m22_nccl_smoke.py
python scripts/m22_run_all.py --help
```

不要写 `python /home/.../scripts/...`。
