# M22 · 操作者拖拽清单（本机 → H20）

> 内部客户端不便直接拉大文件时：在文件管理器中把下列目录/文件拖到 H20 的 `$REPO_ROOT`。  
> 上传代价可忽略；**不要**为省事把这些推进 GitHub。

## 必传（推荐开局包）

| 本地源（Offline Pack） | 拖到 H20 |
|---|---|
| `bundles/m22_runtime_bundle.tar.gz` | `$REPO_ROOT/` |
| `bundles/m22_runtime_bundle.tar.gz.sha256` | `$REPO_ROOT/` |

解压后应出现 `scripts/m22_*.py`、`notes/m22/H20_RUNBOOK.md` 等（见 bundle 文件列表）。

## 模型（按 Path 决策；对照 21:00 门禁）

| 本地源 | 拖到 H20 | 何时 |
|---|---|---|
| `models/Qwen2.5-7B-Instruct/`（~15G） | `$REPO_ROOT/models/Qwen2.5-7B-Instruct/` | Path B，或 Path A 失败 |
| `models/Qwen2.5-1.5B-Instruct/`（~2.9G） | `$REPO_ROOT/models/Qwen2.5-1.5B-Instruct/` | Path C fallback |
| `manifests/qwen2.5-7b.sha256` 等 | `$REPO_ROOT/` 或 `manifests/` | 传完后远端 `sha256sum -c` |

## 可选

| 本地源 | 拖到 H20 | 说明 |
|---|---|---|
| `wheelhouse/vllm-0.24.0-*.whl`（~267M） | `$REPO_ROOT/wheelhouse/` | **candidate only**，不保证兼容 |
| 本仓库已 clone 的文本树 | `$REPO_ROOT/` | 若 GitHub 可用可免拖；仍建议拖 bundle 对齐 Freeze |

## 不要拖 / 不要传 GitHub

- 操作者 `.venv`、HF 缓存整树、本机 `outputs/m7/traces` 等无关大文件  
- 把正式结果 **只** 丢到 GitHub 而不做本机 `sha256sum -c`（错误优先级）

## 回传（H20 → 本机）

优先：本机对 H20 做 rsync/scp **pull**（见 Runbook Stage 3/13）。  
若只能经 GitHub：只提交小摘要（`notes/m22/results.md`、小 CSV）；大 tar/trace/raw 仍走拖拽/拉回本机。
