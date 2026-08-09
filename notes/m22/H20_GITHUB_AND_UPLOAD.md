# M22 · GitHub vs 拖拽上传边界

## 可以进 GitHub（小文件）

| 内容 | 说明 |
|---|---|
| `notes/m22/**`（含本文件、Agent Prompt、Freeze Manifest） | 文本 |
| `scripts/m22_*`、`scripts/reproduce.py` | 与 Freeze 一致（已核对 sha256） |
| `MULTI_GPU_EXPERIMENT_PLAN.md` | 矩阵 |
| `notes/m22/frozen_runtime_snapshot/` | Freeze 版 `check_invariants.py` + `repro_manifest.yaml` |
| 实验后的**小**摘要：`notes/m22/results.md`、小 CSV、`figures/fig9_*`（若有） | 回传后由操作者决定是否 commit |

## 禁止进 GitHub（拖拽 / rsync / 本机保留）

| 内容 | 约略体积 | 去向 |
|---|---:|---|
| Offline Pack 模型 `Qwen2.5-7B-Instruct` | ~15G | 拖拽 → `$REPO_ROOT/models/` |
| Offline Pack 模型 `Qwen2.5-1.5B-Instruct` | ~2.9G | 拖拽（fallback） |
| `vllm-0.24.0-*.whl` | ~267M | 可选拖拽 → `$REPO_ROOT/wheelhouse/` |
| `m22_runtime_bundle.tar.gz` | ~81KiB | **推荐拖拽**（与 Freeze 逐文件一致；虽小但作为主代码路径） |
| H20 正式 `m22_h20_formal_*.tar.gz`、chrome traces、大体量 raw JSONL | 视情况 | **优先本机 pull + sha256**；不要为了「方便」塞进 GitHub |
| `.venv/`、HF/torch/pip 缓存、`*.safetensors` | 大 | 永不 commit |

## 远端 Agent / 操作者回传 GitHub 时的产物规则

推回本仓库前必须检查：

```bash
# 在准备 git add 的目录执行：列出 >1MiB 文件
find . -type f -size +1M | head -100
```

**允许 push 的典型产物**：`notes/m22/results.md`、小 CSV、environment 摘要（截断）、invariants 文本日志。  
**不允许 push**：模型权重、wheel、完整 formal tar、大型 trace、未抽样的超大 JSONL。

若误把大文件推进 Git：立刻停下，报告操作者，**不要**用 `git filter` / force push 自作主张（需授权）。

## 推荐开局组合（诚实优先级）

1. **代码主路径**：拖拽 `m22_runtime_bundle.tar.gz` + `.sha256` → 远端校验解压  
2. **文档 / Agent Prompt**：GitHub clone（若内网 GitHub 可用）或一并拖拽仓库文本  
3. **模型**：GitHub **不可用/太慢** 时拖拽 Offline Pack；可用则 Path A `hf download`（对照 **CP_ENV_MODEL @ 21:00**）  
4. **Wheel**：仅当远端 pip/镜像失败时再拖拽 candidate wheel；**不保证**兼容；**禁止**源码编译作正常 fallback
