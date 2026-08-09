# M22 · GitHub vs 拖拽上传边界

## 可以进 GitHub（小文件）

| 内容 | 说明 |
|---|---|
| `notes/m22/**`（含 Agent Prompt、Freeze Manifest） | 文本 |
| `scripts/m22_*`、`scripts/reproduce.py` | 与 Freeze 一致 |
| `MULTI_GPU_EXPERIMENT_PLAN.md` | 矩阵 |
| `notes/m22/frozen_runtime_snapshot/` | Freeze 版 invariants/repro |
| 实验后的**小**摘要：`notes/m22/results.md`、小 CSV、图 | 回传后由操作者决定是否 commit |

## 禁止进 GitHub（整目录拖拽）

| 内容 | 约略体积 | 去向 |
|---|---:|---|
| **整目录** `m22_offline/`（bundles+models+manifests+wheelhouse） | ~18G | `$REPO_ROOT/m22_offline/` |
| H20 正式 `m22_h20_formal_*.tar.gz`、traces、大体量 raw | 视情况 | **优先本机 pull + sha256** |
| `.venv/`、HF/torch/pip 缓存、`*.safetensors`、`*.whl` | 大 | 永不 commit |

## 远端回传 GitHub 前自检

```bash
find . -type f -size +1M | head -100
```

**允许**：小摘要 / 小 CSV / 截断 environment。  
**不允许**：权重、wheel、formal tar、大型 trace、超大 JSONL。  
误推大文件 → 停下报告操作者，勿擅自 filter/force push。

## 推荐开局

1. GitHub：clone/pull `$REPO_ROOT`  
2. **整目录拖入** `m22_offline` → `$REPO_ROOT/m22_offline/`  
3. Agent：`export OFFLINE=$REPO_ROOT/m22_offline`；从 `$OFFLINE/bundles` 解压代码；`$OFFLINE/models` 作 `$MODEL_PATH`  
4. Wheel 仅作 candidate；**禁止**源码编译作正常 fallback
