# M22 · 远端 / 内部客户端导航（从这里开始）

> **窗口**：2026-08-09 **20:00 Asia/Shanghai** 起，约 4 小时（至约 2026-08-10 00:00）。**不得提前登录 H20。**  
> **状态**：`READY_FOR_H20_EXECUTION` / `FROZEN_FOR_H20_2026-08-09_20-00_CST`  
> **范围**：仅 **TP + collective communication**（D1/D2/D3 + 可选 P1）。**不做** PP/DP/EP/CP/P-D。

## 给内部客户端 Agent 的入口

1. **先读并严格执行**：[REMOTE_AGENT_PROMPT.md](REMOTE_AGENT_PROMPT.md)（可整份粘贴为系统/任务 Prompt）
2. **命令级手册**：[H20_RUNBOOK.md](H20_RUNBOOK.md)（Stage 0–14）
3. **传输与 GitHub 边界**：[H20_GITHUB_AND_UPLOAD.md](H20_GITHUB_AND_UPLOAD.md)
4. **路径约定（禁止本机绝对路径）**：[H20_PATH_CONVENTIONS.md](H20_PATH_CONVENTIONS.md)
5. **操作者拖拽清单**：[OPERATOR_DRAG_DROP.md](OPERATOR_DRAG_DROP.md)
6. **就绪确认**：[H20_READY_CHECKLIST.md](H20_READY_CHECKLIST.md)
7. **冻结记录**：[M22_EXECUTION_FREEZE_MANIFEST.md](M22_EXECUTION_FREEZE_MANIFEST.md)（+ JSON）
8. **理论 T+O**：[tensor_parallel.md](tensor_parallel.md)
9. **矩阵 / 成功线**：[`../../MULTI_GPU_EXPERIMENT_PLAN.md`](../../MULTI_GPU_EXPERIMENT_PLAN.md)

## 代码与冻结快照

| 用途 | 路径 |
|---|---|
| 正式 runner | `scripts/m22_*.py`、`scripts/m22_preflight.sh` |
| 编排 | `scripts/m22_run_all.py` |
| **与 Freeze 一致的** invariants / repro | [`frozen_runtime_snapshot/`](frozen_runtime_snapshot/) |
| 工作区可能已漂移的副本 | 仓库根 `scripts/check_invariants.py`、`config/repro_manifest.yaml`（**H20 正式跑前必须用 snapshot 或 runtime bundle 覆盖**） |

## 双路径心智模型

```text
操作者电脑                              H20（窗口内）
───────────                             ────────────
GitHub clone/pull（小文件）   ──可选──►  $REPO_ROOT
整目录 m22_offline（~18G）    ──拖拽──►  $REPO_ROOT/m22_offline/
                                         ├─ bundles/   → 解压 Frozen 代码
                                         ├─ models/    → $MODEL_PATH
                                         ├─ manifests/ → sha256 校验
                                         └─ wheelhouse/→ 可选 pip
正式结果 tar/raw              ──回传──►  操作者本机校验（优先）
```

**现行约定**：不再要求拆文件零散拖；**一次拖入完整 `m22_offline`**。详见 [OPERATOR_DRAG_DROP.md](OPERATOR_DRAG_DROP.md)。

**诚实性**：没有带回并 `sha256sum -c` 通过的数据 = **没有实验**。不得把本地 5080 TP=1 preflight 写成 H20 多卡结果。
