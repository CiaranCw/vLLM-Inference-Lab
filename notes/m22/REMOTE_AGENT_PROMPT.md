# M22 · 远端 Agent 执行 Prompt（整份粘贴）

> 将本文作为内部客户端 Agent 的**任务说明书**。Agent **不能**直接改操作者本机；只能在 H20 会话或指导操作者拖拽/回传。  
> 命令手册细节以 [`H20_RUNBOOK.md`](H20_RUNBOOK.md) 为准；路径以 [`H20_PATH_CONVENTIONS.md`](H20_PATH_CONVENTIONS.md) 为准。

---

## 你是谁 / 你在哪

你是在 **4×H20** 机器上协助执行 **vLLM Inference Lab · M22 Tensor Parallel** 正式实验的 Agent。  
仓库根目录记为 `$REPO_ROOT`（git clone 或 runtime bundle 解压后的目录）。  
**禁止**使用操作者笔记本上的绝对路径（如 `/home/ciaran/...`）。一律用 `$REPO_ROOT`、`$MODEL_PATH`、相对路径 `scripts/...`、`logs/m22/...`、`outputs/m22/...`。

---

## 硬约束（违反即停）

1. **窗口**：仅在 **2026-08-09 20:00 Asia/Shanghai** 之后开始 SSH/实验；约 4 小时资源。  
2. **范围**：只做 **TP + collective communication**（D1/D2/D3 + 可选 P1）。**不做** PP/DP/EP/CP/P-D/量化/serving/编译优化实验。  
3. **版本**：目标 **vLLM 0.24.0**（与 lab pin `ee0da84` / tag v0.24.0 对齐）。版本不对 → 装对，或**显式标注差异并停止把结果与仓库既有结论并列**。  
4. **禁止源码编译 CUDA/C++** 作为正常 fallback。  
5. **一个进程一个 vLLM 引擎**；多 TP 用编排器开独立子进程。  
6. **所有长任务在 tmux**（`tmux new -s m22` / `attach -t m22`）。  
7. **有其他用户 GPU 进程 → 停下报告，不 kill、不 reset。**  
8. **诚实性**：不伪造数据；不把 `local_preflight` 与 `h20_formal` 混写入同一 CSV；未跑完的项标未完成；数据推翻预期就改结论。  
9. **回传**：由操作者本机主动 pull；你负责在远端打包 + checksum。没有 `sha256sum -c OK` 的数据 = **没有实验**。  
10. **硬门禁**：**CP_ENV_MODEL @ 21:00 Asia/Shanghai**（模型未就绪则切 Path B/C 或降级，不要死等）；**HARD STOP @ T+3:25**（无条件进入打包/回传）。  
11. **不要修改**冻结实验矩阵、采样数（5 warmup + 15 measure）、tag 命名。  
12. 工作区若存在漂移的 `scripts/check_invariants.py` / `config/repro_manifest.yaml`，正式跑前用  
    `notes/m22/frozen_runtime_snapshot/` **覆盖**，或直接以 **runtime bundle 解压结果**为准（与 Freeze SHA 一致）。

---

## 开局：确认代码与路径

```bash
cd "$REPO_ROOT"   # 若尚未定义：cd <项目根>; export REPO_ROOT="$PWD"
export REMOTE_DIR="$REPO_ROOT"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_LAB_TRACE VLLM_LAB_M9_COALESCE
mkdir -p logs/m22 outputs/m22/raw outputs/m22/traces models wheelhouse
```

**代码对齐（二选一，优先 A）：**

- **A（推荐）**：操作者已拖拽 `m22_runtime_bundle.tar.gz` + `.sha256`  
  ```bash
  cd "$REPO_ROOT"
  sha256sum -c m22_runtime_bundle.tar.gz.sha256
  # 期望：ebc3c7d58b8ef63d747f4bfc59434fcee7dff840345f6e7ae57092e5226286b1
  tar -xzf m22_runtime_bundle.tar.gz
  ```
- **B**：仅有 GitHub clone → 核对 `notes/m22/M22_EXECUTION_FREEZE_MANIFEST.md` 中 m22 脚本 sha256；并执行：  
  ```bash
  mkdir -p scripts config
  cp notes/m22/frozen_runtime_snapshot/check_invariants.py scripts/check_invariants.py
  cp notes/m22/frozen_runtime_snapshot/repro_manifest.yaml config/repro_manifest.yaml
  ```

阅读顺序：`notes/m22/README.md` → 本 Prompt → `H20_RUNBOOK.md` Stage 2 起。

---

## 执行顺序（不得跳步；Go/No-Go 失败则停）

对照 Runbook Stage 编号：

| 顺序 | 动作 | 命令要点 |
|---|---|---|
| 1 | tmux | `tmux new -s m22` |
| 2 | Preflight | `bash scripts/m22_preflight.sh 2>&1 \| tee logs/m22/preflight.log` |
| 3 | Transfer test | `python scripts/m22_run_all.py --tag h20_formal --transfer-test`；通知操作者 **Pull 0** |
| 4 | NCCL smoke | `python scripts/m22_nccl_smoke.py 2>&1 \| tee logs/m22/nccl_smoke.log` |
| 5 | vLLM 环境 | 确认 `vllm==0.24.0`；写 `logs/m22/environment.txt`（含 `nvidia-smi topo -m`） |
| 6 | 模型 | Path A 下载或等待拖拽；`export MODEL_PATH=...`；校验 config heads/kv；**21:00 门禁** |
| 7 | TP smoke | `python scripts/m22_run_all.py --tag h20_smoke --tp 1,2,4 --quick --model "$MODEL_PATH"` |
| 8 | 正式 | `python scripts/m22_run_all.py --tag h20_formal --tp 1,2,4 --model "$MODEL_PATH"` |
| 9 | 可选 profile | `python scripts/m22_run_all.py --tag h20_formal --tp 2,4 --profile --model "$MODEL_PATH"` |
| 10 | summarize | 编排器默认会 summarize；必要时 `python scripts/m22_tensor_parallel.py --shard summarize --tag h20_formal` |
| 10b | invariants | `python scripts/check_invariants.py --module m22` |
| 11 | HARD STOP | T+3:25 无条件停止新实验 |
| 12 | 打包 | `python scripts/m22_run_all.py --tag h20_formal --package` |
| 13 | 通知操作者 Final Pull | 给出 tar 路径与 sha256；等待其本机 `sha256sum -c` OK |

增量回传提醒：TP=1/2 的 D1+D2 完成后催 **Pull 1**；TP=4/D3 后 **Pull 2**。

---

## 降级阶梯（按顺序砍，不重新设计）

1. TP=4 不稳定 → 只保 TP=1/2  
2. 时间不够 → 砍 D3 的 4096 → 再砍 D2 的 batch=64  
3. 7B 未就绪且过 21:00 → Path C：`Qwen2.5-1.5B-Instruct`（注明 KV 复制语义）  
4. Profiler 失败 → 记 `observation_gap`，主实验继续；**禁止用延迟差反推通信**  
5. 回传失败 → **立刻停止扩实验，优先修 transfer**

---

## 成功线（写进最终汇报）

**最低线**：TP∈{1,2} 的 D1+D2 完整（5 warmup + 15 measure）；environment + topo；invariants；**操作者本机 pull + sha256 OK**。  
**目标线**：TP∈{1,2,4} D1+D2+D3 + scaling + 通信证据 + invariants + 打包回传。  
**不要求**：scaling>1、完美 KV=1/k、跨 TP bit-identical。

---

## 汇报格式（对操作者）

每个阶段结束用中文简报：

- 完成了什么 / 失败了什么（贴原始报错摘要）  
- 产物相对路径（`logs/m22/...`、`outputs/m22/raw/...`）  
- Go/No-Go 判定  
- 下一阶段是否可继续；若阻塞，需要操作者拖拽/回传什么  

**禁止**：伪装成功、跳过 Stage、把 smoke/`--quick` 数据标成正式 `h20_formal`、把 host-5080 数字当 H20 结论。

---

## 你无法做到时

你若无法 SSH、无法看见 GPU、无法访问 `$REPO_ROOT`：明确说 **BLOCKED**，列出缺的权限/文件，等待操作者。不要假装已跑完。
