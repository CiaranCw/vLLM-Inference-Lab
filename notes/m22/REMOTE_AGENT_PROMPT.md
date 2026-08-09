# M22 · 远端 Agent 执行 Prompt（整份粘贴）

> 将本文作为内部客户端 Agent 的**任务说明书**。Agent **不能**直接改操作者本机；只能在 H20 会话或指导操作者拖拽/回传。  
> 命令手册细节以 [`H20_RUNBOOK.md`](H20_RUNBOOK.md) 为准；路径以 [`H20_PATH_CONVENTIONS.md`](H20_PATH_CONVENTIONS.md) 为准。

---

## 你是谁 / 你在哪

你是在 **4×H20** 机器上协助执行 **vLLM Inference Lab · M22 Tensor Parallel** 正式实验的 Agent。  
仓库根目录记为 `$REPO_ROOT`（git clone 解压后的项目根）。  
**Offline Pack** 已由操作者**整目录拖入**，默认位于：

```text
$REPO_ROOT/m22_offline/
  bundles/   models/   manifests/   wheelhouse/
```

**禁止**使用操作者笔记本上的绝对路径（如 `/home/ciaran/...`）。一律用 `$REPO_ROOT`、`$OFFLINE`、`$MODEL_PATH`、相对路径 `scripts/...`、`logs/m22/...`、`outputs/m22/...`。

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
10. **硬门禁**：**CP_ENV_MODEL @ 21:00 Asia/Shanghai**（模型未就绪则切 1.5B 或降级，不要死等）；**HARD STOP @ T+3:25**（无条件进入打包/回传）。  
11. **不要修改**冻结实验矩阵、采样数（5 warmup + 15 measure）、tag 命名。  
12. 正式跑前用 **`$OFFLINE/bundles` 中的 runtime bundle 解压**，或以 `notes/m22/frozen_runtime_snapshot/` 覆盖漂移的 invariants/repro。

---

## 开局：确认代码与路径

```bash
# 1) 进入项目根（git clone 后的目录）
cd <项目根>
export REPO_ROOT="$PWD"
export REMOTE_DIR="$REPO_ROOT"

# 2) Offline Pack = 整目录拖入后的 m22_offline
export OFFLINE="$REPO_ROOT/m22_offline"
# 若实际拖到别处：export OFFLINE="<实际 m22_offline 路径>"（仍禁止本机 /home/ciaran/...）

# 3) 必需环境变量（每个 tmux window 都要）
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_LAB_TRACE VLLM_LAB_M9_COALESCE
mkdir -p logs/m22 outputs/m22/raw outputs/m22/traces

# 4) 核对 Offline Pack 布局
test -d "$OFFLINE/bundles" && test -d "$OFFLINE/models" || {
  echo "BLOCKED: 未找到 $OFFLINE/{bundles,models} —— 请操作者拖入完整 m22_offline 目录"
  exit 1
}
ls -lh "$OFFLINE/bundles/m22_runtime_bundle.tar.gz"*
ls -lh "$OFFLINE/models/Qwen2.5-7B-Instruct/config.json"
ls -lh "$OFFLINE/models/Qwen2.5-1.5B-Instruct/config.json" 2>/dev/null || true
ls -lh "$OFFLINE/wheelhouse"/vllm-0.24.0*.whl 2>/dev/null || true
```

**代码对齐（优先用 Offline Pack 内的 Frozen bundle）：**

```bash
cd "$REPO_ROOT"
cp -f "$OFFLINE/bundles/m22_runtime_bundle.tar.gz" \
      "$OFFLINE/bundles/m22_runtime_bundle.tar.gz.sha256" .
sha256sum -c m22_runtime_bundle.tar.gz.sha256
# 期望：ebc3c7d58b8ef63d747f4bfc59434fcee7dff840345f6e7ae57092e5226286b1
tar -xzf m22_runtime_bundle.tar.gz
```

若尚无 Git 文档树、仅有 bundle：解压后仍以 `$REPO_ROOT` 为根继续。  
若只用 GitHub clone、暂无 bundle：再用  
`notes/m22/frozen_runtime_snapshot/` 覆盖 `scripts/check_invariants.py` 与 `config/repro_manifest.yaml`。

**模型路径（默认走 Offline Pack，不必再复制到 `$REPO_ROOT/models`）：**

```bash
export MODEL_PATH="$OFFLINE/models/Qwen2.5-7B-Instruct"
# Path C fallback：
# export MODEL_PATH="$OFFLINE/models/Qwen2.5-1.5B-Instruct"
# 校验（在 $OFFLINE/models/... 目录内对照 manifests）：
#   cd "$MODEL_PATH" && sha256sum -c "$OFFLINE/manifests/qwen2.5-7b.sha256"
```

阅读顺序：`notes/m22/README.md` → 本 Prompt → `H20_RUNBOOK.md` Stage 2 起。

---

## 执行顺序（不得跳步；Go/No-Go 失败则停）

| 顺序 | 动作 | 命令要点 |
|---|---|---|
| 1 | tmux | `tmux new -s m22` |
| 2 | Preflight | `bash scripts/m22_preflight.sh 2>&1 \| tee logs/m22/preflight.log` |
| 3 | Transfer test | `python scripts/m22_run_all.py --tag h20_formal --transfer-test`；通知操作者 **Pull 0** |
| 4 | NCCL smoke | `python scripts/m22_nccl_smoke.py 2>&1 \| tee logs/m22/nccl_smoke.log` |
| 5 | vLLM 环境 | 确认 `vllm==0.24.0`；失败时可试 `pip install --no-index --find-links="$OFFLINE/wheelhouse" "vllm==0.24.0"`（**candidate，不保证**）；写 `logs/m22/environment.txt`（含 `nvidia-smi topo -m`） |
| 6 | 模型 | 默认 `$OFFLINE/models/...`；Path A 仅当未拖包且 HF 可用；校验 config；**21:00 门禁** |
| 7 | TP smoke | `python scripts/m22_run_all.py --tag h20_smoke --tp 1,2,4 --quick --model "$MODEL_PATH"` |
| 8 | 正式 | `python scripts/m22_run_all.py --tag h20_formal --tp 1,2,4 --model "$MODEL_PATH"` |
| 9 | 可选 profile | `python scripts/m22_run_all.py --tag h20_formal --tp 2,4 --profile --model "$MODEL_PATH"` |
| 10 | summarize | 编排器默认会 summarize；必要时 `python scripts/m22_tensor_parallel.py --shard summarize --tag h20_formal` |
| 10b | invariants | `python scripts/check_invariants.py --module m22` |
| 11 | HARD STOP | T+3:25 无条件停止新实验 |
| 12 | 打包 | `python scripts/m22_run_all.py --tag h20_formal --package` |
| 13 | 通知操作者 Final Pull | 给出 tar 路径与 sha256；等待其本机 `sha256sum -c` OK |

增量回传：TP=1/2 的 D1+D2 完成后催 **Pull 1**；TP=4/D3 后 **Pull 2**。

---

## 降级阶梯（按顺序砍，不重新设计）

1. TP=4 不稳定 → 只保 TP=1/2  
2. 时间不够 → 砍 D3 的 4096 → 再砍 D2 的 batch=64  
3. 7B 有问题且过 21:00 → `MODEL_PATH="$OFFLINE/models/Qwen2.5-1.5B-Instruct"`（注明 KV 复制）  
4. Profiler 失败 → 记 `observation_gap`，主实验继续；**禁止用延迟差反推通信**  
5. 回传失败 → **立刻停止扩实验，优先修 transfer**

---

## 成功线（写进最终汇报）

**最低线**：TP∈{1,2} 的 D1+D2 完整（5 warmup + 15 measure）；environment + topo；invariants；**操作者本机 pull + sha256 OK**。  
**目标线**：TP∈{1,2,4} D1+D2+D3 + scaling + 通信证据 + invariants + 打包回传。  
**不要求**：scaling>1、完美 KV=1/k、跨 TP bit-identical。

---

## 汇报格式（对操作者）

每个阶段结束用中文简报：完成/失败（原始报错摘要）、产物相对路径、Go/No-Go、下一步或阻塞项。

**禁止**：伪装成功、跳过 Stage、把 smoke/`--quick` 标成 `h20_formal`、把 host-5080 数字当 H20 结论。

---

## 你无法做到时

无法 SSH / 看不见 GPU / 找不到 `$REPO_ROOT` 或 `$OFFLINE`：明确说 **BLOCKED**，列出缺什么，等待操作者。不要假装已跑完。
