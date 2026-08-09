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

仓库为**私有**：H20 需先按 [`H20_GITHUB_PRIVATE_ACCESS.md`](H20_GITHUB_PRIVATE_ACCESS.md) 配好 SSH/PAT 与 `user.name=CiaranCw`，才能 `clone`/`pull`/`push`。密钥与 PAT **禁止**写入本 Prompt 或仓库文件；认证未通时报 **BLOCKED**。

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

---

## Phase 0 · 探索与就绪探测（先做；必要时再请操作者接入）

> **默认模式**：先只做本 Phase 的只读/可逆探测与最小配置，产出一份中文 **探索报告**。  
> **未宣布 Phase 0 = READY 前，禁止**进入上文「执行顺序」里的正式 TP smoke / `h20_formal` / 长时间 GPU 负载。  
> 需要操作者介入时：列出 **精确动作**（复制公钥、拖目录、加 Deploy key 等），然后 **停下等待**；不要空转重试刷屏。

### 0.0 目标

在不烧机时的前提下，弄清：机器能不能跑 M22、代码/Offline Pack 在哪、私有 GitHub 能不能拉、缺什么必须人来补。

### 0.1 发现工作区（只读）

```bash
date -Iseconds; hostname; whoami; pwd; uname -srmo
echo "TMUX=${TMUX:-<unset>}"
# 常见落点：自行 find/ls，不要假设本机路径
ls -la
find "$HOME" -maxdepth 3 -type d \( -name 'vLLM-Inference-Lab' -o -name 'vllm-inference-lab' -o -name 'm22_offline' \) 2>/dev/null | head -40
```

判定并写入报告：

| 发现 | 动作 |
|---|---|
| 已有 git 项目根 | `export REPO_ROOT=<该路径>` |
| 仅有 `m22_offline` | 记路径为候选 `$OFFLINE`；项目根可能需 clone 或仅用 bundle 解压目录 |
| 什么都没有 | **NEED_HUMAN**：请操作者拖入 `m22_offline` 和/或完成 Git 认证后 clone |

### 0.2 时间窗与 GPU 快照（只读）

```bash
date -Iseconds
# 对照：正式实验仅允许 2026-08-09 20:00 Asia/Shanghai 之后
command -v nvidia-smi && nvidia-smi -L
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi topo -m 2>/dev/null | head -40
```

- GPU < 4 → 报告；正式 TP=4 不可用，只能讨论降级。  
- 有他人进程 → **BLOCKED_NEED_HUMAN**（不 kill）。  
- 未到 20:00 → 只允许 Phase 0 探测与文档/Git 配置，**禁止**正式实验。

### 0.3 Offline Pack（整目录）

```bash
# 若已定位
export OFFLINE="${OFFLINE:-$REPO_ROOT/m22_offline}"
ls -la "$OFFLINE" 2>/dev/null || echo "NO_OFFLINE_YET"
for p in bundles models manifests wheelhouse; do
  [ -d "$OFFLINE/$p" ] && echo "OK $p" || echo "MISS $p"
done
ls -lh "$OFFLINE/bundles/m22_runtime_bundle.tar.gz" 2>/dev/null
ls -lh "$OFFLINE/models/Qwen2.5-7B-Instruct/config.json" 2>/dev/null
ls -lh "$OFFLINE/models/Qwen2.5-1.5B-Instruct/config.json" 2>/dev/null
ls -lh "$OFFLINE/wheelhouse"/vllm-0.24.0*.whl 2>/dev/null
```

缺 `bundles` 或 `models` → **NEED_HUMAN：请拖入完整 m22_offline 到 $REPO_ROOT/m22_offline/**。

### 0.4 私有 GitHub 可达性（可做最小配置，密钥勿入库）

对照 `notes/m22/H20_GITHUB_PRIVATE_ACCESS.md`（若尚无该文件，按下列最小步骤）：

```bash
# 身份（可写 global；值为固定要求）
git config --global user.name >/dev/null 2>&1 || true
git config --global user.name "CiaranCw"
git config --global user.email "CiaranCw@users.noreply.github.com"
git config --global --get user.name
git config --global --get user.email

# SSH 探测
ssh -o BatchMode=yes -o ConnectTimeout=8 -T git@github.com 2>&1 || true
# 若失败且疑似拦 22：可试配置 ssh.github.com:443（见私有访问文档），再测一次
```

| 结果 | 你做的事 |
|---|---|
| `Hi ... authenticated` | 继续：`git clone`/`pull`（若还没有 `$REPO_ROOT`） |
| `Permission denied (publickey)` | 若尚无 key：在 H20 `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_github -N ""`，**只把 .pub 内容贴给操作者** → **NEED_HUMAN：请把公钥加到 GitHub SSH/Deploy key** → **停下等待** |
| 连接超时 | 试 443；仍失败 → **NEED_HUMAN：改用 PAT 或放弃 Git，仅拖代码树** |
| HTTPS 才通 | **NEED_HUMAN：请提供/粘贴 PAT 的使用方式（不要让 Agent 把 PAT 写入仓库文件）** |

Clone（仅认证已通时）：

```bash
git clone git@github.com:CiaranCw/vLLM-Inference-Lab.git
cd vLLM-Inference-Lab && export REPO_ROOT="$PWD"
git rev-parse --short HEAD
git status -sb
```

**禁止**：把私钥、PAT、`.pub` 以外的秘密写进 `notes/` 或任何将要 commit 的路径。

### 0.5 Python / vLLM 探测（短、可失败）

```bash
command -v python3; python3 -V
python3 - <<'PY'
import importlib.util
for m in ("torch","vllm"):
    print(m, "YES" if importlib.util.find_spec(m) else "NO")
PY
python3 - <<'PY'
try:
  import torch, vllm
  print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), "ngpu", torch.cuda.device_count())
  print("vllm", vllm.__version__)
except Exception as e:
  print("IMPORT_FAIL", type(e).__name__, e)
PY
```

- 无 vLLM → 记「Stage 5 待装」；可注明 `$OFFLINE/wheelhouse` 是否存在；**Phase 0 不要**开始长编译。  
- 版本不是 0.24.0 → 记入报告，正式阶段再按硬约束处理。

### 0.6 文档与 Freeze 入口是否可见

```bash
cd "${REPO_ROOT:-.}"
ls notes/m22/REMOTE_AGENT_PROMPT.md notes/m22/H20_RUNBOOK.md notes/m22/H20_GITHUB_PRIVATE_ACCESS.md 2>/dev/null
ls scripts/m22_preflight.sh scripts/m22_run_all.py 2>/dev/null
# 若只有 offline bundle：
ls "${OFFLINE:-/nonexistent}/bundles/m22_runtime_bundle.tar.gz" 2>/dev/null
```

### 0.7 探索报告模板（必须输出给操作者）

用中文填写，缺什么写什么，**不要伪装成功**：

```text
## M22 Phase 0 探索报告
- 时间（远端 date）:
- hostname / user / 是否在 tmux:
- REPO_ROOT: <路径|未找到>
- OFFLINE: <路径|未找到>  bundles/models/manifests/wheelhouse: OK/MISS
- GitHub SSH/HTTPS: PASS/FAIL + 原始关键一行报错
- git user.name/email:
- HEAD（若有）:
- GPU 数量与型号:  他人进程: 有/无
- topo 摘要（NVLink/PCIe 一句话）:
- python / torch / vllm:
- 窗口：未到 20:00 / 已到 / 未知
- NEED_HUMAN 列表（每条一个可执行动作）:
  1.
- Phase 0 结论: READY_FOR_STAGE2 | BLOCKED | WAITING_HUMAN
- 下一步建议: <等你接入 | 开始 Runbook Stage2 preflight | 仅补齐 Git/拖包>
```

### 0.8 何时请操作者接入（触发器）

出现任一项就 **停下并 @ 操作者**，不要自行发明绕过：

1. 需要把 SSH **公钥加到 GitHub** / 需要 PAT  
2. 需要拖入或补全 `m22_offline`  
3. GPU 被他人占用或不足 4 卡且你无法判定策略  
4. 网络策略导致 GitHub/HF 全不可达，需改通道  
5. 任何需要 sudo、改系统包、杀进程、reset GPU 的事  
6. 正式窗口未到但有人要求开跑正式实验  

操作者回复并完成动作后：从 **0.1 或失败项** 续跑 Phase 0，直到结论为 `READY_FOR_STAGE2`，再进入上文「开局：确认代码与路径」与「执行顺序」。
