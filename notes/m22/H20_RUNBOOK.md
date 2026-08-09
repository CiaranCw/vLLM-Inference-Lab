# M22 · 4×H20 执行手册（Runbook）

> **状态：`READY_FOR_H20_EXECUTION` / `FROZEN_FOR_H20_2026-08-09_20-00_CST`**
> —— 实验矩阵已冻结；本地 Offline Fallback Pack 已准备（见文末附录 A–D）。
> 拿到 SSH 后**照本执行**，不需要现场写代码、不需要现场重新想统计口径。
>
> **正式窗口**：2026-08-09 **20:00 Asia/Shanghai** 起，总资源约 **4 小时**
> （约至 2026-08-10 00:00 Asia/Shanghai）。**不得提前登录 H20。**
>
> **本文件是命令级操作手册，不讲理论。** 理论与源码依据见
> [`tensor_parallel.md`](tensor_parallel.md)（R1–R7 带 `file:line`），
> 方案与时间预算见 [`../../MULTI_GPU_EXPERIMENT_PLAN.md`](../../MULTI_GPU_EXPERIMENT_PLAN.md)。
>
> **占位符**（执行前替换，**不要把真实凭据写进仓库**）：
> `<USER>` `<H20_HOST>` `<REMOTE_DIR>` `<LOCAL_DEST>`
>
> **三条贯穿全程的铁律**
> 1. **所有长任务必须在 tmux 里跑**（SSH 一断，非 tmux 的任务全丢）。
> 2. **回传只能由本地主动 pull** —— 本地能 SSH 到 H20，**不代表** H20 能反向连到本地。
> 3. **目标是拿到 TP E+H 数据**，不是「非 7B 不做」。见 **CP_ENV_MODEL @ 21:00**。

---

## 阶段总览（14 个 Stage）

| Stage | 在哪执行 | 内容 | 硬门禁 |
|---|---|---|---|
| 0 | **本地** | 准备目录 / 命令模板 / 磁盘 | — |
| 1 | 本地→远端 | SSH + tmux | — |
| 2 | 远端 | preflight | **Go/No-Go** |
| 3 | 远端+**本地** | **Early transfer test（pull 演练）** | **Go/No-Go** |
| 4 | 远端 | torch.distributed / NCCL smoke | **Go/No-Go** |
| 5 | 远端 | vLLM 环境 | **Go/No-Go** |
| 6 | 远端 | 模型就位 + config 校验 | **Go/No-Go** |
| 7 | 远端 | TP quick smoke（1→2→4） | **Go/No-Go（最关键）** |
| 8 | 远端 | 正式实验 D1/D2/D3 | — |
| 9 | 远端 | 通信 profile（P1） | 可跳过 |
| 10 | 远端 | 离线 summarize + invariants | — |
| **11** | — | **T+3:25 HARD STOP** | **无条件** |
| 12 | 远端 | 打包 + checksum | — |
| 13 | **本地** | **pull + 校验** | **成功判据** |
| 14 | 本地 | 落地与文档 | — |

---

## Stage 0 · 本地准备（在 H20 计时开始前完成）

```bash
# 0.1 本地接收目录
export LOCAL_DEST=<LOCAL_DEST>            # 例：~/projects/vllm-inference-lab/_h20_pull
mkdir -p "$LOCAL_DEST"/{transfer_test,pull1,pull2,final}

# 0.2 本地磁盘（runtime bundle 很小；模型上传另计）
df -h "$LOCAL_DEST"
df -h ~/m22_offline

# 0.3 冻结连接参数（写进当前 shell，别写进仓库）
export H20=<USER>@<H20_HOST>
export REMOTE_DIR=<REMOTE_DIR>            # 例：~/vllm-inference-lab
export OFFLINE=~/m22_offline              # Offline Fallback Pack 根目录

# 0.4 本地核对 Offline Pack（计时外）
ls -lh "$OFFLINE"/bundles/m22_runtime_bundle.tar.gz*
ls -lh "$OFFLINE"/models/Qwen2.5-7B-Instruct/config.json
ls -lh "$OFFLINE"/models/Qwen2.5-1.5B-Instruct/config.json
ls -lh "$OFFLINE"/wheelhouse/vllm-0.24.0*.whl 2>/dev/null || echo "wheelhouse optional"

# 0.5 连通性（**不得在 20:00 Asia/Shanghai 前登录**；到点再做）
# ssh -o BatchMode=yes "$H20" 'echo SSH_OK; hostname; nvidia-smi -L'
```

> 若 4 小时**从 SSH 登录起算**，Stage 0 的 0.1–0.4 全部在计时外完成。

---

## Stage 1 · SSH + tmux + 上传 Runtime Bundle

```bash
ssh "$H20"

# ★ 立刻进 tmux，之后所有命令都在里面跑
tmux new -s m22
# 断线后重连：ssh <USER>@<H20_HOST> 然后 tmux attach -t m22
```

```bash
# 远端：优先用本地 Offline Runtime Bundle（不依赖 GitHub / git clone）
mkdir -p "$REMOTE_DIR"
```

```bash
# ★ 本地另开终端上传（开局几分钟内应完成）
# 先探测远端是否有 rsync
ssh "$H20" 'command -v rsync || echo NO_RSYNC'

# 有 rsync：
rsync -avP "$OFFLINE"/bundles/m22_runtime_bundle.tar.gz{,.sha256} "$H20":"$REMOTE_DIR"/
# 无 rsync：
# scp "$OFFLINE"/bundles/m22_runtime_bundle.tar.gz \
#     "$OFFLINE"/bundles/m22_runtime_bundle.tar.gz.sha256 \
#     "$H20":"$REMOTE_DIR"/
```

```bash
# 远端：校验并解压
cd "$REMOTE_DIR"
sha256sum -c m22_runtime_bundle.tar.gz.sha256
tar -xzf m22_runtime_bundle.tar.gz
mkdir -p logs/m22 outputs/m22/raw outputs/m22/traces

# ★ 每个新开的 tmux window 都要 export（脚本会硬断言 VLLM_ENABLE_V1_MULTIPROCESSING）
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_LAB_TRACE VLLM_LAB_M9_COALESCE
```

> 仅当 bundle 不可用且网络可访问 GitHub 时，才退回 `git clone` / 全仓 rsync。
> **不要**在正式窗口里依赖现场从 GitHub 拉代码作为主路径。

---

## Stage 2 · Remote preflight

```bash
bash scripts/m22_preflight.sh 2>&1 | tee logs/m22/preflight.log
echo "preflight exit=$?"
```

覆盖：OS / Python / driver / CUDA / `nvidia-smi` / **`topo -m`** / 4 卡可见性 / 型号显存 /
`torch.cuda.device_count()` / `torch.distributed` / NCCL / 磁盘 / 网络可达性 /
已有 vLLM / 已有 HF 缓存 / **必需环境变量**。

**Go/No-Go**
- ✅ exit 0 → 继续。
- ❌ `[FAIL] 只看到 N 张 GPU`（N<4）→ 若 N≥2 仍可做 TP=1/2（最低成功线），记录后继续；N<2 → **中止**。
- ❌ `[FAIL] VLLM_ENABLE_V1_MULTIPROCESSING` → 补 export 后重跑。
- ⚠ `topo -m` 显示 GPU 间是 PHB/SYS（PCIe）而非 NV# → **不是失败**，但 custom all-reduce 很可能被禁用而回落 NCCL。**原样记录，不要试图调**。

---

## Stage 3 · Early transfer test = **Pull 0**（**必须在这里做，不能拖到打包时**）

```bash
# 远端
python scripts/m22_run_all.py --tag h20_formal --transfer-test
```

```bash
# ★ 本地（另开一个本地终端）—— Pull 0
ssh "$H20" 'command -v rsync || echo NO_RSYNC'
rsync -avP "$H20":"$REMOTE_DIR"/outputs/m22/transfer_test_h20_formal.txt{,.sha256} "$LOCAL_DEST"/transfer_test/
# 无 rsync 时 fallback：
# scp "$H20":"$REMOTE_DIR/outputs/m22/transfer_test_h20_formal.txt" \
#     "$H20":"$REMOTE_DIR/outputs/m22/transfer_test_h20_formal.txt.sha256" "$LOCAL_DEST"/transfer_test/

cd "$LOCAL_DEST"/transfer_test && sha256sum -c transfer_test_h20_formal.txt.sha256
```

**Go/No-Go**
- ✅ `OK` → 回传链路已验证，继续。
- ❌ 任何失败（rsync 不存在 / SSH 参数错 / key 错 / 路径错 / 本地权限错）→ **现在就修**。
  这一步失败而不修，等于 3 小时后必然丢数据。
- 远端无 `rsync` → 立即改用 `scp` 并**把后续所有 Pull 命令一并换成 scp 版本**。

---

## Stage 4 · torch.distributed / NCCL smoke（**不 import vllm**）

```bash
python scripts/m22_nccl_smoke.py --world-size 4 2>&1 | tee logs/m22/nccl_smoke.log
```

4 rank 建 communicator，跨 8 MiB 门槛扫 payload（0.03/0.25/2/8/32 MiB），
**逐 rank 校验 all-reduce 数值**（不是「没报错就算过」）。

**Go/No-Go**
- ✅ `[PASS]` → 继续。
- ❌ 失败 → **不要直接去调 vLLM**。先降级复测切故障域：
  ```bash
  python scripts/m22_nccl_smoke.py --world-size 2 2>&1 | tee logs/m22/nccl_smoke_ws2.log
  ```
  - ws=2 通、ws=4 不通 → 记录原因，**本轮降级为 TP=1/2**（仍达最低成功线）。
  - ws=2 也不通 → 机器/驱动/NCCL 问题，**中止多卡部分并如实报告**，保留完整日志。

---

## Stage 5 · vLLM 环境

```bash
python -c "import vllm, torch, os; print('vllm', vllm.__version__); \
print('path', os.path.realpath(vllm.__file__)); print('torch', torch.__version__, torch.version.cuda)"
```

决策顺序（**先探测，再安装**）：

```text
已有兼容 vLLM（优先 0.24.0）?
  ├─ yes → 使用，写 environment.txt
  └─ no
       ↓
     网络可用?
       ├─ yes → pip install "vllm==0.24.0"
       └─ no → 尝试本地 candidate wheelhouse（不保证兼容）
```

若未安装或版本不是 0.24.0，且**网络可用**：

```bash
python3 -m venv .venv && source .venv/bin/activate   # 或 uv venv && source .venv/bin/activate
pip install "vllm==0.24.0"
python -c "import vllm; print(vllm.__version__)"     # 必须打印 0.24.0
```

若网络不可用 / PyPI 慢，再试 **candidate offline wheel**（本地已准备，**不保证**与未知
H20 Python/CUDA/PyTorch/glibc 兼容）：

```bash
# 本地上传（另开本地终端；先 command -v rsync）
rsync -avP "$OFFLINE"/wheelhouse/ "$H20":"$REMOTE_DIR"/wheelhouse/
# 或：scp "$OFFLINE"/wheelhouse/vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl "$H20":"$REMOTE_DIR"/wheelhouse/

# 远端
pip install --no-index --find-links="$REMOTE_DIR/wheelhouse" "vllm==0.24.0" \
  || pip install "$REMOTE_DIR/wheelhouse/vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl"
```

> **禁止**把「本地已有 wheel」写成「保证 H20 可离线安装」。
> **禁止**在正式窗口以源码编译 CUDA/C++ 作为正常 fallback（Go/No-Go：改容器或中止）。

```bash
# 环境指纹落盘（结论可追溯的前提）
{ date -Iseconds; hostname; uname -srmo;
  python -c "import sys,torch,vllm,os;print('python',sys.version.split()[0]);print('torch',torch.__version__);print('cuda',torch.version.cuda);print('vllm',vllm.__version__);print('path',os.path.realpath(vllm.__file__))";
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv;
  echo '--- topo ---'; nvidia-smi topo -m;
} > logs/m22/environment.txt 2>&1
cat logs/m22/environment.txt
```

**Go/No-Go**
- ✅ `vllm == 0.24.0` 且 import 成功 → 继续。
- ❌ 版本不是 0.24.0 → **不要偷偷换个版本继续采正式数据**。要么装对，要么**在所有产出里显式标注版本差异**且不与本仓库既有结论并列。
- ❌ wheel 与 CUDA/driver 不兼容（import 崩 / CUDA error）→ **不现场编译源码**（会吃掉全部机时）。改用官方容器镜像，或**中止并报告**。

---

## Stage 6 · 模型就位（Path A / B / C）

> 主模型 `Qwen/Qwen2.5-7B-Instruct`，时间/网络 fallback `Qwen/Qwen2.5-1.5B-Instruct`
> （已获授权；**不是**第二个正式对照模型）。
> 本地 Offline Pack：`~/m22_offline/models/` + `~/m22_offline/manifests/`。

### 模型获取决策树

```text
模型已在远端可加载?
  ├─ yes → 校验 config / shards → 继续
  └─ no
       ↓
     HF（或内网镜像）是否足够快?
       ├─ yes → Path A：远端下载 7B
       └─ no
            ↓
          local 7B 上传是否来得及（对照 21:00 门）?
            ├─ yes → Path B：Local 7B → H20
            └─ no → Path C：Local 1.5B → H20
```

### Path A · Remote HuggingFace（首选，不占本地 SSH 上行）

```bash
# 模板 —— 只在远端执行；此处不预跑
hf download Qwen/Qwen2.5-7B-Instruct --local-dir "$REMOTE_DIR/models/Qwen2.5-7B-Instruct"
# 或：export HF_ENDPOINT=<mirror>; hf download ...
MODEL_PATH="$REMOTE_DIR/models/Qwen2.5-7B-Instruct"
```

### Path B · Local 7B → H20

```bash
# 本地（另开终端）
ssh "$H20" 'command -v rsync || echo NO_RSYNC'
# 有 rsync（可断点续传，优先）：
rsync -avP --partial --append-verify \
  "$OFFLINE"/models/Qwen2.5-7B-Instruct/ \
  "$H20":"$REMOTE_DIR"/models/Qwen2.5-7B-Instruct/
# 无 rsync：
# scp -r "$OFFLINE"/models/Qwen2.5-7B-Instruct "$H20":"$REMOTE_DIR"/models/

# 传完后远端校验（对照本地 manifest）
# scp "$OFFLINE"/manifests/qwen2.5-7b.sha256 "$H20":"$REMOTE_DIR"/
# cd "$REMOTE_DIR/models/Qwen2.5-7B-Instruct" && sha256sum -c ../../qwen2.5-7b.sha256
MODEL_PATH="$REMOTE_DIR/models/Qwen2.5-7B-Instruct"
```

### Path C · Local 1.5B → H20（时间/网络 fallback）

```bash
rsync -avP --partial --append-verify \
  "$OFFLINE"/models/Qwen2.5-1.5B-Instruct/ \
  "$H20":"$REMOTE_DIR"/models/Qwen2.5-1.5B-Instruct/
# 或 scp -r ...
MODEL_PATH="$REMOTE_DIR/models/Qwen2.5-1.5B-Instruct"
# 之后所有 runner 加：--model "$MODEL_PATH"
```

```bash
# 一致性校验（优先 local path；sanity check）
python - <<PY
from transformers import AutoConfig
import os
p = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-7B-Instruct")
c = AutoConfig.from_pretrained(p, local_files_only=os.path.isdir(p))
h, kv = c.num_attention_heads, c.num_key_value_heads
print("path", p)
print("heads", h, "kv_heads", kv, "hidden", c.hidden_size,
      "inter", c.intermediate_size, "layers", c.num_hidden_layers,
      "vocab", c.vocab_size, "tie", c.tie_word_embeddings)
for tp in (1, 2, 4):
    print(f"  TP={tp}: heads%tp={h % tp} (须为0)  kv/rank={max(1, kv // tp)}  "
          f"replicated={max(1, kv // tp) * tp > kv}")
PY
```

**期望（7B）**：`heads=28, kv_heads=4`；TP=1/2/4 的 `heads%tp` 均为 0，KV **不会被复制**。
**1.5B 结构边界**：`heads=12, kv_heads=2`；**TP=4 会 KV replication** ——
memory/KV scaling **不能**按 7B 机制解释；这是结构事实，不是实验失败。

**Go/No-Go**
- ✅ 与期望一致（或已记录 1.5B 结构边界）→ 继续。
- ⚠ revision 不一致 → 以**实际配置**为准，重新判断 TP 可行性并记录。
- ❌ 7B 获取失败/过慢 → **切 Path C（1.5B）**，不要拖到耗尽窗口。

### CP_ENV_MODEL @ 21:00 Asia/Shanghai（硬门）

到 **2026-08-09 21:00 Asia/Shanghai**（窗口开始后约 1 小时）必须同时具备：

```text
可工作的 vLLM 环境
+
一个完整可加载的正式实验模型（优先 7B；否则 1.5B）
```

- 若 7B 仍在下载/上传且**无法很快完成** → **立即降级 1.5B**，开始 TP smoke / 正式实验。
- **21:00 之后**不再为「必须使用 7B」持续牺牲 TP 采集时间。
- 目标：**拿到 TP E+H 数据**，不是「非 7B 不做」。

---

## Stage 7 · TP quick smoke（严格 1 → 2 → 4，**不要直接跑 15 repeats**）

```bash
python scripts/m22_run_all.py --tag h20_smoke --tp 1 --quick --no-summarize
python scripts/m22_run_all.py --tag h20_smoke --tp 2 --quick --no-summarize
python scripts/m22_run_all.py --tag h20_smoke --tp 4 --quick --no-summarize
```

每档逐条确认（看 `logs/m22/h20_smoke_tp{N}_run.log`）：

```bash
grep -E 'effective config|D1 ranks|D2 tp|D3 tp|Error|Traceback' logs/m22/h20_smoke_tp4_run.log
```

- `effective config` 里 `tensor_parallel_size` == 本档，`async_scheduling: false`，
  `distributed_executor_backend` 在 TP=1 为 `uni`、TP>1 为 `mp`；
- `D1 ranks=N method=collective_rpc(callable)` —— **N 必须等于 TP**；
- D2/D3 正常完成，无 fatal error；
- 顺带抓一次通信后端选择日志（Stage 9 的分析要用）：
  ```bash
  grep -i "all-reduce backends\|all_reduce" logs/m22/h20_smoke_tp4_run.log | tee -a logs/m22/allreduce_backend_selection.log
  ```

**Go/No-Go（本阶段最关键）**
- ✅ TP=1/2/4 全通 → 进入正式实验。
- ❌ TP=4 失败 → **保留完整日志**，本轮降级为 TP=1/2，**不要现场修到耗尽时间**。
- ❌ TP=2 也失败 → 中止多卡部分，只保留 T+O 结论，如实报告。

---

## Stage 8 · 正式实验（D1 + D2 + D3）+ 增量 Pull

```bash
# 若使用本地模型路径 / 1.5B fallback，记得加 --model
python scripts/m22_run_all.py --tag h20_formal --tp 1,2,4 \
  ${MODEL_PATH:+--model "$MODEL_PATH"} 2>&1 | tee -a logs/m22/run_all.log
# TP=4 已降级时：--tp 1,2
```

性质（已在本地验证）：
- **每档 TP 一个独立子进程**；
- **shard 独立落盘**：`outputs/m22/raw/h20_formal_tp{N}_run.jsonl`；
- **日志实时写独立文件**：`logs/m22/h20_formal_tp{N}_run.log`；
- **已完成的 shard 默认跳过** → 断线/失败后**重跑同一条命令即从中断处继续**，不覆盖已采数据；
- 任一 shard 非零退出 → 立即停止后续档位并打印日志路径。

```bash
# 断线恢复
ssh "$H20" ; tmux attach -t m22
# 若 tmux 也没了：重跑同一条命令即可，已完成的 shard 会被跳过
python scripts/m22_run_all.py --tag h20_formal --tp 1,2,4 ${MODEL_PATH:+--model "$MODEL_PATH"}
```

> 只有确实要重采某档时才加 `--force`（会覆盖）。**正式数据默认不覆盖。**

**进度落后时的降级顺序（预先定死）**：
`D3 的 4096 档 → D2 的 batch=64 档`。（PP 不在此列——见文末。）

### Pull 1 · TP=1/2 的 D1+D2 核心数据完成后（最低成功线）

**立刻**由本地增量 pull（不要等 Final）：

```bash
# 本地
mkdir -p "$LOCAL_DEST"/pull1
rsync -avP \
  "$H20":"$REMOTE_DIR"/outputs/m22/raw/ \
  "$H20":"$REMOTE_DIR"/logs/m22/ \
  "$H20":"$REMOTE_DIR"/logs/m22/environment.txt \
  "$LOCAL_DEST"/pull1/ 2>/dev/null || true
# 更稳妥的写法（两趟）：
rsync -avP "$H20":"$REMOTE_DIR"/outputs/m22/raw/ "$LOCAL_DEST"/pull1/outputs_m22_raw/
rsync -avP "$H20":"$REMOTE_DIR"/logs/m22/ "$LOCAL_DEST"/pull1/logs_m22/
# 无 rsync → 改用 scp -r 对应目录
```

到此已基本达到**最低成功线** —— 即使后续远端崩溃，核心证据已在本地。

### Pull 2 · TP=4 / D3 完成后

```bash
mkdir -p "$LOCAL_DEST"/pull2
rsync -avP "$H20":"$REMOTE_DIR"/outputs/m22/ "$LOCAL_DEST"/pull2/outputs_m22/
rsync -avP "$H20":"$REMOTE_DIR"/logs/m22/ "$LOCAL_DEST"/pull2/logs_m22/
```

---

## Stage 9 · 通信证据（P1）

> **前置：Stage 8 已达最低成功线。** profiler 出问题**不得阻塞主实验**。

```bash
python scripts/m22_run_all.py --tag h20_formal --tp 2,4 --profile 2>&1 | tee -a logs/m22/profile.log
ls -la outputs/m22/traces/ | head -20
```

三个代表性 workload（不做 15 次统计，目标是机制证据）：
小 batch decode(1) / 大 batch decode(16) / 长 prefill(4096)。
每 rank 一份 trace，文件名含 `tp{N}_rank{N}`。

```bash
# 确保后端选择日志被抓下来（trace 分类要与它相互印证）
grep -i "all-reduce backends" logs/m22/h20_formal_tp*_profile.log \
  | tee -a logs/m22/allreduce_backend_selection.log
```

**失败处理**：任何一步失败 → 记录 `observation_gap` 并继续后续 Stage。
**严禁用 TP=1/2 的 wall-time 差反推「通信耗时」。**

> trace 的**分类分析在本地离线做**（`scripts/m22_trace_analyze.py`），不占机时。

---

## Stage 10 · 远端离线校验

```bash
python scripts/m22_tensor_parallel.py --shard summarize --tag h20_formal 2>&1 | tee logs/m22/summarize.log
python scripts/check_invariants.py --module m22 2>&1 | tee logs/m22/invariants.log
echo "invariants exit=$?"

head -3 outputs/m22/h20_formal_decode.csv
head -3 outputs/m22/h20_formal_memory_kv.csv
```

**只在硬红线之前补关键缺口**（缺 TP=1/2 的 D1/D2 才算关键；缺 D3-4096 不算）。
invariants 报 FAIL 时先看是数据问题还是采集问题——**不要为了让它变绿而改数据**。

---

## Stage 11 · 🔴 T+3:25 HARD STOP

**到点无条件停止一切新的 GPU 实验**，无论跑到哪一步。之后只做 Stage 12–14。

---

## Stage 12 · 打包（远端）

```bash
python scripts/m22_run_all.py --tag h20_formal --package
```

打包 `outputs/m22/**` + `logs/m22/**`（含 raw JSONL、CSV、traces、environment.txt、
topology、effective config、各 shard 日志），生成 `m22_h20_formal_<stamp>.tar.gz`
+ `.sha256` + `outputs/m22/package_h20_formal.json`。

```bash
ls -la m22_h20_formal_*.tar.gz*
cat m22_h20_formal_*.tar.gz.sha256
```

---

## Stage 13 · Final Pull（**由本地执行**）

HARD STOP / Stage 12 打包完成后：

```bash
# ★ 本地终端
TAR=$(ssh "$H20" "ls -t $REMOTE_DIR/m22_h20_formal_*.tar.gz | head -1")
echo "$TAR"

rsync -avP "$H20":"$TAR"{,.sha256} "$LOCAL_DEST"/final/
# 无 rsync 时 fallback：
# scp "$H20":"$TAR" "$H20":"$TAR".sha256 "$LOCAL_DEST"/final/

cd "$LOCAL_DEST"/final && sha256sum -c ./*.tar.gz.sha256
```

**成功判据：`sha256sum -c` 输出 `OK`。** 不一致就重传，直到一致。

> **没有带回并校验通过的数据 = 没有实验。** 回传失败时，一切其他工作让路。
> 若 Final Pull 失败但 Pull 1/2 已成功，仍以本地已有 raw/logs 作为最低证据，并如实标注缺口。

---

## Stage 14 · 本地落地（H20 已可释放）

```bash
cd <项目根>
tar -xzf "$LOCAL_DEST"/final/m22_h20_formal_*.tar.gz    # 解到 outputs/m22 与 logs/m22
python scripts/m22_tensor_parallel.py --shard summarize --tag h20_formal
python scripts/m22_trace_analyze.py --trace-dir outputs/m22/traces --tag h20_formal
python scripts/check_invariants.py --module m22
python scripts/m22_figures.py --tag h20_formal
```

然后写 `notes/m22/results.md` 并同步 `ROADMAP.md`（第一阶段验收 #8）、
`notes/evidence_index.md`、`HANDOFF.md`、`README.md`、`config/repro_manifest.yaml`。

**结论纪律**：H20(sm_90) 与本地 RTX5080(sm_120) 的**绝对值不可横向比较**；
只有真实跑过的 TP 档可标 E/H；负收益/退化/无法观测都是合法结果。

---

## 事故处理速查

| 情况 | 处理 |
|---|---|
| **SSH 断线** | 任务在 tmux 里继续跑。重连 `tmux attach -t m22`。tmux 也没了 → 重跑同一条命令（已完成 shard 自动跳过） |
| **某个 shard 失败** | 其他已完成 shard **保留**。看 `logs/m22/<tag>_tp<N>_<shard>.log`，修好后重跑同一条命令 |
| **要重采某档** | 显式 `--force`（**默认不覆盖正式数据**） |
| **TP=4 失败** | 记录真实原因 + 保留日志 → 降级 TP=1/2，**不重新设计实验** |
| **profiler 失败** | 记 `observation_gap`，主实验继续，**禁止用延迟差反推通信** |
| **下载失败** | 切 1.5B（需已获授权），并按 Stage 6 解释 KV 复制 |
| **回传失败** | **立即停止实验，优先解决 transfer** |
| **invariants FAIL** | 先判断是数据问题还是采集问题；**不为了变绿改数据** |
| **发现 GPU 被别人占用** | 停下来报告，**不 kill 他人进程、不 reset** |

---

## 关于 PP（明确不在本流程内）

**PP 不属于本次正式 H20 TP 实验**，不进入：正式时间预算、主实验矩阵、降级阶梯、
最低成功线、目标成功线。本轮**未为 PP 编写任何实验实现**。

只有当以下**全部**满足，才可**另行考虑**：TP=1/2/4 正式数据完整 · 通信证据完成或已明确记为
observation gap · invariants 通过 · **正式数据已成功回传本地** · 距资源结束仍有明确富余时间。

真的做了也只能标：`optional / smoke only / no performance conclusion`。

---

## 附录 A · Offline Fallback Pack 布局（本地，不入 Git）

```text
~/m22_offline/
  bundles/m22_runtime_bundle.tar.gz[.sha256]
  models/Qwen2.5-7B-Instruct/
  models/Qwen2.5-1.5B-Instruct/
  manifests/qwen2.5-7b.sha256 | qwen2.5-7b_files.json
  manifests/qwen2.5-1.5b.sha256 | qwen2.5-1.5b_files.json
  wheelhouse/vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl   # candidate only
```

Freeze 文档：仓库内 `notes/m22/M22_EXECUTION_FREEZE_MANIFEST.md`（+ 同目录 JSON）。

---

## 附录 B · 开局执行链（20:00 Asia/Shanghai 后）

```text
SSH
→ tmux
→ upload runtime bundle
→ checksum + extract
→ early transfer test (Pull 0)
→ preflight
→ distributed / NCCL smoke
→ environment / model decision（对照 CP_ENV_MODEL @ 21:00）
→ TP smoke 1→2→4
→ formal experiment（Pull 1 / Pull 2 增量回传）
→ HARD STOP → package → Final Pull
```

---

## 附录 C · 增量回传一览

| Pull | 时机 | 内容 |
|---|---|---|
| **0** | Stage 3 | transfer_test + sha256 |
| **1** | TP=1/2 D1+D2 完成后 | raw / logs / environment（最低成功线） |
| **2** | TP=4 / D3 完成后 | outputs/m22 + logs/m22 增量 |
| **Final** | HARD STOP + package 后 | 完整 tar.gz + sha256 |

---

## 附录 D · 证据强度声明

- Runtime bundle：**可独立执行** M22 编排（仍需远端 Python/vLLM/模型）。
- 本地模型副本：传输后经 manifest/checksum 校验方可视为就位。
- Wheelhouse：仅 **candidate offline install aid**，**不保证**与未知 H20 环境兼容。
- **禁止**源码编译作为正常 fallback。
