# 4×H20 多卡实验方案（**v3 · 已冻结 · TP 已执行**）

> # 状态：`TP_EH_COMPLETE`（2026-08-09）
>
> **实验矩阵已冻结且已在 4×H20 采集**。结论：[`notes/m22/results.md`](notes/m22/results.md)。  
> 执行留痕：[`notes/m22/execution_record.md`](notes/m22/execution_record.md)。  
> 源码依据：[`notes/m22/tensor_parallel.md`](notes/m22/tensor_parallel.md)（R1–R7）。
>
> **版本演进**：v1 = 计划；v2 = 本地脚本；v3 = 冻结矩阵；**执行后** = 正式 tag `h20_formal`。
>
> 本文件保留设计rationale（为何改 max_model_len、ProfilerConfig、collective 而非纯 NCCL）。  
> **不再作为上机操作手册**（原 Stage 0–14 Runbook 已在收口时删除）。

---

## 0. v1 审查结论：三处会当场失败的设计（已修）

| # | v1 的设计 | 真实源码 | 修法 |
|---|---|---|---|
| **A** | `max_model_len=4096` + prefill 扫到 `prompt_len=4096` | `v1/engine/input_processor.py:423-432`：`prompt_len == max_model_len` 且 runner 为 generate 时**直接 `ValueError`**（因为还要至少产出 1 个 token） | **`max_model_len` 改为 8192**，prefill 仍扫 {256,1024,4096}。不动其他变量 |
| **B** | 用 `VLLM_TORCH_PROFILER_DIR` 开 profiler | 该环境变量在 v0.24.0 **全仓零命中**，已被 `ProfilerConfig` / `--profiler-config` 取代（`config/profiler.py:34-67`，经 `engine/arg_utils.py:662`） | 脚本改用 `ProfilerConfig(profiler="torch", torch_profiler_dir=...)` |
| **C** | P1 = 「NCCL/allreduce evidence」 | 单节点 TP 的 all-reduce **默认优先走 vLLM custom all-reduce**（`cuda_communicator.py:254-311`），NCCL 只是 fallback；custom 的 per-call 门槛是 payload < **8 MiB**（`custom_all_reduce.py:230-243`） | P1 改为「**collective communication evidence**」：先抓 `_log_all_reduce_backend_selection`（`cuda_communicator.py:191`）确定实际路径，再按该路径找 kernel |

> **C 的危害最隐蔽**：decode 的 all-reduce payload 只有几十 KB~几 MB，几乎必然 < 8 MiB
> → 走 custom all-reduce。若按 `ncclDevKernel_*` 去 trace 里找，会**一无所获**，
> 进而误判成「TP 没有通信开销」。

### 另外四处修订

| # | v1 | 修订 | 依据 |
|---|---|---|---|
| D | D2 用整请求 TPOT 间接代表单步 decode | **直接测 steady-state 单步 wall latency**：逐 step 读 `num_scheduled_tokens`，只取「batch 个请求各调度 1 token」的纯 decode step，并丢弃前 3 步瞬态 | ROADMAP 要回答的是「小 batch 时 TP 为何更慢」，TPOT 会把 prefill 和尾部效应混进来 |
| E | D4 单独跑一遍吞吐 | **删除 D4**，speedup / efficiency **全部由 D2 同一批数据离线派生** | D4 与 D2 的 workload 高度重复；少建一次引擎 = 省 3 档 × 启动时间 |
| F | I3/I4 写成「每卡权重必然 ≈1/k」「KV 容量必然上升」的 invariant | **降级为 expected observation**：实测不符时**记录并解释**，不自动判失败 | embedding/lm_head 是否切分随实现而变（M10 已见「非 −50%」先例）；KV head 不足 TP 时会**被复制**（`config/model.py:1259-1270`） |
| G | 「TP 档间第 0 token 分叉 → 自动 FAIL」 | 硬红线只留**输出长度不一致 / 请求未正常结束 / 引擎异常**；第 0 token 分叉升级为「**需要调查**」的强信号，但**不预设为 TP 功能错误** | TP 改变 GEMM 切分与归约顺序，有限精度下贪心 argmax 可能翻转（M5 / M19 已有同类先例） |

---

## 1. 范围（未变，但更明确）

正式优先级**只有一条链**：

```
TP smoke → TP formal data → communication evidence → summarize/invariants → data transfer
```

| 优先级 | 内容 | 本轮 |
|---|---|---|
| **P0** | Tensor Parallel，TP=1/2/4（D1 显存/KV + D2 decode + D3 prefill） | ✅ |
| **P1** | 真实 collective communication evidence | ✅ |
| ✗ | **PP** / DP / EP / CP / Ring Attention / P-D 分离 | ❌ 明确不做，本轮后仍标 ⬜ |

> **PP 已从正式流程中彻底移除**：不进入正式时间预算、主实验矩阵、降级阶梯、
> 最低成功线、目标成功线。本轮**未为 PP 编写任何实验实现**。
> 只有当 TP=1/2/4 正式数据完整、通信证据完成或已记为 observation gap、invariants 通过、
> **且正式数据已成功回传本地**、还剩明确富余时间时，才可**另行考虑**——
> 且只能标 `optional / smoke only / no performance conclusion`。

**诚实性边界（写进每一份产出）**：
- H20(sm_90) 与本地 RTX 5080(sm_120) 的**绝对性能不可横向比较**，只可比趋势与机制；
- 只有**真实跑过的 TP 档**可标 E/H；TP=4 若失败降级到 1/2，**必须记录真实原因**；
- 不为了好看的 scaling 数字改实验设计。

---

## 2. 已冻结的实验矩阵

### 2.1 引擎配置（三档 TP 唯一变量是 `tensor_parallel_size`）

| 项 | 值 | 为什么是这个值 |
|---|---|---|
| `max_model_len` | **8192** | 必须 > 最长 prompt(4096)，否则 `input_processor.py:423-432` 直接 ValueError |
| `gpu_memory_utilization` | 0.85 | H20 显存大，取高些便于观察 KV 容量 |
| `max_num_seqs` / `max_num_batched_tokens` | 256 / 8192 | 支撑 batch=64；显式写死避免按 world size 派生 |
| `enforce_eager` | True | 与既有全部模块口径一致，排除 CUDA Graph 干扰 |
| `enable_prefix_caching` | False | 重复跑相同 prompt，开着会命中缓存污染 prefill |
| **`async_scheduling`** | **显式 False** | 默认 `None`，自动决策依赖 `executor_class.supports_async_scheduling()`（`config/vllm.py:938`），而 **executor 类在 TP=1(uni) 与 TP>1(mp) 下不同** → 不写死就是隐藏变量 |
| **`disable_log_stats`** | **True** | **与 M19 不同**：M19 设 False 是因为投机计数依赖它；本轮**不依赖任何 stats 观测**，故关掉以免引入无关开销。三档一致 |
| `disable_custom_all_reduce` | 默认 False | 保持上游默认；**把实际生效的后端落盘**（见 P1） |

**三层证据**：传入配置 → `effective_cfg` 从引擎 readback → 不变量核对。**不靠「我传了什么」。**

> **不可消除的伴随变量**：`distributed_executor_backend` 必然随 TP 变化（TP=1→`uni`，TP>1→`mp`）。
> 这是 TP 的固有属性，**无法做成单变量**，因此在结论里显式声明，
> 并由不变量脚本作为一条 note 打印出来。

### 2.2 三组测量（一个引擎跑完，不重复建引擎）

| 组 | 内容 | 网格 | 主指标 | 辅助指标 |
|---|---|---|---|---|
| **D1** | 显存 / KV 快照 | TP∈{1,2,4}，引擎建好后采一次 | **per-rank** weight / available_kv / num_gpu_blocks | torch allocated&reserved、non-torch、device UUID、每 rank attn/kv head 数 |
| **D2** | steady-state decode | TP × batch∈{1,4,16,64}，prompt=128，out=64 | **稳态单步 wall latency**（median/stdev/p95） | 完成延迟、首 token 延迟、逐 step 明细、稳态窗口内 batch 是否恒定 |
| **D3** | prefill scaling | TP × prompt∈{256,1024,4096}，batch=1，out=1 | 首 token 延迟 | prefill step 数、chunk 序列 |

每组 **5 warmup + 15 measure**；汇总只用 `phase=measure and success=True`；
差值用 median difference 的 **Bootstrap 95% CI**（10000 次，seed=0，与 M17/M19 同算法同种子）。

**离线派生（不额外跑实验）**：
`TPOT`、`output throughput`、`speedup(k)=T(k)/T(1)`、`efficiency(k)=T(k)/(k·T(1))`
—— 全部来自 D2 的同一批样本。

### 2.3 D2 如何真正隔离 steady-state decode

三重保证，且**第三重是事后独立核对而非假设**：

1. **构造**：batch 内所有请求 prompt 长度相同、`max_tokens` 相同、`ignore_eos=True`
   → 同步推进、同一步结束，**从构造上消除请求陆续完成导致的 batch 缩水**；
2. **筛选**：逐 step 读 `num_scheduled_tokens`，只有「恰好 batch 个请求、每个恰好 1 token」
   才算纯 decode step；再丢弃前 3 个（排除 prefill 刚结束的瞬态）；
3. **核对**：落盘 `steady_batch_stable` —— 稳态窗口内 batch 是否真的恒定。
   **不变量会检查它**，不成立即 FAIL。这样「测到的是纯 decode 而非尾部收缩」是被验证的，不是被假设的。

> 本地预演实测：batch=1 时 step0 调度 128 token（prefill，被正确排除），
> step1+ 各调度 1 token（纯 decode），`steady_batch_stable=True`。

### 2.4 P1 · collective communication evidence

**不预设 NCCL，也不预设 custom all-reduce。**

> ⚠ **表达强度的红线**：`_log_all_reduce_backend_selection`（`cuda_communicator.py:191`）
> 打印的是该 group **「可能被选中」的后端集合**，**不是**「每次调用都走同一个」。
> 真正的选择是 **per-call** 的：`should_custom_ar()`（`custom_all_reduce.py:230-243`）
> 每次都要看**这一次**的 payload 大小、16 字节对齐、弱连续性，不满足就落到下一个后端
> （SymmMem → PyNCCL → torch.distributed）。
> 同一个引擎里，小 payload 的 decode all-reduce 与大 payload 的 prefill all-reduce
> **完全可能走不同后端**。
> → **不得**把 `backend=custom_ar` 写成引擎级不可变事实。

**正式分析单位**：

```
具体某个 workload → 这一轮有哪些 collective → 每次实际走了什么 kernel → 出现在哪
```

步骤：

1. **保存 backend selection / fallback 日志**（Runbook Stage 7、9 都会抓到
   `logs/m22/allreduce_backend_selection.log`）——作为**旁证**，不作为定论；
2. 用 `ProfilerConfig` 采 3 个**代表性** workload（不是把整个网格都 trace）：
   P1-A 小 batch decode(1) / P1-B 大 batch decode(16) / P1-C 长 prefill(4096)；
3. 每 rank 一份 trace，文件名含 `tp{N}_rank{N}`（`distributed/utils.py:664-696`），可映射回 rank；
4. **回传后离线**用 `scripts/m22_trace_analyze.py` 按**真实 kernel 名**分类统计：
   custom AR → `cross_device_reduce_1stage/2stage`；SymmMem → `multimem_all_reduce_`；
   NCCL → `nccl*`；其余 collective → `other_collective`；未识别 → 列出 top 名字，**不静默丢弃**。
   **同一份 trace 里出现多种后端是正常的，要如实分开计数**（已用合成 trace 验证该能力）。

**沿用 M7 的 profiler 纪律**：GPU 时间取 `cat=="kernel"` 的**区间并集**，不对 `key_averages` 求和；
**「通信 kernel 占 GPU busy 的比例」≠「通信造成的净 wall-time 损失」**（可能与计算重叠），
只报前者，不换算成后者。

**若 profiler 不可用** → 记为 **observation gap**（M7 已有先例：WSL2 下 nsys 无法采集），
**严禁用 TP=1/2 的延迟差反推通信时间**。

---

## 3. 模型（**仍需你授权，本轮未下载**）

**硬约束只有一条**（`config/model.py:1163-1170`）：`total_num_attention_heads % tp == 0`。
KV head 不足时**不报错、而是复制**（`:1259-1270`）。

### 3.1 `Qwen/Qwen2.5-7B-Instruct` —— **结构兼容性已静态确认**（不再是未知项）

来源：HuggingFace 公开 `config.json`（663 字节 metadata，**未下载任何模型权重**）：

```json
{"num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584,
 "intermediate_size": 18944, "num_hidden_layers": 28, "vocab_size": 152064,
 "tie_word_embeddings": false, "torch_dtype": "bfloat16"}
```

| TP | `28 % tp` | attn/rank | KV/rank | Σ KV | KV 复制？ | 结论 |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0 ✅ | 28 | 4 | 4 | 否 | ✅ |
| 2 | 0 ✅ | 14 | 2 | 4 | **否** | ✅ |
| 4 | 0 ✅ | 7 | 1 | 4 | **否** | ✅ |

其余维度也整除：`18944/4=4736`、`head_dim=128`、`vocab 152064/4=38016`。
且 vLLM 的 Qwen2 用 `VocabParallelEmbedding`（`models/qwen2.py:358`）与
`ParallelLMHead`（`:513`）——**embedding 与 lm_head 都按 TP 切分**，
加上 `tie_word_embeddings=false`（lm_head 是独立且被切分的权重），
→ **7B 的每卡权重预期相当接近 1/k**（残留复制的只有 layernorm 与 QKV bias，可忽略）。
**但这仍是 expected observation，不是 invariant。**

> 远端仍会读一次实际下载模型的 `config.json` 做一致性校验，
> 但这只是 **sanity check（防拿到不同 revision）**，**不再是 blocker 探索**。

### 3.2 备选与本地预演模型

| 方案 | 模型 | 权重 | TP=2 | TP=4 |
|---|---|---|---|---|
| **A（主）** | `Qwen/Qwen2.5-7B-Instruct` | ~15 GB | ✅ | ✅ **无 KV 复制** |
| **C（保底）** | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | ✅ 12%2=0，KV 2→1/卡 | ⚠ 12%4=0 可跑，但 **KV heads 2<4 → 被复制 2×** |

若真用到 1.5B 保底：「每卡 KV 容量随 TP 上升」的朴素预期**不成立**，
必须按结构事实解释，**不得当成异常或实验失败**。

**本地 0.5B 是 14 heads → `14 % 4 ≠ 0`，TP=4 跑不了**，只能用于 TP=1 预演（已如此使用）。

---

## 4. 已经准备好的东西（本地验证过）

| 文件 | 作用 | 本地验证 |
|---|---|---|
| `notes/m22/tensor_parallel.md` | T+O：R1–R7 全部带 `file:line` | — |
| `scripts/m22_tensor_parallel.py` | D1+D2+D3+profile+summarize | ✅ **TP=1 全链路跑通**（0.5B/quick，17 samples） |
| `scripts/m22_run_all.py` | 编排：每档一进程、**断点续跑**、**transfer-test**、打包+sha256 | ✅ 静态 + dry-run + **transfer-test 实跑 sha256 校验通过** |
| `scripts/m22_preflight.sh` | 上机自检（9 节） | ✅ 本地实跑，正确 FAIL 在「只有 1 张 GPU」 |
| `scripts/m22_nccl_smoke.py` | 独立 NCCL 冒烟（**不 import vllm**） | ✅ 静态；逐 rank 数值校验逻辑就绪 |
| `scripts/m22_trace_analyze.py` | P1 trace 离线分类（区间并集，per-kernel） | ✅ **合成 trace 7/7 校验**（含重叠去重 + 同 trace 多后端） |
| `scripts/m22_figures.py` | 离线出图 | ✅ 只有一档 TP 时**拒绝生成伪图** |
| `scripts/check_invariants.py --module m22` | 12 条硬 invariant + 2 条 expected observation | ✅ 真实数据 15 PASS；**变异测试 4/4 捕获** |
| `config/repro_manifest.yaml` | m22 登记（含冻结配置与 limitations） | ✅ `reproduce.py --module m22 --dry-run` |
| `notes/m22/results.md` / `execution_record.md` | **E/H 结论与执行留痕**（原 Stage Runbook 已收口删除） | ✅ 2026-08-09 |

**本地预演结果（`tag=local_preflight`，RTX 5080 单卡，`--quick`）—— 不是 H20 结果，不进正式 CSV**：
- `collective_rpc(callable)` 的 per-rank 观测**实测可用**（拿到 weight 949.95 MiB / KV 12243 MiB / device UUID）；
- 稳态检测正确区分 prefill 与 decode；
- **batch=1 稳态单步 6.83–7.46 ms，与 M7 实测的 ~7.44 ms 地板吻合** → 交叉验证计时口径正确。

---

## 5. 上机后还剩什么（理想路径）

```
tmux new -s m22
 └─ ① preflight              bash scripts/m22_preflight.sh
    ② distributed smoke      python scripts/m22_nccl_smoke.py --world-size 4
    ③ model availability     （下载 / 镜像 / 离线拷贝，见 §7）+ config.json 门禁
    ④ TP smoke               m22_run_all.py --tag h20_smoke --tp 2,4 --quick
    ⑤ formal shards          m22_run_all.py --tag h20_formal --tp 1,2,4
    ⑥ communication profile  m22_run_all.py --tag h20_formal --tp 2,4 --profile
    ⑦ summarize + check      --shard summarize / check_invariants --module m22
    ⑧ package + rsync        m22_run_all.py --tag h20_formal --package
```

**没有一步需要在机器上写代码或调试脚本。**

---

## 6. 时间预算（修订）

> 净实验时间 ≈ 2 小时。每个检查点有 Go/No-Go，到点没过就降级，**不原地调试**。

> 上机窗口已结束；复现与产物清单见 [`notes/m22/execution_record.md`](notes/m22/execution_record.md)。

| 时刻 | Stage | 内容 | 检查点 |
|---|---|---|---|
| `00:00–00:10` | 1–2 | `tmux` + 拉仓库 + `preflight` | **CP1**：4 卡可见、拓扑已记录、磁盘够、环境变量对 |
| `00:10–00:20` | **3** | **Early transfer test（本地 pull 演练 + sha256）** | **CP2**：回传链路已验证。❌ → **立刻修**，否则 3 小时后必然丢数据 |
| `00:20–00:30` | 4 | `m22_nccl_smoke.py --world-size 4` | **CP3**：4 rank all-reduce **数值校验通过**。❌ → 降 ws=2 切故障域；仍失败 → 中止多卡并报告 |
| `00:30–00:45` | 5 | 装/验 vLLM 0.24.0 + 环境指纹落盘 | **CP4**：`vllm.__version__==0.24.0` 且 import 成功 |
| `00:45–01:05` | 6 | 模型就位 + `config.json` 一致性校验 | **CP5**：与已确认的 28/4 一致。❌ 下载失败 → 降级 1.5B |
| `01:05–01:25` | 7 | **TP smoke（1→2→4，`--quick`）** | **CP6（最关键）**：引擎起得来、`D1 ranks==TP`、内省可用。❌ TP=4 → 降级 1/2；❌ TP=2 → 中止多卡 |
| `01:25–02:45` | 8 | **正式采集** `--tag h20_formal --tp 1,2,4` | **CP7**：各档 raw JSONL 齐全。⚠ 落后 → 见降级阶梯 |
| `02:45–03:10` | 9 | 通信 profile `--profile --tp 2,4` | ⚠ 不可用 → 记 observation gap，**不阻塞主实验** |
| `03:10–03:25` | 10 | 远端 summarize + invariants + 补关键缺口 | 只补 TP=1/2 的 D1/D2 缺口 |
| **`03:25`** | **11** | 🔴 **HARD STOP** | **无条件停止一切新 GPU 实验** |
| `03:25–03:40` | 12 | 打包 + sha256 | |
| `03:40–04:00` | **13** | **本地主动 pull + 校验** | **CP8**：`sha256sum -c` 输出 `OK` = 实验成功 |

### 降级阶梯（预先定死）

1. **TP=4 起不来** → 只做 TP=1/2（**仍闭合第一阶段验收 #8**），保留完整错误日志，
   **记录真实失败原因，不现场修到耗尽时间**
2. **时间不够** → 砍顺序：**D3 的 4096 档 → D2 的 batch=64 档**（PP 不在此列，本轮无 PP）
3. **模型下载超时** → 降级 1.5B（需已获授权，并按 §3.2 解释 KV 复制）
4. **profiler 不可用** → 记 observation gap，**不反推通信时间**
5. **回传失败** → **立即停止实验，一切让路给 transfer**

### SSH 断线保护（已在代码里实现，不只是风险说明）

- **全程 tmux**（`m22_preflight.sh` 第 1 节会检查并警告）；
- 每档 TP 一个独立子进程；**日志实时写入独立文件**（`logs/m22/<tag>_tp<N>_<shard>.log`）；
- **shard 独立落盘**：`outputs/m22/raw/<tag>_tp<N>_<shard>.jsonl`；
- **已完成的 shard 默认跳过**（`m22_run_all.py:shard_done`）→ 断线后**重跑同一条命令即从中断处继续**，
  **不覆盖已采正式数据**；要重采必须显式 `--force`；
- 打包（`--package`）与传输测试（`--transfer-test`）都可独立执行，均带 sha256。

---

## 7. 远端准备（三条路径都备好）

### 7.1 环境安装（首选，不编译）

`ee0da84` 就是 tag v0.24.0，故**不需要编译源码**：

```bash
python3 -m venv .venv && source .venv/bin/activate   # 或 uv venv
pip install "vllm==0.24.0"
python -c "import vllm; print(vllm.__version__)"     # 必须是 0.24.0
```

**停止条件**：若 wheel 与远端 CUDA/driver 不兼容（import 失败、CUDA error），
**不现场编译源码**（那会吃掉全部机时）——改为报告并考虑官方容器镜像。
**是否有可用容器需要你确认**（见 §8）。

### 7.2 vLLM 已存在的情况

若远端已装 vLLM 且版本 **不是** 0.24.0：优先**另建 venv 装 0.24.0**以保证版本可比。
若受限于时间只能用已有版本 → **必须在结论里标注版本差异**，且**不得与本仓库既有结论并列**。

### 7.3 模型获取（三条路径，当前均未执行）

| 路径 | 做法 | 需要你确认什么 |
|---|---|---|
| **A** HF 直连 | `hf download Qwen/Qwen2.5-7B-Instruct` | 远端能否直连 HF、带宽 |
| **B** 镜像 | `export HF_ENDPOINT=https://hf-mirror.com` 后同上 | 镜像是否可用（preflight 会探测） |
| **C** 离线传输 | 本地下载后 `rsync` **推**到远端 | **本地是否有 15 GB 磁盘与带宽**；这条需**提前**做，不能等上机 |

**保底**：1.5B（~3 GB），任何路径都快得多。

> preflight 的第 8 节已经会探测 HF / HF-CDN / hf-mirror / PyPI 的可达性，**但不下载任何模型**。

### 7.4 数据回传：**Local Pull First**（关键设计）

> **本地能 SSH 到 H20，不代表 H20 能反向 SSH 到本地。**
> 因此回传**只能由本地主动 pull**，绝不设计成远端 push。

| 阶段 | 做法 | 时机 |
|---|---|---|
| **Early transfer test** | 远端 `--transfer-test` 生成几 KB 文件 + sha256 → **本地 pull → `sha256sum -c`** | **T+0:10（越早越好）** |
| **Final pull** | 远端 `--package` 生成 tar + sha256 → **本地 pull → 重算 sha256** | T+3:40 |
| **fallback** | 远端无 `rsync` 时改用 `scp`（Runbook 两套命令都给了） | 视 preflight 结果 |

**为什么必须早做 transfer test**：rsync 不存在 / SSH 参数错 / key 错 / 远端路径写错 /
本地目录权限错——这些都必须在 **T+0:10** 暴露，
**绝不能等到 T+3:25 打包时才第一次发现**。那时已经没有时间修了。

**成功判据**：`sha256sum -c` 输出 `OK`。**没有带回并校验通过的数据 = 没有实验。**

---

## 8. 仍然需要你确认的事项

只列**无法从代码/源码/本地环境自行确认**的：

1. **4 小时是否从 SSH 登录即开始计时？** 能否提前进机器装环境与下模型？
   —— 若能提前准备，§6 可整体前移约 40 分钟，TP 主线的把握会明显提高。
2. **HuggingFace / 镜像在远端的实际连通性与带宽**（preflight 能探测可达性，但**探测不出带宽**）。
3. **远端磁盘可用容量**（preflight 会报，但你若已知可提前定 7B 还是 1.5B）。
4. **模型下载授权**：批准下载 7B？还是先用 1.5B？（本轮**未下载任何东西**）
5. **数据回传信息**：SSH host/user、是否允许 `rsync`、回传到本地哪个目录。
6. **远端是否已有可用的 vLLM 环境或容器镜像**（影响 §7.1 的停止条件）。
7. **范围与降级确认**：认可「只做 TP + 通信证据」「TP=4 不通就只做 TP=1/2」吗？

---

## 9. 验收标准

**最低成功线**（达不到即本轮判定未完成）：
TP∈{1,2} 的 **D1 + D2** 数据完整（正式 JSONL）；每组 5 warmup + 15 measure；
环境指纹 + `nvidia-smi topo -m` 落盘；硬 invariant 全过
（`m22_single_variable` / `m22_d2_steady_batch_stable` / `m22_d1_all_ranks_reported` 等）；
**数据成功回传本地并 checksum 校验通过**。
→ TP=1/2 即足以完成原 M11 双卡 TP 核心验收（ROADMAP 第一阶段验收 #8）。
TP=4 失败时**如实记录 + 保留错误日志**，不视为本轮失败。

**目标线**：TP∈{1,2,4} 三档齐全的 D1 + D2 + D3 + scaling efficiency +
collective communication evidence + invariants + report-ready evidence。

**明确不属于成功条件**：PP / DP / EP / CP / P-D；
「scaling 必须 >1」「TP=4 必须比 TP=1 快」「权重必须严格 1/k」
「KV capacity 必须严格单调」「跨 TP token 必须逐 token 完全一致」。
**负收益也是合法数据。**

**交付物**：`notes/m22/results.md`、`outputs/m22/raw/*.jsonl`、`outputs/m22/h20_formal_*.csv`、
`logs/m22/*`、`figures/fig9_m22_tp_scaling_h20_formal.png`，
并同步 `ROADMAP.md`（第一阶段验收 #8）、`notes/evidence_index.md`、`HANDOFF.md`、`README.md`、manifest。

---

## 10. 风险（只写真实存在的）

| 风险 | 缓解 |
|---|---|
| **远端 CUDA/driver 与 0.24.0 wheel 不兼容** | CP2 早期拦截；**不现场编译**；需你确认是否有容器可用 |
| **NVLink 非全连接 → custom all-reduce 被禁、回落 NCCL** | preflight 记录 `topo -m`；这**不是失败**，但会改变通信路径与结论，必须如实记录 |
| **7B 的 head 数不满足 `% 4 == 0`** | 已列为 CP4 门禁；本地无法预先核实，只能上机验；不满足即降级 |
| **模型下载慢/不通** | 三条路径 + 1.5B 保底；路径 C 需**提前**做 |
| **SSH 断线** | tmux + 日志实时落盘 + shard 级断点续跑 |
| profiler 在多进程 worker 下不可用 | 记 observation gap；**禁止用延迟差反推通信** |
| 小模型上 TP 是纯开销 | **这是合法结论**，正是 ROADMAP 要回答的问题，如实写 |

---

## 11. 一句话

**v1 是计划，v2 是已经跑通的脚本 + 操作手册。**
剩下的不确定性只有三类：远端环境是否兼容、模型能否及时到位、NVLink 拓扑如何——
前两类需要你在 §8 拍板，第三类只能上机看，但已经准备好了记录与降级路径。
