# M18 · W4A16（AWQ / GPTQ）量化路径深化 —— 设计冻结

> **状态**：Gate 0 ✅ / Gate 0-b ✅ / **正式实验未执行**。
> 本文件在跑任何正式实验**之前**冻结「测什么 / 写死什么 / 判据是什么 / 不做什么」。
> 冻结之后**不得**为了让结论好看而改网格、改样本量或挑档位。
> 边界：单卡 RTX 5080(sm_120) / WSL2 / `enforce_eager` / in-process / 单卡单副本。

---

## 0. 目标与「不重复 M10」

ROADMAP 给 M18 的定位是**承接 M10**：M10 已做 BF16 / FP8 W8A8 Linear / FP8 KV，
**本模块补 W4A16（AWQ、GPTQ）**，并为 M23 跨引擎对照做预备。

| 轴 | M10 | M18（本模块） |
|---|---|---|
| 量化对象 | 权重 FP8(W8A8) + KV FP8 | **权重 4bit（W4A16），KV 不量化** |
| 量化时机 | 在线（vLLM 启动时动态量化） | **离线 checkpoint**（官方发布的 AWQ / GPTQ 权重） |
| kernel | fp8 GEMM / TRITON_ATTN | **Marlin**（Gate 0 已实测选中） |
| 模型规模 | 仅 0.5B | **0.5B + 1.5B**（见 §3 的 H5，用于控 M7 地板效应） |

**要回答的问题**：
1. W4A16 到底省了多少权重显存？是否也是「理论值达不到」（M10 见过 −36% 而非 −50%）？
2. 权重省下的显存换来了多少 KV 容量？
3. W4A16 对 prefill / decode 的**速度**有没有收益？在哪一侧、什么条件下？
4. AWQ 与 GPTQ 在**同 bits、同 group_size** 下，质量与速度差多少？
5. 上述结论是否**随模型规模变化**（0.5B vs 1.5B）？

---

## 1. Gate 结果（已完成，作为本设计的前提）

**Gate 0（零成本，下载模型之前完成）** —— `outputs/m18/gate0_w4a16_probe.json`、`scripts/m18_probe.py`：

| 检查 | 结果 |
|---|---|
| 设备 capability | sm_120（=120） |
| AutoAWQ / AutoGPTQ 准入门槛 | `min_capability` 75 / 60 → **均通过** |
| kernel 选择（AWQ 带 zero-point / GPTQ 对称 / GPTQ desc_act） | **三种全部选中 `MarlinLinearKernel`** |
| 二进制 op | `marlin_gemm` ✅（本版本已由 `gptq_marlin_gemm` 改名）、`awq_marlin_repack` ✅ |
| **真实 kernel launch（sm_120）** | `awq_marlin_repack` ✅、`marlin_gemm` ✅（输出 shape 正确、数值有限） |

> 为什么必须真 launch：本项目 `VLLM_USE_PRECOMPILED=1`，**不自编译 CUDA**。
> 「源码支持」≠「wheel 里有 sm_120 cubin」。这一步是 M10 的教训（FP8 KV 纸面支持、
> 实测必须换 `TRITON_ATTN` 后端）在本模块的直接应用。

**Gate 0-b（模型下载已获所有者授权）** —— `outputs/m18/gate0b_load_smoke_*.json`：
4 个官方 checkpoint（0.5B/1.5B × AWQ/GPTQ-Int4）**加载 + 生成 smoke 全部 PASS**，
日志确认走 `MarlinLinearKernel`，生成文本连贯。

**Gate 0-c（激活 dtype 可控性，已验证）** —— `logs/m18/dtype_probe_{awq,gptq}_bf16.log`：
AWQ 与 GPTQ checkpoint 都能被强制到 `dtype=bfloat16` 并正常生成
（`effective_dtype=torch.bfloat16`）。这条决定了 §3 的隐藏变量能不能写死。

---

## 2. 冻结的配置矩阵

**每个模型 3 个配置，共 6 个引擎。一个进程只建一个引擎**（AGENTS.md §4：
vLLM 的 KV 显存 `del` + `empty_cache()` 也归还不掉），故每配置一个独立进程。

| 配置 | 模型 | quantization | 说明 |
|---|---|---|---|
| `Q0` | `Qwen2.5-{0.5B,1.5B}-Instruct` | 无 | BF16 基线（**本模块内重测，不引用 M10 旧数**） |
| `Q1` | `Qwen2.5-{0.5B,1.5B}-Instruct-AWQ` | `auto_awq` | W4A16，4bit / group 128 / **非对称（zero_point=true）** |
| `Q2` | `Qwen2.5-{0.5B,1.5B}-Instruct-GPTQ-Int4` | `auto_gptq` | W4A16，4bit / group 128 / **对称（sym=true, desc_act=false）** |

**这是一次受控的 AWQ vs GPTQ 对照**：两者 `bits=4`、`group_size=128` 完全相同，
差别只在**量化算法与是否带 zero-point**（已从各 checkpoint 的 `quantization_config` 逐项核对）。

**为什么 Q0 要重测而不是引用 M10**：M21 已实测本机存在**跨会话性能漂移**
（同配置下高并发 TPOT 比 M12 慢约 30%，根因未定）。跨会话相减会把环境漂移算进量化效应。
→ **Q0/Q1/Q2 必须在同一会话内连续测**，并在实验前后各落一次环境指纹
（复用 M21 的 `scripts/m21_env_fingerprint.py`）。

---

## 3. 必须写死的隐藏变量（跑前冻结）

| # | 变量 | 冻结值 | 为什么 |
|---|---|---|---|
| V1 | **激活 dtype** | **`dtype="bfloat16"`（显式，不用 `auto`）** | 量化 checkpoint 的 `config.json` 声明 `torch_dtype: float16`，BF16 基线声明 `bfloat16`。用 `auto` 会把**量化效应与激活 dtype 混在一起**。Gate 0-c 已验证三侧都支持 bf16 |
| V2 | `kv_cache_dtype` | `auto`（**KV 不量化**） | W4A16 只量化权重；KV 量化是 M10 的正交轴，混进来就说不清 |
| V3 | `enforce_eager` | `True` | 与 M6/M10/M17 一致，且本项目不在 compiled 下下结论 |
| V4 | `enable_prefix_caching` | `False` | 避免 APC 命中干扰 prefill 计时（M17 已量化其影响） |
| V5 | `async_scheduling` | `False` | 同 M10/M19 |
| V6 | `disable_log_stats` | `True` | 同 M10 |
| V7 | attention backend | 不指定（默认 FLASH_ATTN），但**从引擎读回并核对三配置一致** | M10 教训：TRITON_ATTN 本身比 FLASH 慢，后端不同会把后端开销误算成量化效应 |
| V8 | 其余 engine 参数 | `max_model_len=2048`、`gpu_memory_utilization=0.6`、`max_num_seqs=64`、`max_num_batched_tokens=4096`、`seed=0`、`tensor_parallel_size=1` | 与 M10 的 `COMMON` 逐项一致，保证口径可对接 |
| V9 | 采样 | `temperature=0`、`seed=0`、`ignore_eos=True`（性能实验） | 固定输出长度，避免长度差异污染延迟 |

**生效配置必须由 `effective_cfg` 从引擎读回**（`model_config.dtype` / `.quantization` /
attention backend / `kv_cache_size_tokens`），并由不变量核对 —— **不信 CLI 参数，只信引擎回读**。

---

## 4. 测什么（沿用 M10 口径，保证可对接）

### 4.1 显存
- `weight_bytes_by_dtype`：遍历真实 tensor 分类统计（M10 已有实现可复用）。
  W4A16 侧预期出现 **int32 打包权重 + bf16/fp16 scales（+ AWQ 的 zeros）**。
- **`embed` / `lm_head` 是否被量化**：M10 发现它们保持 bf16、占权重约 28%，
  导致 FP8 权重只降 36% 而非理论 50%。W4A16 同样要逐项核对，**不得直接报「理论 −75%」**。
- `kv_cache_size_tokens` / `num_gpu_blocks`：从引擎回读。

### 4.2 性能（进程内 proxy，**不叫 TTFT/TPOT**）
沿用 M10 的指标名，便于跨模块对读：
- `engine_first_step_latency_ms`（prefill proxy）：prompt 长度 **{128, 512, 2000}**
- `steady_decode_step_latency_ms` + `decode_output_tokens_per_second`：
  场景 **{ctx64×b1, ctx64×b64, ctx1024×b1, ctx1024×b8}**（与 M10 完全相同的四个点）
- 每点 **5 warmup + 15 measure**（本项目标准；M10 用的是 8+30，M18 取项目通用口径并在
  结论中标明与 M10 样本量不同，**不与 M10 的绝对值相减**）

### 4.3 质量
- **token 加权 PPL**，直接复用 M10 的固定内置语料与实现（无下载、中英文各若干段、长度 128–1024）。
- 报 **ΔPPL%（相对同会话的 Q0）**，而非 PPL 绝对值（语料有 tile，绝对值偏低，M10 已注明）。

---

## 5. 不变量（跑完逐条核对，失败即实验无效）

| # | 不变量 | 说明 |
|---|---|---|
| I1 | 三配置回读的 `dtype` **均为 `torch.bfloat16`** | V1 是否真的生效 |
| I2 | 三配置回读的 attention backend **相同** | 防止 M10 那种「后端差异被误算成量化收益」 |
| I3 | 三配置的 `kv_cache_dtype` 相同 → **每 token KV 字节数应相同** | KV 不是本模块的变量 |
| I4 | `Σ weight_bytes_by_dtype` == 报告的权重总量 | 分类统计不漏项 |
| I5 | W4A16 侧存在 int32 打包权重张量；**embed/lm_head 未被量化** | 支撑「实际降幅 < 理论值」的解释，而不是猜 |
| I6 | 权重变小 ⇒ `kv_cache_size_tokens` **不减**（同 `gpu_memory_utilization`） | 显存预算守恒的方向性检查 |
| I7 | 性能样本的输出长度**精确等于**请求的 `max_tokens` | 长度不等则延迟不可比 |
| I8 | 汇总只用 `phase=measure and success=True` | 项目通用口径 |
| I9 | PPL 评分 token 数三配置**完全相同** | 否则 ΔPPL 不可比 |

**判据统计口径**：median difference 的 **Bootstrap 95% CI（固定 seed）**。
CI 含 0 只能写「**未观察到超出噪声的差异**」，**不得**写成「没有差异」。

---

## 6. 执行前假设（**可被数据推翻，写在这里是为了防止事后编故事**）

| # | 假设 | 依据 |
|---|---|---|
| H1 | 权重显存显著下降，但**降幅 < 理论 75%** | M10 先例：embed/lm_head 不量化，占权重约 28% |
| H2 | KV 容量随权重腾出的显存而上升 | 显存预算守恒 |
| H3 | **decode 在小 batch 上未必变快，甚至可能变慢** | M7：本机 eager decode 是 CPU 提交受限（~7ms 地板）；Marlin 还要 dequant。M10 的 FP8 就在这里 +36% |
| H4 | 长 prefill（计算受限）**更有可能**受益，但不如 FP8 确定 | W4A16 要把 4bit 反量化回 16bit 再算，收益取决于访存/计算比 |
| H5 | **1.5B 比 0.5B 更容易显现收益** | 地板是固定开销，模型越大占比越小。**这正是加 1.5B 档的理由** |
| H6 | AWQ 与 GPTQ 的 ΔPPL 都在可接受范围，但**谁更好不预判** | 两者算法不同，本模块只报实测 |

---

## 7. 明确**不做**的事（防止范围膨胀 / 防止伪造）

- ⬜ **不自己做量化校准**（不跑 AWQ/GPTQ 的 calibration），只用官方发布的 checkpoint。
- ⬜ **不做 W8A8-INT8**：本版本 Marlin 明确不支持（`auto_gptq.py` 里 assert
  "W8A8-INT8 is not supported by marlin kernel."）。**不为补表格伪造不支持的路径**。
- ⬜ **不与 M10 的 FP8 数字直接相减**：样本量不同（8+30 vs 5+15）、会话不同（M21 已证跨会话漂移）。
  只做**定性**对照（「FP8 在 decode 变慢，W4A16 是否同向」）。
- ⬜ 不做 compiled / CUDA Graph 对照；不做多卡；不做在线 serving（那是 M12/M20/M21 的口径）。
- ⬜ 不做 W4A16 × FP8 KV 的组合（两个轴交叉会让归因困难，且不在 ROADMAP 要求内）。

---

## 8. 执行顺序（每步产出可核验的落盘证据）

1. 实验前落环境指纹（`scripts/m21_env_fingerprint.py`，复用 M21 成果）
2. 实现 `scripts/m18_quant.py`：`--model {0.5b,1.5b} --config {Q0,Q1,Q2}`，一进程一引擎
3. 逐配置跑：显存 + 性能 + 质量（6 次独立进程），逐样本落 `outputs/m18/raw/*.jsonl`
4. `scripts/m18_summarize.py` 离线汇总 → `outputs/m18/*.csv`（含 Bootstrap CI）
5. `check_invariants.py --module m18` 逐条核对 §5
6. 实验后再落一次环境指纹（对照是否发生会话漂移）
7. `scripts/m18_figures.py` 离线出图 → `figures/fig13_m18_*.png`
8. 写 `notes/m18/results.md`，同步 ROADMAP / evidence_index / README / HANDOFF

**产物**：`scripts/m18_{probe,load_smoke,quant,summarize,figures}.py`、
`outputs/m18/{gate0*,raw/*.jsonl,*.csv,configs/*.json}`、`logs/m18/*`、
`notes/m18/{quantization_w4a16,results}.md`、`figures/fig13_m18_*.png`。
