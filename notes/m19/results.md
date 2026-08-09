# M19 · 实测结论（Baseline vs N-gram Speculative Decoding）

> **状态**：**N-gram 主线 CLOSED — T+E+O COMPLETE**（语义收口 2026-08-08；数据采自 2026-08-06）。
> 设计 / 源码定位 / 不变量见 [`speculative_decoding.md`](speculative_decoding.md)；本文件只放结果与结论。
>
> **指标口径（勿与 M12 混淆）**：本文件所有延迟均为 **in-process engine 路径**测量，
> **不含** HTTP / 网络 / tokenization / 异步服务队列。在线 streaming ITL 见 M12。
> 本文件的 **TPOT** = post-first-token **amortized** time per output token，**不是** streaming ITL。
>
> **适用边界**：单卡 RTX 5080（sm_120）/ WSL2 / Qwen2.5-0.5B-Instruct / `enforce_eager=True` /
> in-process / `temperature=0` / 输出长度 128 / batch ≤ 16 / N-gram / `k=3`。
> **所有绝对值仅限本配置，不外推** compiled、大模型、多卡、在线服务、其他 proposer。

## 环境（`logs/m19/environment.txt`）

| 项 | 值 |
|---|---|
| vLLM | `0.24.1.dev0+gee0da84ab` ← `/home/ciaran/projects/vllm-src`（**`ee0da84` / v0.24.0**） |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`（本地已有，未下载新模型） |
| GPU | RTX 5080（sm_120，capability (12,0)），driver 610.62，16303 MiB |
| torch / CUDA / Python / triton | 2.11.0+cu130 / 13.0 / 3.12.13 / 3.6.0 |
| 采样 | 每组 **5 warmup + 15 measure**；**汇总只用 measure 且 success=True**（180 个 measure 样本） |
| 统计 | median difference 的 **Bootstrap 95% CI**（10000 次，固定 seed=0，仅 stdlib），与 M17 同算法同种子 |

**唯一变量**：`speculative_config`。两侧的 `async_scheduling=False`、`disable_log_stats=False`、
`enable_prefix_caching=False`、`enable_chunked_prefill=True`、`max_num_batched_tokens=2048`、
`max_num_seqs=64`、`enforce_eager=True` 逐字相同——由 `effective_cfg`
**从引擎读回**落盘，并由不变量脚本核对（不是「我传了什么」）。

### Workload 命名（保留原名，修正语义）

代码与 CSV 仍使用 `high_repeat` / `low_repeat`，但其**实测语义**是：

| 原名 | 更准确的含义 | 关键证据 |
|---|---|---|
| `high_repeat` | **高初始 prompt 重叠 / 极 N-gram-friendly** | `output_4gram_in_prompt_ratio = 1.00` |
| `low_repeat` | **低初始 prompt 重叠**（**不是**真正低输出重复、也不是 N-gram hostile） | `output_4gram_in_prompt_ratio = 0.00`，但 `output_repetition_4gram ≈ 0.816–0.840` |

N-gram proposer 使用的是 **prompt + generated history**。因此即使初始 prompt 无 4-gram 重叠，
输出自身高度自重复时，后续仍可从 history 获得大量 draft。
**本轮没有覆盖真正低 Proposal Coverage / 低 Acceptance 的退化区域。**

---

## 0. 最终性能模型（本轮统一解释框架）

Speculative Decoding 的核心**不是**「少算 FLOPs」（通常 FLOPs 可能增加），而是：

> **用额外可并行的 proposal / verification 工作，换取更低的串行 Target Execution Depth
> （更少的自回归 execution rounds）。**

统一把最终 wall-time 理解成多因子结果，而不是「Acceptance ↑ → Speedup ↑」：

```text
Proposal Coverage
+ Conditional Acceptance
+ Verification Cost
+ Proposer Cost
+ Rejection / Metadata Cost
+ Batch / Hardware Utilization
+ Runtime Execution Path
        ↓
最终 Speedup
```

粗略关系（decode-dominated）：

```text
T_base ≈ T_prefill + S_base × C_base
T_spec ≈ T_prefill + S_spec × C_spec
Speedup ≈ (S_base / S_spec) × (C_base / C_spec)
```

或用有效进度：`cost/token_spec ≈ C_spec / E`，有收益的粗略条件是 `C_spec / E < C_normal`。
这里的 `E` 是**全局平均有效 output progress / execution round**，不是单纯 Acceptance Length。

---

## 1. Acceptance：随 workload/config 组合变化（不是「完全由 workload 决定」）

数据：`outputs/m19/spec_decode_summary.csv`、`outputs/m19/acceptance_by_position.csv`

| workload | batch | acceptance rate | draft-bearing acceptance length | proposed | accepted | 输出 4-gram 已在 prompt 中的比例 | 输出自身 4-gram 重复 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **high_repeat** | 1 | **1.0000** | **4.000** | 96 | 96 | **1.00** | 0.784 |
| high_repeat | 4 | **1.0000** | **4.000** | 384 | 384 | **1.00** | 0.784 |
| high_repeat | 16 | **1.0000** | **4.000** | 1536 | 1536 | **1.00** | 0.784 |
| **low_repeat** | 1 | **0.7500** | 3.250 | 96 | 72 | **0.00** | 0.816 |
| low_repeat | 4 | **0.7984** | 3.395 | 372 | 297 | **0.00** | 0.832 |
| low_repeat | 16 | **0.8539** | 3.562 | 1431 | 1222 | **0.00** | 0.840 |

（proposed/accepted 为每样本中位数；k=3，故 draft-bearing acceptance length 上界为 4.0。）

**两个独立证据方向一致（但只支持「prompt 重叠 ↔ acceptance 量级」）**：
`output_4gram_in_prompt_ratio` 完全由落盘 token 序列离线算出，与引擎 acceptance 计数互不依赖——
high_repeat 的输出 4-gram **100%** 已在 prompt，acceptance 饱和到 1.0；
low_repeat 该比例为 **0.00**，acceptance 明显更低。
同时 low_repeat 的 **输出自身重复仍很高**，因此它不是「真正低可预测性」workload。

### 1.1 Acceptance Length ≠ global speedup

脚本字段 `acceptance_length = 1 + accepted / num_draft_rounds`（`num_drafts`）描述的是：

> **仅在存在 Draft 的 request-round 中**，平均最终贡献多少正式 output token
> （**draft-bearing acceptance length / draft-bearing round progress**）。

它**不是**：

- 理论 speedup；
- global tokens per target forward；
- 全局 average effective progress `E`。

因为还存在 **no-draft normal rounds**。例如 low_repeat batch=1：

```text
draft-bearing acceptance length = 3.25
但 128 / 56 ≈ 2.286 tokens / execution round   ← 全局 E 的粗 proxy
```

故 **3.25 ≠ global speedup**。

### 1.2 Proposal Coverage（Batch=1 可用近似）

Batch=1 可用：

```text
q = num_draft_rounds / execution_rounds
E ≈ 1 + q × k × A     （近似；request-round 层面）
```

其中 `q` = Proposal Coverage，`k` = speculative token count，`A` = Conditional Acceptance，
`E` = average effective output progress / execution round。

**Batch>1 不要直接用 `num_drafts / engine_steps`**：前者是 request-level draft-bearing count，
后者是 global batch execution rounds——统计空间不同。Batch>1 若谈 Proposal Coverage，
应使用 request-round 概念，而不是把两个不同空间直接相除。

### 1.3 逐位置接受率：较远 speculative position 更容易失败（机制观察，非 k 扫实证）

数据：`outputs/m19/acceptance_by_position.csv`

| workload | batch | pos 0 | pos 1 | pos 2 |
|---|---:|---:|---:|---:|
| high_repeat | 1 / 4 / 16 | 1.000 | 1.000 | 1.000 |
| low_repeat | 1 | 0.781 | 0.750 | 0.719 |
| low_repeat | 4 | 0.895 | 0.879 | **0.621** |
| low_repeat | 16 | 0.891 | 0.874 | 0.797 |

low_repeat 呈靠后位置更低的模式；high_repeat 三个位置全部饱和。
这**提示**更大 `k` 可能存在边际收益递减（更远位置更难命中，但 verification 成本仍付），
但本轮**只实测 k=3、未做 k sweep**，**不能**写成「已证明加大 k 边际收益递减」或「k=3 最优」。

### 1.4 Batch effect 与 Prompt Composition 存在 confounding

low_repeat 的 acceptance 随 batch 为 0.750 → 0.798 → 0.854，
**不能**解释为「batch 增大提高 Acceptance」，也**不能**简单写成「Acceptance 完全由 workload 决定」。

本实验里 batch=1 只用 `样本0`，batch=16 用 `样本0..15`——**prompt composition 不同**。
差异里混着这一项。要干净分离需固定 prompt 集合再扫并发度——**本轮未做**。

更准确：Acceptance 随 **workload/config 组合**变化；high-overlap 组达到 1.0，
lower-prompt-overlap 组为 0.75–0.854。

---

## 2. 机制：串行 execution rounds 下降（`engine.step` 是强 proxy，不是严格 Target Forward）

数据：`outputs/m19/spec_decode_summary.csv` 的 `engine_steps_median` / `steps_per_output_token_median`

在当前 `enforce_eager=True + async_scheduling=False` 配置下，`engine.step()` 是
**execution / model / scheduling round** 的强 proxy，适合观察串行 round 数变化；
**不能一般化为严格的一次纯 Decode Target Forward**。

证据包括：baseline batch=16 有 **130** step 而非 128；prompt token 总量超过有效
batching budget 时会发生 chunked prefill；prefill / boundary / scheduling 语义都可能计入
`engine_steps`。CSV 字段名仍为 `engine_steps`（不改 schema），解释一律用
**execution rounds / engine execution rounds**。

| workload | batch | baseline steps | ngram steps | steps 变化 | ngram 每输出 token 的 step 数 |
|---|---:|---:|---:|---:|---:|
| high_repeat | 1 | 128 | **33** | **−74.22%** | 0.258 |
| high_repeat | 4 | 128 | **33** | **−74.22%** | 0.258 |
| high_repeat | 16 | 130 | **35** | **−73.08%** | 0.273 |
| low_repeat | 1 | 128 | **56** | **−56.25%** | 0.438 |
| low_repeat | 4 | 128 | **62** | **−51.56%** | 0.484 |
| low_repeat | 16 | 130 | **58** | **−55.38%** | 0.453 |

baseline 侧 `steps/token` 恒为 1.0（每步恰好产出 1 个 token，已由不变量核对），
ngram 侧降到 0.26–0.48。**减少串行 execution depth 是加速的主要机制证据**。
它与 draft-bearing acceptance length **方向一致**，但如上所述二者不是同一量。

---

## 3. 性能：完成延迟、amortized TPOT、in-process 吞吐

数据：`outputs/m19/spec_decode_effect.csv`（差值均为 baseline − ngram 的 median difference + Bootstrap 95% CI）

| workload | batch | 完成延迟 base→ngram (ms) | 差值 | CI95 | 显著 | 降幅 | amortized TPOT base→ngram (ms) | 输出吞吐 base→ngram (tok/s) | 增幅 |
|---|---:|---|---:|---|:--:|---:|---|---|---:|
| high_repeat | 1 | 967.19 → **237.02** | +730.17 | [+716.47, +739.22] | ✅ | **75.5%** | 7.556 → **1.807** | 132.3 → **540.0** | **+308%** |
| high_repeat | 4 | 1007.78 → **247.63** | +760.15 | [+753.62, +770.29] | ✅ | **75.4%** | 7.843 → **1.856** | 508.0 → **2067.6** | **+307%** |
| high_repeat | 16 | 1103.76 → **309.56** | +794.19 | [+789.48, +821.77] | ✅ | **72.0%** | 8.533 → **2.276** | 1855.5 → **6615.8** | **+257%** |
| low_repeat | 1 | 961.48 → **409.59** | +551.90 | [+534.85, +588.98] | ✅ | **57.4%** | 7.511 → **3.165** | 133.1 → **312.5** | **+135%** |
| low_repeat | 4 | 1046.26 → **470.23** | +576.03 | [+564.04, +581.46] | ✅ | **55.1%** | 8.147 → **3.612** | 489.4 → **1088.8** | **+123%** |
| low_repeat | 16 | 1094.19 → **494.87** | +599.32 | [+590.43, +611.53] | ✅ | **54.8%** | 8.450 → **3.731** | 1871.7 → **4138.4** | **+121%** |

**12 组差值（完成延迟 6 组 + TPOT 6 组）的 Bootstrap 95% CI 全部排除 0**，即在本配置下
收益是**稳定可测**的，不是噪声。

**Output Throughput** 定义保持：`total output tokens / completion wall time`，解释为
**in-process fixed-batch aggregate output throughput**——不是服务最大吞吐、最大稳定 QPS、
或 production serving capacity（那些属 M12/M20）。

**amortized TPOT** 定义保持：

```text
post_ms = completion_ms - first_token_ms
tpot_ms = post_ms / (output_tokens - 1)
```

这是 **post-first-token amortized time per output token**。
Spec Decode 一个 execution round 可能一次产生多个 token，真实 token 到达是 **bursty** 的；
本轮没有逐 token streaming timestamp，**不能**声称等价于平均 ITL / streaming ITL。

### 3.1 First Token：无稳定收益（机制上也不预期显著加速）

| workload | batch | baseline 首 token (ms) | ngram 首 token (ms) | 差值 (ngram−base) |
|---|---:|---:|---:|---:|
| high_repeat | 1 / 4 / 16 | 7.246 / 11.503 / 20.147 | 7.492 / 11.832 / 20.734 | +0.25 / +0.33 / +0.59 |
| low_repeat | 1 / 4 / 16 | 8.379 / 11.921 / 20.293 | 7.450 / 11.887 / 20.824 | **−0.93** / −0.03 / +0.53 |

得到第一个正式 token 仍需完成 Target 的首次 Prefill/执行；Spec Decode 没有像 Decode 阶段那样
通过 multi-token-per-round 减少这一必要步骤，因此机制上**不预期**显著 First Token 加速。
当前观测差异方向不稳定，属于小幅扰动。
**注意**：low_repeat batch=1 差值约 **−0.93ms**，**不是**「所有 First Token 差异均在 ±0.6ms」。
本轮不对 First Token 宣称稳定收益。

### 3.2 显存：KV capacity planning 与 allocator peak 是不同口径

| 侧 | Available KV cache | GPU KV cache size | 每样本峰值 allocated（batch=16） |
|---|---|---|---|
| baseline | **8.38 GiB** | **731,872 tokens** | 9696.1 MiB |
| ngram | **8.22 GiB** | **717,888 tokens** | 9533.2 MiB |

（KV 数字来自 `logs/m19/{baseline,ngram}.log` 的 `gpu_worker.py:508` / `kv_cache_utils.py:2158`；
峰值来自 `spec_decode_summary.csv`。）

在相同 `gpu_memory_utilization=0.6` 下，Spec-enabled Engine 的**非-KV 预算**增加约 **0.16 GiB**，
使可用于 KV Cache 的容量减少约 **1.9%**（731,872 → 717,888 token）。
这是容量规划口径下的真实代价。

单 sample 的 `torch.cuda.max_memory_allocated` 是**另一统计口径**，不能直接用来
「证明/反驳」KV capacity planning 结论，也**不能**写成「N-gram 总显存一定更高，因为 peak allocated 更高」
（本表 batch=16 的 peak allocated 甚至是 ngram 略低）。

本轮**未做**内存组件 breakdown，不能精确说明 0.16 GiB 分别由 proposer / metadata / rejection 各占多少。

---

## 4. 回答冻结时提出的问题（按修订后的证据强度）

**Q：高初始 prompt 重叠 workload 的 acceptance 是多少？**
1.000（三个 batch 全部饱和），draft-bearing acceptance length 4.000（k=3 上界）。

**Q：低初始 prompt 重叠 workload 的 acceptance 是多少？**
0.750 / 0.798 / 0.854（batch 1 / 4 / 16），draft-bearing acceptance length 3.25–3.56。
注意 §1.4 的 confounding；且该组输出自身仍高重复。

**Q：acceptance 提高是否对应性能改善？**
在本网格内，更高 acceptance 的组确实有更大延迟降幅（high 72–75.5% vs low 54.8–57.4%），
但 **Acceptance 不是充分指标**：high_repeat 的 acceptance 始终 1.0，而 latency reduction
仍从 ~75.5% 降到 ~72%——说明 Round Cost / utilization / tail 等也在起作用。

**Q：batch 增大后收益是否下降或转为退化？**
**在本实验范围（batch ≤ 16）内没有出现退化**，只有轻微衰减（high 75.5%→72.0%，low 57.4%→54.8%）。
原因见 §6。**未触及明确 speedup knee。**

**Q：串行 execution rounds 是否真的减少？**
是，直接计数 `engine.step`（强 proxy）：128→33（high）/ 56–62（low），见 §2。
**不要**无条件把它等同于「纯 Decode Target Forward 次数」。

**Q：固定开销是否抵消了 draft-bearing 上界？**
部分抵消。若**错误地**把 draft-bearing acceptance length 当 global speedup 上界：
high≈4.0 →「应降到 25%」，实测 237/967≈24.5%（几乎打满）；
low≈3.25 →「应降到 30.8%」，实测 410/961≈42.6%（有缺口）。
缺口来自 rejected draft、no-draft rounds、以及 proposer/verification 固定成本
（§6；**未做 profiler 定量拆分**）。

---

## 5. 输出正确性：证据支持一致性，不是严格证明「与 Spec Decode 无关」

数据：`outputs/m19/output_equivalence_detail.csv`（126 个请求级比对）、`outputs/m19/batch_shape_control.csv`

| workload | batch | baseline↔ngram 逐 token 完全一致 | 首个分歧下标（输出长 128） |
|---|---:|---|---|
| high_repeat | 1 / 4 / 16 | **3/3 · 12/12 · 48/48（全一致）** | — |
| low_repeat | 1 | **3/3（全一致）** | — |
| low_repeat | 4 | 3/12 | min 11 / median 19 / max 19 |
| low_repeat | 16 | 42/48 | min 9 / median 10 / max 11 |

**合计 111/126 完全一致。** 分歧全部满足：① 输出长度两侧始终相同；② **没有任何一例在第 0 个
token 就分歧**；③ 分歧位置集中在第 9–19 个 token；④ high_repeat 全部一致；
⑤ 同一 config 跨 repeat 可复现。

**正交对照**（同一侧内部只改 batch 组成，比对 `req_index=0`）：

| 侧 | workload | batch1 vs batch4 | batch1 vs batch16 |
|---|---|---|---|
| **baseline** | high_repeat | 3/3 一致 | 3/3 一致 |
| **baseline** | **low_repeat** | **0/3 一致，首个分歧下标 [19, 19, 19]** | 3/3 一致 |
| ngram | high_repeat | 3/3 一致 | 3/3 一致 |
| ngram | low_repeat | 3/3 一致 | 3/3 一致 |

**最终正确性结论（证据强度）**：

> Batch-shape control 证明：有限精度下 **Batch Shape 变化本身足以导致** Greedy output divergence；
> 因此当前 baseline↔ngram divergence **与 batch-shape numerical sensitivity 一致**；
> 现有证据**没有发现**明确的 Spec Decode correctness bug。

明确区分：

```text
CAN cause divergence   ≠   WAS the sole cause of every divergence
```

**不要**写成「已严格证明每一个 divergence 都完全不是 Spec Decode 导致」或
「这不是 Spec Decode 的问题 / 已证明无关」。

本轮正确性判据仍是：长度一致 + 不得在第 0 位分歧 + 完整记录一致率（设计笔记 §3.6）。

---

## 6. Batch>1 与 Round Cost：从「Acceptance 唯一决定」升级

Batch>1 时一个 global engine round 内可包含多个 request-round。因此：

- request-level Acceptance **不等于** global engine-step reduction；
- 不同请求推进速度不同会产生 **tail / straggler**；
- batch 增大后 Verification work 更难被固定 CPU/launch overhead 掩盖；
- high_repeat acceptance 始终 1.0，但 latency reduction 从 ~75.5%→~72%，
  本身说明 Acceptance 不是唯一变量。

**aggregate round-cost proxy**（完成延迟 / engine_steps）显示 Spec Round 相对成本随 batch
增大开始上升——这是观测到的 aggregate 现象。

**证据强度**：当前**没有 profiler 分解**证明 b16 的额外时间分别来自 verification / proposer /
metadata 哪一部分。可以说「相对成本 proxy 上升」，**不能**说「已精确测出 verification cost 上升了多少」。

**low_repeat 相对 draft-bearing 上界的缺口**（若误把 3.25 当 global 上界）可定性归因于：

1. **被拒绝的 draft 仍付算力**（`proposed == num_drafts × 3` 已由不变量核对）；
2. **逐位置衰减**放大浪费（§1.3）；
3. **proposer / verification / metadata 固定成本**（smoke 可见 `rejection_greedy_sample_kernel` JIT）——
   **未单独测量绝对值**。

**为何 batch≤16 只轻微衰减、未退化**：M7 已证本机 eager decode 在小 batch 下是
**CPU 提交受限**（GPU 仅忙 23%）。batch=16 时每步 token 数仍远未把 GPU 推到算力受限，
所以「多算 verification token」几乎不额外花钱，而「少提交几次」的收益依旧成立。
→ **理论上「batch 增大 → 收益消失甚至退化」在本配置下没有被观察到**，
**不等于该规律不成立**——需要更大 batch / 更大模型 / compiled 才能触及拐点（均未做）。

---

## 7. 实验边界 / Scope Boundary（Optional · Future Extension）

这些是**范围边界**，不是「M19 没完成」：

- 未做真正低 Proposal Coverage workload；
- 未做真正低 Acceptance degradation workload；
- 未扫 `k`（仅 k=3）；
- 未拆 proposer / target verification / rejection / metadata overhead（无 profiler breakdown）；
- 未做 compiled / CUDA Graph × Spec Decode；
- batch 最大只到 16；未触及明确 speedup knee；
- batch 与 prompt composition 存在 confounding；
- 未做 small draft model；
- 未做 EAGLE / MTP / Medusa；
- 当前结果仅限：Qwen2.5-0.5B-Instruct / RTX 5080（本环境）/ eager / N-gram / k=3 /
  当前两个 workload；
- `num_accepted` 由上游从输出 token 数反推（`scheduler.py:1850`），本轮用守恒交叉验证自洽，
  但未独立复算每个位置的接受掩码；
- 一致性 probe 每组仅 3 repeat（126 请求级比对），足以定性，不足以给精确分歧率。

### Optional / Future（不阻塞主线，按优先级）

**深化**：`k` sweep；true low-proposal / low-acceptance workload；profiler breakdown；
larger batch / larger model 找 knee；compiled × Spec Decode。

**Proposer extension**：small draft model；EAGLE；MTP；Medusa。

---

## 8. 最终认识（M19 主线结论）

1. Spec Decode 优化的是**串行 execution depth**，而非保证减少总计算/FLOPs。
2. Acceptance Rate **不是** speedup 的充分指标。
3. Proposal Coverage 与 Conditional Acceptance 必须分开；Acceptance Length 是
   **draft-bearing** 进度，不是 global `E`。
4. Verification / Proposer / Round Cost 决定「减少的 rounds」是否转化为 wall-time gain。
5. CPU/launch-bound decode 更容易从 multi-token verification 受益（本机证据链接 M7）。
6. batch / GPU utilization 提高后 verification cost 更难被隐藏（本轮仅有 aggregate proxy）。
7. `k` 是 effective progress 与 verification cost 的 trade-off；本轮仅提示机制，未扫 k。
8. Spec Decode 主要改善 post-first-token generation，**不天然**改善 First Token。
9. amortized TPOT **不是** streaming ITL。
10. Spec Decode 有额外显存/KV capacity 代价（本轮约 −1.9% KV tokens）。
11. Greedy divergence 需结合有限精度与 batch-shape control 判断；现有证据未发现明确 correctness bug，
    但也不是「逐例排除 Spec Decode」的严格证明。
12. 当前实验**没有覆盖**真正 degradation knee。

---

## 9. 复现

```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
nvidia-smi                                                   # 先确认 GPU 空闲
python scripts/m19_spec_decode.py --shard baseline           # 引擎 1
python scripts/m19_spec_decode.py --shard ngram              # 引擎 2（等显存释放后）
python scripts/m19_spec_decode.py --shard summarize
python scripts/m19_spec_decode.py --shard equiv --mode baseline
python scripts/m19_spec_decode.py --shard equiv --mode ngram
python scripts/m19_spec_decode.py --shard summarize-equiv
python scripts/check_invariants.py --module m19
python scripts/m19_figures.py                                # 离线出图，不需 GPU
```

## 10. 产出文件

- 逐样本（权威证据）：`outputs/m19/raw/{m19_baseline,m19_ngram,m19_equiv_baseline,m19_equiv_ngram}.jsonl`
- CSV：`outputs/m19/{spec_decode_summary,spec_decode_effect,acceptance_by_position,output_equivalence,output_equivalence_detail,batch_shape_control}.csv`
- 日志：`logs/m19/*`（含 `environment.txt`、smoke 日志）
- 图：`figures/{fig7_m19_acceptance,fig8_m19_speedup}.png`（离线由正式 CSV 生成）
- 不变量：`python scripts/check_invariants.py --module m19`

**零 vLLM 产品源码改动**（计数只用上游公开的 `stat_loggers=` 扩展点）。

# **M19 CLOSED — N-GRAM SPECULATIVE DECODING T+E+O COMPLETE**
