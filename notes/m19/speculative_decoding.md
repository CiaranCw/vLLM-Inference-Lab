# M19 · Speculative Decoding（N-gram 主线）—— 理论 / 源码定位 / 冻结设计

> **模块状态**：**N-gram 主线 T+E+O COMPLETE / CLOSED**（见 [`results.md`](results.md)）。
> Draft model / EAGLE / MTP / Medusa、`k` sweep、profiler breakdown、compile 对照等均为
> **Optional / Future Extension**，**不阻塞**主线完成状态。
>
> **模块目标**：建立一个**小而完整**的闭环——普通 decode baseline 对照 N-gram 投机解码，
> 把 Proposal Coverage、Conditional Acceptance、execution rounds、Round Cost、延迟与吞吐
> 串成一条可从原始数据回溯到机制的证据链。
>
> **边界**：单卡 RTX 5080（sm_120）/ WSL2 / Qwen2.5-0.5B-Instruct / `enforce_eager=True` /
> in-process。所有绝对值仅限本配置，不外推。
> 源码行号基于 pinned `ee0da84` / v0.24.0，**全部逐条实读核对过**（非凭记忆）。

---

## 1. T · 理论

### 1.1 Spec Decode 优化的是串行 depth，不是「少算」

自回归解码每次只能推进有限进度，而小 batch 下一次 execution round 的成本**主要不是算力**，
而是固定开销（M7 已证：本机 eager decode 的 ~7ms 地板 = CPU kernel 提交开销，GPU 仅忙 23%）。

投机解码的思路是：

1. **Proposer（草稿器）**用廉价方式先猜接下来的 `k` 个 token（draft）；
2. **Target** 在同一轮里对「当前 token + k 个 draft」做 verification，得到各位置真实分布；
3. **Verification（验收）**逐位置比对：猜对就接受，猜错就从该位置起丢弃并用 target 结果纠正。

**核心表述**：

> Speculative Decoding 通过增加可并行执行的 proposal/verification 工作，
> 减少自回归生成所需的串行 Target Execution Rounds。

通常 **FLOPs 可能增加**；收益来自更低的 **serial execution depth**，不是保证减少总计算。

在当前实验配置（`enforce_eager=True + async_scheduling=False`）下，`engine.step()` 是
execution/model/scheduling round 的**强 proxy**，适合观察串行 round 数变化，
但**不能一般化为严格的一次纯 Decode Target Forward**（chunked prefill / boundary /
scheduling 都可能计入）。

### 1.2 N-gram proposer 怎么提候选

N-gram 不需要任何额外模型：它把**当前上下文（prompt + 已生成部分）的结尾**当作一个 pattern，
回头在上下文里找同样的片段，找到就把该片段**后面紧跟的 k 个 token** 当作 draft。

vLLM 的实现（`vllm/v1/spec_decode/ngram_proposer.py:207`
`_find_longest_matched_ngram_and_propose_tokens`）用了一个技巧：把 token 序列**反转**，
于是「找与后缀匹配的最长 n-gram」变成「找与前缀匹配的位置」，再用 KMP 的 LPS（前缀函数）
一次扫描解决。匹配长度被限制在 `[prompt_lookup_min, prompt_lookup_max]` 内。

→ 因此 N-gram 受益于：**上下文（含已生成 history）里存在可匹配片段**。
仅「初始 prompt 无重叠」不等于「全程无法 draft」——输出自身高重复时，history 仍可供给 draft。
本轮 `low_repeat` 正属此类（见 results § workload 命名）。真正低 Proposal Coverage 的退化区**未覆盖**。

### 1.3 Acceptance、Acceptance Length、Proposal Coverage

- **proposed**：proposer 提出的 draft token 总数；
- **accepted**：verification 通过、被采纳的 draft token 数；
- **rejected**：`proposed − accepted`；
- **Acceptance Rate (Conditional Acceptance)** = `accepted / proposed ∈ [0,1]`；
- **draft-bearing acceptance length** = `1 + accepted / num_drafts`：
  **仅在存在 Draft 的 request-round** 中，平均最终贡献多少正式 output token。
  加的 1 是「该 round 总会产出至少一个正式 token」（全对时的 bonus / 猜错时的纠正）。

它**不是** global tokens per forward，也**不是**理论 speedup。
全局有效进度 `E` 还需计入 no-draft rounds。

**Proposal Coverage**（Batch=1 近似）：`q = num_draft_rounds / execution_rounds`，
且在近似条件下 `E ≈ 1 + q × k × A`。Batch>1 不要直接 `num_drafts / engine_steps`
（统计空间不同）。

最终 speedup 由多因子共同决定（Acceptance 不是充分指标）：

```text
Proposal Coverage + Conditional Acceptance + Verification/Proposer/Rejection Cost
+ Batch/Hardware Utilization + Runtime Path  →  Speedup
```

### 1.4 为什么高 Acceptance、低 Batch、CPU-bound decode 更可能受益

小 batch 的 decode 是**固定开销受限**的（M7 实证）：把「每 round 推进 1 token」换成
「每 round 推进多个 token」，**GPU 侧增量往往很小，而省掉的是多次固定提交开销**。
因此在 CPU/launch-bound 区，减少 execution rounds 更容易转化为 wall-time gain。

### 1.5 为什么大 Batch / 低 Coverage·Acceptance / 高固定开销可能无收益甚至退化

- **Batch 增大**：每 round 的 verification work 增大，一旦越过「固定开销受限 → 算力/带宽受限」
  拐点，多算的 token 就要真金白银付算力；被拒绝的部分是**纯浪费**；还有 tail/straggler。
- **Coverage / Acceptance 低**：提了 k 个只接受很少，等于多算却少推进，还要付 proposer/verification 成本。
- **固定开销高**：proposer、rejection sampler、spec metadata 都有开销；省下的 rounds 不多时可能反噬。

→ **这三条要用数据检验，而不是预设**；本轮在 batch≤16 未触及明确 knee。

---

## 2. O · 源码定位（v0.24.0 / `ee0da84` 真实 file:line，逐条实读核对）

### 2.1 配置

| 环节 | 位置 | 事实 |
|---|---|---|
| 配置类 | `vllm/config/speculative.py:74` `SpeculativeConfig` | pydantic dataclass，`extra="forbid"`（`vllm/config/utils.py:58-59`）→ **拼错 key 会直接报错**，不会被静默忽略 |
| 方法枚举 | `vllm/config/speculative.py:60-69` | `"ngram"` 与 `"ngram_gpu"` 是**两条不同实现**，本轮用前者 |
| 投机 token 数 | `vllm/config/speculative.py:81` | `num_speculative_tokens: int = Field(default=None, gt=0)` |
| n-gram 窗口 | `vllm/config/speculative.py:141` / `:144` | `prompt_lookup_max` / `prompt_lookup_min` |
| **默认值改写（陷阱）** | `vllm/config/speculative.py:633-659` | **两个都不给 → 静默变成 min=max=5**；只给一个 → 另一个跟随；`min > max` → 抛错 |
| Python API 入口 | `vllm/engine/arg_utils.py:621` | `speculative_config: dict[str, Any] \| None = None`，**接受 dict 而不是 dataclass** |
| dict 被原地改写 | `vllm/engine/arg_utils.py:1721-1733` | 会 `update()` 注入 target config → **复用同一个 dict 跑多配置会污染，每次要新建** |

> **实验含义**：做单变量对照时 `prompt_lookup_min/max` 必须**两个都显式写死**，
> 否则上游的默认值改写会让自变量在你不知情的情况下变成 5/5。本脚本已写死 3/5，
> 并从引擎**读回**生效值落盘（`effective_cfg`），由不变量脚本核对。

### 2.2 提议与验收的真实路径

| 环节 | 位置 |
|---|---|
| proposer 构造 | `vllm/v1/worker/gpu_model_runner.py:665-668`（`method == "ngram"` → `NgramProposer`） |
| proposer 调用 | `vllm/v1/worker/gpu_model_runner.py:5088-5099`（ngram 吃 CPU 上的 sampled token，故安排在 bookkeeping **之后**，见 `:4809-4812`） |
| n-gram 匹配算法 | `vllm/v1/spec_decode/ngram_proposer.py:207` `_find_longest_matched_ngram_and_propose_tokens`（反转 + KMP LPS） |
| 窗口/步长参数 | `vllm/v1/spec_decode/ngram_proposer.py:19` `min_n` / `:21` `max_n` / `:25` `k` |
| 验收（rejection sampling） | `vllm/v1/sample/rejection_sampler.py:743-768` greedy kernel |
| **greedy 判据** | `vllm/v1/sample/rejection_sampler.py:756-757`：`token_id = target_argmax_id; rejected = draft_token_id != target_argmax_id` —— 即**逐位置严格相等**；一旦不等，该位置写 target 的 argmax，后续位置作废 |

> ngram 不产生 draft 概率（`draft_probs is None`），所以 `temperature=0` 时走的是最干净的
> greedy 分支：**接受 ⇔ draft token 恰等于 target 的 argmax**。本实验全程 `temperature=0`，
> 正是为了让语义可复现、便于逐位置核对。

### 2.3 计数从哪里来（决定了怎么观测）

**`num_accepted` 不是 kernel 直接吐出的接受掩码，而是 Scheduler 从输出 token 数反推的**
（`vllm/v1/core/sched/scheduler.py:1845-1869`）：

```python
num_draft_tokens = len(scheduled_spec_token_ids)
num_sampled = self.num_sampled_tokens_per_step      # 非 diffusion 模型恒为 1（scheduler.py:204-206）
num_accepted = max(len(generated_token_ids) - num_sampled, 0)
num_rejected = num_draft_tokens - num_accepted
```

统计对象 `SpecDecodingStats`（`vllm/v1/spec_decode/metrics.py:17-49`）：
`num_drafts` / `num_draft_tokens` / `num_accepted_tokens` /
`num_accepted_tokens_per_pos` / `num_draft_tokens_per_pos`。
注意 **`num_drafts` 是「本 step 中出过 draft 的请求数」**，不是 draft token 数。

三道 `log_stats` 门禁，关掉就什么都读不到：
`scheduler.py:2597`（`make_spec_decoding_stats`）、`scheduler.py:2558`（`make_stats`）、
`llm_engine.py:452`（`get_metrics`）。

### 2.4 in-process 怎么读到（本实验采用的方式）

| 路径 | 结论 |
|---|---|
| `engine.step()` 返回值 | **拿不到**。`llm_engine.py:407` 只返回 `request_outputs`，`scheduler_stats` 在 `:400-404` 喂给 logger 后即丢弃 |
| `engine.get_metrics()` | 可以，但是**进程级累计的 Prometheus counter**，读取不清零，按样本切窗口要自己前后做差 |
| `engine_core.engine_core.scheduler` | **拿不到**。`spec_decoding_stats` 是 `scheduler.py:1782` 的局部变量，每 step 新建、交出即弃 |
| **自定义 `StatLoggerBase`** ✅ | **本实验采用**。`vllm/v1/metrics/loggers.py:44-71` 是上游公开的扩展点，经 `LLMEngine.from_engine_args(stat_loggers=[...])`（`vllm/v1/engine/llm_engine.py:215`）注册；拿到的是**per-step 原始 `SpecDecodingStats`**，可精确归属到每个样本 |

自定义 logger 会被 `PerEngineStatLoggerAdapter` 以 `factory(vllm_config, engine_index)` 构造
（`vllm/v1/metrics/loggers.py:376-378`），因此实验代码用类变量登记单例取回。
**全程不修改任何 vLLM 产品源码。**

### 2.5 两个会变成隐藏变量的开关（必须两侧都写死）

1. **async scheduling**：`vllm/config/vllm.py:976-986` —— 当 `async_scheduling is None` 且
   方法是 ngram 时，上游会**自动把它关掉并 warning**。若 baseline 侧留 `None`（可能被开启），
   开 ngram 就顺带改变了 scheduler 行为 → 对照不成立。
   **本实验两侧都显式 `async_scheduling=False`。**
2. **`disable_log_stats`**：投机计数依赖它。若只有 ngram 侧打开统计，两侧的统计开销就不对称。
   **本实验两侧都 `disable_log_stats=False`。**

另：`enable_prefix_caching=False`（两侧）——本实验重复跑相同 prompt，开着 APC 会让后续 repeat
命中缓存，同时污染 prefill 成本与 ngram 的上下文。

### 2.6 与 `enforce_eager` 的关系

未发现任何不兼容：`SpeculativeConfig.enforce_eager`（`speculative.py:78`）只被 draft-model 类
方法消费，ngram 不走那些分支；上游自带的 ngram 测试本身就用 `enforce_eager=True`
（`tests/v1/spec_decode/test_max_len.py:25`）。

---

## 3. 冻结实验设计（跑前冻结）

### 3.1 引擎配置（两侧严格单变量）

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct`（本地已有，**未下载新模型**） |
| 执行模式 | `enforce_eager=True` |
| `max_model_len` / `gpu_memory_utilization` / `max_num_seqs` | 4096 / 0.6 / 64 |
| `async_scheduling` / `disable_log_stats` / `enable_prefix_caching` | False / False / False（**两侧相同**） |
| 采样 | `temperature=0`、`seed=0`、`ignore_eos=True` |
| **唯一变量** | `speculative_config`：baseline 为 `None`；ngram 为下表 |

```python
{"method": "ngram", "prompt_lookup_min": 3, "prompt_lookup_max": 5,
 "num_speculative_tokens": 3}
```

（照抄上游自带测试的可用组合：`tests/v1/e2e/spec_decode/test_spec_decode.py:165-170`。）

### 3.2 网格

| 维度 | 取值 |
|---|---|
| config | baseline / ngram |
| workload | `high_repeat` / `low_repeat` |
| batch（并发请求数） | 1 / 4 / 16 |
| 输出长度 | 128（`ignore_eos=True`，恰好等于此值） |
| prompt 长度 | **两个 workload 都恰好 256 token**（隔离 prefill 成本差异） |
| 重复 | 每组 **5 warmup + 15 measure**，汇总只用 `phase=measure and success=True` |

合计 2×2×3 = 12 组 × 15 = **180 个 measure 样本**。

### 3.3 两类 workload（原名保留；语义以实测为准）

| workload（原名） | 构造 | 实测语义 |
|---|---|---|
| `high_repeat` | 一个 27-token 的结构化句式重复 40 遍后截断到 256 | **高初始 prompt 重叠 / 极 N-gram-friendly**（`output_4gram_in_prompt_ratio=1.00`） |
| `low_repeat` | 22 句内容互不相关的中文陈述，**只用一遍**截断到 256 | **低初始 prompt 重叠**（ratio=0.00），但输出自身仍高重复（`output_repetition_4gram≈0.82–0.84`）；**不是**真正低可预测性 / N-gram hostile |

> ⚠ **这里踩过一个坑并已修正**：`low_repeat` 首版是把 12 句文本 **×3** 凑到 256 token，
> 结果同一段话在 prompt 里出现三遍 —— 「低初始重叠」对照根本不成立，smoke 实测 acceptance
> 也是 1.0。已改为一次就够长的 22 句文本。**这个缺陷是 smoke 抓出来的，不是事后发现的。**
>
> 另：N-gram 使用 **prompt + generated history**，因此即便初始 prompt 无重叠，自重复输出仍可 draft。
> **真正低 Proposal Coverage / 低 Acceptance 退化区未覆盖。**

### 3.4 正交的第二证据源

acceptance 来自 vLLM 内部的 scheduler 统计。为避免「用引擎计数解释引擎计数」，
另记两个**完全由落盘 token 序列离线算出**的指标：

- `output_repetition_4gram`：输出自身的 4-gram 重复度；
- **`output_4gram_in_prompt_ratio`**：输出的 4-gram 有多大比例已出现在 prompt 里
  —— 这正是 ngram 能否命中的直接前提。

若某 workload 的 acceptance 高而该比例低（或反之），说明机制解释站不住，必须回查。

### 3.5 不变量（跑后逐条核对，已固化进 `scripts/check_invariants.py --module m19`）

- 两侧使用**完全相同的 prompt**（`batch_prompt_hash` 相等）；
- 每个请求的输出长度**恰好** 128；
- `0 <= accepted <= proposed`；`acceptance_rate ∈ [0,1]`；
- **baseline 侧 proposed/accepted/num_drafts 恒为 0**，且 `steps/token ≈ 1.0`；
- `proposed == num_drafts × k`；
- 逐位置计数之和 == 总数（`Σ accepted_per_pos == accepted`，draft 同理）；
- 生效配置**从引擎读回**且等于冻结值（防上游默认值改写）；
- 两侧除 `speculative_config` 外生效配置逐字相同（单变量）；
- 每组 measure 样本数 = 15，warmup 不进汇总；
- 输出差异**如实记录**（见 §3.6），不静默忽略。

### 3.6 输出正确性怎么判

greedy + rejection sampling 在**精确算术**下应当与 baseline 输出一致，但两侧每步的
batch 形状不同（baseline 每步算 B 个 token，ngram 每步算至多 B×(k+1) 个），
浮点不满足结合律 → **即使实现完全正确，也不保证逐 token 位级相同**
（M5 已观察到同类现象：compiled 与 eager 的贪心输出从第 5 个 token 起分叉，是数值差异而非 bug）。

因此本轮**不预设「必须完全一致」**，而是：

1. 用独立的 `--shard equiv` probe 落盘**完整 output token ids**，逐请求算首个分歧下标；
2. 硬红线只保留两条：**输出长度必须一致**、**分歧不得出现在第 0 位**；
3. 增加**正交对照 `batch_shape_control`**：在**同一侧内部**只改 batch 组成，看输出是否也会分歧
   —— 若 baseline 自己分歧，则证明 batch-shape **可以**导致 divergence，并与
   baseline↔ngram 现象一致；这**支持**「现有证据未发现明确 Spec Decode bug」，
   但**不是**「已严格证明每一个 divergence 都与 Spec Decode 无关」；
4. 同时验证同配置跨 repeat 完全可复现，排除「随机性」这一竞争解释。

---

## 3.7 主线退出标准（已满足）

| 等级 | 内容 | 状态 |
|---|---|---|
| **T** | proposer / verification / rejection；Scheduler rollback；跨-step speculative lifecycle；Proposal Coverage / Acceptance / Round Cost；适用与退化条件 | **PASS** |
| **E** | Baseline vs N-gram 正式实验；Acceptance / per-position；execution rounds；completion；amortized TPOT；throughput；memory/KV；output equivalence；Bootstrap CI；invariants | **PASS** |
| **O** | N-gram proposer；proposer invocation；rejection sampler；scheduler spec stats；configuration semantics；logging/observability path（均 file:line） | **PASS** |

→ **N-gram Speculative Decoding 主线达到 T+E+O。** Optional proposer extensions 见 `results.md` §7。

---

## 4. 复现

```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
nvidia-smi          # 先人工确认 GPU 空闲

# 自检（很小，确认 ngram 真的生效、计数字段可观测）
python scripts/m19_spec_decode.py --shard smoke --mode ngram
python scripts/m19_spec_decode.py --shard smoke --mode baseline

# 正式（一个进程一个引擎；两次之间等显存释放）
python scripts/m19_spec_decode.py --shard baseline
python scripts/m19_spec_decode.py --shard ngram
python scripts/m19_spec_decode.py --shard summarize          # 不建引擎、不需 GPU

# 输出一致性 probe（存完整 token ids）
python scripts/m19_spec_decode.py --shard equiv --mode baseline
python scripts/m19_spec_decode.py --shard equiv --mode ngram
python scripts/m19_spec_decode.py --shard summarize-equiv    # 不建引擎

python scripts/check_invariants.py --module m19
python scripts/m19_figures.py                                # 离线出图
```

## 5. 产出文件

- 逐样本：`outputs/m19/raw/{m19_baseline,m19_ngram,m19_equiv_baseline,m19_equiv_ngram}.jsonl`
- CSV：`outputs/m19/{spec_decode_summary,spec_decode_effect,acceptance_by_position,output_equivalence,output_equivalence_detail,batch_shape_control}.csv`
- 日志：`logs/m19/*`（含 `environment.txt`）
- 图：`figures/{fig7_m19_acceptance,fig8_m19_speedup}.png`
- 结论：[`results.md`](results.md)

**零 vLLM 产品源码改动**（只用了上游公开的 `stat_loggers=` 扩展点）。
