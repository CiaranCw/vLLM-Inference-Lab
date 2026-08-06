# M17 · 实测结论（E1 APC / E3-single 分块 / E3-concurrency ITL）

> **状态**：**已按审计修复并全部重跑**（2026-08-05）。设计/源码定位/不变量见同目录
> `prefix_cache_and_chunked_prefill.md`；本文件只放结果与结论。
> **⚠ 本文件已取代 2026-08-04 的首版结论**：首版 E3 每配置只有 **1 个样本且无 warmup、
> 被首次 Triton JIT 编译污染**，其「+5% / +15.6% / ≈3×7ms」性能结论**已撤回**（见 §6）。
> 旧数据保留在 `backups/m17_pre_fix_20260805T032832Z/`，未删除。

## 环境（`logs/m17/environment.txt`）
| 项 | 值 |
|---|---|
| vLLM | `0.24.1.dev0+gee0da84ab` ← `/home/ciaran/projects/vllm-src`（**`ee0da84` / v0.24.0**） |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| GPU | RTX 5080（sm_120, capability (12,0)），driver 610.62 |
| torch / CUDA / Python | 2.11.0+cu130 / 13.0 / 3.12.13 |
| 执行模式 | `enforce_eager=True`，`seed=0`，`temperature=0` |
| 采样 | 每配置 **5 warmup + 15 measure**；**汇总只用 measure 且 success=True** |

**指标口径（勿与 M12 混淆）**：本文件所有延迟均为 **in-process engine 路径**测量，
字段名 `in_process_first_token_latency_ms`。**不含** HTTP / 网络 / tokenization /
异步服务队列 / 跨请求排队。**在线 TTFT 口径见 M12**。

---

## 1. E1：APC（长共享前缀 on/off）

数据：`outputs/m17/prefix_cache_sweep.csv`（每组 n=15）、`outputs/m17/prefix_cache_effect.csv`

| P | off median | hot median | **median 差** | **Bootstrap 95% CI** | CI 排除 0？ | 相对降幅 | off sd | hot sd |
|---:|---:|---:|---:|---|:--:|---:|---:|---:|
| 256 | 7.02 | 7.57 | −0.55 | **[−1.16, +0.10]** | ❌ 否 | −7.8%（hot 反而略慢） | 0.49 | 0.80 |
| 512 | 8.21 | 8.15 | +0.05 | **[−0.65, +1.30]** | ❌ 否 | 0.6% | 0.91 | 0.80 |
| **1024** | 12.19 | 8.24 | **+3.95** | **[+3.26, +4.49]** | ✅ **是** | **32.4%** | 0.42 | 0.84 |
| **2048** | 22.59 | 9.39 | **+13.20** | **[+12.75, +13.86]** | ✅ **是** | **58.5%** | 0.38 | 0.88 |

> 数据：`outputs/m17/prefix_cache_effect.csv`（**2026-08-05 第二轮**，字段修正后重跑）。
> **跑间波动提示**：同日第一轮同一实验得 P=1024 +4.95ms（−39.9%）、P=2048 +14.20ms（−61.3%）；
> 两轮**方向与显著性完全一致**（1024/2048 显著、256/512 不显著），但绝对值有约 1–3ms 的跑间漂移。
> 因此下文只对「显著/不显著」与量级下结论，不把某一位小数当稳定常数。

> 统计口径：median difference 的 **Bootstrap 95% CI**（10000 次重采样，固定 `seed=0`，
> 仅用 stdlib `random`+`statistics`，无重依赖）。**不使用 σ 表述**（首版的 "3.5σ/8.8σ"
> 不是正式标准化统计量，已删除）。

### 1.1 可支持的结论
- **P=1024 / 2048：APC 的 in-process 首 token 延迟收益真实存在**（CI 明确排除 0），本轮降
  **32.4% / 58.5%**（第一轮 39.9% / 61.3%，方向与显著性一致）。
- **P=256 / 512：在本配置与当前样本量（n=15）下，未观察到超出运行噪声的稳定延迟收益**（CI 含 0）。
  → **这不等于「完全没有收益」**，只是本实验无法区分它与噪声；本轮 P=256 的点估计甚至为负
  （hot 比 off 慢 0.55ms，但 CI 含 0）。

### 1.2 机制（与 M7 打通）
off 侧 median 随 P 的斜率（新数据重算）：

| 区间 | 斜率（本轮） | 第一轮 |
|---|---:|---:|
| 256 → 512 | **+4.62 us/tok** | +1.09 |
| 512 → 1024 | **+7.78 us/tok** | +8.36 |
| 1024 → 2048 | **+10.16 us/tok** | +10.52 |

- **短 prompt 区斜率明显偏低**（两轮分别 +1.09 / +4.62 us/tok，远低于长区的 ~10 us/tok）：延迟被
  **M7 已证的 ~7ms 级 CPU kernel 提交开销**摊薄 → 此时省掉计算换不出可测的延迟。
  （该区斜率本身跑间波动较大，故只说「明显偏低」，不引用某一具体值。）
- **长 prompt 区转为稳定线性增长**（+7.78 → +10.16 us/tok，两轮一致）：进入计算受限区 → 省掉的计算
  才直接变成省下的时间。
- **hot 侧几乎不随 P 增长**（本轮 7.57 → 8.15 → 8.24 → 9.39 ms）：命中后每次只算 **16 个尾 token**。

→ **APC 的收益必须等 prompt 长到穿透那条固定开销地板才显现。**

### 1.3 不变量（I1–I3，180 个 measure 样本全过）
| 不变量 | 结果 |
|---|---|
| **I1** `hit == floor(P/16)*16` | ✅ 四档 hit 集合分别为 {256}/{512}/{1024}/{2048} |
| **I2** `hit == actual − scheduled` | ✅ **0/180 违反** |
| **I3** APC off → `hit == 0` | ✅ 60 个 off 样本 hit 集合 = {0} |

命中后 `scheduled_prompt_tokens` 恒为 **16**（= 未命中的后缀长度）→ 直接证明命中的前缀**完全跳过** prefill 计算。

**⚠ 字段修正（审计发现）**：E1 原先记录 `chunked = 0 <= scheduled < actual`，这会把**全部 60 个 hot
样本误标为 `chunked=True`**——但它们各自只跑 **1 个 prefill step**，`scheduled < actual` 的原因是
**prefix 命中**，不是「未命中部分被切成多个 chunk」。已删除该字段，改为两个语义明确的字段：
`prefix_cache_hit`（hit_tokens > 0）与 `prefill_step_count`（E1 恒为 1），并已**重跑 E1 使数据落实**
（实测 hot: `prefix_cache_hit=True, prefill_step_count=1`；cold/off: `prefix_cache_hit=False`）。

### 1.4 token 构造已加断言（审计要求）
每个 cold/hot pair 断言：`len(cold)==len(hot)==P+16`、`cold[:P]==hot[:P]`、`cold[P:]!=hot[P:]`；
每 repeat 用独立前缀族（`族{P}_{r}：`）避免跨 repeat 意外命中；记录 `prefix_hash`/`suffix_hash`（不写全量 ids）。

---

## 2. E3-single：长 prompt 分块（重复测量，已排除 JIT）

数据：`outputs/m17/chunked_prefill.csv`（prompt_len=2048、APC off、每组 n=15）

| chunked | mbt | prefill steps | chunk_sizes | Σ | median | sd | p95 | min | max |
|:--:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| False | 4096 | 1 | `[2048]` | 2048 | **23.58** | 0.40 | 24.23 | 23.12 | 24.27 |
| True | 512 | **4** | `[512,512,512,512]` | 2048 | **32.31** | 1.33 | 34.75 | 30.79 | 35.37 |
| True | 1024 | **2** | `[1024,1024]` | 2048 | **25.55** | 0.81 | 26.85 | 24.78 | 27.41 |
| True | 2048 | 1 | `[2048]` | 2048 | **22.59** | 0.71 | 23.93 | 22.07 | 24.38 |
| True | 4096 | 1 | `[2048]` | 2048 | **23.60** | 0.46 | 24.29 | 22.95 | 24.43 |

### 2.1 I4 分块守恒 ✅
**每个配置的 Σchunks 恒为 2048**，每块 ≤ 预算，步数恰为 `ceil(2048/预算)`（512→4、1024→2、≥2048→1）。
**prefill 与 decode 已分离统计**：按每请求 `num_scheduled_tokens` 累计到 prompt 算完为止才算 prefill chunk，
之后每步 1 token 计入 `decode_steps`——不再把 decode step 误计为 prompt chunk（首版缺陷）。

### 2.2 单请求下分块的固定成本（新数据，n=15）
以 `chunked on, mbt=2048`（1 步，22.59ms）为同侧基准：
- **2 块（mbt=1024）**：25.55ms，**+2.96ms**
- **4 块（mbt=512）**：32.31ms，**+9.72ms**

→ 每多一个 prefill step ≈ **+3.0 ~ 3.2ms**（`(32.31−22.59)/3 = 3.24`、`(25.55−22.59)/1 = 2.96`）。
量级与 M7 所证的「每 step 固定 CPU 提交开销」一致，但**本轮实测的 per-step 成本约 3ms，
不是首版所写的 ~7ms**——首版那个数字来自被 JIT 污染的单样本，已撤回。

**边界**：这是**单请求**测量。分块的真实价值在多请求并发（见 §3），**不能据此说 chunked prefill 有害**。

---

## 3. E3-concurrency：Decode ITL 抖动（**本轮新增，回答 chunked prefill 的核心卖点**）

数据：`outputs/m17/chunked_prefill_concurrency.csv`、`outputs/m17/raw/e3_conc_*.jsonl`（每侧 n=15）

**构造**：① 加入 decode 请求 D（`max_tokens=96`，`ignore_eos`）→ 跑 4 步进入稳定 decode；
② 插入长 prompt 请求 P（2048 token）；③ 记录 P 运行期间 **D 每个输出 token 的时间戳** → ITL。

**单变量公平性（已核实生效配置）**：
| | chunked | nochunk |
|---|---|---|
| `max_num_batched_tokens` | **4096** | **4096**（相同） |
| `enable_chunked_prefill` | True | False |
| `long_prefill_token_threshold` | **512** | 0 |
| `max_num_partial_prefills` | 1 | 1（相同） |
| 实测 P 的 prefill 步数 | **4** × 512 | **1** × 2048 |
| **D 的 ITL 样本数/repeat** | **92** | **92**（相同，可公平比较） |

### 3.1 结果（median difference + Bootstrap 95% CI，nochunk − chunked）

| 指标 | nochunk | chunked | 差值 | CI95 | 显著？ |
|---|---:|---:|---:|---|:--:|
| **D 的 ITL max**（每 repeat 最大 stall） | **22.87** | **10.51** | **+12.36** | **[+11.84, +12.60]** | ✅ **显著** |
| D 的 ITL p95 | 9.30 | 9.19 | +0.12 | [−0.06, +0.41] | ❌ 不显著 |
| D 的 ITL median | 7.58 | 7.54 | +0.04 | [−0.11, +0.12] | ❌ 不显著 |
| **长 prefill 的 TTFT** | **22.87** | **33.69** | **−10.82** | **[−11.83, −10.28]** | ✅ **显著（代价）** |

（另：`itl_max_worst` chunked 13.41ms vs nochunk 23.49ms；total latency median 708 vs 720ms。）

### 3.2 可支持的结论 ⭐
**在本配置下，chunked prefill 确实把并发 decode 的最坏 stall 砍掉约一半**（22.87 → 10.51ms，
**−54%**，CI 明确排除 0），代价是**长 prefill 自身 TTFT 变慢 47%**（22.87 → 33.69ms）。

**且它只改善尾部、不改善中位**：ITL median / p95 的 CI 都含 0（几乎不变）。
→ 这正符合 chunked prefill 的设计意图：**用长请求自身的 TTFT，换取其他请求 decode 的平滑性**。
这条因果证据是首版缺失的核心内容。

### 3.3 机制（以原始逐步记录为准）
先看实测的 `steps_detail`（`outputs/m17/raw/e3_conc_*.jsonl`，长 prefill 占用的 step）：

| mode | 该 step 的 `d`（D 被调度的 token） | `p`（长 prefill） | `total` |
|---|---:|---:|---:|
| nochunk | **1** | 2048 | **2049** |
| chunked | **1**（× 4 步，每步都有） | 512 | 513 |

**准确说法**：nochunk 时 **D 仍与长 prefill 同批被调度了 1 个 decode token**（`d=1`），
**但这个 token 的返回被这个「2049-token 的长联合 step」整体推迟** → D 观察到的最大 ITL ≈ 该 step
时长（22.87ms，与 §2 中 `[2048]` 单步的 23.58ms 吻合）。
chunked 时长 prefill 被切成 4×512，每个 step 都是 513 token 的小联合 step，D 同样每步拿到 1 个
token，但**每次等待的 step 变短** → 最大 ITL 缩到约一个 512-chunk step 的时长（10.51ms）。

→ 所以 chunked prefill 改善的**不是「D 是否被调度」，而是「D 的 token 要等多长的一个 step 才返回」**。
（⚠ 本文件首版曾写「D 在这一步里拿不到任何输出」，与逐步记录 `d=1` 矛盾，**已修正**。）

---

## 4. 上游硬约束（重新验证）

`chunked=off` + `max_num_batched_tokens=512` 仍被配置校验拒绝（`logs/m17/e3_off_512_expected_reject.log`）：

```
ValueError: max_num_batched_tokens (512) is smaller than max_model_len (4096).
```

源码 `vllm/config/scheduler.py:279`（条件在 `:268` 起）：
```python
if (self.max_num_batched_tokens < max_model_len
        and not self.enable_chunked_prefill):
    raise ValueError(...)
```

**表述（按审计收紧）**：这是**当前 vLLM 的配置语义要求** —— 关闭 chunked prefill 时，scheduler 必须
能在**单步**容纳最大序列长度，否则长请求会被静默拒绝，故上游在配置层直接报错。
→ 因此「off + 小预算」这个对照**无法构造**，off 侧只能用 `mbt=4096`。
**不把它夸大为 chunked prefill 性能价值的完整证明**（性能价值由 §3 单独给出）。

---

## 5. 与 M3 / M6 / M7 的关系

| 旧结论 | 本轮 |
|---|---|
| **M6**：prefix 命中省 94% 计算 token，但首 token 延迟改善**落在噪声内** | **深化**：仅在**短前缀**成立。M6 用 body=256（正落在固定开销地板区）；本轮 1024/2048 的收益 CI 明确排除 0。 |
| **M6**：prefix **ON 短 prompt 反略慢于 OFF** | **部分复现（见下）**：本轮（第二轮）cold 与 off 的差值为 P=256 **+0.60ms（CI [+0.22,+0.98]，显著，cold 更慢）**、P=512 −0.08、P=1024 +0.38、P=2048 +0.25（后三档 CI 均含 0，不显著）。→ **在 P=256 上确实复现了「APC 开启但未命中时略慢」**，量级 <1ms；其余档次无法与噪声区分。⚠ 本文件首版曾写「4 档 cold median 全部 ≤ off，P=256 cold=7.42」，**该陈述与 CSV 不符（第一轮实际 cold=8.0051 > off=7.8465），已修正**；且两轮 cold−off 的符号并不稳定，说明多数档次确实是噪声。 |
| **M7**：decode ~7ms 地板 = CPU 提交开销 | **复用为解释工具**：同一固定开销掩盖了短前缀 APC 收益（§1.2），也解释 §2 每 step ≈3ms 的分块成本与 §3 的 stall 量级。 |

---

## 6. 首版（2026-08-04）已撤回的结论

| 首版写法 | 问题 | 现状 |
|---|---|---|
| E3「+5% / +15.6% TTFT」「≈多付 3 次 ~7ms 固定开销」 | **每配置仅 1 个样本、无 warmup、被首次 Triton JIT 污染**（实测首个 warmup 样本 128–145ms vs measure max 23–35ms，慢 4–5×） | **撤回**。重跑 n=15 后，per-step 成本约 **3.0–3.2ms**（§2.2） |
| 「3.5σ / 8.8σ」 | σ 非正式标准化统计量，且 P=1024 的数字与表中 stdev 不自洽 | **删除**，改用 **Bootstrap 95% CI**（§1） |
| 「ttft_proxy_ms」含糊近似在线 TTFT | 口径不清，易与 M12 在线 TTFT 混淆 | 改名 **`in_process_first_token_latency_ms`**，并显式声明不含 HTTP/队列等（§环境） |
| `chunked = scheduled > actual` | **判断方向反了**（分块时 scheduled < actual） | 修为 `0 <= scheduled < actual`；请求是否真跨多步以 **prefill step 数 > 1** 为准 |
| E3 计时 `t0` 在 `synchronize()` 之前 | 把先前排队的 GPU 工作计入本次测量 | 修为**先 sync 再取 t0**，收尾同样先 sync |
| 「M17 全部完成」 | 并发 ITL 实验（冻结设计要求）当时未做 | 本轮**已完成**（§3） |
| 逐 step 把所有 `total_num_scheduled_tokens>0` 当 prompt chunk | 会把 decode step 误计为 chunk | 改为按**每请求** `num_scheduled_tokens` 且只累计到 prompt 算完 |

| 「D 在长 prefill 那一步拿不到任何输出」 | 与原始 `steps_detail` 矛盾（实测 `d=1, p=2048, total=2049`，D 确被同批调度） | **改为**「D 仍被调度 1 个 token，但返回被长 step 推迟」（§3.3） |
| E1 记录 `chunked = 0 <= scheduled < actual` | 把 **60/60 hot 样本误标 chunked=True**；实际只跑 1 个 prefill step，`scheduled<actual` 是**prefix 命中**所致 | **删除该字段**，改 `prefix_cache_hit` + `prefill_step_count`，并**重跑 E1** 落实（§1.3） |
| 「4 档 cold median 全部 ≤ off，P=256 cold=7.42」 | **事实错误**：第一轮 CSV 实为 cold 8.0051 > off 7.8465；7.42 不存在于该轮数据。且我的"14/14 自动核对"只覆盖主结论数字，**未覆盖这句正文** | 改为按 CI 陈述（§5），并承认自动核对存在覆盖盲区 |
| 设计笔记 §3 仍写 `chunked = scheduled>actual` | 与已修正的脚本相互矛盾 | 同步修正设计笔记 |
| I5 被整项勾选 `[x]` | 冻结的 I5 含**定量实验**（扫 `max_tokens` 看相对收益下降），本轮未做 | 拆成两项：**理论边界=完成 / 输出长度定量实验=未执行** |

**另修复一个本轮新代码里的隐患**：vLLM 会给 `add_request` 的 id **追加后缀**（`"p"` → `"p-8ef2061f"`），
若用 `num_scheduled_tokens.get(rid)` 精确查找会**静默返回 0**，导致所有 per-request 计数失效。
已改为前缀容错匹配（`_same_req`），并在探针中实测确认（`[512,512,512,512]`）。

---

## 7. 局限（如实）
- **单卡 / eager / 0.5B / WSL2 / in-process**：所有绝对值限本配置。**不外推 compiled**（CUDA Graph 会
  改变每 step 固定开销，从而移动 §1.2 的所有分界线）、**不外推大模型 / 多卡 / 在线服务**。
- **非在线服务**：无 HTTP / 网络 / 异步队列；在线 TTFT 口径见 M12。
- **E3-concurrency 仅 1 个 decode 请求 + 1 个长 prefill**；未扫并发度、未扫 prompt 长度组合。
- **`max_num_partial_prefills` 无法调**：本版本 `arg_utils.py:2350` 对非默认值直接
  `NotImplementedError("Concurrent Partial Prefill")`，故 chunked 侧靠 `long_prefill_token_threshold`
  单独强制分块（已实测生效）。
- **E2（block 生命周期 ref_cnt/驱逐内省）仍未做**（可选 TODO，需先验证 `block_pool` 内省 API 可读）。
- APC 的 `hit_tokens` 用 `actual − scheduled` 间接推断（M6 已验证可靠），未直接读 block hash 表。
- E1 短前缀（256/512）结论是"未观察到超出噪声的稳定收益"，**不是"没有收益"**。
- **I5 只完成了理论边界确认，未做定量实验**：冻结设计要求「加长 `max_tokens` 时 APC 的 E2E 相对收益
  应变小」，本轮所有请求都用 `max_tokens=1`（只测 prefill），**未扫输出长度** → 该定量实验**未执行**。
- **跑间波动**：同一实验同日两轮，P=1024/2048 的 median 差有约 1–3ms 漂移（方向与显著性一致）；
  cold−off 的符号在两轮间甚至反转 → 亚毫秒级差异不可当稳定结论。
- **文档数字自动核对有覆盖盲区**：上一轮报告称"文档数字与 CSV 14/14 一致"，但那 14 项只覆盖主结论
  表格，未覆盖正文叙述句（正是 cold/off 那句出错的地方）。本轮已把正文关键数字一并纳入核对。

---

## 8. 复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
# 一个进程只建一个引擎（同进程内多引擎无法归还 KV 显存）
python scripts/m17_prefix_cache.py --shard e1-on
python scripts/m17_prefix_cache.py --shard e1-off
for mbt in 512 1024 2048 4096; do
  python scripts/m17_prefix_cache.py --shard e3 --e3-chunked on --e3-mbt $mbt; sleep 20; done
python scripts/m17_prefix_cache.py --shard e3 --e3-chunked off --e3-mbt 4096
python scripts/m17_prefix_cache.py --shard e3-conc --conc-mode chunked
python scripts/m17_prefix_cache.py --shard e3-conc --conc-mode nochunk
python scripts/m17_prefix_cache.py --shard summarize   # 不建引擎、不需 GPU
```
（`--shard e3` 缺 `--e3-chunked/--e3-mbt` 会直接报错退出，防止同进程建多引擎。）

## 9. 产出文件
- CSV：`outputs/m17/{prefix_cache_sweep,prefix_cache_effect,chunked_prefill,chunked_prefill_concurrency}.csv`
- 逐样本：`outputs/m17/raw/{e1_on,e1_off,e3_single_*,e3_conc_*}.jsonl`（含 `phase`/`success`/`metric_scope`）
- 日志：`logs/m17/{environment.txt,e1_on.log,e1_off.log,e3_single_*.log,e3_conc_*.log,summarize.log,e3_off_512_expected_reject.log}`
- 旧数据备份：`backups/m17_pre_fix_20260805T032832Z/`、`outputs/m17/raw_old_20260805/`（均未删除）
