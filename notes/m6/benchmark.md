# M6 · 离线 / 闭环引擎 Benchmark（★核心验收 #3）

> 目标：把能力从"能解释执行路径"推进到**"能可信测量执行路径"**。在**固定环境**下，用**进程内闭环**方式测量 prefill / decode / 吞吐 / prefix / KV 与显存，并对 **eager vs compiled**（接 M5）做性能对照。
> **边界（务必牢记）**：本模块**只测量 + 解释现象**，**不做 profiler 根因、不断言 compute/memory-bound、不改算法、不做在线 serving、不下跨机跨模型结论**（留 M7 及后续）。所有数字**仅在本机、本模型、本配置下成立**。
>
> 证据：`outputs/m6/baseline.csv`（汇总）、`outputs/m6/raw/*.jsonl`（逐样本，可回查）、`outputs/m6/config/*_meta.json`（引擎/环境/KV/显存指纹）、`outputs/m6/config/diag_modes_*.jsonl`（逐长度/逐 batch 实际执行模式，正式诊断入口）、`outputs/m6/summary/summary.json`（含 prefix 配对分析）、`logs/m6/*.log`。复现命令见文末。
>
> 注：本文数字来自**修订版重跑**（decode 改为 window=1 使 seq_len 严格恒定；补跑 prefix OFF 对照；补逐 batch/逐长度执行模式诊断）。修订点见文末「审查修复记录」。

---

## 0. 固定环境指纹
RTX 5080(sm_120, 16 GiB) / torch 2.11.0+cu130 / vLLM `ee0da84`(v0.24.0) / Qwen2.5-0.5B-Instruct / bf16 / seed=0 / temperature=0 / TP=1 / 进程内(`VLLM_ENABLE_V1_MULTIPROCESSING=0`) / `async_scheduling=False` / `VLLM_USE_FLASHINFER_SAMPLER=0`。引擎参数：`max_model_len=2048, max_num_seqs=64, max_num_batched_tokens=4096, gpu_memory_utilization=0.6`。

## 1. 指标定义与测量边界（先说清楚"测的是什么"）
统一计时：`torch.cuda.synchronize()` → `perf_counter()` → `engine.step()` → `torch.cuda.synchronize()` → `perf_counter()`。它覆盖**整个 engine step**：调度、worker 调用、输入张量准备、forward、attention、logits、sampling、状态更新、RequestOutput 构造（含 detokenize）。因此：

| 指标 | 含义 |
|---|---|
| `prefill_step_latency` / `engine_first_token_latency` | 单请求首步（整段 prompt 一次 prefill + 出第 1 token）的 step 耗时 |
| `decode_step_latency` | 稳态 decode 单步 step 耗时（batch 固定、每请求 1 token） |
| `decode_output_tokens_per_second` | `= batch / decode_step_seconds`，稳态 decode token 吞吐 |

**这些是进程内闭环 engine 指标**，不含 API server / HTTP / 网络 / 排队 / 流式 / 客户端接收。**`engine_first_token_latency` ≠ 在线 TTFT**（在线 TTFT/ITL/E2E 留 M12）。

## 2. 方法学（保证可信）
- 进程内手动 `LLMEngine` + `step()` 循环；**正式计时区间无 trace、无文件写、无打印**。
- prompt 用**精确 token 数**（tokenizer 编码后裁剪、直接传 `prompt_token_ids` 绕过 chat template）；每样本断言 `actual==target`。
- 只读包裹 `scheduler.schedule` 仅存最近 `SchedulerOutput` 引用；校验信息（`total_num_scheduled_tokens`、running 数、状态）在**计时区间之外**读取。
- **预热与正式分离**：prefill 每长度 warmup 5 + 有效样本 15；decode 每 batch 15 独立 repeat（方案 A：每 repeat 重建 batch→prefill→4 步 decode warmup→**只测 1 个固定位置的 decode step**，故 **seq_len 严格恒=38**，跨 repeat 完全可比）。
- 统计报 median/mean/stdev/p95/min/max；**保留全部逐样本原始数据**。本次共 535 样本，**465 有效 measure，70 dropped 全为 prefill warmup，0 个无效 measure**。
- 每种 (mode,prefix) 配置**独立子进程**运行（防 CUDA context / allocator / CUDA Graph 池 / torch.compile 缓存污染）。
- 执行模式（FULL/PIECEWISE/NONE）由**正式诊断入口** `--experiments diag`（临时开 exec trace 的 throwaway step，不进入计时样本）**逐长度、逐 batch**确认，结果落 `outputs/m6/config/diag_modes_<tag>.jsonl`。

## 3. KV Cache 容量 / 显存构成（每项标来源；区分"容量等价值"与"运行时上限"）

**配置直读**（`vllm_config.cache_config`），**eager 与 compiled 不同**：
| 配置 | num_gpu_blocks | KV 容量(tokens) | KV 最大并发等价值 |
|---|---:|---:|---:|
| eager+off | **45,737** | 731,792 | 357.3× |
| compiled+off | **45,118** | 721,888 | 352.5× |

- **compiled 的 KV 块更少**：因 compiled 额外占 **0.10 GiB CUDA Graph 池**（日志 "CUDA graph pool memory"），KV 可用显存相应减少 → 少 ~619 块 / ~1 万 token。这是"用少量显存换 CUDA Graph 加速"的直接代价。
- **⚠️ "最大并发 357×" 是 KV 容量等价值，不是运行时并发上限**：它表示"若每请求占满 2048 上下文，KV 池最多容 357 条"。**实际同时 running 的请求数被 `max_num_seqs=64` 硬性限制**（本 benchmark 的 decode 最大 batch 也就取到 64）。两者含义不同：357× 是"KV 装得下多少"，64 是"调度器允许同时跑多少"。

**KV 每 token 字节（三路交叉验证）**：
| 来源 | 每 token 字节 | 说明 |
|---|---:|---|
| 实测（KV tensor `Σ numel×element_size`，经 `collective_rpc`） | **12,288** | 总 KV ≈ 8.99 GB |
| 理论（`2×L(24)×kv_heads(2)×head_dim(64)×dtype(2B)`） | **12,288** | 与实测**完全一致** ✓ |
| 配置推导（`num_gpu_blocks×page_size_bytes`，naive） | 512（**错**） | ⚠️ `page_size_bytes=8192` 是**每层每块**，naive 公式漏乘 24 层，故偏小 24× |

→ **以实测=理论为准：每 token KV ≈ 12 KiB**（K 与 V 各半），验证了 M4 的 GQA 结构（2 kv-head×64）。诚实记录：naive 配置公式踩了"page_size 是 per-layer"的坑，已弃用。

**显存构成**（16 GiB 卡，util=0.6）：模型权重 0.93 GiB（日志 "Model loading took"）；KV 可用 ~8.3 GiB（日志 "Available KV cache memory" ≈ 实测 8.99 GB）；CUDA Graph 池 0.10 GiB（仅 compiled，日志）。`MemorySnapshot` 只能分 torch / non-torch，**不能**直接拆 weights/KV/cudagraph，上述拆分来自 vLLM 启动日志。

## 4. Decode：step latency & 吞吐 vs batch（seq_len 恒=38，prefix off）

| batch | eager lat(ms) | compiled lat(ms) | eager 吞吐(tok/s) | compiled 吞吐(tok/s) | compiled 加速 | 实际模式(eager/compiled) |
|---:|---:|---:|---:|---:|---:|:---|
| 1  | 7.09 | 2.49 | 141   | 402    | 2.85× | NONE / **FULL** |
| 2  | 7.52 | 2.65 | 266   | 756    | 2.84× | NONE / **FULL** |
| 4  | 7.31 | 2.63 | 547   | 1,524  | 2.78× | NONE / **FULL** |
| 8  | 7.44 | 2.69 | 1,075 | 2,975  | 2.77× | NONE / **FULL** |
| 16 | 7.47 | 2.79 | 2,141 | 5,735  | 2.68× | NONE / **FULL** |
| 32 | 7.85 | 2.94 | 4,077 | 10,883 | 2.67× | NONE / **FULL** |
| 64 | 8.40 | 3.36 | 7,621 | 19,053 | 2.50× | NONE / **FULL** |

> 「实际模式」一列由**逐 batch 诊断**（`diag_modes_*.jsonl`）实测确认：**compiled 在 batch 1–64 全部走 FULL CUDA Graph**（uniform_decode=True），eager 全部 NONE。此前只有 batch=1 的证据，现已逐 batch 覆盖。

**实测观察**：
- decode 单步 latency **几乎不随 batch 增长**：batch 1→64（64×），eager 仅 7.09→8.40ms（≈1.2×），compiled 2.49→3.36ms（≈1.3×）。
- 因此**吞吐近线性增长**：compiled 402→19,053 tok/s（约 47×），eager 141→7,621（约 54×）。
- compiled decode **每个 batch 都比 eager 快 ~2.5–2.85×**（全走 FULL 图）。

**可解释的机制（假设，非断言）**：一步 decode 要把**整套模型权重**读一遍，但只算 B 个 token；batch 增大时权重读取被更多 token 摊销 → 每 token 成本下降、吞吐上升，而"读权重"的固定成本让单步 latency 近乎不变。这与"decode 受权重/访存主导"的常见解释一致，但**当前只有 end-to-end step latency，尚不能证明已到显存带宽上限**——需 M7 的 kernel 时间线 + 硬件计数器验证。compiled 到 batch 64 latency 抬头（2.94→3.36）提示计算开始显现，也留 M7 看。

## 5. Prefill：step latency vs 输入长度（prefix off）

| 长度(token) | eager(ms) | compiled(ms) | compiled 实际模式 |
|---:|---:|---:|---|
| 32   | 6.89 | 4.13 | PIECEWISE |
| 128  | 6.48 | 4.78 | PIECEWISE |
| 256  | 6.91 | 6.34 | **NONE** |
| 512  | 7.37 | 7.12 | **NONE** |
| 1024 | 11.93| 11.89| **NONE** |
| 1536 | 16.49| 16.18| **NONE** |
| 2000 | 22.04| 21.82| **NONE** |

> 「compiled 实际模式」由正式诊断入口逐长度实测（`diag_modes_compiled_off.jsonl`）：**≤128 → PIECEWISE（分段图），≥256 → NONE**。原因：本配置 `max_cudagraph_capture_size=128`（由 `max_num_seqs=64` 决定），prefill token 数超 128 即无对应捕获图 → 回退 NONE（仍是 inductor 编译过的 kernel，但**无图回放**）。

**实测观察**：
- **eager prefill 在 ≤512 token 近乎持平（~6.5–7.4ms 一个"地板"），512 之后才明显增长**（1024→11.9、1536→16.5、2000→22ms，约每 +512 token 加 ~5ms）。
- **compiled 只在 ≤128 token 明显更快（4.1–4.8ms，比 eager 快 ~1.4–1.7×）**（走 PIECEWISE 图）；**256 起与 eager 基本相同**（都 NONE，仅剩 inductor 融合的微弱差异）。

**可解释的机制（假设）**：短 prompt 的 prefill（以及所有 decode）似乎被一个 **~6.5–7ms 的固定 per-step 开销"地板"**主导；CUDA Graph 回放（compiled ≤128 的 PIECEWISE、decode 的 FULL）**削掉了这个地板**，于是短任务提速。一旦 prompt 变长（>128 无图、或 >512 计算量上来），**计算逐渐主导**，图回放不再帮忙（且 >128 根本没捕获），compiled≈eager。这条"开销地板"假设同时解释了 §4 decode 近平和 §5 prefill 短区平。**但"地板"到底是 CPU/Python/kernel-launch 哪一块，必须 M7 用 profiler 拆**——本模块不下定论。也**不把 512→2000 的增长写成 O(n²)**：这些尺寸 + 0.5B 下更接近线性增长，注意力二次项尚未显性。

## 6. Prefix Cache 对首 token 延迟（含 OFF 对照 + 配对分析）

三组同长度（256 token body）请求：`off`（prefix OFF 引擎）、`cold`（prefix ON，全新前缀，必 miss）、`hot`（prefix ON，与对应 cold 完全相同 token ids，应命中）。cold/hot 用**独立前缀族**逐 repeat 隔离，杜绝跨 repeat 污染。

| 组 | 本轮 scheduled tokens | prefix 命中 tokens | first-token latency(ms, median) | p95 |
|---|---:|---:|---:|---:|
| off（prefix 关） | 256 | 0 | **7.00** | 7.81 |
| cold（prefix 开，未命中） | 256 | 0 | 8.17 | 9.87 |
| hot（prefix 开，命中） | **16** | **240** | 7.88 | 10.25 |

**cold vs hot 配对分析**（同 repeat 逐配对，正=hot 更快，n=15）：`mean=+0.033ms, median=+0.083ms, stdev=1.54ms, min=−3.97ms, max=+2.29ms, hot 更快仅 8/15 次`。

**实测观察 + 机制**：
1. **命中确实大幅减少计算 token**：hot 的 scheduled 从 256 降到 **16（省 94%）**，命中 240=15 整块×16（呼应 M3 的整块对齐命中，末 16 token 仍需算）。
2. **但命中没有带来可测的延迟下降**：cold−hot 配对差**均值仅 +0.033ms、跨零（−3.97~+2.29）、只有约一半 repeat hot 更快** → **落在测量噪声内**。原因：256-token 短 prompt 的 prefill 本就被 §5 的 ~7ms 开销地板主导，省掉 94% 计算 token 也几乎不动延迟。
3. **prefix ON 在短 prompt 下反而略慢于 OFF**：cold 8.17 / hot 7.88ms 都 > off 7.00ms。说明在这种"计算省不出延迟"的场景，prefix caching 的**块 hash / 查表等簿记开销**反而占了上风。
4. **结论**：**"省计算 ≠ 省延迟"**——prefix cache 的收益要在**计算主导**（长 prompt / 大命中比例）时才转化为延迟下降；短、overhead-bound 的请求上，命中省下的计算被固定开销"吃掉"，甚至因簿记开销略微变慢。命中证据来自 `scheduled_tokens` 骤降（计时区外读，非由 latency 反推）。

## 7. eager vs compiled 执行路径确认（接 M5，正式诊断入口）
- eager：所有 prefill 长度、所有 decode batch 均 `cudagraph_mode=NONE`（纯 eager）。
- compiled：`mode=VLLM_COMPILE + cudagraph_mode=FULL_AND_PIECEWISE`（inductor）；**decode batch 1–64 全 FULL、prefill ≤128 PIECEWISE、prefill ≥256 NONE**（`diag_modes_*.jsonl` 逐项实测）。
- 启动成本：eager init≈9.9s；compiled init≈13.9s（含 torch.compile，命中磁盘缓存后仅 ~1.3s，加图捕获；CUDA Graph 池 0.10 GiB，代价是 KV 块少 ~619）。
- 数值：M5 已知 compiled 与 eager 贪心 token 会分叉，本模块跨模式**不要求 token 一致**，只确认无 NaN/Inf、无 EOS 提前结束、batch 与 seq_len 可比。

## 8. 结论（按"观察→机制→证据强度→M7 待验证"）
1. **compiled 大幅加速短任务**：decode 全 batch（1–64 全 FULL）快 2.5–2.85×，prefill 仅 ≤128（PIECEWISE）快 ~1.4–1.7×。机制假设：CUDA Graph 回放削掉固定 per-step 开销地板。证据强度：latency 差异明确、逐 batch/逐长度模式已确诊；**"地板=CPU/launch"仍是假设**。
2. **batching 提 decode 吞吐**：单步 latency 近平（seq_len 恒=38）、吞吐近线性。机制假设：权重读取被更多 token 摊销（访存主导）。证据强度：吞吐曲线明确；**是否触及显存带宽上限待 M7 证**。
3. **prefill 短区被开销主导、长区转计算**：≤512 近平（~7ms），>512 增长。**不写死 O(n²)**（当前尺寸更近线性）。
4. **prefix cache 省计算 ≠ 省延迟**：命中省 94% 计算 token，但短 prompt 下 first-token latency 配对差在噪声内（均值 +0.03ms），prefix ON 甚至因簿记开销略慢于 OFF；收益需在计算主导场景才显。
5. **KV 每 token ≈12 KiB**（实测=理论）；KV 池 eager 73.2 万 / compiled 72.2 万 token（compiled 少因 cudagraph 占显存）；**"最大并发 357×"是 KV 容量等价值，运行时并发由 `max_num_seqs=64` 决定**。

**M7 待验证**：用 PyTorch Profiler + Nsight 拆开"~7ms 开销地板"到底是 CPU/Python、kernel launch gap、H2D 还是 sampler；验证 decode 是否真的到显存带宽上限；观察 CUDA Graph replay 在时间线上如何消除 launch gap。

## 9. 局限
- 单卡、单模型（0.5B）、单 dtype（bf16）；结论不外推。
- decode 固定 seq_len=38（短上下文）；长上下文 decode（attention 读全历史增大）未扫。
- prefill >128 在 compiled 下无图，故"compiled prefill"曲线主要反映 inductor 编译效果，非图回放。
- `engine_first_token_latency` 含 detokenize，非在线 TTFT；计时含 CPU+GPU 端到端 step，非单 kernel 时间（那是 M7）。
- 各测点 run-to-run 有系统抖动（±0.5ms 量级）；以 median 为主、辅以 p95/stdev。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
# 全部配置（各自独立进程）+ 汇总
python scripts/m6_run_all.py            # eager+off(mem,prefill,decode,prefix,diag) / compiled+off(…,diag) / eager+on(mem,prefix)
python scripts/m6_summarize.py          # -> outputs/m6/baseline.csv + summary/（含 prefix 配对）
# 单配置 / 单实验示例（diag = 逐长度/逐 batch 执行模式诊断，正式入口）
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m6_benchmark.py --mode compiled --prefix off --experiments diag
```
原始逐样本：`outputs/m6/raw/*.jsonl`；指纹/KV/显存：`outputs/m6/config/*_meta.json`；逐长度/逐 batch 模式：`outputs/m6/config/diag_modes_*.jsonl`；prefix 配对：`outputs/m6/summary/summary.json` 的 `prefix_paired`。

---

## 审查修复记录（针对 7 点，均已据真实环境核实并修复）
1. **decode seq_len 非恒 38**：原 window=2 使每 repeat 出 38 与 39 各一。→ 改 `--decode-window` 默认 1，只测 1 个固定位置，**seq_len 现严格恒=38**（重跑）。
2. **prefix "降 6%" 不稳**：原用非配对 median 比较且 hot 方差大。→ 新增**逐 repeat 配对分析**，cold−hot 均值 +0.03ms、跨零、hot 仅 8/15 更快，结论改为**落在噪声内 / 省计算≠省延迟**。
3. **prefix OFF 对照未跑**：→ 在 `eager+off` 配置补跑 `prefix` 实验（off 组，median 7.00ms），纳入对照表。
4. **"decode 全走 FULL" 缺逐 batch 证据**：原仅 batch=1 诊断。→ 新增正式诊断入口 `--experiments diag`，**逐 batch 1–64 实测确认全 FULL**（eager 全 NONE）。
5. **最大并发 357 的含义**：→ §3 明确 357× 是 **KV 容量等价值**，运行时并发由 `max_num_seqs=64` 限制。
6. **compiled/eager KV 块数不同**：→ §3 补 compiled **45,118** 块（vs eager 45,737），并解释 CUDA Graph 池占显存导致 KV 块减少。
7. **`_diag_prefill_lengths.jsonl` 无正式入口**：→ 用 `run_diag()` + `--experiments diag` 正式生成 `diag_modes_<tag>.jsonl`（逐长度 prefill + 逐 batch decode），删除旧的临时手工文件。
