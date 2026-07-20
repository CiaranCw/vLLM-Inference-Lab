# M6 · 离线 / 闭环引擎 Benchmark（★核心验收 #3）

> 目标：把能力从"能解释执行路径"推进到**"能可信测量执行路径"**。在**固定环境**下，用**进程内闭环**方式测量 prefill / decode / 吞吐 / prefix / KV 与显存，并对 **eager vs compiled**（接 M5）做性能对照。
> **边界（务必牢记）**：本模块**只测量 + 解释现象**，**不做 profiler 根因、不断言 compute/memory-bound、不改算法、不做在线 serving、不下跨机跨模型结论**（留 M7 及后续）。所有数字**仅在本机、本模型、本配置下成立**。
>
> 证据：`outputs/m6/baseline.csv`（汇总）、`outputs/m6/raw/*.jsonl`（逐样本，可回查）、`outputs/m6/config/*_meta.json`（引擎/环境/KV/显存指纹）、`outputs/m6/summary/summary.json`、`logs/m6/*.log`（启动日志含 KV/显存/编译行）。复现命令见文末。

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
- **预热与正式分离**：prefill 每长度 warmup 5 + 有效样本 15；decode 每 batch 15 独立 repeat（方案 A：每 repeat 重建 batch→prefill→4 步 decode warmup→固定位置测 2 步，**seq_len 恒=38**，保证跨 repeat 可比）→ 30 样本。
- 统计报 median/mean/stdev/p95/min/max；**保留全部逐样本原始数据**。共 730 样本，**660 有效 measure，70 dropped 全为 prefill warmup，0 个无效 measure**。
- 每种 (mode,prefix) 配置**独立子进程**运行（防 CUDA context / allocator / CUDA Graph 池 / torch.compile 缓存污染）。
- runtime_mode 在**诊断相**（临时开 exec trace 的 throwaway step）确认，不进入计时样本。

## 3. KV Cache 容量 / 显存构成（每项标来源）

**配置直读**（`vllm_config.cache_config`）：`num_gpu_blocks=45737`，`block_size=16`，`kv_cache_size_tokens=731,792`，`kv_cache_max_concurrency=357.3x`（@ max_model_len=2048）。
→ 即：这套配置的 KV 池能装 **73 万 token**，若每请求占满 2048 上下文，理论最大并发 **357 条**。

**KV 每 token 字节（三路交叉验证）**：
| 来源 | 每 token 字节 | 说明 |
|---|---|---|
| 实测（KV tensor `Σ numel×element_size`，经 `collective_rpc`） | **12,288** | 总 KV = 8.99 GB |
| 理论（`2×L(24)×kv_heads(2)×head_dim(64)×dtype(2B)`） | **12,288** | 与实测**完全一致** ✓ |
| 配置推导（`num_gpu_blocks×page_size_bytes`，naive） | 512（**错**） | ⚠️ `page_size_bytes=8192` 是**每层每块**，naive 公式漏乘 24 层，故偏小 24× |

→ **以实测=理论为准：每 token KV ≈ 12 KiB**（K 与 V 各半）。这条也验证了 M4 的 GQA 结构（2 kv-head×64）。诚实记录：naive 配置公式踩了"page_size 是 per-layer"的坑，已弃用。

**显存构成**（16 GiB 卡，util=0.6）：
| 项 | 值 | 来源 |
|---|---|---|
| 设备总显存 / 已用 | 16,302 / ~11,268 MiB | 实测（`mem_get_info`） |
| torch reserved | ~9,846 MiB | 实测 |
| 模型权重 | 0.93 GiB | 日志（"Model loading took"） |
| KV cache 可用 | ~8.3 GiB | 日志（"Available KV cache memory"）≈ 实测 KV 8.99 GB |
| CUDA Graph 池（仅 compiled） | 0.10 GiB | 日志（"CUDA graph pool memory"） |
> 注：`MemorySnapshot` 只能分 torch / non-torch，**不能**直接拆 weights/KV/cudagraph；上表 weights/KV/graph 来自 vLLM 启动日志。

## 4. Decode：step latency & 吞吐 vs batch（seq_len 恒=38，prefix off）

| batch | eager lat(ms, median) | compiled lat(ms) | eager 吞吐(tok/s) | compiled 吞吐(tok/s) | compiled 加速 |
|---:|---:|---:|---:|---:|---:|
| 1  | 7.45 | 2.49 | 134   | 401    | 3.0× |
| 2  | 8.10 | 2.82 | 247   | 710    | 2.9× |
| 4  | 8.13 | 2.71 | 492   | 1,476  | 3.0× |
| 8  | 7.64 | 2.77 | 1,047 | 2,883  | 2.8× |
| 16 | 7.89 | 2.84 | 2,028 | 5,630  | 2.8× |
| 32 | 8.73 | 3.07 | 3,667 | 10,426 | 2.8× |
| 64 | 8.32 | 3.67 | 7,695 | 17,432 | 2.3× |

**实测观察**：
- decode 单步 latency **几乎不随 batch 增长**：batch 从 1→64（64×），eager 仅 7.45→8.32ms（≈1.1×），compiled 2.49→3.67ms（≈1.5×）。
- 因此**吞吐近线性增长**：compiled 401→17,432 tok/s（约 43×），eager 134→7,695（约 57×）。
- compiled decode **每个 batch 都比 eager 快 ~2.3–3.0×**（decode 走 **FULL CUDA Graph**，见 §7）。

**可解释的机制（假设，非断言）**：一步 decode 要把**整套模型权重**读一遍，但只算 B 个 token；batch 增大时权重读取被更多 token 摊销 → 每 token 成本下降、吞吐上升，而"读权重"的固定成本让单步 latency 近乎不变。这与"decode 受权重/访存主导"的常见解释一致，但**当前只有 end-to-end step latency，尚不能证明已到显存带宽上限**——需 M7 的 kernel 时间线 + 硬件计数器验证。compiled 到 batch 64 latency 抬头（3.07→3.67）提示计算开始显现，也留 M7 看。

## 5. Prefill：step latency vs 输入长度（prefix off）

| 长度(token) | eager(ms) | compiled(ms) | compiled 实际模式 |
|---:|---:|---:|---|
| 32   | 7.31 | 3.02 | PIECEWISE |
| 128  | 7.29 | 3.52 | PIECEWISE |
| 256  | 7.40 | 7.33 | **NONE** |
| 512  | 7.59 | 7.65 | **NONE** |
| 1024 | 12.17| 12.52| **NONE** |
| 1536 | 17.16| 16.99| **NONE** |
| 2000 | 22.44| 22.73| **NONE** |

**实测观察**：
- **eager prefill 在 ≤512 token 近乎持平（~7.3–7.6ms 一个"地板"），512 之后才明显增长**（1024→12、1536→17、2000→22ms，约每 +512 token 加 ~5ms）。
- **compiled 只在 ≤128 token 明显更快（3.0–3.5ms，~2×）**；**256 起与 eager 几乎相同**。
- 关键：诊断相实测发现 compiled prefill 的**实际执行模式随长度变化**——`≤128` 走 **PIECEWISE**（分段 CUDA 图），`≥256` 因**超过本配置 CUDA Graph 捕获上限 128**（`max_cudagraph_capture_size=128`，由 `max_num_seqs=64` 决定）而**回退 NONE**（仍是 torch.compile/inductor 编译过的 kernel，但**无图回放**）。

**可解释的机制（假设）**：短 prompt 的 prefill（以及所有 decode）似乎被一个 **~7ms 的固定 per-step 开销"地板"**主导；CUDA Graph 回放（compiled ≤128 的 PIECEWISE、decode 的 FULL）**消除了这个地板**，于是短任务提速 2–3×。一旦 prompt 变长（>128 无图、或 >512 计算量上来），**计算逐渐主导**，图回放不再帮忙（且 >128 根本没捕获），compiled≈eager。这条"开销地板"假设同时解释了 §4 decode 近平和 §5 prefill 短区平。**但"地板"到底是 CPU/Python/kernel-launch 哪一块，必须 M7 用 profiler 拆**——本模块不下定论。也**不把 512→2000 的增长写成 O(n²)**：这些尺寸 + 0.5B 下更接近线性增长，注意力二次项尚未显性。

## 6. Prefix Cache 对首 token 延迟（eager，prefix on；独立前缀族防冷污染）

| | 本轮 scheduled tokens | prefix 命中 tokens | first-token latency(ms, median) |
|---|---:|---:|---:|
| cold（全新前缀，必 miss） | 256 | 0 | 7.20 |
| hot（同 ids，命中） | **16** | **240** | 6.80 |

**实测观察**：命中把**需计算 token 从 256 降到 16（省 94%）**（240=15 整块×16 命中，仅剩末 16 token 重算——呼应 M3 的整块对齐命中），但 **first-token latency 只从 7.20→6.80ms（约 6%）**。

**可解释的机制**：在这种**短 prompt（256）下 prefill 本就被 §5 的 ~7ms 开销地板主导**，所以即便省掉 94% 的计算 token，latency 也几乎不动。→ **prefix cache 的"省计算"要转化为"省延迟"，得在计算主导（长 prompt / 大命中）时才明显**。命中证据来自 `scheduled_tokens` 骤降（计时区外读，非由 latency 反推）。

## 7. eager vs compiled 执行路径确认（接 M5，诊断相）
- eager：所有步 `cudagraph_mode=NONE`（纯 eager）。
- compiled：`mode=VLLM_COMPILE + cudagraph_mode=FULL_AND_PIECEWISE`（inductor 后端）；诊断实测 **decode→FULL、prefill(≤128)→PIECEWISE、prefill(≥256)→NONE**。
- 启动成本：eager init≈9.9s；compiled init≈13.9s（含 torch.compile 1.29s——本次命中磁盘缓存故比 M5 首次的 9.44s 快，另加图捕获；CUDA Graph 池仅 0.10 GiB）。
- 数值：M5 已知 compiled 与 eager 贪心 token 会分叉，本模块跨模式**不要求 token 一致**，只确认无 NaN/Inf、无 EOS 提前结束、batch 与 seq_len 可比。

## 8. 结论（按"观察→机制→证据强度→M7 待验证"）
1. **compiled 大幅加速短任务**：decode 全 batch 快 2.3–3.0×，prefill 仅 ≤128 快 ~2×。机制假设：CUDA Graph 回放消除固定 per-step 开销地板。证据强度：latency 差异明确、模式已确诊；**"地板=CPU/launch"仍是假设**。
2. **batching 提 decode 吞吐**：单步 latency 近平、吞吐近线性。机制假设：权重读取被更多 token 摊销（访存主导）。证据强度：吞吐曲线明确；**是否触及显存带宽上限待 M7 证**。
3. **prefill 短区被开销主导、长区转计算**：≤512 近平，>512 增长。**不写死 O(n²)**（当前尺寸更近线性）。
4. **prefix cache 省计算 ≠ 省延迟**（短 prompt 下 94% 计算省下仅换 6% 延迟）；长/计算主导场景才显收益。
5. **KV 每 token ≈12 KiB**（实测=理论），KV 池 73 万 token / 最大并发 357×（本配置）。

**M7 待验证**：用 PyTorch Profiler + Nsight 拆开"~7ms 开销地板"到底是 CPU/Python、kernel launch gap、H2D 还是 sampler；验证 decode 是否真的到显存带宽上限；观察 CUDA Graph replay 在时间线上如何消除 launch gap。

## 9. 局限
- 单卡、单模型（0.5B）、单 dtype（bf16）；结论不外推。
- decode 固定 seq_len=38（短上下文）；长上下文的 decode（attention 读全历史增大）未扫，另留。
- prefill >128 在 compiled 下无图，故"compiled prefill"曲线主要反映 inductor 编译效果，非图回放。
- `engine_first_token_latency` 含 detokenize，非在线 TTFT。
- 计时含 CPU+GPU 端到端 step，非单 kernel 时间（那是 M7）。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
# 全部配置（各自独立进程）+ 汇总
python scripts/m6_run_all.py            # eager+off / compiled+off / eager+on
python scripts/m6_summarize.py          # -> outputs/m6/baseline.csv + summary/
# 单配置示例
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m6_benchmark.py --mode compiled --prefix off --experiments mem,prefill,decode
# 快速自测（小矩阵，数据不入正式报告）
python scripts/m6_run_all.py --quick && python scripts/m6_summarize.py --quick
```
原始逐样本：`outputs/m6/raw/*.jsonl`；指纹与 KV/显存：`outputs/m6/config/*_meta.json`；compiled prefill 各长度模式诊断：`outputs/m6/config/_diag_prefill_lengths.jsonl`。
