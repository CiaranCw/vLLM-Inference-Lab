# vLLM Inference Lab · 学习进度记录（Progress Log）

> 本文件是**执行进度流水**，与 `study_path.md`（原始学习路线/计划，保持不改）分离，避免混淆。
> 每完成一个 Day/场景在此追加一节；详细数据与解读见 `notes/`、`outputs/`、`logs/` 对应目录。

## ✅ 第一周 Day 1 —— 单请求主链路 + Prefill/Decode 运行时 Trace（2026-07-14）

**环境**：RTX 5080(sm_120) / torch 2.11.0+cu130 / vLLM v0.24.0 (`ee0da84`, editable) / Qwen2.5-0.5B-Instruct (bf16)。
单进程 `VLLM_ENABLE_V1_MULTIPROCESSING=0`；WSL 无 nvcc 故 `VLLM_USE_FLASHINFER_SAMPLER=0` 规避 FlashInfer 采样器 JIT。

**确认的真实执行组件**：V1 LLM engine（async scheduling，batch queue=2，延迟执行）；`Scheduler`（chunked prefill，prefix caching 默认开）；`UniProcExecutor` + `Worker`；`GPUModelRunner`；Attention backend **FLASH_ATTN / FlashAttentionImpl，FA v2**（sm_120 候选 `[FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION]` 自动选择）；enforce_eager → torch.compile/CUDAGraph 关；**KV cache block_size=16, num_gpu_blocks=34801**（实测）；GQA 14 q head / 2 kv head / head_dim 64。

**Prefill vs Decode（实测）**：prefill scheduled=11、q=T(11,14,64)、query_start_loc=[0,11]、max_query_len=11；decode scheduled=1、q=T(1,14,64)、positions=[11]、max_seq_len=12、slot_mapping=T(1,)。Trace 开启不改变输出（`[84648,19479]` 与基线一致）。

**产出**：
- `scripts/day1_baseline.py`、`outputs/day1/baseline_result.json`、`outputs/day1/trace_result.json`
- `logs/day1/{baseline_debug.log, single_request_trace.log, nvidia_smi_before/after.txt, vllm_collect_env.txt, source_status_before/after.txt}`
- `notes/day1/{baseline_analysis.md, runtime_trace.md}`（含真实调用链 + Prefill/Decode 对照表 + Mermaid）
- `vllm-src/vllm/_lab_trace.py`（Trace 工具，默认关）+ 4 文件 7 处 `# VLLM_INFERENCE_LAB_TRACE` 插桩；清单见 `outputs/day1/trace_changes_manifest.md`；插桩前备份在 `backups/day1/`。

## ✅ 第一周 Day 2 —— Scheduler 与 Continuous Batching（2026-07-17）

**方法**：手动驱动 V1 `LLMEngine`（`from_engine_args` + `add_request`/`step()` 循环，按到达表注入请求），`async_scheduling=False` 使每 step 干净一次 schedule→execute→update；`max_num_seqs=2` 逼出排队，`max_num_batched_tokens=64` 逼出 chunked prefill；复用 Day1 `scheduler/schedule` 钩子，仅给 `_lab_trace.py` 加 `VLLM_LAB_TRACE_COMPONENTS` 与 `VLLM_LAB_TRACE_JSONL`（未改 4 个 vLLM 源文件）。采样器仍 `VLLM_USE_FLASHINFER_SAMPLER=0`。

**nvcc/FlashInfer 实测**：nvcc 12.8 已装，但即使设 `CUDA_HOME`/`TORCH_CUDA_ARCH_LIST=12.0`，FlashInfer 采样器 JIT 仍在 `check_cuda_arch` 失败（`TARGET_CUDA_ARCHS` 为空 → "requires sm75 or higher"）——本机 sm_120 上 flashinfer 采样器仍不可用，保留原生采样器规避。

**四场景实测结论**（`notes/day2/continuous_batching.md` + `outputs/day2/E*_timeline.md`）：
- E1 单请求：手动驱动与 Day1 结构一致。
- E2 错峰：**step4 同一 batch 内 A(decode D1) + B(prefill P8) 混跑** = continuous batching 本质。
- E3 四请求 max_num_seqs=2：running 封顶 2，C/D `waiting=2` 排队 10 步，A/B 结束后才准入。
- E4：长 prompt(288) 被 `max_num_batched_tokens=64` 切成 **64+64+64+64+32** 五步 chunked prefill。
- **附带发现（Day3 伏笔）**：相同前缀的后到请求 prefill 被前缀缓存缩短（`computed_before=16`=1 block, block_size=16）。

**产出**：`scripts/day2_engine_driver.py`、`scripts/day2_build_timeline.py`；`logs/day2/{nvcc_flashinfer_check,E1_single,E2_stagger2,E3_queue4,E4_chunked_prefill}.log`；`outputs/day2/{E1..E4}.jsonl`、`{E1..E4}_timeline.md`、`trace_changes_manifest.md`；`notes/day2/continuous_batching.md`。

### ✅ Day 2 追加 · E5/E5b —— blocked decode + chunked prefill 的调度优先级（2026-07-17）

问题：decode 请求 D 被临时阻塞、让 chunked prefill P 先跑，解除后调度器是继续 P 还是转去调度 D、把 P 搁置？为此新增**实验性、env 门控（`VLLM_LAB_BLOCK`）、默认关闭、可回退**的 scheduler「按请求跳过」钩子（`# VLLM_INFERENCE_LAB_EXPERIMENT`，回归验证零副作用）。

- **E5（max_num_seqs=2，预算争用）**：block 步 P 独占 **P64**；解除步起 **D1 + P63**——恢复的 decode 与在跑的 prefill **同一步共享 64 预算**（running 序 [D,P]，D 先拿 1，P 拿剩余 63），**prefill 不被搁置、decode 不被推迟**。
- **E5b（max_num_seqs=1，槽位争用）**：阻塞 D **不释放槽位** → P 只能 waiting，D 恢复后独占槽位跑到结束，**P 直到 D 完成、槽位释放才被准入**。即**阻塞(skip)≠抢占(free slot)**。
- 结论：decode 与 chunked prefill 的"共存"发生在**同一 running 集合的 token 预算分配**层面；能否共存取决于 **max_num_seqs 是否允许二者同时 running**。
- 产出：`scripts/day2_engine_driver.py`(+E5/E5b)、`outputs/day2/{E5,E5b}.jsonl`、`{E5,E5b}_timeline.md`、`notes/day2/e5_blocked_decode_vs_prefill.md`、备份 `backups/day2/*.pre_e5`。

## ✅ 第一周 Day 3 —— KV Cache 与 Block 管理（2026-07-17）

**方法**：手动驱动 V1 引擎（`scripts/day3_kv_driver.py`，支持 `num_gpu_blocks_override` / `enable_prefix_caching`），在 `kv_cache_manager` / `block_pool` / `scheduler._preempt_request` 加**被动只读** trace（component `kv`；E1 回归验证零副作用）。基本事实：block_size=16、物理块 id 从 1 起（块 0 为 null）。

- **K1 块分配/block_table/跨块边界**：prompt=24 → prefill 分配 2 块 `[1,2]`；decode 到 `computed_before=32`（跨入逻辑块 2）那步新分配物理块 `[3]`。逻辑→物理 = `[1,2,3]`。
- **K2 prefix 命中 vs 关闭**：B（与 A 同 prompt）`get_computed_blocks` 命中 **32 token、hit_block_ids=[1,2]（A 的物理块，共享）**，只 prefill 剩余 16。命中 32≠48 因 `max_cache_hit_length=num_tokens-1` + 整块对齐（解释 Day2 的 16）。OFF 命中 0。
- **K3 释放→cached-free→复用**：A 结束 `free`，带 hash 整块 `freed_cached_tail=[3,2,1]` 保留、无 hash `freed_evict_first=[4]`；B（相同 prompt）命中复用 A 释放但仍 cached 的 `[1,2]`。
- **K4 KV 不足→抢占**（`num_gpu_blocks_override=48`, `max_model_len=512`, 4 请求）：step26 `allocate_slots_FAILED(available=0)` → `free` victim D → `preempt D, num_computed_tokens=0`。**prefix OFF**：D 恢复 `get_computed_blocks=0` → 从头重算；**prefix ON**：D 恢复命中自己 cached 的 176 token → 几乎不重算。两条不变量验证成立（活跃请求块不被 free；恢复不 append 失效 block_table）。
- **要点**：block_table=逻辑→物理映射；prefix 命中=整块 hash+共享物理块(ref_cnt++)；释放的 cached 块可复用；抢占 victim 归零重算，而 prefix caching 能让抢占-恢复几乎免重算。

**产出**：`scripts/day3_kv_driver.py`；`logs/day3/K{1,2,3,4}.log`、`K2_off/K4_on.log`；`outputs/day3/K*.jsonl`、`trace_changes_manifest.md`；`notes/day3/kv_cache_block_management.md`；`backups/day3/*.pre_day3`。

**下一步（待确认）**：Day 4 —— Attention Backend（自动选择逻辑、Prefill/Decode shape、GQA、block_table/slot_mapping 如何被 backend 消费、Python 层与 kernel 边界）。

### ✅ Day 3 追加 · K5 —— 抢占后 block chain 尾部优先淘汰、前缀优先保留（2026-07-17）

问题：请求被抢占后，其 cached-free 的 block chain 是否尾部先被驱逐、从而保留最长连续前缀？为此新增两个可控机制（`VLLM_LAB_BLOCK` 门控、默认关、回归零副作用）：**priority 调度**让持块多的 D 成为抢占 victim；**waiting 准入循环实验性 skip**（`# VLLM_INFERENCE_LAB_EXPERIMENT`）在 D 被抢占后阻止其重新准入，使其块保持 cached-free；并给 evict 钩子补记 `block_hash_num_tokens`（覆盖 token 数）。

- D 抢占前链 `[1..13,34,37]`（15 块），`free()` 逆序释放 → `freed_cached_tail=[37,34,...,1]`（尾块进队头）。
- **驱逐严格按 `block_hash_num_tokens` 递减：37(240)→34(224)→…→1(16)**（K5_full 全 15 块单调），即**逻辑块从尾到头**。
- 抢占尾部 M=8 块后放行 D → **恢复命中前 N−M=7 个连续头部块 `[1..7]`，`num_new_computed_tokens=112=7×16`**（K5）。
- 结论：**"尾部优先淘汰、前缀优先保留"成立**；命中恒为从头连续前缀，无"头部先驱逐却命中后部"。机制：`free` 逆序 + `popleft` 取块使尾块先被 `_maybe_evict_cached_block`；prefix 命中要求从头连续整块。
- 产出：`scripts/day3_kv_driver.py`(+run_k5)；`logs/day3/K5.log`、`K5_full.log`；`outputs/day3/K5.jsonl`、`K5_full.jsonl`；`notes/day3/k5_eviction_order.md`；备份 `backups/day3/*.pre_k5`。

## ✅ M4 —— Attention Backend：SchedulerOutput→KV写入→Attention读取 闭环（2026-07-18）

（进入模块制后第一个模块；进度/覆盖以 `ROADMAP.md` 为准，本文件仅记流水。）

在 `flash_attn.py` 加 2 处只读钩子（`attention/kv_write` 在 `do_kv_cache_update`、`attention/attn_read` 在 `forward` 的 `flash_attn_varlen_func` 前，均 layer0；E1 回归零副作用），一次运行串起完整闭环：

- backend：本机自动选中 **FLASH_ATTN / FlashAttentionImpl / FA2**（未主动指定）。
- Q/K/V：通用层 `T(N,896)/T(N,128)` → backend `T(N,14,64)/T(N,2,64)`（GQA 14 q / 2 kv / head_dim 64）。
- KV cache：`T(34801,2,16,2,64)`=[blocks,K&V,block_size16,kv_heads2,head_dim64]；unbind 后 key/value_cache=`T(34801,16,2,64)`。
- **写**：`reshape_and_cache_flash` 用 `slot_mapping`（prefill T(11,)/decode T(1,)）把新 K/V 散写进 cache。
- **读**：`flash_attn_varlen_func` 用 `block_table`（T(1,128)）从 cache 取历史 K/V；Prefill `max_seqlen_k=11`、Decode `max_seqlen_k=12`（读全历史）。
- Python↔kernel 边界：Python 造张量+metadata；两个 CUDA 算子（写=reshape_and_cache_flash、读=flash_attn_varlen_func）是边界。
- 产出：`notes/m4_attention_backend.md`、`notes/evidence_index.md`(新建证据索引)、`outputs/m4/*`、`logs/m4/*`、`backups/m4/flash_attn.py.pre_m4`。

**下一步（待确认）**：M5 —— 执行模式（本机实际启用哪种：enforce_eager 改了什么、是否 torch.compile、CUDA Graph FULL/PIECEWISE/NONE、capture/replay/非图区），只验证执行路径，性能结论留到 M6/M7。

## ✅ M5 —— 执行模式（本机实际路径验证）（2026-07-20）

（笔记规范化：本模块起每阶段独立目录 `notes/mN/`；`notes/evidence_index.md` 仍为跨模块索引置于 `notes/` 根。）

方法：`scripts/m5_exec_mode.py` 用**同一单请求**分别以 `enforce_eager=True/False` 运行，从**存活引擎**读出解析后的 `compilation_config`（真相来源），并在 `gpu_model_runner.py` 加 2 处只读钩子（`exec/dispatch` 每步模式+padding、`exec/capture` 捕获描述符；E1 回归零副作用）。

- **eager（基线，enforce_eager=True）**：`mode=NONE / cudagraph_mode=NONE`，纯 eager；`vllm/config/vllm.py:1061-1067` 一票否决并打印 "Enforce eager set..." WARNING；无编译、无捕获（capture_sizes=[]）；每步 dispatch=NONE、pad0。
- **compiled（vLLM 默认 O2，enforce_eager=False）**：`mode=VLLM_COMPILE`(inductor) + `cudagraph_mode=FULL_AND_PIECEWISE`，**sm_120 上真实跑通、未降级**。启动链：torch.compile 9.44s（Dynamo 3.09+Inductor 4.36，AOT 磁盘缓存，只编一张按 attention 切过的图）→ 捕获 **51 PIECEWISE(mixed prefill-decode)+35 FULL(decode)**、0.22 GiB、约 2s（init 合计≈26.6s）。
- **运行时每步实测**：**prefill(11 token,非均匀)→PIECEWISE**，token pad 11→16（num_paddings=5），`pad_attn=False`（attention 在图外走 eager varlen）；**decode(1 token,均匀)→FULL**，无 pad，`pad_attn=True`（整步含 attention 入图 replay）。对上 `FULL_AND_PIECEWISE`=规整 decode 用 FULL、其它用 PIECEWISE。
- **非图区**：`splitting_ops`（attention 系列 + `unified_kv_cache_update`，16 个）= piecewise 的图边界，attention 留在图外。
- **如实记录**：compiled 与 eager 贪心输出**第 5 token 起分叉**（`custom_ops` all→none + Inductor 融合 + CUDA Graph 的浮点数值细微差异，非 bug）；跨模式不能用逐 token 相等做正确性判据。回归：trace 关时 eager 输出与 Day1 基线逐 token 一致（IDENTICAL）。
- 产出：`scripts/m5_exec_mode.py`；`notes/m5/execution_mode.md`；`outputs/m5/{eager,compiled}_config.json`、`{eager,compiled}.jsonl`、`regress_eager.json`；`logs/m5/{eager,compiled,regress_eager}.log`；备份 `backups/m5/gpu_model_runner.py.pre_m5`。

**下一步（待确认）**：M6 —— 离线/闭环引擎 Benchmark（★核心验收）：Prefill/Decode latency、throughput、显存/KV 容量，对照 batch/长度/prefix，预热与正式分离、重复≥3 次报方差/中位数/P95。

## ✅ M6 —— 离线/闭环引擎 Benchmark（★核心验收 #3）（2026-07-20）

先出《M6 最终方案审查报告》（对 ee0da84 逐条核查接口后定稿，见批准计划），再实现。**本模块零 vLLM 源码改动**（纯 driver + 引擎自省 + 只读 wrapper）；trace 全程关闭以保证计时干净。

方法：进程内手动 `LLMEngine`+`step()`；`torch.cuda.synchronize()` 夹 `perf_counter()` 计时（进程内闭环 engine 指标，非在线 TTFT）；精确 token（绕 chat template + 断言 actual==target）；只读包裹 `scheduler.schedule` 在计时区外读校验；每种 (mode,prefix) **独立子进程**（防显存/编译缓存污染）；预热与正式分离、重复≥15、报 median/mean/stdev/P95/min。共 730 样本，660 有效 measure，70 dropped 全为 prefill warmup，0 无效。

- **decode（vs batch，seq_len 恒=38）**：单步 latency 近平（eager 7.1→8.4ms、compiled 2.5→3.4ms，batch 1→64），吞吐近线性（compiled 402→19,053 tok/s）；compiled 全 batch 快 **2.5–2.85×**（**逐 batch 1–64 实测全走 FULL 图**，非仅单请求）。
- **prefill（vs 长度）**：eager ≤512 近平（~7ms 固定开销地板），>512 才增长（1024→11.9、2000→22ms）；compiled 仅 **≤128 提速 ~1.4–1.7×**（走 PIECEWISE 图），**≥256 因超本配置捕获上限 128 回退 NONE**（正式诊断入口逐长度实测）→ ≈eager。
- **prefix（含 OFF 对照 + 配对分析）**：命中把 computed token **256→16（省 94%，240=15整块×16）**，但 first-token latency **配对差在噪声内**（cold−hot 均值 +0.03ms、hot 仅 8/15 更快）；prefix ON（cold 8.17/hot 7.88ms）短 prompt 下反**略慢于 OFF（7.00ms）**——**省计算≠省延迟**（簿记开销在 overhead-bound 时占上风）。
- **KV/显存**：eager `num_gpu_blocks=45737`(731,792 tokens)、compiled `45118`(721,888，少因 cudagraph 占 0.10 GiB 显存)；**KV≈12,288 B/token（实测 tensor-sum = 理论）**；naive 配置公式(×page_size_bytes)踩"per-layer"坑偏小 24×，已弃用；**"最大并发 357×"是 KV 容量等价值，运行时并发由 max_num_seqs=64 限制**。
- **结论按"观察→机制→证据强度→M7 待验证"**表达，不预设 compute/memory-bound、不写死 O(n²)。**"~7ms 开销地板=CPU/launch?"、"decode 是否达带宽上限" 留 M7 profiler。**
- **审查修复（7 点，均据真实环境核实并修复）**：decode 改 window=1 使 seq_len 恒定；prefix 加配对分析（结论改为"噪声内"）+ 补 OFF 对照；新增正式诊断入口 `--experiments diag` 逐 batch/逐长度确认执行模式（补 decode 全 FULL 证据 + 取代临时手工 diag 文件）；澄清最大并发含义；补 compiled 块数。
- 产出：`scripts/m6_{benchmark,run_all,summarize}.py`；`notes/m6/benchmark.md`；`outputs/m6/{baseline.csv,benchmark_config.yaml,raw/,config/(含 diag_modes_*.jsonl),summary/}`；`logs/m6/*`。

**下一步（待确认）**：M7 —— Profiler 定位瓶颈（★核心验收，≥2 类 workload：decode-heavy 小 batch / prefill-heavy 长输入），用 PyTorch Profiler + Nsight 拆开 M6 观察到的"开销地板"与 decode 访存假设，每份形成"现象→证据→假设→单变量实验→结论"。

## ✅ M7 —— Profiler 定位瓶颈（★核心验收 #4）（2026-07-20）

**方法**：`scripts/m7_{profile,run_all}.py`，复用 M6 进程内引擎+step 基础设施，每 mode 独立子进程。**两相分离**：干净计时相（cuda.sync 夹 perf_counter）测 wall；torch.profiler 相取 CPU/GPU/launch 构成 + top kernel + chrome trace。预热 10、正式 20 步/点。单变量对照 = eager vs compiled（唯一差别是有无 CUDA Graph）。**本模块零 vLLM 源码改动**。

**工具实测（如实记录）**：**nsys 在本 WSL2 无法采集 GPU kernel**（`cuda_gpu_kern_sum` 空，CUPTI device-trace 限制）；**torch.profiler 的 CUPTI 通道本机可正常取 CPU+CUDA**，故以其为主力，chrome trace 提供时间线/launch gap。

- **分析一 decode-heavy 小 batch（现象→证据→假设→实验→结论）**：M6 现象=decode ~7ms 地板。证据(eager b=1)：wall 7.44ms 但 GPU 忙碌并集仅 1.70ms(**gpu_frac 0.23**)，**每步 257 次 CPU 提交、头号 CPU 开销 `cudaLaunchKernel` 1.77ms**，GPU 头号是 `gemvx`(GEMV=矩阵×向量/batch1 典型)。假设=CPU/提交受限。单变量实验(→compiled FULL 图)：**总提交 257→15(含 1 次 graph 回放)、wall 7.44→2.50ms**，gpu_frac→≈1(转为 GPU 受限)。**结论：地板=CPU 逐 kernel 提交开销，CUDA Graph 一次回放抹掉之（M6 假设证实且更强）**。
- **分析二 prefill-heavy 长输入**：M6 现象=长 prefill eager≈compiled、随长度增长。证据(L=2000 eager)：GPU 忙碌并集 18.92ms、wall 21.60ms(**gpu_frac 0.88**)，**cutlass tensor-core GEMM 14.6ms + flash attention 2.9ms 主导**。假设=GPU 计算受限。单变量实验(→compiled)：**总提交 257→255 基本不变(只是 runtime↔driver API 转移，prefill≥256 是 NONE 无图)、wall 不变(21.6→22.2ms)**→提交非瓶颈。**结论：长 prefill=GPU 计算受限（M6 假设证实）**。
- **发散**：按 gpu_frac 分界——CPU/提交受限区(≈0.23：decode 任意 batch、短 prefill L32，图能减半 wall)vs GPU 计算受限区(≈0.88：长 prefill，图无用)；decode b1 vs b64 GPU 仅微增(权重被摊销)→印证 M6"单步近平/吞吐近线性"，侧面支持"decode 访存受限"(GEMV 读权重)但缺 kernel 级带宽计数器(ncu 不可用)故仅**受支持假设**。
- **统计口径（经审查修订）**：GPU 时间取 chrome trace `cat=="kernel"` 事件**区间并集**(不对 key_averages 求和，避免高层算子/底层 kernel 双计——修订前 decode 双计到 3.46ms、真实 1.70ms)；CPU 提交覆盖 runtime+driver(`cuLaunchKernel`)+graph 三类(修订前漏计 driver，误报"launch 161→62"，实为总提交 257→255 基本不变)。
- **局限**：nsys GPU 追踪不可用(WSL)；本机单 stream 故并集=求和；gpu_frac 仅 <1 时是干净占比；ncu 不可用故访存受限不下定论；单卡单模型。
- 产出：`scripts/m7_{profile,run_all}.py`、`notes/m7/profiling.md`（含 Profiler 概念讲解 + 统计实现修订记录）、`outputs/m7/profile_{eager,compiled}.json`、`logs/m7/*`（chrome trace 体积大未入库、可复现）。

**下一步（待确认）**：进入两条并行能力线 **M8（Triton kernel，slot-mapping 类）∥ M9（profiler 驱动的真实 vLLM 修改，热点由 M7 证据决定）**。

## ✅ M8 —— Triton Kernel 能力建设（★核心验收 #5）（2026-07-21）

先出《M8 最终计划审计报告》（据 study_path 原典 630-660 + vLLM 真实源码，**纠正主 kernel 定位**：是 slot_mapping **生成**侧 `_compute_slot_mapping_kernel`，非消费侧 `reshape_and_cache` 散写）。实机验证 Triton 3.6.0 在 sm_120 正常 JIT。**本模块零 vLLM 源码改动**。

- **基础练习（达概念即止）**：vector_add / act_and_mul(SiLU-gate，即 M7 见到的 act_and_mul_kernel) / row-wise softmax（数值稳定，测非 2 次幂列宽 127/128/255/512/1024）——均与 PyTorch allclose。
- **主验收 slot_mapping 生成 kernel**：grid=(num_reqs+1)，每 program 处理一请求 token 区间、末 program 填 padding(-1)；公式 `slot=block_table[req,pos//bs]*bs+off`；参数化支持多请求(query_start_loc)、padding、CP interleave。
- **三路功能交叉验证 22/22 全绿**（自写 Triton / PyTorch 语义 Oracle / vLLM 官方 `_compute_slot_mapping_kernel`，int64 `torch.equal`）：随机多请求/单token/块内首末/跨块/non_contiguous_physical_blocks/shuffled/尾部mask(小tile=8)/ragged/padding(哨兵-999检测漏写,pad区全-1)/CP(world2 rank0·1、interleave2、world4)。
- **microbench(do_bench，计时仅含 kernel)**：自写 Triton **≈ vLLM 官方**(全尺寸重合，n=16384 28.4≈28.4us，**无数量级退化**)；PyTorch ref 小尺寸慢 3–4×(多 op 多次 launch，~30us 平台)、n=16384 收敛；两 Triton ≤2048 有 ~7us launch 地板→之后转访存主导（呼应 M7"小任务被 launch 开销主导"）。指标 `effective_payload_bandwidth_GBps` 明确为逻辑有效载荷、非硬件 DRAM 带宽。
- **如实说明**：生成侧允许重复 slot（前缀共享，属合法 index 计算；写竞争是消费侧问题）；CP 仅单卡纯 index 验证、未做真实多卡；未覆盖消费侧散写/多 KV-group/FP8/head-major。
- 产出：`scripts/m8_{kernels,test,bench}.py`、`notes/m8/triton_kernels.md`（含 Triton 编程模型讲解 + 生成/消费之分）、`outputs/m8/{correctness.json,microbench.csv,bench_summary.json}`、`logs/m8/*`。

**下一步（待确认）**：M9 —— Profiler 驱动的真实 vLLM 修改（★核心验收，优化对象由 M7 证据决定：decode/短 prefill 的 CPU 提交/H2D 路径 或 长 prefill 计算侧）。

## ✅ M9 —— Profiler 驱动的真实 vLLM 修改（★核心验收 #6，诚实无收益报告）（2026-07-21）

先出《M9 最终计划审计报告》并用 AskUserQuestion 确认方向（用户选"安全微优化+诚实报告"）。**关键诚实结论（实测支撑）**：decode 路径已被 vLLM 高度优化——compiled decode step ≈2.68ms 中 `_prepare_inputs`(CPU) 仅占 16%(0.42ms)，H2D buffer 已 pinned+non_blocking、`from_numpy` zero-copy、token_ids 不 H2D → **无大赢面**。

- **选定改动**：由 M7 证据选"合并 decode 路径两处同 dtype/同长度的 H2D 拷贝 `req_indices`+`query_pos`"（二者 `.gpu` 仅在同一行被消费、下游无其它读取 → 合并安全）。`gpu_model_runner.py` 加 `VLLM_LAB_M9_COALESCE` 门控（默认关、可回退、`# VLLM_INFERENCE_LAB_M9`）；备份 `backups/m9/`；隔离 diff `outputs/m9/m9_change.diff`。
- **isolated microbench**（do_bench）：合并省 ~3.5us/step（两次拷贝 vs 一次）→ 占 decode step ~0.13%，**提前量化天花板并预判 E2E 无效**。
- **correctness 完整**：关时 eager 输出与基线逐一致（**零副作用**）；开时输出与基线逐一致；compiled 下 on/off 输出完全相同；`collective_rpc` 验证 flag 生效。
- **E2E 单变量对照**（compiled decode，≥40 重复）：**机制成立**（cudaMemcpyAsync/step **11→10**，每步 −1）；但 **decode latency 差异 ±0.01-0.02ms ≪ stdev 0.11-0.30ms → 落在噪声内、无可测收益**；prefill 无退化。
- **结论（诚实）**：机制成立、正确性完整，但 E2E 收益被噪声淹没（根因：H2D 发行仅占 decode step ~0.13%，路径已高度优化）。按 ROADMAP **保留为无收益报告**，改动留作 env 门控可选项，**不伪造收益**。价值在完整的 profiler 驱动闭环方法论（热点→假设→microbench 预判天花板→toggle 可回退改动→correctness→单变量 E2E+机制计数→诚实结论）。
- 产出：`scripts/m9_{microbench,e2e}.py`、`notes/m9/optimization.md`、`outputs/m9/*`、`backups/m9/`、`logs/m9/*`；`vllm-src/gpu_model_runner.py`（门控改动，默认关零副作用）。

**下一步（待确认）**：M10 —— 量化对比（★核心验收，单卡可做）：BF16 baseline / 权重量化 / KV 量化，对比显存/吞吐/TTFT/TPOT/质量，源码聚焦反量化位置、Q/K/V 与 KV dtype、scale 存放与消费。

## 后续 Day 3–7 固定路线（来自 study_path.md，待逐日执行）
- **Day 3**：KV Cache 与 Block 管理（block 分配/释放/复用、block_table、prefix cache 命中、KV 不足与 preemption）。
- **Day 4**：Attention Backend（自动选择逻辑、P/D shape、GQA、Python 层与 kernel 边界）。
- **Day 5**：Worker / Model Runner / 执行模式（eager vs CUDA Graph / compile 对照）。
- **Day 6**：性能指标（TTFT/TPOT/延迟/吞吐/显存/利用率），仅在主链理解后进行。
- **Day 7**：学习总结（完整调用图、P/D 对照、Scheduler 状态图、KV 生命周期图、Attention 边界图、面试化表达、CUDA/Kernel 入口）。
