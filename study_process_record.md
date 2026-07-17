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

## 后续 Day 3–7 固定路线（来自 study_path.md，待逐日执行）
- **Day 3**：KV Cache 与 Block 管理（block 分配/释放/复用、block_table、prefix cache 命中、KV 不足与 preemption）。
- **Day 4**：Attention Backend（自动选择逻辑、P/D shape、GQA、Python 层与 kernel 边界）。
- **Day 5**：Worker / Model Runner / 执行模式（eager vs CUDA Graph / compile 对照）。
- **Day 6**：性能指标（TTFT/TPOT/延迟/吞吐/显存/利用率），仅在主链理解后进行。
- **Day 7**：学习总结（完整调用图、P/D 对照、Scheduler 状态图、KV 生命周期图、Attention 边界图、面试化表达、CUDA/Kernel 入口）。
