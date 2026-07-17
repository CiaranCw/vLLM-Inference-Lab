# Day 1 · 基线真实执行组件分析

> 数据来源：`logs/day1/baseline_debug.log`（`VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_LOGGING_LEVEL=DEBUG`）、`outputs/day1/baseline_result.json`、`logs/day1/vllm_collect_env.txt`，并对照 `vllm-src` @ `ee0da84` (v0.24.0) 源码。所有结论均给出日志/源码依据，非纯静态推测。

claude --resume 16706085-d792-413c-af83-8776ca83b24b

## 运行环境

- OS：Ubuntu 26.04 WSL2；GPU：NVIDIA GeForce RTX 5080（Blackwell，sm_120，16 GiB）。
- Python 3.12.13；torch 2.11.0+cu130（CUDA 13.0）；vLLM `0.24.1.dev0+gee0da84ab`（editable → `vllm-src`）。
- `vllm.__file__ = /home/ciaran/projects/vllm-src/vllm/__init__.py`（确认运行的是本地源码）。
- **环境规避**：WSL 无 nvcc，FlashInfer 采样模块在 `profile_run` 的 dummy sampler 里触发 JIT 编译失败（`flashinfer/jit/cpp_ext.py:get_cuda_path → RuntimeError: Could not find nvcc`）。设 `VLLM_USE_FLASHINFER_SAMPLER=0` 改用原生采样器规避；不改模型、不改 attention backend、不编译 vLLM 原生扩展。

## 基线参数
- model=`Qwen/Qwen2.5-0.5B-Instruct`（HF 原始格式，非量化）；dtype 自动解析为 `torch.bfloat16`。
- max_model_len=2048；max_tokens=16；gpu_memory_utilization=0.50；tensor_parallel_size=1；temperature=0；seed=0；enforce_eager=True。
- 结果：num_prompt_tokens=**11**，num_output_tokens=**16**（finish_reason=`length`），输出文本 `' KV Cache 是一种用于在内存中缓存键值对的缓存'`。
- 耗时：engine_init≈14.2s；generate≈74.1s（**冷启动**，主要被推理期 Triton JIT 编译拖慢，见下）；显存 after_init：proc_allocated≈7550 MiB，device_used≈9652 MiB。

## Engine
- **V1 LLM engine**，进程内运行（`VLLM_ENABLE_V1_MULTIPROCESSING=0`）。
  - 依据：`v1/engine/core.py:258` “Initializing a V1 LLM engine (…)”。
- 入口 → 引擎链：`vllm/entrypoints/llm.py::LLM.generate` → `offline_utils.py::_run_engine` 循环调用 `llm_engine.step()` → `v1/engine/llm_engine.py::LLMEngine.step` → `v1/engine/core.py::EngineCore.step`（schedule→execute→update）。
- **异步调度 + batch queue**：`config/vllm.py:1006` “Asynchronous scheduling is enabled.”；`v1/engine/core.py:349` “Batch queue is enabled with size 2”。对应 V1 的**延迟执行**：`execute_model()` 准备并缓存状态后返回，采样在 `sample_tokens()` 完成。
- Executor：**UniProcExecutor**（单进程单 worker）——`v1/executor/uniproc_executor.py::collective_rpc` 在 driver_worker 上直接 `run_method`（首跑 traceback 可见）。

## Scheduler
- 类：`vllm/v1/core/sched/scheduler.py::Scheduler`。
- **Chunked prefill 启用**，`max_num_batched_tokens=8192`（`config/scheduler.py:252`）；prefix caching 默认启用（`arg_utils.py:2495` “Enabling prefix caching by default”，`enable_prefix_caching=True`）。
- 关键方法：`schedule()`（返回 `SchedulerOutput`，定义于 `v1/core/sched/output.py`）与 `update_from_output()`（回填 `num_computed_tokens`、追加 output token、判定 finish）。
- Prefill/Decode 判别（调度层）：`request.num_computed_tokens < num_prompt_tokens` 为 prefill，`==` 后为 decode（`v1/request.py`）。本轮日志中 prefill 一次性调度 11 token，之后每步 decode 调度 1 token。

## Worker
- 类：`vllm/v1/worker/gpu_worker.py::Worker`。
- `Worker.execute_model` 委托给 `self.model_runner`；`determine_available_memory()` 在 `profile_run` 中测显存（`gpu_worker.py:438/496-508`）。
- 显存画像：weights 0.93 GiB，peak activation 0.58 GiB，non-torch 0.08 GiB；**Available KV cache memory 6.37 GiB**（`gpu_worker.py:508`）。

## Model Runner
- 类：`vllm/v1/worker/gpu_model_runner.py::GPUModelRunner`。
- 主流程：`execute_model()`（延迟执行，返回 None）→ `_update_states` → `_prepare_inputs`（构造 input_ids/positions/query_start_loc/seq_lens/slot_mapping/block_table 与 attn metadata）→ `_model_forward`（调用 `self.model(...)`）→ `compute_logits` → 缓存 `ExecuteModelState`；采样在 `sample_tokens()` → `_sample` → `self.sampler(...)`。
- 每步执行日志：`gpu_model_runner.py:4307` “Running batch with cudagraph_mode: NONE, batch_descriptor: BatchDescriptor(num_tokens=…)”。
  - `num_tokens=11` → **Prefill**（11:28:02）；随后 `num_tokens=1` ×15 → **Decode**。
- Model：`vllm/model_executor/models/qwen2.py`，架构 `Qwen2ForCausalLM`（`config/model.py:598` “Resolved architecture: Qwen2ForCausalLM”）；24 层，注意力模块 `Qwen2Attention.forward` 内 `self.attn(q,k,v)` 进入通用 `Attention` 层。

## Attention Backend
- **实际启用：`FLASH_ATTN`（`vllm/v1/attention/backends/flash_attn.py::FlashAttentionBackend` / `FlashAttentionImpl`），FlashAttention version 2。**
  - 依据：`platforms/cuda.py:480` “Using FLASH_ATTN attention backend out of potential backends: ['FLASH_ATTN','FLASHINFER','TRITON_ATTN','FLEX_ATTENTION'].”；`flash_attn.py:673` “Using FlashAttention version 2”。
- 选择逻辑：`vllm/platforms/cuda.py::_get_backend_priorities`——sm_120（`device_capability.major==12`）非 MLA 分支的候选优先级为 `FLASH_ATTN → FLASHINFER → TRITON_ATTN → FLEX_ATTENTION`（cuda.py:154-157）；`get_valid_backends` 剔除无效项（`cuda.py:439` 显示 `TURBOQUANT` 因 kv_cache_dtype 不支持被排除），取最高优先级有效 backend。
- **未主动设置** `VLLM_ATTENTION_BACKEND`，为自动选择结果。selector 配置：head_size=64、dtype=bf16、use_mla=False、attn_type=DECODER。
- 采样器：因 `VLLM_USE_FLASHINFER_SAMPLER=0`，走原生 top-k/top-p（非 FlashInfer JIT）。

## KV Cache 配置
- kv_cache_dtype=auto（bf16）；**GPU KV cache size 556,816 tokens**（`v1/core/kv_cache_utils.py:2158`）；对 2048-token 请求 **Maximum concurrency 271.88x**（556816/2048）。
- block_size：selector 阶段为 `None`，随后按 V1 FlashAttention 默认解析（**16**，待 Trace 中实测确认）；由此推算 num_gpu_blocks ≈ 556816/16 ≈ **34,801**（推算值，Trace 复核）。
- 管理链：`v1/core/kv_cache_manager.py` / `block_pool.py` / `single_type_kv_cache_manager.py`；paged attention 通过 block_table + slot_mapping 读写。

## CUDA Graph 与 compile 状态
- **均关闭**（enforce_eager=True）：`config/vllm.py:1062` “Enforce eager set, disabling torch.compile and CUDAGraphs”；`CompilationMode.NONE`、`cudagraph_mode=CUDAGraphMode.NONE`；每步日志 `cudagraph_mode: NONE`。
- 但**推理期仍有 Triton JIT**：`triton_utils/jit_monitor.py:106` 警告 `_compute_slot_mapping_kernel` 在推理中 JIT 编译（延迟尖峰）。时间线：prefill 11:28:02 → 首个 decode 11:28:29（~27s，slot mapping kernel JIT）→ 第二个 decode 起集中在 11:29:16（其余 decode 内核 JIT 后，剩余 step 同一秒内完成）。这解释了 generate≈74s 的冷启动开销主要来自 JIT 而非算力。

## 当前仍不确定的问题（留待 Day1 后续 / Day2）
1. block_size 的确切值与 num_gpu_blocks（Trace 中直接读 `attn_metadata` / block_table 确认）。
2. Prefill 与 Decode 在 `_prepare_inputs` 的 `query_start_loc`/`seq_lens`/`positions` 及 backend 入口 q/k/v/kv_cache/block_table 的确切 shape（Step 5 Trace 验证）。
3. GQA 具体切分：Qwen2.5-0.5B 的 num_attention_heads 与 num_key_value_heads 在运行时的 q/k/v head 维度（Trace layer 0 记录）。
4. async scheduling（batch queue size 2）对单请求 step 序列的影响（是否与同步调度一致）。
5. 推理期 Triton JIT 的完整清单与可否通过 warmup 覆盖（性能话题，留待 Day6）。
