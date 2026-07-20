# M5 trace 改动清单

> 与前模块一致的改动登记；权威改动记录见 `patches/vllm-src.patch`（基准 commit ee0da84）。

## vLLM 源码改动（只读钩子，`VLLM_LAB_TRACE` 门控，默认关，回归零副作用）
- `vllm/v1/worker/gpu_model_runner.py`
  - `execute_model` 内、`pad_attn = cudagraph_mode == CUDAGraphMode.FULL` 之后（line ~4368）：
    `exec/dispatch` —— 每步 `num_reqs / num_reqs_padded / num_tokens_unpadded / num_tokens_padded / num_paddings / cudagraph_mode / uniform_decode / pad_attn`。
  - `_capture_cudagraphs` 的 `for batch_desc in batch_descriptors:` 首行（line ~6913）：
    `exec/capture` —— 每张被捕获图的 `runtime_mode / num_tokens / uniform_decode`。
  - 标记：`# VLLM_INFERENCE_LAB_TRACE`。备份：`backups/m5/gpu_model_runner.py.pre_m5`。

## 新增脚本
- `scripts/m5_exec_mode.py`：构建引擎后从存活 `vllm_config.compilation_config` 读解析后配置 + 跑单请求。

## 运行开关
- `VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_COMPONENTS=exec VLLM_LAB_TRACE_JSONL=<path>`；
  本机附加 `VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0`。

## 零副作用验证
- 不设 `VLLM_LAB_TRACE` 跑 `day1_baseline.py --max-tokens 16`，输出 token 与 Day1 基线 IDENTICAL（`outputs/m5/regress_eager.json`）。
- `logs/m5/source_status_{before,after}.txt`：vLLM-src 文件集合不变（21 M + `_lab_trace.py` 未跟踪）。
