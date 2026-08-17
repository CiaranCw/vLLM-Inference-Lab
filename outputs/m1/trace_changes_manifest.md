# M1 · Trace 改动清单（trace_changes_manifest.md）

本轮为观察运行时主链路而加入的 Trace 代码。**默认完全关闭**（`VLLM_LAB_TRACE` 未设或为 0 时零输出、零行为改变）。所有改动位于 `vllm-src`，与用户已有的中文注释改动相互独立。

## 开关（环境变量）

| 变量 | 默认 | 作用 |
|---|---|---|
| `VLLM_LAB_TRACE` | 0（关） | =1 才输出 |
| `VLLM_LAB_TRACE_MAX_STEPS` | 5 | 只追踪前 N 个调度 step |
| `VLLM_LAB_TRACE_LAYER` | 0 | attention 只追踪该 layer |

> 注：vLLM 会对未注册的 `VLLM_` 前缀变量打印一条 `Unknown vLLM environment variable detected` 警告（`envs.py:2004`），**无害**，不影响运行。

## 新增文件（1）

| 文件 | 说明 |
|---|---|
| `vllm/_lab_trace.py` | 统一 Trace 工具。`trace_event(component, event, fields, *, layer, once_key)`、`bump_step()`、`current_step()`、`layer_index()`、`enabled()`。写 stderr，前缀 `[VLLM_LAB_TRACE]`，含 pid/step；只打印 shape/dtype/标量；自身异常全部吞掉。git 状态：`?? vllm/_lab_trace.py`（未跟踪）。 |

## 插桩位置（7 处，均标 `# VLLM_INFERENCE_LAB_TRACE`）

| # | 文件 | 类 / 函数 | 位置 | 记录内容 | 目的 |
|---|---|---|---|---|---|
| 1 | `vllm/v1/core/sched/scheduler.py` | `Scheduler.schedule` | 方法开头（`self.current_step+=1` 后） | `bump_step()` | 推进全局 step |
| 2 | 同上 | `Scheduler.schedule` | `_update_after_schedule` 前 | total_scheduled、n_new/n_cached、running/waiting、preempted/finished、per_req(scheduled/computed_before/num_prompt/phase) | 观察每步调度决策与 prefill/decode |
| 3 | 同上 | `Scheduler.update_from_output` | `return engine_core_outputs` 前 | n_sampled_tokens、running/waiting、computed_after | 观察回填后的 num_computed_tokens |
| 4 | `vllm/v1/worker/gpu_model_runner.py` | `GPUModelRunner._prepare_inputs` | `return (logits_indices, ...)` 前 | num_reqs、total_scheduled、num_scheduled_tokens、query_start_loc、seq_lens、positions（均 CPU 侧，无 GPU sync） | 观察输入张量构造与 P/D 判据 |
| 5 | 同上 | `GPUModelRunner.execute_model` | `self._model_forward(...)` 调用前 | input_ids/positions shape、num_tokens_padded、cudagraph_mode | 观察前向输入与执行模式 |
| 6 | `vllm/model_executor/layers/attention/attention.py` | `Attention.forward`（通用层，backend 无关） | docstring 后、`if self.calculate_kv_scales` 前 | layer_name、num_heads/num_kv_heads/head_size、q/k/v shape（**仅 layer 0**） | 观察进入 backend 前的 q/k/v |
| 7 | `vllm/v1/attention/backends/flash_attn.py` | `FlashAttentionImpl.forward` | profiling 早返回后、`attn_type=` 前 | layer_name、q/k/v/kv_cache shape、num_actual_tokens、max_query_len、max_seq_len、query_start_loc/seq_lens/block_table/slot_mapping shape、fa_version、phase（**仅 layer 0**） | 观察 backend 入口与 KV/paged attn metadata |

> 说明：#4 只取 CPU 侧已算好的小数组（`query_start_loc.np`、`optimistic_seq_lens_cpu`、`positions_np`），不触发 GPU→CPU 同步；#6/#7 的 GPU 张量只记 `tuple(shape)`，不 `.tolist()`、不 sync。

## 是否影响正常逻辑

- **不影响**。每处均为 `try/except` 包裹、先判 `enabled()`；关闭时除一次 `enabled()` 的 env 读取外无开销，不改任何返回值/异常/控制流。
- 实证：`VLLM_LAB_TRACE=1` 与基线（关闭）产生的前两个 output token 完全一致（`[84648, 19479]`）→ Trace 未改变推理结果。

## 如何关闭 / 如何删除

- **关闭**：不设 `VLLM_LAB_TRACE`（或设 0）。代码保留但零输出。
- **删除**：
  1. 删除 7 处标有 `# VLLM_INFERENCE_LAB_TRACE` 的代码块（可用 `grep -n "VLLM_INFERENCE_LAB_TRACE"` 定位）；
  2. 删除 `vllm/_lab_trace.py`；
  3. 如需逐处核对，原始文件备份在 `vllm-inference-lab/backups/day1/*.orig`（md5 记录见该目录）。
- **禁止**用 `git restore/checkout/reset/clean/stash` 恢复（会破坏用户已有注释）。

## 备份与 git 状态

- 备份：`backups/day1/{scheduler,gpu_model_runner,attention,flash_attn}.py.orig`（插桩前原文件）。
- git 状态变化：BEFORE 21 个 `M` → AFTER 21 个 `M`（同一批注释文件，插桩加在其内）+ 1 个 `?? vllm/_lab_trace.py`。无其他非预期改动。未执行任何 `git add/commit/stash`。
