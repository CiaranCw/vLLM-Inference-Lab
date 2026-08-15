# M24 · 4×H20 实验矩阵（**已冻结 · 上机前**）

> **Freeze ID**：`FROZEN_FOR_H20_M24_2026-08-15`
> **状态**：READY_FOR_H20_EXECUTION。未跑过任何多卡数字；本文件不含结果。
> **硬件**：与 M22 同一类窗口——4× NVIDIA H20 / 单节点。上次实测见 `notes/m22/execution_record.md`。
> **版本**：H20 上次用的是 **vLLM 0.24.0 wheel**（非本机 editable `ee0da84`）。
> 结论不得与 RTX 5080 / 0.5B / `local_preflight` 混写。

面试纪律（写入每一份产出）：**在 vLLM 里看到 CP 配置 ≠ 掌握 Ring Attention。**
本树 DCP decode = Q-allgather + 本地 KV shard + LSE 合并，**不是**环形传 KV。

---

## 0. 为什么正式模型必须是 1.5B，不能是 7B / 0.5B

GQA 开 DCP 的源码门槛（`vllm/config/model.py:1184-1207`，非 MLA）：

1. `tensor_parallel_size > H_kv`
2. `dcp ≤ tp / H_kv`
3. `(H_q / H_kv) % dcp == 0`
4. 另：`tp % dcp == 0`（`parallel.py:503-507`）；DCP **不增加** GPU 数

| 模型 | H_q / H_kv | 4 卡上 DCP=2 |
|---|---|---|
| Qwen2.5-7B | 28 / **4** | **非法**：`tp=4` 不大于 `H_kv=4`。且 TP=4 时 KV **没有复制**，DCP 无事可做 |
| Qwen2.5-0.5B | 14 / 2 | **非法**：`H_q/H_kv=7`，`7 % 2 != 0` |
| **Qwen2.5-1.5B** | **12 / 2** | **合法**：`tp=4>2`，`max_dcp=2`，`6%2==0`。TP=4 时 KV 复制 2×；DCP=2 去掉复制 |

脚本在建引擎前读 `config.json` 做同样的算术，不合法则 **立刻退出**，不建引擎。

本机 HF 缓存里 **没有完整 7B 权重**（hub 目录仅 12K）。正式包只带 1.5B。

---

## 1. 范围与优先级（时间不够就停在 P0）

```
preflight → DCP smoke → DCP formal (P0) → DCP profile (P1) → PCP 负例 (P2) → [可选] P-D smoke (P3)
```

| 优先级 | 内容 | 本轮 |
|---|---|---|
| **P0** | TP=4 固定，**只改 `decode_context_parallel_size ∈ {1,2}`**。D1 KV/显存 + D2 长上下文稳态 decode | **必须** |
| **P1** | DCP=2 的 collective 证据（不预设 kernel 名；记录实际路径） | 有时间就做 |
| **P2** | `prefill_context_parallel_size=2` **预期失败**（本版本无 `supports_pcp=True`），保存完整 traceback | ≤10 分钟 |
| **P3** | P-D 1P1D / ExampleConnector smoke | **仅当 P0 数据已落盘且仍有余时**；不得把未完成的 P-D 写成落地 |
| ✗ | Ring Attention 实现或 Nsight | **禁止**（本版本未实现） |
| ✗ | 下载任何新模型 / 编译 CUDA/C++ / 改 vLLM 产品源码 | **禁止** |
| ✗ | 用 DCP 数字和 M22 7B TP 数字相减 | **禁止**（模型不同） |
| ✗ | 把 DCP 加速写成 Ring Attention 收益 | **禁止** |

---

## 2. 冻结配置（唯一变量 = DCP）

| 项 | 值 | 为什么 |
|---|---|---|
| 模型 | **本地目录** `Qwen2.5-1.5B-Instruct`（HF 名仅作对照） | 见 §0 |
| `tensor_parallel_size` | **4**（两档相同） | DCP 复用 TP GPU |
| `decode_context_parallel_size` | **1 vs 2** | 唯一自变量 |
| `prefill_context_parallel_size` | **1**（P0/P1） | PCP 未实现 |
| `cp_kv_cache_interleave_size` | **1** | token 级交错；不扫 interleave |
| `dcp_comm_backend` | 默认 `ag_rs` | 不扫 a2a |
| `max_model_len` | **16384** | 最长 prompt 8192，且必须 **严格大于** prompt（`input_processor.py:423-432`） |
| `gpu_memory_utilization` | 0.85 | 与 M22 同，便于看 KV 容量 |
| `max_num_seqs` / `max_num_batched_tokens` | 256 / 8192 | 写死，不随 world size 派生 |
| `enforce_eager` | True | 与既有模块一致，排除 Graph |
| `enable_prefix_caching` | False | 重复 prompt 会污染 |
| `async_scheduling` | **False** | 默认 None 会随 executor 变（M22 已踩） |
| `disable_log_stats` | True | 不依赖 stats |
| seed / temperature | 0 / 0 | 可复现 |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | **0** | 否则 scheduler 内省失效 |
| `VLLM_USE_FLASHINFER_SAMPLER` | 0 | 与既有实验一致 |

伴随变量（必须在结论里声明，无法做成单变量）：`distributed_executor_backend` 在 TP=4 下为 `mp`。两档 DCP 都是 TP=4，这条应对齐。

**三层证据**：传入 → `effective_cfg` 引擎回读 → 不变量。不信「我传了什么」。

---

## 3. 测量（一个引擎跑完 D1+D2，每档一个进程）

### D1 · 显存 / KV（建引擎后采一次）

per-rank：weight / available_kv / num_gpu_blocks / torch allocated。走 `collective_rpc`，**不用 driver 的 `torch.cuda.memory_allocated()`**。

**预期观察（不是 invariant）**：DCP=2 相对 DCP=1，KV 可服务 token 数应明显上升（复制因子 2→1）。不符则记录并解释，不自动 FAIL。

### D2 · 长上下文稳态 decode（主指标）

DCP 的卖点是沿 T 切 KV，短 prompt 上看不到。网格：

| 维 | 值 |
|---|---|
| `prompt_len`（= 上下文） | **1024, 4096, 8192** |
| `batch` | **1, 8** |
| `max_tokens` | 32 |
| warmup / measure | **5 / 15**（`--quick` 才 1/3） |

构造：同 batch 内 prompt 等长、`ignore_eos=True`、同步结束。
筛选：纯 decode step = batch 个请求各 1 token；丢弃前 3 个瞬态。
核对：`steady_batch_stable` 必须为 True。

主指标：`steady_step_ms_median`。差值用 median difference 的 Bootstrap 95% CI（10000，seed=0）。

假说（**待数据检验，不是结论**）：

- 短上下文 / 小 batch：DCP=2 可能更慢（Q-allgather + LSE 固定税）。
- 长上下文：本地 KV 为 1/2，decode step 有机会更快或持平。
- CI 含 0 → 只能写「未观察到超出噪声的差异」，不能写「没有收益」。

口径：in-process 延迟 **不要叫 TTFT**。

### P1 · 通信证据

只对 **DCP=2** 开 `ProfilerConfig(profiler="torch")`。不预设 `ncclDevKernel_*` 或 Ring 的 send/recv。
记录 chrome trace 里实际出现的 collective / allgather / lse 相关事件。找不到就标 observation gap，**不要编**。

### P2 · PCP 负例

`prefill_context_parallel_size=2`，预期在 `check_attention_cp_compatibility` 断言失败。
保存命令 + 原始 stderr。成功建起来反而是需要调查的事（版本漂移），**不当成功**。

### P3 · P-D（可选）

官方：P-D **不提升吞吐**；当前注释限制 1P1D；ExampleConnector 走共享目录，**不是**生产 NIXL。
若做：只允许 smoke（KV 能从 producer 落到 consumer、decode 能继续）。
**不得**在 ExampleConnector 上报告「P-D 加速」。

---

## 4. 产物

| 类型 | 路径 |
|---|---|
| 逐样本 | `outputs/m24/raw/<tag>_dcp{1,2}_run.jsonl` |
| 汇总 | `outputs/m24/<tag>_{memory_kv,decode,decode_effect}.csv` |
| 环境 | `logs/m24/environment.txt` |
| PCP 负例 | `outputs/m24/raw/<tag>_pcp_negative.json` |
| 回传包 | 编排器 `--package`（或整目录 `outputs/m24` + `logs/m24` 拷回） |

回传必须由**本地主动 pull**（H20 未必能反连）。T+10 分钟先做 `--transfer-test`。

---

## 5. 诚实边界

- H20(sm_90) 绝对值 **不可** 与 5080(sm_120) 横向比较。
- 只有真实跑过的 DCP 档可标 E/H。DCP=2 建不起来 → 记录原因，**可以只交 DCP=1**，不得伪造 DCP=2。
- 不为好看的 scaling 改网格。
- Ring Attention / PCP 实践：本轮结束后仍标 ⬜，除非 P2 以外另有成功证据（本版本预期没有）。
