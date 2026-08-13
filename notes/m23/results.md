# M23 · 实测结论（vLLM vs TensorRT-LLM，2×2 execution-mode）

> **状态**：**L1 COMPLETE**（2026-08-13）。Gate 0/1/2 + V-ctr preflight + 四组正式对照均已执行。
> **边界**：单卡 RTX 5080(sm_120) / WSL2 / Qwen2.5-0.5B-Instruct / in256-out128 / 两侧均在容器内 / 单副本。
> 设计冻结见 [`cross_engine_plan.md`](cross_engine_plan.md) v4；Gate 证据见 [`gate_results.md`](gate_results.md)；
> 上游后端迁移的理解见 [`backend_migration.md`](backend_migration.md)。

---

## 0. 一页结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | **TRT-LLM 的 PyTorch backend 在本机 sm_120 上可用**：engine 加载 + 真实生成均正常，无 kernel 缺失报错 | `outputs/m23/gate1.json` |
| 2 | **两条 lane 下 vLLM 都更快**：OFF lane 单请求 **+45.0%**、batch **+29.9%**；ON lane **+20.1%** / **+24.7%**（T 相对 V，正=更慢，四项 Bootstrap CI 全部排除 0） | `outputs/m23/cross_engine_summary.json` |
| 3 | **两侧从 optimized lane 获益都很大且幅度接近**：vLLM 吞吐 **2.85×**、TRT-LLM **2.97×** | 同上 |
| 4 | **冷启动 TRT-LLM 明显更慢**：**60.5–60.8s** vs vLLM **15.6s(OFF) / 26.6s(ON)** | 同上 |
| 5 | 整卡显存峰值 TRT-LLM 略低（约 −520 MiB），但**两侧显存预算参数语义不同**，这不是等价口径的显存对照 | §4 |
| 6 | **交叉验证成立**：vLLM 的 ON/OFF 吞吐比 **2.85×** 与 M6 实测的 **2.5–2.85×** 吻合；V-E 单 token 约 **7.8ms** 与 M7 的 **~7ms CPU 提交地板**吻合 | §5 |

**贯穿全文的表述纪律**（冻结稿 v4 §5.0.2）：
正式结论**只在 lane 内做**；**禁止跨 lane 相减**；
**任一差值都不得称为「纯 engine intrinsic speedup」**；本对照是 **stack-vs-stack，不是 kernel 级 A/B**。

---

## 1. 四组配置（生效值由引擎回读，不信入参）

| | **V-E** | **T-E** | **V-O** | **T-O** |
|---|---|---|---|---|
| 引擎 | vLLM `0.24.0+092c4842.dev` | TRT-LLM `1.3.0rc24` | 同左 | 同左 |
| 镜像 | `tritonserver:26.07-vllm-python-py3` | `tensorrt-llm/release:1.3.0rc24` | 同 | 同 |
| **execution mode** | `cudagraph_mode=NONE`<br>`compilation_mode=0` | `cuda_graph_config=None` | `cudagraph_mode=FULL_AND_PIECEWISE`<br>`compilation_mode=3` | `cuda_graph_config=CudaGraphConfig()`（默认） |
| dtype | `torch.bfloat16` | `bfloat16` | 同 | 同 |
| batch 上限 | `max_num_seqs=64` | `max_batch_size=64` | 同 | 同 |
| token 上限 | `max_num_batched_tokens=4096` | `max_num_tokens=4096` | 同 | 同 |
| 序列长度 | `max_model_len=2048` | `max_seq_len=2048` | 同 | 同 |
| chunked prefill | 关 | `enable_chunked_prefill=False` | 同 | 同 |
| 前缀复用 | `enable_prefix_caching=False` | `enable_block_reuse=False` | 同 | 同 |
| 显存预算 | `gpu_memory_utilization=0.6` | `free_gpu_memory_fraction=0.6` | 同 | 同 |

**输入完全一致**：两侧用**同一份 token ids**（host 侧 tokenize 一次落盘，`outputs/m23/bench_inputs.json`，
64 条 × 精确 256 token，互不相同）。Gate 2 已验证 TRT-P 接受 token ids 输入，这才使严格对齐成为可能。

**定长输出的实现方式不同但结果一致**：vLLM 用 `ignore_eos=True`，
TRT-LLM 的 `SamplingParams` **没有 `ignore_eos`**，改用 `min_tokens=max_tokens=128`。
四组的输出长度校验（`out_len_all_equal`）**全部通过**。

---

## 2. Primary metrics（只有这些承担正式结论）

| group | 冷启动(s) | 单请求 e2e 中位(ms) | batch64 墙钟(ms) | 请求吞吐(req/s) | 输出 token 吞吐(tok/s) | 整卡显存峰值(MiB) |
|---|---|---|---|---|---|---|
| **V-E** | 15.6 | **996.4** | 1359.1 | 47.1 | **6027.7** | 11869 |
| **T-E** | 60.8 | **1444.4** | 1764.9 | 36.3 | **4641.7** | 11352 |
| **V-O** | 26.6 | **252.9** | 476.3 | 134.4 | **17199.9** | 11974 |
| **T-O** | 60.5 | **303.9** | 593.9 | 107.8 | **13793.1** | 11414 |

样本量：单请求 3 warmup + **10 measure**；batch64 2 warmup + **5 measure**。

### 2.1 OFF lane：`T-E vs V-E`

| 指标 | V-E | T-E | ΔT−V | Bootstrap 95% CI | CI 排除 0 |
|---|---|---|---|---|---|
| 单请求 e2e 中位 | 996.4ms | 1444.4ms | **+448.0ms（+45.0%）** | [433.8, 463.0] | ✅ |
| batch64 墙钟中位 | 1359.1ms | 1764.9ms | **+405.8ms（+29.9%）** | [395.3, 436.6] | ✅ |
| 冷启动 | 15.6s | 60.8s | +45.2s | — | — |
| 输出 token 吞吐 | 6027.7 | 4641.7 | −23.0% | — | — |

### 2.2 ON lane：`T-O vs V-O`

| 指标 | V-O | T-O | ΔT−V | Bootstrap 95% CI | CI 排除 0 |
|---|---|---|---|---|---|
| 单请求 e2e 中位 | 252.9ms | 303.9ms | **+50.9ms（+20.1%）** | [48.8, 54.0] | ✅ |
| batch64 墙钟中位 | 476.3ms | 593.9ms | **+117.6ms（+24.7%）** | [104.8, 129.4] | ✅ |
| 冷启动 | 26.6s | 60.5s | +33.9s | — | — |
| 输出 token 吞吐 | 17199.9 | 13793.1 | −19.8% | — | — |

**→ 两条 lane 上 vLLM 都更快，且 ON lane 的相对差距比 OFF lane 小**（45.0%→20.1% 单请求）。
**不得**把这两个数字相减或合并，也**不得**称为引擎的内在速度差 ——
OFF lane 的差值里仍混着两套 runtime 的实现差异；ON lane 还叠加了**两侧优化栈内容本就不同**这一事实。

---

## 3. 一个两侧都成立的观察：optimized lane 的收益

| 引擎 | OFF → ON 单请求 | OFF → ON 输出吞吐 |
|---|---|---|
| vLLM | 996.4 → 252.9ms（**−74.6%**） | 6027.7 → 17199.9 tok/s（**2.85×**） |
| TRT-LLM | 1444.4 → 303.9ms（**−79.0%**） | 4641.7 → 13793.1 tok/s（**2.97×**） |

**两侧的相对收益量级接近**（2.85× vs 2.97×）。
这也说明：如果只跑一条 lane 就下结论，会把「有没有开图优化」这个变量混进「谁的引擎更快」里 ——
所以 2×2 设计是必要的，不是冗余。

---

## 4. 显存：只报整卡，分项 unavailable

整卡峰值：V-E 11869 / T-E 11352 / V-O 11974 / T-O 11414 MiB（TRT 侧低约 **520 MiB**）。

**但这不是等价口径的显存对照**，有两个必须声明的原因：

1. **预算参数语义不同**：vLLM 的 `gpu_memory_utilization=0.6` 是「**总显存**的 60% 给整个引擎」；
   TRT-LLM 的 `free_gpu_memory_fraction=0.6` 是「**可用显存**的 60% 给 **KV cache**」。
   数值同为 0.6，**含义不同** —— 所以这个 −520 MiB 只能读作「**在各自这样设置下**的峰值差」。
2. **分项不可得**：Gate 2 实测 TRT 侧主进程 `torch.cuda.memory_allocated()` 读到 **0**
   （executor 在独立 runtime 内）。因此 weights / KV / activations **无法给出两侧同口径的分项**，
   一律标 `unavailable`，**并禁止 `activation = total − weights − KV` 这种残差推导**
   （冻结稿 §5.3；否则只是把「不知道的部分」改名叫「激活」）。

---

## 5. 交叉验证：这批数字与既有模块自洽

两个独立的一致性检查（都不是刻意构造的）：

1. **vLLM 的 ON/OFF 吞吐比 = 2.85×**，与 **M6** 实测的「compiled 比 eager 快 **2.5–2.85×**」**吻合**。
2. **V-E 的单请求 996.4ms / 128 token ≈ 7.8ms/token**，与 **M7** 证实的
   「eager decode 的 **~7ms 地板 = CPU kernel 提交开销**」**同量级**。

→ 这说明容器内的 vLLM 行为与本 lab 历史主线一致，本批数据不是异常会话的产物。
（注：这是**量级一致性**检查，不是严格复现 —— 容器内 vLLM 版本与 host pinned 版本不同。）

---

## 6. 必须随 V-O / T-O 结论一起呈现的 limitation

V-ctr preflight 实测（`outputs/m23/vctr_preflight.json`）：

- V-E 与 V-O **各自** within-process **6/6**、across-process **6/6** —— 每条 lane 内部**完全确定性可复现**；
- 但**跨 lane exact-match 仅 1/6**。

因为每条 lane 各自确定，所以这 1/6 **不是随机抖动，而是两条 execution path 之间系统性、可复现的数值差异**。
人工检视 5 处分歧的完整文本：**全部连贯、切题、无退化**。

→ **只能说「未观察到明显 functional degradation」，不得升级成「两条 execution path 数值等价」** ——
本次证据恰恰表明它们**不等价**。

---

## 7. 局限与未做

- ⚠️ **stack-vs-stack，不是 kernel 级 A/B**：两个容器的 CUDA / PyTorch / 驱动栈不同
  （vLLM 容器 torch 2.x-NGC vs TRT-LLM 容器 torch 2.12.0a0/CUDA 13.2），
  引擎版本也属不同软件世代（vLLM `0.24.0+092c4842.dev` vs TRT-LLM `1.3.0rc24`，后者是 **Release Candidate**）。
- ⚠️ **显存不是等价口径**（§4）；分项 `unavailable`。
- ⬜ **未做跨引擎输出一致性比对**：本轮只在 vLLM 内部比了 V-E vs V-O，
  **没有**比较 TRT-LLM 与 vLLM 生成的 token 是否一致 —— 因此**不能**声称两引擎输出等价。
- ⬜ **未做在线服务口径**（L2）、**未做 Triton 集成**（L3）、**未做 Transformers 基线**（L4）。
- ⬜ 未扫模型规模（只有 0.5B）、未扫 shape（只有 in256/out128）、未扫并发度（只有 1 与 64）。
- ⬜ 未做 kernel 级 profiling，因此**没有解释**「为什么 TRT-LLM 在本机更慢」——
  只有现象与区间，没有机制归因。
- ⚠️ **`V-host` 未参与正式差值**：仅作与 M6–M18 历史主线连接的 supplementary anchor。

---

## 附：复现

```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate

python scripts/m23_gate.py --gate 0            # dry-run 复核；加 --run 才执行
~/bin/docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24
python scripts/m23_gate.py --gate 0 --run      # Gate 0/1/2
python scripts/m23_gate.py --gate 1 --run
python scripts/m23_gate.py --gate 2 --run
python scripts/m23_vctr_preflight.py           # V-ctr correctness 前置门
python scripts/m23_bench.py --build-inputs
python scripts/m23_bench.py --all              # 四组：V-E / V-O / T-E / T-O
python scripts/m23_summarize.py                # -> cross_engine.csv / _summary.json
python scripts/m23_figures.py                  # -> figures/fig14_m23_cross_engine.png
```
