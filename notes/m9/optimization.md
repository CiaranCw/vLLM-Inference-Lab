# M9 · Profiler 驱动的真实 vLLM 修改（★核心验收 #6）

> 目标：由 M7 的 profiler 证据出发，做一次**真实的 vLLM 代码优化**，走完整闭环：
> **现象 → 根因 → 假设 → microbenchmark → 改代码 → correctness → E2E benchmark → 收益/退化（诚实）**。
> ROADMAP 明确：**无收益也保留失败报告**。本模块正是一个"**机制成立、正确性完整、但 E2E 收益落在测量噪声内**"的诚实结果——价值在于完整方法论 + 忠实结论 + 清晰的根因（为何天花板低）。
>
> 证据：`outputs/m9/{microbench.csv, e2e_off.json, e2e_on.json, correctness.json, m9_change.diff}`、`logs/m9/*`、`backups/m9/gpu_model_runner.py.pre_m9`。改动 env 门控（`VLLM_LAB_M9_COALESCE`）**默认关闭、关闭时零副作用**。复现见文末。

---

## 0. 现象与热点（承接 M7）
M7 结论：compiled decode 的残留 CPU 头号是 `cudaMemcpyAsync`（发行 + 可能同步）；每个 decode step 发行 **~6–11 次独立小 H2D `copy_to_gpu`**。M9 实测补充（monkeypatch，未改源码）：**compiled decode step ≈2.68ms 中 `_prepare_inputs`(CPU) 仅 0.42ms（16%）**，其余 84% 为 CUDA graph replay(GPU)+sampling+memcpy 发行+scheduler 更新。

## 1. 根因分析（据真实源码，排除伪优化）
逐步 H2D 的**实际拷贝带宽极小**（M7 trace：HtoD 0.002ms/step），成本在"**发行 N 次独立 async 拷贝**"的 CPU 开销。排除的伪优化（据 `ee0da84` 源码核实）：
- `torch.from_numpy(...)` 是 **zero-copy**（不分配/不拷贝）→ 预分配无收益。
- `token_ids_cpu_tensor` 只做 **CPU 上 `index_select`**、结果写入已 pinned 的 `input_ids.cpu` 再 H2D，**自身不 H2D** → pinning 无效。
- 主元数据 buffer 经 `CpuGpuBuffer`（`vllm/v1/utils.py:110`，默认 `pin_memory=PIN_MEMORY`）**已全部 pinned + non_blocking**；`copy_to_gpu(n)` **已按 slice 拷贝**。
→ **唯一 profiler 可辩护的机制 = 减少 H2D 发行次数（coalescing）**。

## 2. 假设
把 decode 路径两处**同 dtype(int64)、同长度(`total_num_scheduled_tokens`)**的 H2D——`req_indices` 与 `query_pos`（`gpu_model_runner.py` 每步各 `copy_to_gpu` 一次，且这两个 `.gpu` 只在同一行 `self.positions = num_computed_tokens[req_indices_gpu] + query_pos_gpu` 被消费，下游无其它读取）——**合并为一块 pinned 缓冲、一次 `copy_to_gpu`**，即可 **每步少发 1 次 `cudaMemcpyAsync`**。假设：这能带来可测的 decode step 加速。

## 3. Microbenchmark（isolated，先量天花板）
脱离引擎，`triton.testing.do_bench` 对比"两次独立 pinned int64 `copy_(non_blocking)`" vs "一次合并 `copy_(2N)`"（`outputs/m9/microbench.csv`）：

| total N | two_copies (us) | one_copy (us) | 节省 (us/step) |
|---:|---:|---:|---:|
| 1 | 63.78 | 60.26 | 3.52 |
| 8 | 62.75 | 60.61 | 2.14 |
| 64 | 64.03 | 60.16 | 3.87 |
| 256 | 65.84 | 59.78 | 6.06 |
| 1024 | 63.42 | 60.42 | 3.01 |

> 说明：do_bench 在本机 WSL 有 ~60us 的测量地板（单次拷贝实际是亚微秒级），但**相对节省 ~2–6us/step（中位 ~3.5us）是可信的每步量级**。
> **天花板判断**：compiled decode step ≈2680us，省 ~3.5us = **约 0.13%** → **理论上就低于测量噪声**。此时已可预判 E2E 无可测收益，但仍按 ROADMAP 完成"改代码→correctness→E2E"以坐实结论。

## 4. 改代码（toggle + 备份 + 隔离 diff）
- 位置：`vllm/v1/worker/gpu_model_runner.py`，标 `# VLLM_INFERENCE_LAB_M9`；env 门控 `VLLM_LAB_M9_COALESCE`（默认关）。**关闭时逐字节原行为**。
- 开启：init 分配 `self._m9_combined = _make_buffer(2*max_num_tokens, int64)`；每步把 `req_indices`、`query_pos` 写入其 `[:t]`、`[t:2t]` 后**一次** `copy_to_gpu(2t)`，切出两个视图供 `positions` 计算。
- 完整 diff 见 `outputs/m9/m9_change.diff`（由 `backups/m9/gpu_model_runner.py.pre_m9` 对比生成，**已与学习注释隔离，只含 M9 改动**）。核心：
```python
if self._m9_coalesce:
    _t = total_num_scheduled_tokens
    self._m9_combined.np[:_t] = req_indices
    self._m9_combined.np[_t : 2 * _t] = self.query_pos.np[:_t]
    self._m9_combined.copy_to_gpu(2 * _t)          # ← 一次 H2D，替代原来两次
    req_indices_gpu = self._m9_combined.gpu[:_t]
    query_pos_gpu   = self._m9_combined.gpu[_t : 2 * _t]
else:  # 原行为（两次 copy_to_gpu）
    ...
```

## 5. Correctness（完整）
`outputs/m9/correctness.json`：
- **零副作用（关闭时）**：`VLLM_LAB_M9_COALESCE` 未设，eager 单请求输出 token 与既有基线 **逐一致 IDENTICAL**（`[84648,19479,…,24360]`）。
- **正确性（开启时）**：`VLLM_LAB_M9_COALESCE=1` 输出与基线 **逐一致**；compiled 下 on/off 输出 token **完全相同**。
- **开关生效**：经 `collective_rpc` 读 `model_runner._m9_coalesce` → off=False / on=True。

## 6. E2E benchmark（单变量对照 + 机制确认）
compiled decode，M9 开 vs 关（独立进程，flag 在 init 读取），≥40 重复取 median（`outputs/m9/e2e_{on,off}.json`）：

| 指标 | M9 关 | M9 开 | 差异 | stdev（量级） |
|---|---:|---:|---:|---:|
| **cudaMemcpyAsync / step** | **11** | **10** | **−1（机制成立）** | — |
| decode b=1 median | 2.513ms | 2.526ms | +0.013 | ~0.15–0.30 |
| decode b=8 median | 2.783ms | 2.767ms | −0.016 | ~0.11 |
| decode b=64 median | 3.447ms | 3.464ms | +0.017 | ~0.19 |
| prefill 512 median | 7.333ms | 7.152ms | −0.18 | — |
| prefill 2000 median | 22.24ms | 21.98ms | −0.26 | — |

- **机制确认**：H2D 发行次数 **11→10（每步 −1）**，正是设计目标。
- **E2E 收益**：decode 各 batch 的差异均为 **±0.01–0.02ms，远小于 stdev（0.11–0.30ms）→ 落在测量噪声内**，**无可测收益**。prefill 差异同属噪声（M9 只改 decode 相关的两次拷贝，prefill 也走此路径但 total 更大、影响更小），**无退化**。

## 7. 结论（诚实）
- **机制成立、正确性完整**：合并确实每步少发 1 次 H2D；输出逐一致；关闭零副作用。
- **E2E 无可测收益**：省下的 ~3.5us/step 约为 decode step 的 **0.13%**，低于测量噪声 —— 与 §3 microbench 的天花板判断一致。
- **根因**：decode step 的时间由 **CUDA graph replay(GPU) + 不可约的每步开销**主导；H2D 发行只占极小比例，且 vLLM 已用 pinned+non_blocking+overlap 把它压到很低。**这条路径已被高度优化，安全的局部改动没有 E2E 空间**。
- **按 ROADMAP 保留为"失败/无收益报告"**：改动正确但收益在噪声内，不建议默认开启（保留为 env 门控的可选项）。
- **方法论价值**（本模块真正的产出）：完整走通了 profiler 驱动闭环——用 profiler 定位热点、用 isolated microbench **提前量化天花板并预判 E2E 无效**、用 toggle+备份+隔离 diff 做**零副作用、可回退**的真实改动、用单变量 E2E + 机制计数（memcpy/step −1）**坐实"机制成立但收益被噪声淹没"**，并诚实报告。

## 8. 退化场景与边界
- 仅改 decode 输入准备的两次同型 H2D；不改 GPU kernel / 控制流 / CUDA graph 语义（关闭时零变化，开启时 graph 仍正常，已验证输出一致）。
- 未覆盖：合并更多不同 dtype/长度的拷贝（收益更小、风险更高，不做）；async scheduling / spec decode / mrope 下 `req_indices/query_pos.gpu` 无其它下游读取（已核实），故合并安全；若未来这些路径新增对它们的读取，需重新评估。
- 数字仅本机（RTX 5080/sm_120）本配置成立。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
python scripts/m9_microbench.py                       # 机制天花板 -> outputs/m9/microbench.csv
# 正确性（关=零副作用 / 开=正确性；均应等于基线）
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/day1_baseline.py --max-tokens 16 --out outputs/m9/corr_off.json
VLLM_LAB_M9_COALESCE=1 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/day1_baseline.py --max-tokens 16 --out outputs/m9/corr_on.json
# E2E 单变量对照 + 机制计数
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/m9_e2e.py --m9 off --out outputs/m9/e2e_off.json
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/m9_e2e.py --m9 on  --out outputs/m9/e2e_on.json
```
改动隔离 diff：`outputs/m9/m9_change.diff`；回退：删除 `# VLLM_INFERENCE_LAB_M9` 两处或用 `backups/m9/gpu_model_runner.py.pre_m9` 还原。
