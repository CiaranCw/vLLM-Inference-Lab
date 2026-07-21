# ROADMAP —— vLLM Inference Lab · 第一阶段学习覆盖清单

> **范围声明**：本 ROADMAP **只覆盖当前学习路线的第一阶段** —— vLLM 单机（单卡）主链路理解、性能分析、第一次真实优化，以及基础求职项目。
> 以下**后续扩展路线不在本文件范围**（待第一阶段完成后另行规划，用户尚未提供该部分路线）：Triton Inference Server 深度部署、PP/DP/EP、TensorRT-LLM 完整实践、Context Parallel/Ring Attention、Prefill/Decode 分离、模型压缩专题、百亿或多节点集群规划。
>
> **设计**：按**阶段与验收条件**推进，**不按天/周/月排期**；做完一项直接开下一项。
> **状态图例**：✅ 完成　🔄 进行中　⬜ 未开始　⏸ 后续扩展（不在本阶段）
> **使用约定**：可对某轮限速——`[推进]`=只往下一个未完成模块走、不加额外实验；`[深化]`=允许在当前模块内追加实验。每轮结束我会标注本轮是"推进"还是"深化"，并给出剩余覆盖。

---

## 能力自评（分项，非笼统）
| 领域 | 层级 |
|---|---|
| Scheduler / Continuous Batching | **能证** ✅ |
| KV Cache / Block 管理 | **能证** ✅ |
| Attention Backend | **能证** ✅（M4 闭环 trace） |
| 执行模式（eager / compile / CUDA Graph） | **能证** ✅（M5：eager=NONE；默认 O2=VLLM_COMPILE+FULL_AND_PIECEWISE，实测捕获/回放/padding） |
| 端到端主链路（入口→输出闭环） | **能证** ✅（M4 已把单请求整条链串成一次运行的证据） |
| 性能测量（latency / throughput / 显存·KV） | **能测量** ✅（M6：进程内闭环 benchmark，prefill/decode/吞吐/prefix/KV，warmup+重复+P95） |
| 瓶颈定位（profiler / CPU vs GPU / launch） | **能定位** ✅（M7：torch.profiler 拆 step，证实 decode=launch 受限 / 长 prefill=计算受限） |
| 写 kernel（Triton） | **能写** ✅（M8：slot_mapping 生成 kernel 三路交叉 22/22，性能≈vLLM 官方） |
| 能改 / 能优 | **能改** ✅（M9：profiler 驱动、可回退、正确性完整的真实改动；本例 E2E 收益在噪声内，诚实保留为无收益报告）；**能优**：方法论完备，待有正空间的目标再证正收益 |

能力阶梯：能读 → 能串 → 能证 → **能改** → **能优**。第一阶段目标是把主链路推到"能改/能优"。

---

## 持续维护任务（贯穿所有模块，不是单独模块）
- **`contribution_candidates.md`（待建）**：从现在起持续记录可回馈上游的候选——测试缺口、文档与行为不一致、明确 bug、Benchmark/可观测性改进、性能修复点。M13 只负责把其中一个**正式提交**，而非到 M13 才开始找。
- **`notes/evidence_index.md`（已建，持续维护）**：证据索引，汇总"每个模块结论 → 对应 trace/log/jsonl/图 文件"，便于随时核查。
- **每完成一个模块**：同步更新 `ROADMAP.md`、`README.md`、`evidence_index.md`，并做一次 git commit（保持仓库与进度一致）。
- **项目整理不积压**：README、图表、复现说明随做随更新，M16 只做最终删减、重组、简历与面试表达。

---

## 阶段 A · 环境 + 主链路理解（对应 study_path 第三~五章）

### M0 环境与可复现基础设施　🔄
- ✅ 固定 vLLM commit（`ee0da84` / v0.24.0）、模型（`Qwen/Qwen2.5-0.5B-Instruct`）与关键配置（enforce_eager、`VLLM_USE_FLASHINFER_SAMPLER=0`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、seed=0、temperature=0）。
- ✅ 环境记录：`vllm_dev_environment.md`、`logs/day1/vllm_collect_env.txt`、`logs/day1/nvidia_smi_*.txt`（GPU/Driver/CUDA/PyTorch/vLLM）。editable 安装已验证 `vllm.__file__ → vllm-src`。
- ✅ 统一 trace 基础设施：`vllm-src/vllm/_lab_trace.py`（统一事件 schema `component/event/fields`、`VLLM_LAB_TRACE*` 开关、JSONL 输出、组件过滤、layer/once/step 限制）。
- ✅ trace 默认关闭不改变执行语义：多次 E1 回归验证"归一化后逐行一致"（钩子零副作用）；每次改源码前备份到 `backups/`；`logs/*/source_status_{before,after}*.txt` 记录 git 状态。
- 🔄 待补（据实标记，尚未落地）：
  - 统一**复现入口**（目前分散在各 notes 的复现命令；`scripts/smoke_import.py` 仅做导入自检）。
  - **正式 correctness invariants + 回归脚本**（目前 E1 回归是内联 bash 的 ad-hoc 比对，未固化为脚本）。
  - `notes/evidence_index.md`（已建）；`contribution_candidates.md`（见"持续维护任务"，待建）。
  - 目录/命名当前为 `dayN` 制（M1–M3 沿用）；后续模块建议改为**主题命名**，此不一致待统一。

### M1 单请求推理主链路 + Prefill/Decode 运行时 Trace　✅
- 覆盖：LLM.generate→Engine→Scheduler→Worker→ModelRunner→Qwen2 forward→Attention→Backend→Sampling→输出；Prefill/Decode 调度数据与 Tensor shape 对照；实际 backend 确认。
- 产出：`scripts/day1_baseline.py`、`notes/day1/{baseline_analysis,runtime_trace}.md`、`outputs/day1/*`、`logs/day1/*`。

### M2 Scheduler 与 Continuous Batching　✅
- 覆盖：多请求错峰混批、token 预算、running/waiting 排队准入、chunked prefill；E5 追加 blocked-decode vs prefill 的预算共享 / 槽位争用。
- 产出：`scripts/day2_engine_driver.py`、`scripts/day2_build_timeline.py`、`notes/day2/{continuous_batching,e5_blocked_decode_vs_prefill}.md`、`outputs/day2/*`。

### M3 KV Cache 与 Block 管理　✅
- 覆盖：block 分配 / block_table 逻辑→物理 / 跨块边界；prefix 命中(hash/ref_cnt/共享/对齐)；释放→cached-free→复用；KV 不足→抢占→重算(prefix on/off)；K5 追加抢占后尾部优先淘汰、前缀优先保留。
- 产出：`scripts/day3_kv_driver.py`、`notes/day3/{kv_cache_block_management,k5_eviction_order}.md`、`outputs/day3/*`。

### M4 Attention Backend　✅
- **必做（当前真实 backend 主线）**：当前机器最终选中的 backend；Q/K/V reshape；KV Cache tensor shape；slot_mapping 如何进入 KV cache update；block_table 如何进入 attention kernel；Prefill/Decode 如何通过 metadata 表达；Python 层与底层 kernel 的边界。
- **可选（不作退出条件）**：穷举所有 backend、强制多 backend 完整性能对照、深挖 FlashInfer 与 sm_120 兼容、其他平台/量化 Query/KV sharing 等支线。
- **退出标准（已达成）**：以真实 Trace 证明 **SchedulerOutput → KV 写入(slot_mapping) → Attention 读取(block_table)** 的完整闭环。
- 产出：在 `flash_attn.py` 加 2 处只读钩子（`attention/kv_write`、`attention/attn_read`，layer0，回归零副作用）；`notes/m4_attention_backend.md`、`outputs/m4/m4_closed_loop.jsonl`、`logs/m4/m4_closed_loop.log`；备份 `backups/m4/flash_attn.py.pre_m4`。

### M5 执行模式（本机实际路径验证）　✅
- 覆盖：确认当前 commit + 当前硬件**实际启用**的执行模式；`enforce_eager` 具体改变了什么；是否使用 torch.compile；CUDA Graph 是 FULL / PIECEWISE / NONE；观察 capture、padding、replay 与非图区域；**只对本环境实际可运行的模式提供证据**。
- **边界**：M5 只验证执行路径；正式性能结论放到 M6/M7。
- **实测结论（本机 sm_120）**：`enforce_eager=True`（基线）→ `mode=NONE / cudagraph_mode=NONE`（纯 eager，`vllm/config/vllm.py:1061` 一票否决 + WARNING）；`enforce_eager=False`（vLLM 默认 O2）→ `mode=VLLM_COMPILE`(inductor) + `cudagraph_mode=FULL_AND_PIECEWISE`，**sm_120 上真实跑通未降级**：torch.compile 9.44s（Dynamo 3.09+Inductor 4.36，AOT 磁盘缓存）；捕获 **51 PIECEWISE(mixed)+35 FULL(decode)**、0.22 GiB；运行时 **prefill→PIECEWISE(pad 11→16, attn 图外) / decode→FULL(pad_attn=True, 整步含 attn 入图)**；compiled 与 eager 贪心输出第 5 token 起分叉（编译数值差异，非 bug）。
- 产出：`gpu_model_runner.py` 加 2 处只读钩子（`exec/dispatch`、`exec/capture`，回归零副作用）；`scripts/m5_exec_mode.py`；`notes/m5/execution_mode.md`、`outputs/m5/*`、`logs/m5/*`；备份 `backups/m5/gpu_model_runner.py.pre_m5`。

---

## 阶段 B · 从"能解释"到"能测量 / 能改 / 能优"（对应 study_path 第六~七章，价值主体）

### M6 离线 / 闭环引擎 Benchmark　✅　★核心验收
- 覆盖：Prefill latency、Decode step latency、token throughput、显存与 KV 容量；对照 batch size / 输入长度 / prefix cache / eager-vs-compiled；预热与正式分离、重复≥15、报 median/mean/stdev/P95/min、固定 model/commit/dtype/seed。
- **实测结论（本机，仅本配置成立）**：decode 单步 latency 近平、吞吐近线性（compiled 全 batch 快 2.5–2.85×，逐 batch 1–64 实测全走 FULL 图）；prefill ≤512 有 ~7ms 固定开销地板，compiled 仅 ≤128（PIECEWISE 图）提速 ~1.4–1.7×、≥256 超本配置捕获上限 128 回退 NONE≈eager；prefix 命中省 94% 计算 token 但首 token 延迟配对差在噪声内（+0.03ms）、prefix ON 短 prompt 反略慢于 OFF（省计算≠省延迟）；KV≈12 KiB/token（实测 tensor-sum = 理论），KV 池 eager 73.2 万 / compiled 72.2 万 token（compiled 少因 cudagraph 占显存），**"最大并发 357×"是 KV 容量等价值，运行时并发由 max_num_seqs=64 限制**。**"开销地板=CPU/launch?" 与 "decode 是否达带宽上限" 留 M7 profiler 证。**
- **边界**：只测量+解释现象，不做 profiler 根因、不断言 compute/memory-bound、不下跨机结论；`engine_first_token_latency`≠在线 TTFT（在线留 M12）。
- 产出：`scripts/m6_{benchmark,run_all,summarize}.py`、`notes/m6/benchmark.md`、`outputs/m6/{baseline.csv,benchmark_config.yaml,raw/,config/,summary/}`、`logs/m6/*`。**本模块零 vLLM 源码改动**（纯 driver + 引擎自省 + 只读 wrapper）。
- 与 M12 区别：M6 = 离线闭环引擎基准；M12 = 在线开环 serving + SLO。**不重复做两遍相同压测。**

### M7 Profiler 定位瓶颈　✅　★核心验收（≥2 类 workload）
- 覆盖两类不同 workload，各出一份分析：
  - **Decode-heavy / 小 batch**：CPU 开销、kernel launch gap、H2D、Sampler、CUDA Graph。
  - **Prefill-heavy / 长输入**：GEMM、Attention、GPU 利用率、带宽、大计算 kernel。
  - 工具：PyTorch Profiler（CPU/GPU 时间、kernel、launch、chrome trace）。**Nsight Systems 在本 WSL2 无法采集 GPU kernel（已实测 CUPTI device-trace 不可用），如实记录并以 torch.profiler 替代覆盖时间线/gap。**
- **每份分析必须形成**：现象 → Profiler 证据 → 原因假设 → 单变量对照实验 → 最终结论。**仅导出 timeline 不算完成。**
- **实测结论（本机，仅本配置成立）**：① **decode 的 ~7ms 地板 = CPU kernel 提交开销**——eager gpu_frac 0.23（GPU 仅忙 1.70ms/7.44ms wall）、每步 257 次 CPU 提交、`cudaLaunchKernel` 1.77ms 为头号 CPU、GPU 头号是 GEMV；单变量对照 compiled(FULL 图) 总提交 257→15（含 1 次 graph 回放）、wall 7.44→2.50ms（M6 假设证实且更强）。② **长 prefill = GPU 计算受限**——gpu_frac 0.88、cutlass tensor-core GEMM 14.6ms + flash attention 主导；compiled 总提交 257→255 基本不变（只是 runtime↔driver API 转移，NONE 模式无图）而 wall 不变（证明提交非瓶颈）。③ decode 访存受限（GEMV 读权重、batch↑ GPU 仅微增）为受支持假设，ncu 不可用故不下定论。
- **统计口径（经审查修订）**：GPU 时间取 chrome trace `cat=="kernel"` 事件的**区间并集**（不对 key_averages 求和，避免高层算子/底层 kernel 双计）；CPU 提交计数覆盖 **runtime(cudaLaunchKernel)+driver(cuLaunchKernel)+graph(cudaGraphLaunch)** 三类。
- 产出：`scripts/m7_{profile,run_all}.py`、`notes/m7/profiling.md`、`outputs/m7/profile_{eager,compiled}.json`、`logs/m7/*`（chrome trace 体积大未入库、可复现）。**本模块零 vLLM 源码改动**（纯 profiler 外挂）。

> **M7 之后分出两条并行能力线（M8 与 M9 无硬串行依赖）：**

### M8 Triton Kernel 能力建设　✅　★核心验收（slot-mapping 类 kernel）
- 基础练习（**达到核心概念即停，不无限扩展**）：Vector Add、Fused Activation(SiLU-gate)、Row-wise Softmax（含非 2 次幂列宽）。
- **主要验收 = 简化版 slot mapping kernel**：明确为**生成侧**（simplified `_compute_slot_mapping_kernel`：`positions+block_table+query_start_loc → slot_mapping`），**非消费侧** `reshape_and_cache`（散写，M4 已 trace）。
- **实测结论**：自写 Triton 生成 kernel 通过**三路功能交叉验证**（自写 / PyTorch 语义 Oracle / vLLM 官方 `_compute_slot_mapping_kernel`）**22/22 全绿**——覆盖随机/单token/块内首末/跨块/非连续物理块/乱序/尾部mask/ragged 多请求(query_start_loc)/padding(哨兵检测漏写)/CP interleave(world 2·4，纯 index 数学)。microbench(do_bench)：**自写 ≈ vLLM 官方**(n=16384 28.4≈28.4us，无数量级退化)，torch ref 小尺寸慢 3–4×(多 op 多 launch)，≤2048 有 ~7us launch 地板→之后转访存主导。
- 产出：`scripts/m8_{kernels,test,bench}.py`、`notes/m8/triton_kernels.md`（含 Triton 编程模型讲解 + 生成/消费之分）、`outputs/m8/{correctness.json,microbench.csv,bench_summary.json}`、`logs/m8/*`。**零 vLLM 源码改动**（官方 kernel 作参照调用）。CP interleave 为生成侧真实特性（不再标 N/A）；未覆盖：消费侧散写、真实多卡 CP、多 KV-group、FP8/head-major。

### M9 Profiler 驱动的真实 vLLM 修改　✅　★核心验收（诚实无收益报告）
- 优化对象**只由 M7 的 profiler 证据决定**。若瓶颈在 Python / CPU metadata / H2D / 同步 / Tensor 分配，**可先完成 M9，不要求先做完全部 Triton（M8）**。
- 流程：热点→假设→microbenchmark→确认可复现→改代码→correctness→E2E benchmark。
- 产出：修改报告（改前 timeline / 根因 / diff / 单测 / microbench / E2E / 收益区间 / 退化场景）；**无收益也保留失败报告**。
- **实测结论（诚实）**：由 M7 证据选定"合并 decode 路径两处同型 H2D 拷贝（req_indices+query_pos）"作 profiler 驱动优化（`gpu_model_runner.py`，`VLLM_LAB_M9_COALESCE` 门控、默认关、可回退）。**机制成立**（cudaMemcpyAsync/step 11→10，每步 −1）、**正确性完整**（关=零副作用、开=输出与基线逐一致），但 **E2E 收益落在测量噪声内**（decode 各 batch 差异 ±0.01–0.02ms ≪ stdev 0.11–0.30ms）。根因：省下的 ~3.5us/step 仅占 decode step ~0.13%；该路径已被 vLLM 高度优化（pinned+non_blocking+overlap）。isolated microbench **提前量化天花板并预判 E2E 无效**，结论坐实。→ 按 ROADMAP **保留为无收益报告**，改动留作 env 门控可选项。
- 产出：`vllm-src/gpu_model_runner.py`（M9 门控改动，`# VLLM_INFERENCE_LAB_M9`）、`backups/m9/`、`scripts/m9_{microbench,e2e}.py`、`notes/m9/optimization.md`、`outputs/m9/{microbench.csv,e2e_{on,off}.json,correctness.json,m9_change.diff}`、`logs/m9/*`。**价值在于完整的 profiler 驱动方法论 + 忠实结论（不伪造收益）**。

---

## 阶段 C · 岗位第二层能力（单卡可推进 + 资源门控）（对应 study_path 第八~九章）

### M10 量化对比　⬜　★核心验收（单卡可做）
- 覆盖：BF16 baseline / 权重量化 / KV 量化（本机不支持 FP8 则走 INT4/AWQ/GPTQ）；对比显存/吞吐/TTFT/TPOT/质量；源码聚焦：权重在哪反量化、Q/K/V 与 KV dtype、scale 存放、backend 如何消费 scale。

### M11 双 GPU Tensor Parallel　⬜　★核心验收（**资源门控，非硬前置**）
- 覆盖：单卡 vs TP=2；每卡存哪些权重、QKV 切分、attention 后 collective、NCCL 占比、小 batch 时 TP 为何可能更慢。
- **门控规则**：本机为**单卡**。M10/M12 可在单卡继续，不被 M11 阻塞。获得双卡资源后再执行；无硬件时**只允许完成理论 + 源码定位**，不得标记为"实践完成"。

### M12 在线 / 开环 Serving 与 SLO　⬜
- 覆盖：OpenAI 兼容 server；固定 QPS / Poisson 到达 / 并发；请求排队；TTFT、ITL、P50/P95/P99；最大稳定 QPS；超时、过载、背压与容量规划。
- 与 M6 区别：M12 = 在线开环 + SLO。

### M13 首次开源 PR　⬜　★核心验收
- 从 `contribution_candidates.md`（持续维护）中选一个**正式完成一次上游贡献**：复现→定位→改→测试→benchmark→提交→回应 review→改 PR（涉及 kernel 需 custom op 注册 + meta function + op 检查）。目标是完整经历，非必须合入。

---

## 阶段 D · 求职收口（对应 study_path 第九章后半的收口部分）

### M14 ⏸ 第二引擎横向对比　—— 后续扩展，不在第一阶段（本文件不展开）
### M15 ⏸ 高级专题（Spec Decoding / MoE / CP / KV Offload / P-D 分离 等）　—— 后续扩展，不在第一阶段（本文件不展开）

### M16 项目整理 + 面试化表达　⬜　★验收 #1 收口
- **只做最终删减、重组、简历与面试表达**（组织工作已在"持续维护任务"里随做随更新，不积压到此）。
- README 让招聘者 5 分钟看懂（问题/源码路径/环境/基线/瓶颈/改动/性能变化/回归/复现）；核心图≤5 张；每主题 30 秒 / 3 分钟 / 10 分钟三版表达；重点题（continuous batching 为何提吞吐、PagedAttention 解决什么、block_table vs slot_mapping、P/D 瓶颈、KV 与最大并发、如何确认优化有效、profiler 区分 CPU/GPU 瓶颈、为何 TP=2 不一定更快）。

---

## 第一阶段核心验收线（按项，不设时间）
| # | 验收项 | 对应模块 | 状态 |
|---|---|---|---|
| 1 | 完整 vLLM Trace & Optimization 项目 | M0 起 / M16 收口 | 🔄 进行中 |
| 2 | ≥5 个可复现实验 workload | M1–M3 | ✅ 已超额 |
| 3 | 一套离线/闭环 Benchmark 数据 | M6 | ✅ |
| 4 | ≥2 份 Profiler 分析（现象→证据→假设→实验→结论） | M7 | ✅ |
| 5 | 1 个 slot-mapping 类 Triton kernel（含 correctness+microbench） | M8 | ✅ |
| 6 | 1 次 profiler 驱动、有测试+性能数据的 vLLM 改动 | M9 | ✅（改动+测试+E2E 数据齐全；E2E 收益在噪声内，诚实保留无收益报告） |
| 7 | 1 次开源 PR | M13 | ⬜ |
| 8 | 分布式(TP)对比报告 | M11（资源门控） | ⬜ |
| 9 | 量化对比报告 | M10 | ⬜ |

**总体：M1–M9 完成、M0 进行中（能力已到"能证 + 能测量 + 能定位 + 能写 kernel + 能改"）；核心验收 6/9。第一阶段主线优先级：~~M4~~ → ~~M5~~ → ~~M6~~ → ~~M7~~ → ~~M8~~ → ~~M9~~ → **M10（下一步）** →（M11 资源允许时）→ M12 → M13 → M16。**

---

## 暂不做（study_path 第十章，避免分心）
进入 M6 之前不要：无目标通读全部源码 / 同时啃多套引擎 / 深入多模态·Mamba·MLA·DCP / 没有 benchmark 就改性能代码 / 死磕复杂 kernel / 只做笔记不跑实验 / 追 main 分支结构变化。第一阶段锁定：decoder-only、单机单卡、FlashAttention、Scheduler / KV / ModelRunner / BlockTable / Attention / Profiler。

---

## 本次修订说明
- **新增**：
  - **M0 环境与可复现基础设施**（据实：环境固定/记录、统一 trace 工具、trace-off 零副作用、备份/回归快照均已完成；复现入口脚本、正式 invariants/回归脚本、`contribution_candidates.md`、`notes/evidence_index.md`、主题化命名标为**待补**，未虚构）。
  - **持续维护任务**一节（`contribution_candidates.md`、`evidence_index.md`、每模块同步 ROADMAP/README/commit）。
- **调整的模块关系**：
  - **M8 与 M9 解除硬串行**，改为 M7 之后的两条并行能力线；M9 优化对象仅由 profiler 证据决定，可先于 Triton 完成。
  - **M6 与 M12 明确区分**：M6=离线/闭环引擎基准，M12=在线/开环 serving + SLO，避免重复压测。
  - **M11 TP 改为资源门控**（非硬前置；单卡下 M10/M12 照常推进；无双卡只做理论+源码定位）。
- **删除 / 弱化的过度要求**：
  - 删除时间导向表述（原"三个月最低验收线"→"第一阶段核心验收线"，全文无天/周/月排期与 "Week N" 残留）。
  - **M4 收窄**：退出标准仅为"trace 证明 SchedulerOutput→KV 写入→Attention 读取闭环"；穷举 backend、多 backend 性能对照、FlashInfer/sm_120 深挖列为可选。
  - **M5 改写**：不再要求分别完成 eager/compile/CUDA Graph 三模式，只验证本机实际执行路径，性能结论后移到 M6/M7。
  - **M7 细化**：≥2 类 workload，每份须形成"现象→证据→假设→单变量实验→结论"，仅导出 timeline 不算完成。
  - **M8 设停止条件**：基础 kernel 达概念即止，主验收集中在 slot-mapping kernel。
  - **移出本阶段**：原 M14（第二引擎横向对比）、M15（高级专题）标为 ⏸ 后续扩展，不在第一阶段展开；范围声明收窄为"仅第一阶段"。
- **能力自评修正**：由笼统"整体能证"改为分项（Scheduler/KV=能证；Attention Backend/执行模式=能串待能证；端到端=能证进行中；能改/能优=未完成）。
- **当前下一步应执行的模块**：**M4 Attention Backend**（退出标准见上）。
