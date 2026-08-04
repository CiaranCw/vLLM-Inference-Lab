# ROADMAP —— vLLM Inference Lab · 学习覆盖清单（第一阶段 M0–M16 + 第二阶段 M17–M26）

> **范围声明**：本 ROADMAP 覆盖两个阶段。**第一阶段**（阶段 A–D，M0–M16）= vLLM 单机（单卡）主链路理解、性能分析、第一次真实优化与基础求职项目。**第二阶段**（阶段 E–J，M17–M26）= 面向推理系统岗位的技术栈扩展路线（vLLM 高级优化、服务化/Triton、分布式并行、TensorRT-LLM 跨引擎、长上下文/解耦、模型压缩），**整合自用户提供的 `study_path_with_tech_stack_roadmap.md`（§13–§24）**（该扩展路线此前标为"待规划、用户尚未提供"，现已落地为下方【第二阶段】）。
> **优先级铁律**：vLLM 单卡核心 ＞ 服务化与性能分析 ＞ 多 GPU 分布式 ＞ TensorRT-LLM/Triton ＞ 高级解码与长上下文 ＞ 非推理主线的模型压缩。时间不足时**不要平均分配**（最低五项见第二阶段末）。
>
> **设计**：按**阶段与验收条件**推进，**不按天/周/月排期**；做完一项直接开下一项。（`study_path` 原文的"第 N 周"仅作粗略顺序参考，本文件不锁定周次。）
> **状态图例**：✅ 完成　🔄 进行中　⬜ 未开始　⏸ 资源/时机门控（非硬前置，条件具备再做）
> **学习等级（第二阶段每模块标注，诚实区分能力边界）**：**T**=理论理解（能画图/解释/推导开销）｜**E**=可复现实验（脚本+配置+原始数据+对照+结论）｜**O**=开源项目实践（能定位/改真实源码与配置）｜**H**=特定硬件·多卡·集群实践（真实通信/显存/吞吐/故障数据）。简历"熟悉"≥**T+E**；"有实践经验"≥**E+O**；"分布式部署经验"必须 **E+O+H**。**严禁把 T/O 写成 H**（如单卡读 TP 源码 ≠ "TP 部署经验"）。
> **使用约定**：可对某轮限速——`[推进]`=只往下一个未完成模块走、不加额外实验；`[深化]`=允许在当前模块内追加实验。

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
| 在线服务与容量/SLO | **能测容量** ✅（M12：真实 serve + 开环 Poisson，真在线 TTFT/ITL/E2EL，最大稳定 QPS≈40、饱和~46、队列/过载/容量随 shape） |
| 能改 / 能优 | **能改** ✅（M9：profiler 驱动、可回退、正确性完整的真实改动；本例 E2E 收益在噪声内，诚实保留为无收益报告）；**能优**：方法论完备，待有正空间的目标再证正收益 |
| 上游贡献（回馈社区） | **能提交** ✅（M13：PR [#51017](https://github.com/vllm-project/vllm/pull/51017) 为零覆盖的 `compute_slot_mapping` 补单测，18 passed / DCO pass / test-only；含语义漂移重核与变异测试验证 oracle 独立性。**未合入，等审查**） |

能力阶梯：能读 → 能串 → 能证 → **能改** → **能优**。第一阶段目标是把主链路推到"能改/能优"。

---

## 持续维护任务（贯穿所有模块，不是单独模块）
- **`contribution_candidates.md`（已建，持续维护）**：持续记录可回馈上游的候选——测试缺口、文档与行为不一致、明确 bug、Benchmark/可观测性改进、性能修复点。**C1（slot_mapping 单元测试）已于 M13 落地为 PR [#51017](https://github.com/vllm-project/vllm/pull/51017)**；**剩余备选 C2–C4** 仍有效（C2 `calculate_kv_scales` 弃用信息陈旧、C3 安装文档缺 consumer Blackwell、C4 fp8 KV 默认 scale 文档），第二阶段若再提 PR 从此挑选（动手前须按文件内 kill 标准对最新 main 重核）。
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
  - `notes/evidence_index.md`（已建）；`contribution_candidates.md`（已建，见"持续维护任务"）。
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

### M10 量化对比　✅　★核心验收（单卡可做）
- 覆盖：BF16 baseline / 权重量化 / KV 量化（本机不支持 FP8 则走 INT4/AWQ/GPTQ）；对比显存/吞吐/TTFT/TPOT/质量；源码聚焦：权重在哪反量化、Q/K/V 与 KV dtype、scale 存放、backend 如何消费 scale。
- **实测结论（本机 sm_120，eager，Qwen2.5-0.5B；单变量对照 C1−C0/C2−C0T，同后端基线 C0T）**：**本机 FP8 可用**（在线 `quantization="fp8"` = W8A8 动态；`kv_cache_dtype="fp8"` 需 `attention_backend="TRITON_ATTN"` 避开 sm_120 FlashInfer 失败；无需下载模型）。
  - **显存**：C1 权重 942→**601 MiB(−36%)**（linear 半减、embed/lm_head 260MiB 保 bf16 故非 −50%）；C2 KV 容量 **~2×**（C2/C0T tokens 1.993×）。
  - **质量(token 加权 PPL)**：FP8 Linear **+0.51%**、FP8 KV(未校准 scale=1.0) **+2.17%**、FP8 KV(动态 scale) **+1.72%**、后端影响 +0.02%。
  - **速度(诚实含反常)**：fp8 Linear 长 prefill **−27%**（计算受限受益）但 decode **+36%（反而变慢**，小 GEMV+动态激活量化开销+sm_120 未优化）；fp8 KV **不提速**（价值在容量）。
  - **scale 审计**：C2 全层 scale=1.0（未校准）；C2dyn `calculate_kv_scales=True` 动态非 1.0。
- 产出：`scripts/m10_{quant,run_all,summarize}.py`、`notes/m10/quantization.md`、`outputs/m10/{configs,memory,quality,raw,summary.json,latency.csv}`、`logs/m10/*`。**零 vLLM 源码改动**。结论限 eager/单卡/0.5B，不外推 compiled。

### M11 双 GPU Tensor Parallel　⬜　★核心验收（**资源门控，非硬前置**）
- 覆盖：单卡 vs TP=2；每卡存哪些权重、QKV 切分、attention 后 collective、NCCL 占比、小 batch 时 TP 为何可能更慢。
- **门控规则**：本机为**单卡**。M10/M12 可在单卡继续，不被 M11 阻塞。获得双卡资源后再执行；无硬件时**只允许完成理论 + 源码定位**，不得标记为"实践完成"。

### M12 在线 / 开环 Serving 与 SLO　✅
- 覆盖：OpenAI 兼容 server；固定 QPS / Poisson 到达 / 并发；请求排队；TTFT、ITL、P50/P95/P99；最大稳定 QPS；超时、过载、背压与容量规划。
- 与 M6 区别：M12 = 在线开环 + SLO。
- **实测结论（本机 sm_120，eager，Qwen2.5-0.5B；真实 vllm serve + 官方 vllm bench serve 开环 Poisson）**：Calibration(rate=1 无排队)→冻结实验 SLO(P99 TTFT≤300ms 等，跑前写死)。Shape A(in256/out128)：QPS≤32 队列恒空、P99 TTFT ~25-32ms；**最大稳定 QPS≈40 req/s（容量区间 (40,44]，3 seed worst-seed 判 SLO）**；44/48 时 P99 TTFT 尾破线(383-663ms)但吞吐仍跟随、队列不堆积→**过载是"尾延迟劣化"非"吞吐崩溃"，SLO 实用容量(40)< 饱和吞吐(~46)**。Burst(inf 一次性)：peak_running=64(=max_num_seqs)、peak_waiting=236、P99 TTFT 5.3s、饱和吞吐~46.6 req/s。Shape B(in1024/out32 prefill-heavy)：req/s 容量相近(≥32)但 total token 吞吐~33k tok/s(高一量级)、TTFT 更高→**容量随负载 shape 变化**。
- 产出：`scripts/m12_{run_sweep,summarize}.py`、`notes/m12/serving.md`、`outputs/m12/{config.json,knee.json,sweep.csv,summary.json,metrics/,metadata/}`（raw 逐请求明细体积大、gitignore、可复现）、`logs/m12/*`。**零 vLLM 源码改动**。结论限 eager/单卡/0.5B，不外推 compiled。

### M13 首次开源 PR　✅　★核心验收（PR 已提交，未合入）
- 从 `contribution_candidates.md`（持续维护）中选一个**正式完成一次上游贡献**：复现→定位→改→测试→benchmark→提交→回应 review→改 PR。目标是完整经历，非必须合入。
- **已达成**：**[vllm-project/vllm#51017](https://github.com/vllm-project/vllm/pull/51017)** `[Core] Add unit tests for BlockTable slot mapping` —— Ready for review（2026-08-04），base `24c939c` / head `4dd4fa7`、**1 commit / 1 file / +384 / −0、零产品代码改动**、**18 passed**、DCO pass、changed-file pre-commit 13 Passed 0 Failed。落点 `tests/v1/worker/test_block_table_slot_mapping.py`。
- **覆盖**：3 类手算 golden（普通 / DCP 跨虚拟块 / hybrid 跨第二 kernel 块）+ 边界/ragged/prefix offset + 跨 1024 Triton tile padding + DCP world{2,4}×interleave{1,4} 全 rank（含「每 token 恰属一个 rank」守恒律）+ hybrid (32,16)&(64,16) + `SlotMappingMode.NONE`。
- **关键工程结论（比 PR 本身更值钱）**：
  - **不盲信旧笔记**：候选研究基于 v0.24.0，动手前对最新 main 重核，发现 **3 处语义漂移**——slot mapping 改为**只按 DCP 分片**（PCP 不再参与）、新增 **hybrid 块三分**（`kv_cache_block_size`/`kernel_block_size`/`blocks_per_kv_block`）、新增 **`SlotMappingMode`**；按旧假设写的测试跑不起来。
  - **变异测试证明 oracle 必须独立**：初版参照实现复用产品 `map_to_kernel_blocks`，注入 `+1` 后 hybrid 用例**仍通过（漏检）**；改为测试内自建展开后 **3/3 正确失败**。→「测试通过 ≠ 测试有效」。
  - **主动规避冲突**：open PR #39591 会创建同名文件（但只测 row-tail 清零、不碰 slot mapping），故改名 `test_block_table_slot_mapping.py` 并在 PR 说明、表示愿合并。
  - **前车教训**：stale PR #30683 同方向，**死于 pre-commit 未过后无人跟进**（非评审否决）→ 本次把 pre-commit 全绿当硬门槛。
  - **上游贡献树与学习树隔离**：`vllm-src` 带中文注释且 pin 在 v0.24.0（脏树/shallow），不可提 PR；另建 `vllm-m13-upstream`（干净 clone + `uv` 独立 venv + 跟最新 main）。
- 产出：`notes/m13/{STATUS.md,pr_slot_mapping_test.md}`、`notes/m13/review/*`（逐行审查 / 手算 golden / 清单 / 复现命令 / diff patch）、`notes/m13/validation/*`（pytest / pre-commit / 变异回归 / 环境）、`notes/m13/pr_draft/{test_block_table_slot_mapping.py,PR_BODY_FINAL.md}`。
- **待回填**：maintainer review 往返与迭代、是否最终合入。

---

## 阶段 D · 求职收口（对应 study_path 第九章后半的收口部分）

> **原 M14（第二引擎横向对比）、M15（高级专题）已由【第二阶段】细化展开，不再单列**：第二引擎/跨引擎对比 → **M23**（TensorRT-LLM）；Speculative Decoding → **M19**；MoE/EP → **M22**；Context Parallel / Ring Attention / P-D 分离 → **M24**；KV Offload 归入 M24 相关连接器（`docs/features/kv_offloading_usage.md`）。

### M16 项目整理 + 面试化表达　✅　★验收 #1 收口
- **只做最终删减、重组、简历与面试表达**（组织工作已在"持续维护任务"里随做随更新，不积压到此）。
- README 让招聘者 5 分钟看懂（问题/源码路径/环境/基线/瓶颈/改动/性能变化/回归/复现）；核心图≤5 张；每主题 30 秒 / 3 分钟 / 10 分钟三版表达；重点题（continuous batching 为何提吞吐、PagedAttention 解决什么、block_table vs slot_mapping、P/D 瓶颈、KV 与最大并发、如何确认优化有效、profiler 区分 CPU/GPU 瓶颈、为何 TP=2 不一定更快）。
- **已完成**：① 招聘者向 README 重写（30 秒速览 + 能力阶梯 + 逐模块成果表 M0–M13 + 亮点洞察 + 复现）；② `notes/m16/interview_and_presentation.md`（项目 30s/3min/10min + 8 重点题，全部锚定本 lab 证据）；③ **4 张核心图**（`scripts/m16_figures.py` 由既有 CSV **离线生成、无需 GPU**：M6 decode / M6 prefill / M10 量化反常 / M12 容量拐点）；④ evidence_index 补 M12/M13/M16。
- **已回填（M13 提交后完成）**：README §7 由占位改为**真实成果**（PR #51017 链接 / 规模 / CI / 5 条工程判断）；30 秒速览、能力阶梯、逐模块表同步（并清除 v0.24.0 时代作废的「26/26」与旧文件名）；evidence_index 的 M13 行改写为已提交事实 + 变异测试证据；`notes/m16/interview_and_presentation.md` 把 M13 升级为真实上游经历，**新增 Q9「怎么保证测试不是自证」**（9 道重点题）并补「PR 未合入」的诚实边界；README 工作区结构补 `vllm-m13-upstream` 隔离说明。

---

## 第一阶段核心验收线（按项，不设时间）
| # | 验收项 | 对应模块 | 状态 |
|---|---|---|---|
| 1 | 完整 vLLM Trace & Optimization 项目 | M0 起 / M16 收口 | ✅（招聘者向 README + 面试三档表达/9 题 + 4 核心图 + 证据索引；M13 PR 已回填） |
| 2 | ≥5 个可复现实验 workload | M1–M3 | ✅ 已超额 |
| 3 | 一套离线/闭环 Benchmark 数据 | M6 | ✅ |
| 4 | ≥2 份 Profiler 分析（现象→证据→假设→实验→结论） | M7 | ✅ |
| 5 | 1 个 slot-mapping 类 Triton kernel（含 correctness+microbench） | M8 | ✅ |
| 6 | 1 次 profiler 驱动、有测试+性能数据的 vLLM 改动 | M9 | ✅（改动+测试+E2E 数据齐全；E2E 收益在噪声内，诚实保留无收益报告） |
| 7 | 1 次开源 PR | M13 | ✅ **PR [#51017](https://github.com/vllm-project/vllm/pull/51017) 已提交**（Ready for review，18 passed / DCO pass / test-only；未合入，等审查） |
| 8 | 分布式(TP)对比报告 | M11（资源门控） | ⬜ |
| 9 | 量化对比报告 | M10 | ✅（BF16/FP8 W8A8/FP8 KV：显存·性能·质量·源码机制，含 scale 审计与诚实性能反常）|

**总体（第一阶段）：M1–M13 + M16 完成、M0 进行中；核心验收 8/9（仅差 M11 双卡 TP，属硬件资源门控）。推进：~~M4~~→~~M5~~→~~M6~~→~~M7~~→~~M8~~→~~M9~~→~~M10~~→~~M12~~→~~M13（PR #51017 已提交）~~→~~M16 收口~~；M11 待多卡资源。**第一阶段已收口，当前进入【第二阶段】阶段 E（M17 起，单卡可做；GPU-free 部分已完成）。**

---

# ═══ 第二阶段 · 面向推理系统岗位的技术栈扩展路线（M17–M26，整合自 `study_path` §13–§24）═══

> **定位**：第一阶段把 vLLM 单机主链路推到"能改/能优"；第二阶段在其之上**分层扩展**到岗位要求的分布式、服务化、跨引擎、长上下文与压缩能力。**每个模块标 T/E/O/H 等级与硬件门控**（见顶部图例），诚实区分"验证过 / 只读过源码 / 只懂理论"。
> **版本接地**：仍以 pinned `ee0da84`/v0.24.0 为主完成实验——扩展主题的官方文档在该版本**均存在**（已核对：`docs/serving/parallelism_scaling.md`、`docs/features/{disagg_prefill,speculative_decoding,automatic_prefix_caching,kv_offloading_usage,lora}.md`、`docs/features/quantization/`）；上游已演进到 v0.25.0+，跨版本结构差异待各模块动手前核对。

## 技术栈覆盖总表（推荐最低等级 / 硬件门控）
| 技术栈 | 最低等级 | 需开源项目 | 硬件 | 支撑项目 |
|---|---|---|---|---|
| Continuous Batching / PagedAttention / KV Cache | T+E+O | 必须 | 单卡 | vLLM（=第一阶段，已达成） |
| 张量并行 TP | T+E+O+H | 必须 | ≥双 GPU | vLLM、torch.distributed |
| 流水线并行 PP | T+E+O+H | 必须 | 多 GPU（最好多节点） | vLLM |
| 数据并行 DP + 请求路由 | T+E+O+H | 必须 | ≥双 GPU/双实例 | vLLM、负载均衡 |
| Expert Parallel / MoE | T+E+O+H | 必须 | 多 GPU | vLLM、DeepEP/NCCL |
| Triton Inference Server | T+E+O | 必须 | 单 GPU 起步 | Triton、vLLM Backend |
| 权重量化 / KV Cache 量化 | T+E+O | 必须 | 单卡可做部分 | vLLM、TensorRT-LLM、llm-compressor |
| Speculative Decoding | T+E+O | 必须 | 单卡(N-gram)；双模型看显存 | vLLM |
| Medusa / EAGLE / MTP | T+E+O | 必须 | 看模型/draft 方案 | vLLM、speculators |
| Context Parallel / Ring Attention | T+E+O+H | 必须 | 多 GPU | vLLM、Ring Attention 实现 |
| 百亿模型服务 | T+E+O+H | 必须 | 多 GPU 服务器 | vLLM、Triton、监控压测 |
| 千亿/MoE 服务 | T+E+O+H | 必须 | 多节点集群 | vLLM、TRT-LLM、Ray/K8s |
| TensorRT-LLM | T+E+O | 必须 | NVIDIA GPU | TensorRT-LLM |
| P/D 分离、KV 传输 | T+E+O+H | 必须 | ≥双实例（最好多 GPU） | vLLM、LMCache/NIXL |
| 生产可观测性与容量规划 | T+E+O | 必须 | 单机入门/集群更完整 | Prometheus、Grafana、压测 |
| 知识蒸馏 / 剪枝 / 低秩 | 推理岗 T；压缩岗 T+E(+O) | 视岗位 | 单卡起步 | Transformers/PyTorch/SparseML |

## 阶段 E · vLLM 高级优化（单卡可做，T+E+O）

### M17 Prefix Caching & Chunked Prefill 深化（承接 M3/M6）　✅　T+E+O
- 覆盖：共享前缀 workload 证明——哪些 KV blocks 被缓存、第二请求命中多少完整 block、少算多少 prefill token、TTFT 变化、**什么情况下没有收益**；chunked prefill 对调度与 TTFT 的影响。
- **T+O（GPU-free）**：APC + chunked prefill 源码逐点定位（block hash 链式 / 命中 / ref_cnt / LRU 驱逐「尾部优先淘汰·前缀优先保留」/ 登记；token 预算 / 长阈值 / 预算共享），均带 file:line；实验设计与不变量跑前冻结。
- **E 实测结论（已完成，`notes/m17/results.md`）**：
  - **长前缀下 APC 收益真实且显著**：P=1024 省 **5.02ms（−39.0%，≈3.5σ）**、P=2048 省 **14.96ms（−62.4%，≈8.8σ）**；而 **P≤512 省 0.9–1.3ms 落在噪声内**。
  - **机制打通（接 M7）**：off 侧延迟斜率 256→512 **≈0**（被 M7 已证的 ~7ms CPU 提交地板主导）、512→2048 转为 **8.85→10.85 us/tok**（计算受限）；hot 侧几乎不随 P 增长（命中后每次只算 16 个尾 token）。→ **APC 的收益必须等 prompt 长到穿透地板才显现**。
  - **修正 M6 两处**：① 「省计算≠省延迟」只是**短前缀**现象；② 「prefix ON 短 prompt 反略慢」**未复现**（4 档 ×15 样本 cold 全部 ≤ off），判定为噪声。
  - **不变量 I1–I4 全过**（命中对齐 / 省计算守恒 `hit=actual−scheduled` / off 基线 hit=0 / 分块守恒 Σchunks=prompt_len）。
  - **E3**：分块守恒成立（512→4 块、1024→2 块）；**单请求下分块本身有成本**（4 块 +15.6% TTFT ≈ 多付 3 次 ~7ms 级 step 固定开销）——但其真实价值在多请求并发降 ITL 抖动，本轮**未覆盖**，故不下"chunked 有害"的结论。
  - ⭐ **意外发现（上游硬约束）**：`enable_chunked_prefill=False` 时 vLLM **强制 `max_num_batched_tokens ≥ max_model_len`**（`vllm/config/scheduler.py:268-279` 直接 raise）——这是 chunked prefill 存在意义的硬编码证明，也使「off + 小预算」对照**无法构造**（据实调整网格）。
- **实测暴露的脚本缺陷（已修）**：同进程内建多个引擎时 `del engine + empty_cache()` **不足以归还 KV cache 显存**（nvidia-smi 显示 12GB 空闲、vLLM 只见 5.08GB）→ 改为 **M6 已验证的「每配置一个独立进程」**（`--shard`）+ `summarize` 阶段不建引擎。
- 产出：`scripts/m17_prefix_cache.py`、`notes/m17/{prefix_cache_and_chunked_prefill.md,results.md}`、`outputs/m17/{prefix_cache_sweep.csv,chunked_prefill.csv,raw/*.jsonl}`、`logs/m17/*`。**零 vLLM 源码改动。**
- **未做（如实）**：E2 block 生命周期 ref_cnt/驱逐内省（可选）；E3 的并发场景（ITL 抖动）；不外推 compiled。
- 产出：`workloads/prefix_cache_comparison.py`（本仓库实为 `scripts/m17_prefix_cache.py`）、`results/prefix_cache_on_off.csv`（实为 `outputs/m17/prefix_cache_sweep.csv`）、`reports/06_prefix_cache.md`。
- **单卡足够**。注：M3/M6 已初步触及 prefix 命中（省计算≠省延迟）；本模块系统化为 on/off 定量对照 + 长前缀真实受益场景。

### M18 量化执行路径深化（承接 M10）　⬜　T+E+O
- 理论：W8A8/W4A16、对称/非对称、per-tensor/channel/group、scale 与 zero-point、权重量化 vs KV 量化、误差·显存·带宽·吞吐关系。
- 实验：BF16 vs 权重量化 vs（硬件可行时）FP8 KV；比显存/KV 容量/启动/TTFT/TPOT/吞吐/质量。源码聚焦：`QuantizationConfig`、量化 Linear 替换、权重加载、scale 保存传递、实际 GEMM/dequant kernel、backend 如何消费 KV scale。
- 注：M10 已做 BF16 / FP8 W8A8 / FP8 KV（含 scale 审计与"decode +36% 反常"诚实结论）；本模块补 **W4A16/AWQ/GPTQ** 与跨引擎对照预备。**不为补表格伪造不支持的 FP8/FP4 路径**。

### M19 Speculative Decoding　⬜　T+E+O（`docs/features/speculative_decoding/`）
- 理论：target/draft(proposer)、一次提议 K token、target verification、accepted/rejected、acceptance rate、**为何中低 QPS / memory-bound decode 更受益、batch 增大为何 verification 成本抵消收益**。
- 顺序：普通 decode baseline → N-gram → 小 draft model →（显存允许）EAGLE/MTP/Medusa。**不要一上来啃 Medusa 源码**。
- 记录：proposed/accepted 数、acceptance rate、target forward 次数、TPOT、吞吐、显存、不同 batch 下收益或退化。硬件：N-gram 单卡即可；draft+target 共存看显存。仅读论文只算 T。

## 阶段 F · 在线服务与生产化（T+E+O）

### M20 服务化生产部署 + Triton（承接 M12）　⬜　T+E+O
- vLLM OpenAI server 生产化：并发/固定 QPS/Poisson/streaming/prefix；测 TTFT/TPOT/ITL/E2E/P50-99/请求·输出吞吐/错误率/GPU·CPU·显存·KV 使用率。**M12 已做开环 SLO 与容量曲线**，本模块承接并加 streaming 与真实 server 治理。
- Triton Inference Server（`Triton vLLM Backend`）：① 单模型部署（容器、model repository、backend 配置、健康检查、指标）；② vLLM 原生 server **vs** Triton+vLLM backend 对照（部署复杂度/启动/吞吐/TTFT/P95-99/指标暴露/故障恢复）；③ 动态配置与服务治理。单卡可入门；高可用/多副本/路由需多实例或 K8s。
- 产出：`serving/{load_generator.py,workload_config.yaml}`、`results/online_serving_baseline.csv`、`reports/07_online_serving.md`。

### M21 监控与容量规划　⬜　T+E+O
- 搭 Prometheus + Grafana + GPU 指标 + vLLM/Triton 服务指标 + 压测结果。回答：**最大稳定 QPS、并发到哪 P99 失控、瓶颈是 GPU/显存/KV/CPU/队列、一张卡承载多少用户、OOM/超时/worker 退出如何定位**。理论只能定义指标，替代不了真实压测与面板。

## 阶段 G · 分布式与并行（T+E+O+H，★资源门控）　`docs/serving/parallelism_scaling.md`

### M22 Tensor / Pipeline / Data / Expert Parallel（承接 M11）　⬜　T+E+O+H
- **门控**：本机单卡。无多卡时**只允许 T + O**（理论 + 源码定位：vLLM 并行组、切分、collective、EP dispatch 路径），**不得标 E/H**；获多卡（建议短租双/四卡云）后补 E+H。
- **TP**：Column/Row Parallel Linear、QKV/head 切分、MLP 切分、AllReduce/ReduceScatter/AllGather 位置、通信量、**小 batch 为何 TP 更慢**；TP=1 vs 2 每卡权重·显存·KV、NCCL、kernel 与通信重叠、不同 batch 的 scaling efficiency。
- **PP**：按 layer 分 stage、activation 传输、pipeline bubble、microbatch、与 TP 组合、stage 负载不均。**DP**：复制实例扩吞吐 + 请求路由/负载均衡/会话黏性/故障 replica 摘除。**EP/MoE**：Router/Top-K、token dispatch、All-to-All、load imbalance、capacity/dropped、DeepEP。
- 产出：`results/{tp1_tp2,dp_scaling}.csv`、`profiles/nccl_timeline/`、`figures/parallelism_topology.png`、`reports/08_distributed_inference.md`（**区分：TP/PP 扩模型容量、DP 扩吞吐、EP 扩专家、CP 扩上下文**）。

## 阶段 H · 跨引擎对比（T+E+O）

### M23 TensorRT-LLM 与跨引擎对照　⬜　T+E+O
- TRT-LLM：checkpoint 准备 → engine 构建 → 加载 → 离线推理 → 在线服务 → benchmark；**理解 vLLM 动态 PyTorch 路径 vs TRT-LLM engine 构建/tactic/runtime 路径**。
- 同模型同硬件严格对照 Transformers vs vLLM vs TRT-LLM（固定精度/GPU/长度/并发/warmup/采样/benchmark 方法）；比安装转换复杂度、模型准备、启动、TTFT、TPOT、吞吐、显存、量化能力、动态请求适配、调试、扩展性。
- Triton + TRT-LLM Backend 把 vLLM / TRT-LLM / Triton 串成完整栈。产出：`reports/09_vllm_vs_tensorrt_llm.md`、`results/cross_engine_benchmark.csv`、`scripts/reproduce_cross_engine.sh`。

## 阶段 I · 长上下文与解耦（T+E+O+H，★资源门控）

### M24 Context Parallel / Ring Attention / Disaggregated P-D　⬜　T+E+O+H
- **CP**：序列/上下文维切分、每卡持哪部分 token、Attention 为何跨卡换 K/V、通信随上下文长度变化、prefill vs decode 并行策略差异（多卡才 E/H）。
- **Ring Attention**：标准 Attention 数据依赖、序列切分、环形传 K/V block、online softmax、通信计算重叠、causal mask 分块处理；路径 论文→PyTorch reference→读开源实现→多卡→Nsight 看 P2P/NCCL 与计算重叠。**在 vLLM 里看到 CP 配置 ≠ 掌握 Ring Attention**。
- **P/D 分离**（`docs/features/disagg_prefill.md`）：prefill compute-bound / decode bandwidth-bound、为何适合不同 GPU 配置、KV 如何跨实例传输、**传输延迟何时抵消分离收益**、routing/backpressure/容错；至少两实例，资源隔离最好多卡。产出含 Prefill/Decode worker + Connector/KV transfer + Router + TTFT/ITL/throughput@latency 对照。

## 阶段 J · 模型压缩（岗位定向；推理岗多为 T，压缩岗 T+E+O）

### M25 知识蒸馏 / 结构化剪枝 / 低秩分解　⬜　T（推理岗）/ T+E+O（压缩岗）
- **推理系统岗**：**T** 即可——蒸馏(teacher/student、logit/feature/sequence、温度、如何降推理成本)、剪枝(非结构化稀疏为何不必然加速、结构化剪 head/channel/layer、硬件/kernel 能否利用)、低秩(SVD、秩与误差、大 GEMM→两小 GEMM)。
- **压缩/算法岗**：T+E(+O)——真做小模型实验并在真实引擎 benchmark，**验证 wall-clock 加速**（模型变小≠推理加速；理论 FLOPs 降≠GPU 延迟降）。**LoRA（低秩更新/Adapter）≠ 对原权重做低秩分解压缩，勿混**。

## 百亿 / 千亿落地门槛（诚实边界，勿夸大）
- **单卡可积累（可迁移基础）**：Scheduler/CB、PagedAttention/KV、小模型高并发、压测方法、profiler、量化、spec decode、Triton 单机、TRT-LLM 单机、故障定位与监控。
- **必须多卡/多节点（不得单卡模拟后写成"落地"）**：百亿权重跨卡切分、真实 TP/PP 通信、MoE EP、多节点 NCCL、跨节点 KV 传输、千亿容量规划、节点故障恢复、集群负载均衡、大规模 P/D。
- 简历如实写「在 4×GPU 完成 TP/EP **缩小规模验证**」，**不写**「拥有千亿模型生产集群落地经验」。

## 最终项目组合（4 个，替代十几个零散 demo）
- **A · vLLM V1 Trace & Optimization Lab**（=第一阶段全部，**T+E+O**）：Scheduler/CB/KV/PagedAttention/backend/profiler/CUDA·Triton/一次改动或 PR。
- **B · Distributed LLM Inference Lab**（**T+E+O+H**）：TP/PP/DP/EP(可选)/NCCL/多卡 serving/扩展效率。
- **C · Production Serving Stack**（**T+E+O**）：vLLM Server + Triton vLLM Backend + Prometheus/Grafana + 压测 + P95/99 + 容量规划 + 故障处理。
- **D · Cross-Engine Optimization**（**T+E+O**）：vLLM vs TRT-LLM + Triton TRT Backend + 量化 + spec decode + 同模型同硬件 benchmark。

## 各技术栈最终掌握标准（自评对照）
| 技术 | "了解" | "熟悉" | "有实践经验" |
|---|---|---|---|
| vLLM | 能解释架构 | 能追踪并修改主链路 | 有实验/Profiler/修改/PR（≈已达成） |
| TP/PP/DP | 能解释切分与通信 | 能定位分布式源码 | 多 GPU 上有 benchmark 与通信数据 |
| Triton Server | 知道 Backend 架构 | 能配 model repo 与客户端 | 部署+压测+监控+排障 |
| 量化 | 理解误差与低精度类型 | 能追踪量化执行路径 | 有 BF16/量化对照数据 |
| Speculative Decoding | 理解 proposer/verification | 能解释 vLLM 路径 | 有 acceptance rate/TPOT 数据 |
| Ring Attention | 理解环形 K/V 交换 | 能读开源实现 | 多卡通信与性能数据 |
| TensorRT-LLM | 知道 engine 工作流 | 能构建并运行 engine | 跨引擎 benchmark + 服务部署 |
| 百亿/千亿服务 | 能容量估算 | 能设计 TP/PP/DP 拓扑 | 多卡运行+压测+故障（明确规模边界） |

## 第二阶段核心验收线（按项，硬件门控标注）
| # | 验收项 | 模块 | 硬件门控 |
|---|---|---|---|
| 1 | Prefix cache on/off 定量对照报告 | M17 | 单卡 |
| 2 | 量化路径深化对照（含 W4A16/AWQ/GPTQ） | M18 | 单卡 |
| 3 | Speculative decoding acceptance/TPOT 数据 | M19 | 单卡起 |
| 4 | 在线 serving + Triton 部署对照报告 | M20 | 单卡/多实例 |
| 5 | 监控 + 容量规划报告 | M21 | 单机起 |
| 6 | TP/PP/DP/EP 实验（多卡 E+H；单卡仅 T+O） | M22 | ★多卡 |
| 7 | TensorRT-LLM 跨引擎对照 | M23 | NVIDIA GPU |
| 8 | CP / Ring Attention / P-D 分离设计与实践 | M24 | ★多卡 |
| 9 | 压缩专题（按岗位深度） | M25 | 单卡起 |
| 10 | 四大项目组合 + 面试三档表达 | M26 | 无新增 |

### M26 第二阶段收口：项目组合 + 掌握标准 + 面试化表达（扩展 M16）　⬜
- 把 A/B/C/D 四项目 README 化（招聘者 5 分钟看懂：问题/源码路径/环境/基线/瓶颈/改动/性能变化/回归/复现），核心图 ≤5 张（请求时序 / CB / KV block 生命周期 / profiler timeline / 优化前后）；每主题 **30s / 3min / 10min** 三档表达；**明确标注能力边界**（已验证 / 只读源码未获硬件 / 仅理论）——这种区分反而证明系统工程严谨性。

## 时间不足时的最低五项（优先级铁律的落地）
1. vLLM Trace & Optimization Lab（=项目 A / 第一阶段）；2. 一次多卡 TP 实验（M22）；3. 一次 Triton+vLLM 部署（M20）；4. 一次量化或 spec decode 对照（M18/M19）；5. 一次开源贡献（M13）。**这五项比"每个名词都看过一点"更有可验证的求职竞争力。**

## 第二阶段总验收线（走完全程应拥有）
1 个 vLLM 源码 Trace 与优化项目、1 个 Triton 在线服务项目、1 个 TRT-LLM 跨引擎对照、1 套多 GPU TP/DP 实验、1 个量化对照、1 个 spec decode 实验、1 份长上下文或 P/D 分离设计与实践报告、1 次真实开源贡献、1 套 Prometheus/Grafana 监控与容量报告；**并能清楚标记：已真实运行验证 / 已读源码未获硬件验证 / 仅理论**。

---

## 暂不做（study_path 第十章，避免分心）——**第一阶段聚焦纪律**
> 本节是**第一阶段**的防分心纪律；第二阶段（M17+）才按优先级铁律逐步解锁多引擎/分布式/长上下文等。
进入 M6 之前不要：无目标通读全部源码 / 同时啃多套引擎 / 深入多模态·Mamba·MLA·DCP / 没有 benchmark 就改性能代码 / 死磕复杂 kernel / 只做笔记不跑实验 / 追 main 分支结构变化。第一阶段锁定：decoder-only、单机单卡、FlashAttention、Scheduler / KV / ModelRunner / BlockTable / Attention / Profiler。

---

## 修订说明

### 第二阶段整合（本次）
- **整合 `study_path_with_tech_stack_roadmap.md` §13–§24**：新增【第二阶段】（M17–M26，阶段 E–J）＋技术栈覆盖总表＋T/E/O/H 学习等级＋最终项目组合(A–D)＋各技术栈掌握标准表＋第二阶段验收线＋百亿/千亿诚实边界；**范围声明由"仅第一阶段"改为覆盖两阶段**，加"优先级铁律"与"时间不足最低五项"。
- **接地核对**：扩展主题官方文档在 pinned v0.24.0 **均存在**（parallelism_scaling / disagg_prefill / speculative_decoding / automatic_prefix_caching / kv_offloading / quantization），扩展路线可落到实处。
- **原 M14/M15 占位**由第二阶段 M19（Spec Decoding）/M22（TP·PP·DP·EP）/M23（TensorRT-LLM 跨引擎）/M24（CP·Ring·P-D）细化取代。
- **第一阶段状态刷新**：M12 已完成、M13 研究完成待提交（候选 C1）。

### 早期修订（第一阶段成型）
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
- **当前下一步**：第一阶段已收口；**第二阶段 M17 已完成（T+E+O）**。→ **M18（量化路径深化：W4A16/AWQ/GPTQ，单卡可做）** 或 **M19（Speculative Decoding：N-gram 起步，单卡可做）**；之后阶段 F（M20 服务化+Triton、M21 监控容量）。多卡门控模块（M22/M24）待资源。M13 若收到 maintainer review，优先回应迭代。
