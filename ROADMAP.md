# ROADMAP —— vLLM Inference Lab · 学习覆盖清单（第一阶段 M0–M16 + 第二阶段 M17–M26 + 阶段 K · M27 + 阶段 L · M28）

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
- **`contribution_candidates.md`（本地维护，不入库）**：持续记录可回馈上游的候选——测试缺口、文档与行为不一致、明确 bug、Benchmark/可观测性改进、性能修复点。**C1（slot_mapping 单元测试）已于 M13 落地为 PR [#51017](https://github.com/vllm-project/vllm/pull/51017)**；**剩余备选 C2–C4** 仍有效（C2 `calculate_kv_scales` 弃用信息陈旧、C3 安装文档缺 consumer Blackwell、C4 fp8 KV 默认 scale 文档），第二阶段若再提 PR 从此挑选（动手前须按文件内 kill 标准对最新 main 重核）。
- **`notes/evidence_index.md`（已建，持续维护）**：证据索引，汇总"每个模块结论 → 对应 trace/log/jsonl/图 文件"，便于随时核查。
- **每完成一个模块**：同步更新 `ROADMAP.md`、`README.md`、`evidence_index.md`，并做一次 git commit（保持仓库与进度一致）。
- **项目整理不积压**：README、图表、复现说明随做随更新，M16 只做最终删减、重组、简历与面试表达。

---

## 阶段 A · 环境 + 主链路理解（对应 study_path 第三~五章）

### M0 环境与可复现基础设施　🔄
- ✅ 固定 vLLM commit（`ee0da84` / v0.24.0）、模型（`Qwen/Qwen2.5-0.5B-Instruct`）与关键配置（enforce_eager、`VLLM_USE_FLASHINFER_SAMPLER=0`、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、seed=0、temperature=0）。
- ✅ 环境记录：`logs/m1/vllm_collect_env.txt`、`logs/m1/nvidia_smi_*.txt`（GPU/Driver/CUDA/PyTorch/vLLM）；本地备忘 `vllm_dev_environment.md` 不入库。editable 安装已验证 `vllm.__file__ → vllm-src`。
- ✅ 统一 trace 基础设施：`vllm-src/vllm/_lab_trace.py`（统一事件 schema `component/event/fields`、`VLLM_LAB_TRACE*` 开关、JSONL 输出、组件过滤、layer/once/step 限制）。
- ✅ trace 默认关闭不改变执行语义：多次 E1 回归验证"归一化后逐行一致"（钩子零副作用）；每次改源码前备份到 `backups/`；`logs/*/source_status_{before,after}*.txt` 记录 git 状态。
- ✅ **统一复现入口已落地**：`config/repro_manifest.yaml`（模块 → 命令 / 产物 / 日志 / 环境变量 / 预期样本数 / 不变量集合）
  + `scripts/reproduce.py`（`--list` / `--dry-run` / `--verify-only` / `--run`）。
  安全设计：`--run` 默认**只跑 `status != done` 的步骤**（不覆盖已核验数据），涉及 GPU 必须显式 `--confirm-gpu`，
  覆盖已有产物必须显式 `--confirm-overwrite`，每个建引擎的步骤起独立子进程，失败立即停止，且**不做任何 git 操作**。
- ✅ **正式 correctness invariants + 回归脚本已落地**：`scripts/check_invariants.py`（不需 GPU、不 import vllm）。
  当前覆盖 M17：I1–I4、字段语义（`prefix_cache_hit`/`prefill_step_count`/`scheduled_prompt_tokens`）、
  每组 measure 样本数、warmup 已分离、**CSV↔JSONL 的 n 一致**、**`prefix_cache_effect.csv` 可由 JSONL 逐位重算**，
  以及**文档数字反向溯源**（E1 主表与 E3-single 表的每个数字都必须能从 CSV 重算；另做正向核对以覆盖正文叙述句
  ——正是 M17 审计暴露的盲区）。三种结论严格区分 **PASS / FAIL / PENDING**，PENDING（实验未执行）**不计为通过**。
  已用变异测试验证其有效性（改文档数字 / 删一个 measure 样本 / 破坏 hit 守恒 → **3/3 均被捕获并非零退出**）。
- 🔄 待补（据实标记，尚未落地）：
  - `check_invariants.py` 目前登记了 **m0 / m17 / m18 / m19 / m21 / m22 / m23 / m24 / m27 / m28**（`python scripts/check_invariants.py` → 190 PASS / 0 FAIL / 0 PENDING，2026-10-06 集成后，见 `logs/m27/invariants_final_integration_full.log`；本地没有 11 个外部 M27 raw 时，M27 有 2 项 PENDING，不算通过。2026-10-12 登记 M28 后为 192 PASS / 0 FAIL / 3 PENDING，3 项 PENDING 都是 M28 未执行的 GPU 实验，见 `logs/m28/invariants_full_cpu_validated.log`）；**M1–M16 的逐样本规则仍未迁移**（架构已支持按模块注册，只是没写）。
  - 文档溯源目前覆盖 E1 主表 + E3-single 表（反向）与关键结论数字（正向）；**§1.2 斜率、§5 对照表等正文推导值尚未覆盖**（已在脚本输出中如实声明，不当作全覆盖）。
  - `config/repro_manifest.yaml` 目前**只登记 m0 / m17 / m19 / m22 / m27**（m27 为 `not_executed`，H20 步骤 `formal_only`）；**M18 / M20 / M21 / M23 / M24 虽已完成，复现命令仍只写在各自 `notes/mN/results.md` 里**，未收敛进统一入口。
  - `notes/evidence_index.md`（已建）；`contribution_candidates.md`（本地维护、不入库，见"持续维护任务"）。
  - ✅ 目录/命名已统一：M1–M3 由 `dayN` 改为 `mN`（`notes/m1–m3`、`outputs/m1–m3`、`logs/m1–m3`、`scripts/m{1,2,3}_*.py`，2026-08-17）。磁盘上 `backups/dayN/` 是历史备份目录名，未改。

### M1 单请求推理主链路 + Prefill/Decode 运行时 Trace　✅
- 覆盖：LLM.generate→Engine→Scheduler→Worker→ModelRunner→Qwen2 forward→Attention→Backend→Sampling→输出；Prefill/Decode 调度数据与 Tensor shape 对照；实际 backend 确认。
- 产出：`scripts/m1_baseline.py`、`notes/m1/{baseline_analysis,runtime_trace}.md`、`outputs/m1/*`、`logs/m1/*`。

### M2 Scheduler 与 Continuous Batching　✅
- 覆盖：多请求错峰混批、token 预算、running/waiting 排队准入、chunked prefill；E5 追加 blocked-decode vs prefill 的预算共享 / 槽位争用。
- 产出：`scripts/m2_engine_driver.py`、`scripts/m2_build_timeline.py`、`notes/m2/{continuous_batching,e5_blocked_decode_vs_prefill}.md`、`outputs/m2/*`。

### M3 KV Cache 与 Block 管理　✅
- 覆盖：block 分配 / block_table 逻辑→物理 / 跨块边界；prefix 命中(hash/ref_cnt/共享/对齐)；释放→cached-free→复用；KV 不足→抢占→重算(prefix on/off)；K5 追加抢占后尾部优先淘汰、前缀优先保留。
- 产出：`scripts/m3_kv_driver.py`、`notes/m3/{kv_cache_block_management,k5_eviction_order}.md`、`outputs/m3/*`。

### M4 Attention Backend　✅
- **必做（当前真实 backend 主线）**：当前机器最终选中的 backend；Q/K/V reshape；KV Cache tensor shape；slot_mapping 如何进入 KV cache update；block_table 如何进入 attention kernel；Prefill/Decode 如何通过 metadata 表达；Python 层与底层 kernel 的边界。
- **可选（不作退出条件）**：穷举所有 backend、强制多 backend 完整性能对照、深挖 FlashInfer 与 sm_120 兼容、其他平台/量化 Query/KV sharing 等支线。
- **退出标准（已达成）**：以真实 Trace 证明 **SchedulerOutput → KV 写入(slot_mapping) → Attention 读取(block_table)** 的完整闭环。
- 产出：在 `flash_attn.py` 加 2 处只读钩子（`attention/kv_write`、`attention/attn_read`，layer0，回归零副作用）；`notes/m4/m4_attention_backend.md`、`outputs/m4/m4_closed_loop.jsonl`、`logs/m4/m4_closed_loop.log`；备份 `backups/m4/flash_attn.py.pre_m4`。

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
- 产出：`notes/m13/pr_slot_mapping_test.md`、`notes/m13/review/{LINE_BY_LINE_REVIEW,MANUAL_GOLDEN_CASES,REPRO_COMMANDS}.md`、`REVIEW_DIFF.patch`、`notes/m13/validation/{pytest_block_table,precommit_changed_files,mutation_final}.txt`、`notes/m13/pr_draft/{test_block_table_slot_mapping.py,PR_BODY_FINAL.md}`。
- **待回填**：maintainer review 往返与迭代、是否最终合入。

---

## 阶段 D · 求职收口（对应 study_path 第九章后半的收口部分）

> **原 M14（第二引擎横向对比）、M15（高级专题）已由【第二阶段】细化展开，不再单列**：第二引擎/跨引擎对比 → **M23**（TensorRT-LLM）；Speculative Decoding → **M19**；MoE/EP → **M22**；Context Parallel / Ring Attention / P-D 分离 → **M24**；KV Offload 归入 M24 相关连接器（`docs/features/kv_offloading_usage.md`）。

### M16 项目整理 + 面试化表达　✅　★验收 #1 收口
- **只做最终删减、重组、简历与面试表达**（组织工作已在"持续维护任务"里随做随更新，不积压到此）。
- README 让招聘者 5 分钟看懂（问题/源码路径/环境/基线/瓶颈/改动/性能变化/回归/复现）；核心图≤5 张；每主题 30 秒 / 3 分钟 / 10 分钟三版表达；重点题（continuous batching 为何提吞吐、PagedAttention 解决什么、block_table vs slot_mapping、P/D 瓶颈、KV 与最大并发、如何确认优化有效、profiler 区分 CPU/GPU 瓶颈、为何 TP=2 不一定更快）。
- **已完成**：① 招聘者向 README 重写（30 秒速览 + 能力阶梯 + 逐模块成果表 M0–M13 + 亮点洞察 + 复现）；② 第一阶段三档表达 + 重点题（个人备考文档，已移出仓库）；③ **4 张核心图**（`scripts/m16_figures.py` 由既有 CSV **离线生成、无需 GPU**：M6 decode / M6 prefill / M10 量化反常 / M12 容量拐点）；④ evidence_index 补 M12/M13/M16。
- **已回填（M13 提交后完成）**：README §7 由占位改为**真实成果**（PR #51017 链接 / 规模 / CI / 5 条工程判断）；30 秒速览、能力阶梯、逐模块表同步（并清除 v0.24.0 时代作废的「26/26」与旧文件名）；evidence_index 的 M13 行改写为已提交事实 + 变异测试证据；README 工作区结构补 `vllm-m13-upstream` 隔离说明。

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
| 8 | 分布式(TP)对比报告 | M11（资源门控）→ 并入 M22 | ✅ **TP T+E+O+H COMPLETE**（4×H20 / 7B / TP=1/2/4；见 `notes/m22/results.md`；PP/DP/EP 仍 ⬜） |
| 9 | 量化对比报告 | M10 | ✅（BF16/FP8 W8A8/FP8 KV：显存·性能·质量·源码机制，含 scale 审计与诚实性能反常）|

**总体（第一阶段）：M1–M13 + M16 完成；核心验收 **9/9**（原 M11 双卡 TP 已由第二阶段 **M22** 在 4×H20 闭合）。推进：~~M4~~→…→~~M16~~。**第二阶段**：~~M17~~ / ~~M18~~ / ~~M19~~ / ~~M20~~ / ~~M21~~ / ~~M23(L1)~~ / ~~M25(T)~~ / ~~M26~~ / **M22(TP)** / **M24(DCP)** 已收口；余 M22 的 PP/DP/EP 与 M24 的 Ring（P-D 转入阶段 K / **M27**：新实例 fresh E0–E4 已完成，T+E+O+H COMPLETE）。**

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

### M17 Prefix Caching & Chunked Prefill 深化（承接 M3/M6）　✅　T+E+O　**主线已收口**
> **收口口径**：核心实验（E1 / E3-single / E3-concurrency）完成并核验；
> **P0 no-chunk correctness 与 I5-b 已于 2026-08-13 执行完毕；仅 E2 block 生命周期内省仍未做（脚本未实现）**
> （已在 `config/repro_manifest.yaml` 标为 `status: optional`，`reproduce.py --run` 不会自动执行）。
> **no-chunk 限制说明（须随结论一起呈现）**：当前模型不正式支持关闭 chunked prefill
> （`vllm/engine/arg_utils.py:2476-2480` 有明确告警），因此现有结果**主要支持调度形态与延迟观察，
> 不外推为普适的生成正确性保证**。
- **T+O**：APC + chunked prefill 源码逐点定位（block hash 链式 / 命中 / ref_cnt / LRU 驱逐「尾部优先淘汰·前缀优先保留」/ 登记；token 预算 / 长阈值 / 预算共享），均带 file:line。
- **E 实测（2026-08-05 按审计修复后重跑；每配置 5 warmup + 15 measure，汇总只用 measure）**：
  - **E1 APC**：**P=1024 / 2048 收益显著**（median difference 的 Bootstrap 95% CI 明确排除 0）——
    第二轮 **+3.95ms（−32.4%）/ +13.20ms（−58.5%）**，第一轮 +4.95ms（−39.9%）/ +14.20ms（−61.3%），
    **两轮方向与显著性一致、绝对值有 1–3ms 跑间漂移**；**P=256/512 的 CI 含 0**
    → 在本配置与 n=15 下**未观察到超出噪声的稳定收益**（不等于"没有收益"）。
  - **机制（接 M7）**：off 侧短 prompt 区斜率明显偏低（两轮 +1.09 / +4.62 us/tok），长区稳定在
    **+7.78 → +10.16 us/tok**（计算受限）；hot 侧几乎不随 P 增长（命中后只算 16 尾 token）
    → **APC 收益必须等 prompt 长到穿透固定开销地板才显现**。
  - **E3-single 分块守恒（I4）**：512→4 块 / 1024→2 块 / ≥2048→1 块，**Σchunks 恒 2048**、每块 ≤ 预算；
    **prefill 与 decode 已按每请求 `num_scheduled_tokens` 分离统计**（不再把 decode step 误计为 chunk）。
    单请求下每多一个 prefill step 约 **+3.0–3.2ms**。
  - **⭐ E3-concurrency（本轮新增，回答 chunked prefill 的核心卖点）**：D 稳定 decode 后插入 2048-token
    长 prefill，**两侧 `max_num_batched_tokens=4096` 严格单变量**（chunked 侧靠
    `long_prefill_token_threshold=512` 强制分块，实测 4×512；两侧 D 的 ITL 样本数均为 92）：
    **D 的最坏 ITL 22.87 → 10.51ms（−54%，CI [+11.84,+12.60] 显著）**，而 **ITL median/p95 的 CI 含 0
    （几乎不变）**；代价是**长 prefill 自身 TTFT 22.87 → 33.69ms（+47%，CI 显著）**。
    → **chunked prefill 用长请求自身的 TTFT，换取其他请求 decode 的尾部平滑性**——只改尾部、不改中位。
    **机制（以原始 `steps_detail` 为准）**：nochunk 那一步是 `d=1, p=2048, total=2049`——**D 仍被同批
    调度了 1 个 decode token，只是该 token 的返回被这个长联合 step 整体推迟**；chunked 侧每步
    `d=1, p=512, total=513`，D 同样每步产出，但每次等待的 step 更短。（改善的是「等多长的 step」，
    不是「是否被调度」。）
  - **上游配置语义（重新验证）**：`enable_chunked_prefill=False` 时 `max_num_batched_tokens < max_model_len`
    被拒（`vllm/config/scheduler.py:268-279`）——关闭分块时 scheduler 必须能在单步容纳最大序列长度。
    据此调整网格（off 侧用 mbt=4096）；**不把它当作 chunked prefill 性能价值的证明**（性能价值由并发实验单独给出）。
- **与 M6 的关系**：①「省计算≠省延迟」仅是**短前缀**现象（长前缀收益显著）；
  ②「prefix ON 短 prompt 反略慢」**在 P=256 上部分复现**——第二轮 cold−off = **+0.60ms
  （CI [+0.22,+0.98]，显著，cold 更慢）**，其余三档 CI 含 0；且两轮 cold−off 符号不稳定，
  说明多数档次是噪声。（首版曾写「4 档 cold 全部 ≤ off、P=256 cold=7.42」，**与 CSV 不符，已修正**。）
- **已撤回首版（2026-08-04）不可支持的结论**：E3「+5%/+15.6%/≈3×7ms」（单样本 + 无 warmup + 首次 Triton JIT
  污染，实测首个 warmup 样本 128–145ms vs measure max 23–35ms）、"3.5σ/8.8σ"（非正式统计量）、
  `chunked = scheduled > actual`（方向反）、E3 计时 sync 顺序错、含糊的 "在线 TTFT" 称呼。旧数据留存备份未删。
- **第二轮又修 5 处（独立审计发现，均已在真实数据核实）**：① 「D 在长 prefill 那步拿不到输出」与原始
  `steps_detail`（`d=1`）矛盾；② 设计笔记 §3 仍留反向公式；③ **E1 的 `chunked` 字段把 60/60 hot 样本
  误标 True**（真因是 prefix 命中、非分块）→ 改 `prefix_cache_hit`+`prefill_step_count` 并重跑 E1；
  ④ **I5 被整项勾选**但其定量实验（扫输出长度）未做 → 拆为「理论边界=完成 / 定量实验=未执行」；
  ⑤ **「4 档 cold 全部 ≤ off、cold=7.42」是事实错误**（实为 8.0051 > 7.8465），且暴露我的"14/14 自动
  核对"只覆盖主表、未覆盖正文叙述句这一盲区。
- 产出：`scripts/m17_prefix_cache.py`（`--shard` 单引擎模式）、`scripts/m17_figures.py`（离线出图，无需 GPU）、
  `notes/m17/{prefix_cache_and_chunked_prefill.md,results.md}`、
  `outputs/m17/{prefix_cache_sweep,prefix_cache_effect,chunked_prefill,chunked_prefill_concurrency}.csv` + `raw/*.jsonl`、
  `logs/m17/*`（含 `environment.txt`）、`figures/{fig5_m17_apc,fig6_m17_chunked,m17_apc_output_length}.png`（已并入 README 核心图）、`outputs/m17/{apc_output_length,apc_output_length_effect,e3_correctness}.csv`。**零 vLLM 源码改动。**
- **✅ 两项可选扩展已于 2026-08-13 执行完毕**（`check_invariants.py --module m17`：**53 PASS / 0 FAIL**）：
  - `[x]` **E3 no-chunk 生成正确性** —— **9/9 个 case 输出 token 逐一相同**（场景 A 单长请求 2048/64 +
    场景 B 复现 E3-concurrency 的 D/P 两侧，各 3 repeat；两侧 prompt hash 逐组一致）。
    按先行冻结的处置规则：**保留 §2/§3 性能结论并注明「有限 case 验证」**。
    边界：9 case / greedy / 单模型，**不外推**为「关闭 chunked prefill 普遍安全」——
    上游告警（`vllm/engine/arg_utils.py:2476-2480`）依然成立。证据 `outputs/m17/e3_correctness.csv`。
  - `[x]` **I5-b 输出长度定量实验** —— 固定 2048 前缀扫 `max_tokens ∈ {1,8,32,128}`，180 measure 样本。
    相对降幅 **61.2% → 22.9% → 9.3% → 3.4%** 单调下降 → **定量支持 I5-b**；
    `max_tokens=128` 档完成延迟 CI 含 0（该档未观察到超出噪声的 E2E 收益）。
    **口径按实测修正**：恒定的是**首 token** 绝对收益（14.6–15.2ms，四档 CI 全排除 0）；
    完成延迟绝对差反而随输出变长增大（14.6→35.7ms），来自约 2% TPOT 差累积，**不声称机制**。
    证据 `outputs/m17/apc_output_length{,_effect}.csv`、`figures/m17_apc_output_length.png`。
- **未做（如实）**：E2 block 生命周期 `ref_cnt`/驱逐内省（**连脚本都未实现**，不为完成度强行实现）；
  并发实验仅 1 decode + 1 长 prefill，未扫并发度；
  `max_num_partial_prefills` 本版本不可调（`arg_utils.py:2350` 直接 NotImplementedError）；不外推 compiled/大模型/多卡/在线服务。

### M18 量化执行路径深化（承接 M10）　✅　**T+E+O COMPLETE**（2026-08-13）
- 理论：W8A8/W4A16、对称/非对称、per-tensor/channel/group、scale 与 zero-point、权重量化 vs KV 量化、误差·显存·带宽·吞吐关系。
- 实验：BF16 vs 权重量化 vs（硬件可行时）FP8 KV；比显存/KV 容量/启动/TTFT/TPOT/吞吐/质量。源码聚焦：`QuantizationConfig`、量化 Linear 替换、权重加载、scale 保存传递、实际 GEMM/dequant kernel、backend 如何消费 KV scale。
- 注：M10 已做 BF16 / FP8 W8A8 / FP8 KV（含 scale 审计与"decode +36% 反常"诚实结论）；本模块补 **W4A16/AWQ/GPTQ** 与跨引擎对照预备。**不为补表格伪造不支持的 FP8/FP4 路径**。
- **✅ Gate 0（2026-08-13，零成本源码+二进制探针，未下模型时完成）**：sm_120 上 AutoAWQ(min_cap 75)/AutoGPTQ(min_cap 60) 门槛均过；
  三种 W4A16 配置（AWQ 带 zero-point / GPTQ 对称 / GPTQ desc_act）**全部选中 `MarlinLinearKernel`**；
  二进制中 `marlin_gemm`（本版本已由 `gptq_marlin_gemm` 改名）与 `awq_marlin_repack` **在 sm_120 上真实 launch 成功**
  —— 这一步是必需的，因为 `VLLM_USE_PRECOMPILED=1` 下「源码支持」≠「wheel 里有 sm_120 cubin」。
  证据 `outputs/m18/gate0_w4a16_probe.json`、`scripts/m18_probe.py`。
- **✅ Gate 0-b（2026-08-13，模型下载已获所有者授权）**：4 个官方 checkpoint（0.5B/1.5B × AWQ/GPTQ-Int4，共 4.0GB）
  **加载 + 生成 smoke 全部 PASS**，日志确认走 `MarlinLinearKernel`。证据 `outputs/m18/gate0b_load_smoke_*.json`。
- **⚠️ 已发现必须写死的隐藏变量**：AWQ/GPTQ checkpoint 的 `config.json` 声明 `torch_dtype: float16`，
  而 M10 基线模型声明 `bfloat16`，两边都用 `dtype="auto"` 的话会把**量化效应与激活 dtype 混在一起**。
  M18 设计阶段必须显式冻结激活 dtype（Gate 0 探针已验证 Marlin GEMM 接受 bf16 激活）。
- **✅ 正式实验完成**（2 模型 × 3 配置 = 6 引擎，串行、一进程一引擎；激活 dtype 显式冻结 bfloat16）：
  - **显存**：整体权重 0.5B **−53.6%** / 1.5B **−62.8%**；但 `embed`/`lm_head` **完全未被量化**（不变量核对：三配置字节数相同），
    剔除后**可量化部分 −74.0%**（≈4bit 理论值）。**模型越小，未量化 embedding 占比越高（0.5B 27.6% vs 1.5B 15.1%），整体收益被稀释越多**。
  - **KV 容量**：+5.9%（0.5B）/ **+30.1%**（1.5B）。
  - **速度**：**28 个对照点 0 个变快**，23 个显著变慢（Bootstrap 95% CI 排除 0）。
    **连 M7 已证为计算受限的长 prefill(n=2000) 也更慢**（+9.2% / +6.6%）→ 慢的来源包含 Marlin GEMM 本身，不只是 CPU 提交地板的伪影。
  - **执行前假设 H5 被推翻**：预期「模型越大越易显现收益」，实测 1.5B 的 decode 惩罚反而更大（+28.6% vs 0.5B +12.9%），如实记录。
  - **质量**：ΔPPL 0.5B +5.0%(AWQ)/+4.8%(GPTQ)；1.5B **+1.4%(AWQ)** / +4.4%(GPTQ)。评分 token 数六配置相同。
  - 与 M10 只做**定性**对照（样本量与会话不同，不相减）：FP8 至少在长 prefill 换到 −27% 真加速，**W4A16 连这一段都没换到**；
    两者共同点是 decode 都更慢、价值都主要在显存。
- **⬜ 未做**：kernel 级 profiling（故「Marlin 为何慢」只有现象与区间证据、无时间线归因）；W8A8-INT8（Marlin 明确不支持，不伪造）；
  W4A16×FP8 KV 组合；compiled/多卡/在线口径；自做量化校准。质量只用 PPL，非下游任务准确率。
- 产出：`scripts/m18_*.py`、`outputs/m18/*`、`logs/m18/*`、`notes/m18/{quantization_w4a16,results}.md`、`figures/fig13_m18_w4a16.png`；`check_invariants.py --module m18`（11 PASS）。

### M19 Speculative Decoding　✅　T+E+O（**N-gram 主线已闭环**；draft-model family 为 Optional / Future）

- **M19-A · Baseline vs N-gram** ✅ T+E+O COMPLETE（见 `notes/m19/`）
- **M19-B · Small Draft Model** ⬜ Optional / Future（需显存评估；下载须授权）
- **M19-C · EAGLE / MTP / Medusa** ⬜ Optional / Future
- 理论：proposer/verification/rejection；**用额外并行 verification 换更少串行 execution rounds**（通常不保证少 FLOPs）；
  Proposal Coverage + Conditional Acceptance + Round Cost；为何 CPU/launch-bound decode 更受益、
  batch 增大为何 verification 更难被隐藏。
- **不要一上来啃 Medusa 源码**；主线已完成，后续 proposer extensions **不阻塞**模块完成状态。
- **T+O（已完成）**：ngram 的 config 语义、proposer 算法（反转 + KMP LPS，`vllm/v1/spec_decode/ngram_proposer.py:207`）、
  proposer 调用点（`gpu_model_runner.py:665-668` / `:5088-5099`）、greedy 验收判据
  （`rejection_sampler.py:756-757`：`rejected = draft_token_id != target_argmax_id`）、
  以及**计数其实由 Scheduler 从输出 token 数反推**（`scheduler.py:1845-1869`，`num_accepted = len(generated)-1`）——均带 file:line 逐条实读核对。
- **E 实测（2026-08-06；2×2×3=12 组，每组 5 warmup+15 measure，180 measure；语义收口 2026-08-08）**：
  - **Acceptance 随 workload/config 组合变化**（**不是**「完全由 workload 决定」）：
    `high_repeat`（高初始 prompt 重叠）**1.000**（draft-bearing acceptance length 4.000），
    `low_repeat`（低初始 prompt 重叠，**非**真正低输出重复）**0.750 / 0.798 / 0.854**。
    正交第二证据：`output_4gram_in_prompt_ratio` **1.00 / 0.00**；但 low 的
    `output_repetition_4gram≈0.82–0.84`。**Batch↔prompt composition confounding**（§1.4）。
  - **机制**：`engine.step` = execution-round **强 proxy**（非严格纯 Decode Target Forward）——
    **128→33**（high）、**128→56–62**（low）。draft-bearing length ≠ global `E`
    （例：low b1 length=3.25，但 128/56≈2.286 tok/round）。
  - **性能**：完成延迟 −**72–75.5%**（high）/ −**54.8–57.4%**（low）；
    **amortized TPOT**（≠ streaming ITL）7.5–8.5ms → **1.8ms** / **3.2–3.7ms**；
    in-process 输出吞吐 **+257~308%** / **+121~135%**。**12 组 CI 全排除 0**。
    high 的 acceptance 恒为 1.0 但降幅 75.5%→72% → Acceptance **非充分指标**。
  - **逐位置**：low b4 pos0/1/2 = 0.895/0.879/0.621 → **提示**更大 k 可能边际递减；
    **未扫 k，不能实证最优 k**。
  - **显存（KV capacity planning）**：非-KV 预算约 +0.16 GiB → KV tokens **−1.9%**；
    与 sample `max_memory_allocated` **不同口径**；未做组件 breakdown。
  - **First Token**：无稳定收益；low b1 差值约 **−0.93ms**（不是「全部 ±0.6ms」）。
  - **batch≤16 未触及 knee**（仍 CPU 提交受限，M7）；≠「大 batch 也安全」。
  - **输出一致性**：111/126 exact match；batch-shape control 证明 shape 变化**可以**导致
    divergence，与现象一致；**未发现明确 Spec Decode bug**，也**不是**严格逐例排除。
- **两个必须写死的隐藏变量**：① 两侧 `async_scheduling=False`；② 两侧 `disable_log_stats=False`。
  生效配置由 `effective_cfg` 从引擎读回并被不变量核对。
- **计数观测**：自定义 `StatLoggerBase` → per-step `SpecDecodingStats`。**零产品源码改动**。
- 产出：`scripts/m19_{spec_decode,figures}.py`、`notes/m19/*`、`outputs/m19/*.csv`+`raw/*.jsonl`、
  `logs/m19/*`、`figures/fig{7,8}_m19_*.png`；`check_invariants.py --module m19`。
- **Optional / Future（不阻塞）**：`k` sweep；true low-coverage/acceptance workload；profiler breakdown；
  larger batch/model 找 knee；compiled×Spec；small draft model；EAGLE/MTP/Medusa。

## 阶段 F · 在线服务与生产化（T+E+O）

### M20 服务化生产部署 + Triton（承接 M12）　✅　**T+A+B+C COMPLETE**（2026-08-10）
- **目标**：求职向「能部署并解释 Triton+vLLM Backend，并在 **same-runtime** 下与 Native 对照」。**不重复** M12 Online SLO 网格；不把 host-M12 与 container 混减称 overhead。
- **✅ 已完成**：
  - Gate0→Docker 29.7.2 + NVIDIA Container Toolkit 1.19.1；镜像 `tritonserver:26.07-vllm-python-py3`（digest `sha256:31e20bf…`）。
  - **M20-A** host smoke；**M20-B** Triton health/load/`generate_stream`/metrics PASS。
  - **M20-C** 同镜像串行对照 C0=`vllm serve` vs C1=`tritonserver+backend`：Shape A in256/out128，Poisson QPS **20/32/40**，每点 30 warmup+200 measure，success=1.0；TTFT/E2E 的 C0−C1 Bootstrap CI **全程排除 0**；高 QPS 下差距主要在 TTFT 排队（ITL 接近）。见 `notes/m20/results.md`、`figures/fig10_m20_same_runtime.png`。
- **容器 vs host**：container vLLM `0.24.0+092c4842…` ≠ lab `ee0da84` → C **必须**同镜像对照（已遵守）。
- **⬜ 可选未做**：按 M12 SLO 重测 Triton「最大稳定 QPS」；Triton OpenAI frontend；compiled。
- 产出：`serving/triton_model_repository/`、`scripts/m20_*.py`、`outputs/m20/`、`logs/m20/`、`notes/m20/results.md`。

### M21 监控与容量规划　✅　**T+E+O COMPLETE**（2026-08-12）
- **栈**：`vllm serve`(8014) + `dcgm-exporter`(9400) + 自写 `m21_host_exporter`(9836) → Prometheus(1s 抓取, 7 条告警) → Grafana(provisioning 加载面板)。配置全部入库 `monitoring/`。
- **E2 指标可信度先过关**：21 个压测窗口的服务端 counter 增量与客户端 bench 汇总**逐条精确相等**；但**服务端 `histogram_quantile` 近似分位数 ≠ 客户端 exact percentile**（TTFT 中位 +9.7%，最大 +68%；差异同时来自统计表示、观测边界与窗口聚合，**不可归因到单一项**）→ 面板数字只能看趋势/告警，SLO 验收以客户端 exact percentile 为准。
- **E3 容量**：Shape A 7 档 × 3 seed × 60s offered-arrival horizon，本次**最后一个达标档 24 / 第一个不达标档 32**（拐点定位到 (24,32]；真实容量 **C ∈ [24,32)**）；过载先堆队列（waiting 705）而**粗粒度 GPU util 窗口均值仅 25→32%**。**到达时长决定结论**：固定 200-request 短实验（nominal horizon 随 QPS 从 25.0s 变到 4.2s）把 QPS32 测成 35ms（达标），60s horizon 是 415ms（破线）。
- **与 M12(≈40) 的差异**：E7 单变量 A/B 证明**监控栈开销 ≲1%**（非原因）；NVML 时钟/温度排除降频；单流 ITL 9.40ms 说明机器没整体变慢；差异出在**高并发 TPOT +30%**（9.3→12.1ms）→ Little's law 所需并发 48→62 顶到 `max_num_seqs=64` 硬闸 → 正反馈。**根因未定**（M12 无环境指纹），本模块补了 `env_fingerprint.json`。**容量跨会话不稳定（24–32），不宣称复现 40，也不宣称 40 是错的。**
- **E4 瓶颈签名**：B1 CPU 提交受限(GPU 31%/进程 CPU 110%)、B2 prefill 计算压力更强(粗粒度 GPU util 66%；**无 SM-active 指标，不断言算力饱和**)、B3b KV 受限(kv 0.9997 + 抢占 290)、B4 队列受限(waiting 1196)。**负结果**：只压 `--gpu-memory-utilization` 到 0.10 也造不出抢占（需求侧被 `max_num_seqs×384 tokens` 封顶），必须用长输出从需求侧构造。
- **E6 故障演练 4/4 PASS，7 条告警规则全部真实 firing**（其中 "OOM" 实为**启动期显存预算拒绝** `startup_memory_rejection`，**未注入** runtime CUDA allocator OOM）。核心发现：**`SIGKILL` vLLM API server 不会杀死持有显存的 `EngineCore` 子进程** —— `up==0` 但显存 90s 不回收，新实例撞 `No available memory for the cache blocks`；清掉已记录的子进程 PID 后 5s 回收、28s 恢复。
- **⬜ 未做**：K8s/多副本/长时 soak/跨机 SLA/Triton 面板/compiled 对照；`DCGM_FI_PROF_*`(SM active) 在本环境全 N/A，**拿不到 SM 占用率**；WSL2 下无进程级显存归因。
- **E5**：Interactive Response Time Law。X=24 时平均**在系统内请求数** L≈**41.4**；闭环**用户总数** N（含思考中用户）按 Z=5/15/30s 为 **161/401/761**——N 是 population，**不是**并发请求数。
- 产出：`monitoring/*`、`scripts/m21_*.py`、`outputs/m21/*`、`logs/m21/*`、`notes/m21/{monitoring_and_capacity,results}.md`、`figures/fig1{1,2}_m21_*.png`；`check_invariants.py --module m21`（14 PASS）。
- **2026-08-12 evidence audit**：修正 E5 闭环 population 语义、E6 timeout 秒/毫秒单位（19.57ms→**19568ms**）、E2 因果归因过强、「5s 窗口」表述、OOM 与 GPU-compute 术语、容量记号；**未改原始数据、未重跑 E3/E4/E7**，derived 由 raw 离线重算（旧版备份 `outputs/m21/pre_audit_20260812/`）。逐条见 `notes/m21/results.md` §10。

## 阶段 G · 分布式与并行（T+E+O+H，★资源门控）　`docs/serving/parallelism_scaling.md`

### M22 Tensor / Pipeline / Data / Expert Parallel（承接 M11）　✅　**TP：T+E+O+H COMPLETE**；PP/DP/EP ⬜
- **门控**：本机单卡时只允许 T+O；多卡 E/H 已在 **4×H20（2026-08-09）** 完成 TP 部分。
- **✅ TP 的 T+O**（`notes/m22/tensor_parallel.md`，R1–R7，`file:line`，v0.24.0）：
  MultiprocExecutor / InprocClient、per-rank `collective_rpc`、ProfilerConfig（无 `VLLM_TORCH_PROFILER_DIR`）、
  custom all-reduce 优先路径、7B 在 TP=1/2/4 结构兼容（28/4 heads，KV 不复制）。
- **✅ TP 的 E/H**（`notes/m22/results.md`，tag=`h20_formal`，7B / eager / in-process）：
  D1 权重≈1/TP、KV tokens 随 TP 升；D2 稳态 decode 单步 TP 显著更慢（全 batch Bootstrap CI 排除 0）；
  D3 长 prefill TP 加速（P=4096 TP4≈3.39×）；大 batch 吞吐 speedup **主要来自并发 prefill** 而非 decode 单步；
  P1 见 custom AR/NCCL；invariants **22 PASS**；图 `figures/fig9_m22_tp_scaling_h20_formal.png`。
- **⬜ PP/DP/EP**：**未编写实现、未实测**，不得勾选。DCP 已由 **M24** 在 4×H20 闭合（1.5B，与本模块 7B TP **不得相减**）。P-D 仍 ⬜。方案原文 [`MULTI_GPU_EXPERIMENT_PLAN.md`](MULTI_GPU_EXPERIMENT_PLAN.md)。
- **TP 要点**：Column/Row Parallel、每层 2× all-reduce、**小 batch 更慢的机制与数据**、scaling efficiency、collective 后端。
- **PP/DP/EP（未做）**：stage/bubble；副本扩吞吐；MoE dispatch / All-to-All。
- 产出：`outputs/m22/h20_formal_*.csv`、`outputs/m22/raw/*`、`outputs/m22/traces/`、`notes/m22/{tensor_parallel,results,execution_record}.md`。

## 阶段 H · 跨引擎对比（T+E+O）

### M23 TensorRT-LLM 与跨引擎对照　✅　**L1 FROZEN**（2026-08-14）　（设计冻结 v5 + 结论见 [`notes/m23/`](notes/m23/)）
- **Gate 0/1/2 全 PASS**：TRT-LLM `1.3.0rc24` 在本机 **sm_120 上可用**（真实生成，无 kernel 缺失报错）；接受 token ids 输入故可与 vLLM 严格对齐。
- **2×2 execution-mode 对照**（所有者决策）：OFF / ON 两条 lane 分别报告。
  - 正式跨引擎比较只允许 `T-E vs V-E`、`T-O vs V-O`；允许同引擎 `V-O vs V-E` / `T-O vs T-E` 作为 execution-mode ablation；**禁止交叉 lane 做 engine speedup**。
  - `T-E vs V-E`：单请求 **+45.0%**、batch **+29.9%**（TRT 更慢，CI 排除 0）
  - `T-O vs V-O`：单请求 **+20.1%**、batch **+24.7%**（TRT 更慢，CI 排除 0）
  - 冷启动 TRT **60.5–60.8s** vs vLLM **15.6/26.6s**；输出吞吐 vLLM 6028→17200、TRT 4642→13793 tok/s
- **两侧 optimized 收益接近**：vLLM **2.85×**、TRT **2.97×** —— 只跑一条 lane 会把「有没有开图优化」混进「谁更快」。
- **交叉验证**：vLLM 的 2.85× 与 M6 的 2.5–2.85× 吻合；V-E 单 token ~7.8ms 与 M7 的 ~7ms CPU 提交地板吻合。
- **跨引擎 generation consistency**（6 条固定 token-id）：`T-E vs V-E` exact **2/6**、`T-O vs V-O` **1/6**；mismatch 只记 numerical divergence，**不得直接判为 correctness failure**。
- **纪律**：stack-vs-stack 非 kernel 级 A/B；任一跨引擎差值**不得**称 intrinsic engine speedup；显存**只报整卡**、分项 unavailable、禁止残差推导。
- ⬜ **明确不做**：L2 在线服务、L3 Triton 集成、L4 Transformers 基线、kernel 级 profiling（故无机制归因）、模型规模/shape/并发度扫描。

> **⚠️ 上游事实修正（2026-08-13，所有者指出）**：本节下面两条原文写于 TensorRT backend 尚存的时期，**前提已失效**：
> **TensorRT engine backend 已被上游移除，PyTorch backend 成为唯一执行后端**；
> **v1.3.0rc20 是最后一个支持 TensorRT engine backend 的版本**（release notes 原文：
> "This RC version will be the last one supporting the TensorRT backend, in the next version the TensorRT backend will be removed!"）。
> 随之移除的还有 `trtllm-build` / `trtllm-refit` / `trtllm-prune`、每模型 `convert_checkpoint.py`、
> `LLM(backend="tensorrt")`、`tensorrt` pip 依赖，**以及服务 legacy TensorRT engine 的那条 Triton backend**
> （**注意边界**：这不等于 Triton 与当前 TRT-LLM PyTorch/LLM-API 集成全部消失）。
> → **M23 必须区分「当前 TRT-LLM（PyTorch backend）」与「legacy TensorRT engine backend（已退役，止于 v1.3.0rc20）」**，
> **不得**再把 engine 构建 / tactic / 固定 runtime 描述成当前 TRT-LLM 的唯一或主执行路径。
> 另修正一处技术表述：legacy engine 的 build-time `max_batch_size` / `max_seq_len` 是
> **optimization profile 的上界**，**不是固定 shape** —— runtime 在 profile 范围内仍可动态执行，超界才需重建。
> 下面两条保留为**历史原文**，实际执行以**设计冻结稿 v3** 为准。

- TRT-LLM：checkpoint 准备 → engine 构建 → 加载 → 离线推理 → 在线服务 → benchmark；**理解 vLLM 动态 PyTorch 路径 vs TRT-LLM engine 构建/tactic/runtime 路径**。
- 同模型同硬件严格对照 Transformers vs vLLM vs TRT-LLM（固定精度/GPU/长度/并发/warmup/采样/benchmark 方法）；比安装转换复杂度、模型准备、启动、TTFT、TPOT、吞吐、显存、量化能力、动态请求适配、调试、扩展性。
- Triton + TRT-LLM Backend 把 vLLM / TRT-LLM / Triton 串成完整栈。产出：`reports/09_vllm_vs_tensorrt_llm.md`、`results/cross_engine_benchmark.csv`、`scripts/reproduce_cross_engine.sh`。
  （**被移除的是 legacy TensorRT-engine 的 Triton backend**，**不等于** Triton 与**当前** TRT-LLM
  （PyTorch / LLM-API）的集成全部消失 —— 该类现行集成路线仍存在，只需按现行官方方式重新定义。
  故此项是**「可行但本轮不批准」，不是「不可行」**；待 L1/L2 结果后再评估边际价值。）
- **授权状态（2026-08-13 冻结）**：**只批准 L1**（Gate 0–2 + 离线跨引擎对照）；L2/L3/L4 **未批准**
  （L3 属**可行但未批准**）。**唯一镜像已冻结为 `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24`（TRT-P 路线）**；
  **container-only，禁止 host 新建 TRT-LLM venv**；**禁止自动换 tag**；**禁止自动租机**。
  Gate 0/1 任意失败 → **保存完整证据 → 停止 → 回报所有者**，不自行进入下一补救路线。
- **规划稿已冻结**：[`notes/m23/cross_engine_plan.md`](notes/m23/cross_engine_plan.md) 状态为 **DESIGN FROZEN (v3)**。
  技术主线定为：**「既然当前 vLLM 和 TRT-LLM 都以 PyTorch runtime 为核心，它们为什么仍是两套不同的 inference engine？」**
  —— 研究 Scheduler / KV management / kernel-runtime 优化 / 量化支持 / 请求适配 / 内存行为 / 部署复杂度 / 可观测性；
  **弃用**「vLLM=dynamic PyTorch vs TRT-LLM=prebuilt TensorRT engine」这一已被上游推翻的对立。

## 阶段 I · 长上下文与解耦（T+E+O+H，★资源门控）

### M24 Context Parallel / Ring Attention / Disaggregated P-D　✅ **T COMPLETE**（2026-08-14）　✅ **DCP E+O+H**（2026-08-16）　⬜ Ring / P-D E
- **T 已完成**：见 [`notes/m24/`](notes/m24/)。钉死本树（`ee0da84` / v0.24.0）：PCP 乘进 `world_size`、DCP 复用 TP GPU；DCP decode = Q-allgather + 本地 KV shard + LSE 合并，**不是** Ring Attention；PCP 官方 under active development 且本树无 `supports_pcp=True`；P-D experimental、官方写明 **不提吞吐**、当前注释限制 1P1D。
- **面试纪律**：**在 vLLM 里看到 CP 配置 ≠ 掌握 Ring Attention。**
- **DCP E/H（4×H20 / 1.5B / TP=4 / tag=`h20_formal`）**：唯一变量 `decode_context_parallel_size ∈ {1,2}`。D1 KV tokens **5.90M → 11.80M（恰好 2.0×）**、权重不变；D2 六档稳态 decode 单步 **全部 DCP=2 更慢**（+10.6～+13.8 ms，Bootstrap CI 全排除 0）；P1 chrome trace 见 `vllm::all_gather` / reduce-scatter / LSE merge，`AllGather_RING_LL` 是 NCCL 算法名。不变量 **13 PASS**。图 `figures/fig15_m24_dcp.png`。详见 [`notes/m24/results.md`](notes/m24/results.md)。
- **P2 PCP 负例已跑**：4 卡 TP=4×PCP=2 在 `world_size=8` 被拒，未走到 `supports_pcp`。不当性能成功。
- **仍为 ⬜**：Ring Attention 实现/Nsight；P-D 双实例（P3 未跑）。不用单卡模拟后写成落地。
- **M24 P-D experimental work → continued in M27**（阶段 K）。M24 的冻结计划与已完成的结论不做改动。

## 阶段 J · 模型压缩（岗位定向；推理岗多为 T，压缩岗 T+E+O）

### M25 知识蒸馏 / 结构化剪枝 / 低秩分解　✅ **T COMPLETE**（2026-08-14）　⬜ E（压缩岗）
- **推理系统岗**：**T 已完成**——见 [`notes/m25/`](notes/m25/)。蒸馏(teacher/student、logit/feature/sequence、温度；**蒸馏不天然缩词表/context**，推理收益来自 student architecture 变小)、剪枝(非结构化：dense GEMM shape 不变则零权重不会自动少干活，须匹配 sparse kernel/hardware)、低秩(SVD、逐层可压缩性 model-dependent、计算节省 vs 额外 GEMM launch/中间激活；**CUDA Graph ≠ kernel fusion**)。**LoRA ≠ 低秩压缩**。
- **压缩/算法岗**：T+E(+O)——真做小模型实验并在真实引擎 benchmark，**验证 wall-clock 加速**（模型变小≠推理加速；理论 FLOPs 降≠GPU 延迟降）。**本轮明确不做 E**。

## 阶段 K · Distributed Serving Deep Dive（T+E+O+H，★多卡）

### M27 Prefill–Decode Disaggregation & NIXL KV Transfer　✅ **T+E+O+H COMPLETE**（2026-10-05，8×H20 单节点同新实例 Gate 0→E4；正式实验最多 4 GPU；E0–E4 COMPLETE 为最终范围，E5 未执行、P1 未做；2026-10-06 post-hoc 审计 PASS_WITH_DOCUMENTATION_CORRECTION，修正已闭环）
- **目标**：在 H20 上用 vLLM 0.24.0 + NIXL 1.2.0 测量 P-D 分离的**传输/编排成本**与**干扰隔离收益**；**“没有收益”是允许的结果**（官方文档写明 P-D 不提升吞吐，合理配置的 chunked prefill 也可能达到同样效果）。
- **T**：在 `ee0da84` 上重新核实源码（file:line）：NIXL 是 D 主动 READ（`NixlPullConnector`）、`kv_load_failure_policy` 默认 fail、NIXL 要求 HND 布局而无 connector 时默认 NHD（所以所有 arm 统一设 HND）、`kv_rank` 的“1P1D”注释不适用于 NIXL；对 M24 笔记的更正记录在 [`notes/m27/pd_disagg_nixl_theory.md`](notes/m27/pd_disagg_nixl_theory.md)。KV 理论值 57344 B/token（Qwen2.5-7B bf16），与 M22 的 KV 容量交叉一致；**理论值 ≠ 实测传输量**。
- **O / DESIGN**：冻结计划 [`notes/m27/M27_H20_PLAN.md`](notes/m27/M27_H20_PLAN.md)（每实例 TP1、bf16、eager、APC 关、async 关、chunked 开、mbt8192；A 层因果对照 colocated vs 1P1D，B 层调优后的 colocated mbt∈{2048,4096,8192} 对 PD，B 层不是单变量对照；chunked-off 不作为正式 arm）；E0 正确性/fail-closed → E1 传输随长度变化 → E2 事件驱动干扰 → E3 在线 Poisson（3 seeds）→ E4 4 卡 P:D 配比 → Gate 8 → E5（可选）。脚本 `scripts/m27_*.py` + `scripts/m27_preflight.sh`；不变量 `check_invariants.py --module m27`（本轮 M27 35 PASS、0 FAIL/0 PENDING）；manifest 中 H20 步骤为 `formal_only`，`reproduce.py --run` 会拒绝启动。
- **资源**：7B 模型身份已冻结（逐文件 sha256，内容 == HF `Qwen/Qwen2.5-7B-Instruct@a09a3545`；找回的 M22 副本，见 [`notes/m27/model_artifact_manifest.md`](notes/m27/model_artifact_manifest.md)）；NIXL wheel sha256 已核对。证据边界：E2 是机制实验（1 卡 vs 2 卡），同 GPU 预算的部署结论只来自 E3 / E4；结论只适用于 HND，不与 M22 / M24 绝对数相减。
- **E/H 已实测**：新实例 `ins-a8qac2ve`（单节点 8×H20，Gate 0 核实 GPU0–7 两两 NV18）完成 fresh Gate0、E0–E4；NIXL GPU KV `cuda_ipc/cuda`。**规模边界**：正式实验覆盖 1/2/4 GPU——E0–E3 按 arm 用 1 或 2 GPU，P:D ratio 实验 E4 每 arm 4 GPU；各 arm 顺序执行，最大正式并发 4 GPU；没有 8-GPU P-D / scaling 实验。E2 8K 注入 colo−PD max stall +780.98ms（1 vs 2 GPU，仅机制）；E3/E4 同成本：PD 尾部 ITL 更低（对调优后的 colocated 差距缩小、方向不变），但高负载 TTFT/排队和 achieved rate 更差，见 [`notes/m27/results.md`](notes/m27/results.md)。Gate 8 选择跳过可选 E5；P1 profiler 未执行，不声称 compute/communication overlap proven。旧机器中途数据保留但不 pool。
- **验证与证据**：H20 收口时 `outputs/m27/final_evidence_audit.json` 73/73 PASS、M27 35 PASS、全仓 187 PASS；raw、派生 CSV/图及负结果见 `notes/m27/results.md`。正式执行约 12.44 h ≈ 99.5 GPU·h（计费口径为整机墙钟 ×8，与正式实验最多同时用 4 GPU 不是一回事）；之后实例空闲分配至 168 GPU·h 额度耗尽（约 68.5 GPU·h，非实验负载）。
- **Post-hoc 审计与集成（2026-10-06）**：独立审计 `M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`（CRITICAL 0 / MAJOR 0 / MINOR 7，见 [`notes/m27/audit/`](notes/m27/audit/)），7 条修正已闭环（[`FINAL_CLOSURE_ADDENDUM.md`](notes/m27/FINAL_CLOSURE_ADDENDUM.md)）；H20 原提交规范化见 [`H20_COMMIT_PROVENANCE.md`](notes/m27/H20_COMMIT_PROVENANCE.md)；11 个 >100MB raw 改为外部证据身份（[`H20_RAW_ARCHIVE_MANIFEST.tsv`](notes/m27/H20_RAW_ARCHIVE_MANIFEST.tsv)）；新增离线生成器 `scripts/m27_final_evidence_audit.py`（73/73 重算）。
- **未做（据实）**：E5（冻结设计中的可选 8-GPU P:D 配比确认）未执行；P1 profiler 时间线未执行，因此不声称 compute/communication overlap。M27 以 E0–E4 为最终范围，不再补 8 卡实验。

## 阶段 L · 单卡 KV Offloading 与 Serving 可靠性（T+E+O，单卡 RTX 5080）

### M28 KV Offloading & Serving Reliability　🔄 **DESIGN FROZEN v1 · CPU_VALIDATED · GPU NOT EXECUTED**（2026-10-12）
- **目标**：A = 原生 `OffloadingConnector`（GPU→CPU，`/dev/shm` mmap）三条路径（GPU 热命中 / GPU 未命中但 CPU 命中 / 全量重算）逐样本由指标证明，并与独立重算对照正确性；B = `vllm serve` 生命周期故障：B1 SIGKILL EngineCore 子进程（区别于 M21 E6 杀 API 父进程）、B3 SIGTERM × `--shutdown-timeout ∈ {0,2,60}`、B2 SIGSTOP/SIGCONT（可选，需单独授权）。
- **T**：源码核查见 [`notes/m28/source_review.md`](notes/m28/source_review.md)（file:line；预测 P-A1..A4、P-B1/B3/B2，均待实测）。
- **O**：冻结计划 [`notes/m28/M28_PLAN.md`](notes/m28/M28_PLAN.md)；脚本 `scripts/m28_*.py`（`--help` / `--dry-run` / GPU 门禁 / 故障授权门禁）；validator `scripts/m28_validate.py`（自检 12 个反例全部检出）；`check_invariants.py --module m28` 2 PASS / 3 PENDING；manifest 中 GPU 步骤为 `formal_only`。执行记录 [`notes/m28/execution_record.md`](notes/m28/execution_record.md)。
- **未做（据实）**：A 冒烟 / A 正式 / A evict、B1 / B3 / B2 均**未执行**（等待 GPU 授权与故障注入授权）；没有任何实测数字或结论。mock 演练只证明脚本链路，不是实验数据。

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
| 3 | Speculative decoding acceptance/TPOT 数据 | M19 | 单卡起 ✅ **N-gram T+E+O 主线完成**；draft-model family / EAGLE / MTP / Medusa **未做**（Optional） |
| 4 | 在线 serving + Triton 部署对照报告 | M20 | 单卡/多实例 |
| 5 | 监控 + 容量规划报告 | M21 | 单机起 |
| 6 | TP/PP/DP/EP 实验（多卡 E+H；单卡仅 T+O） | M22 | ★多卡 |
| 7 | TensorRT-LLM 跨引擎对照 | M23 | NVIDIA GPU |
| 8 | CP / Ring Attention / P-D 分离设计与实践 | M24 | ★多卡　✅ T / ✅ DCP E / ⬜ Ring E；P-D E → M27 |
| 9 | 压缩专题（按岗位深度） | M25 | 单卡起　✅ T（推理岗）/ ⬜ E |
| 10 | 四大项目组合 + 面试三档表达 | M26 | 无新增 |
| 11 | P-D 分离 + NIXL KV 传输实测（阶段 K） | M27 | ★多卡　✅ T+E+O+H COMPLETE（8×H20 单节点，正式实验最多 4 GPU）；E5 未执行；P1 未做 |

### M26 第二阶段收口：项目组合 + 掌握标准 + 面试化表达（扩展 M16）　✅　**收口完成（含明示占位）**（2026-08-13）
- 把 A/B/C/D 四项目 README 化（招聘者 5 分钟看懂：问题/源码路径/环境/基线/瓶颈/改动/性能变化/回归/复现），核心图 ≤5 张；每主题 **30s / 3min / 10min** 三档表达；**明确标注能力边界**（已验证 / 只读源码未获硬件 / 仅理论）——这种区分反而证明系统工程严谨性。
- **✅ 产出**：[`notes/m26/portfolio.md`](notes/m26/portfolio.md)（四项目 + 能力边界总表 + 5 张核心图选择理由）；
  第二阶段三档表达 + 重点题为个人备考文档，已移出仓库。
- **依赖核查结论（做之前先查的）**：A（第一阶段）与 C（M12+M20+M21）**不依赖 M23–M25，已完整收口**；
  **B 部分占位** —— TP 已在 4×H20 实测，PP/DP/EP **阻塞于多卡机时**（非模块依赖）；
  **D 必须占位** —— 缺 **M23 TensorRT-LLM** 与跨引擎 benchmark，**整项标为未完成**，
  **不把「量化 + spec decode 两条腿」包装成完整的跨引擎项目**。
- **核心图如实说明**：原设想的「请求时序 / CB / KV block 生命周期」在本 lab 里的证据是 **trace 日志与不变量、不是图**，
  **没有为了凑图而画图**；实际选了 5 张结论价值最高的（M6 decode / M21 瓶颈签名 / M21 容量 / M18 量化 / M22 TP）。
- **✅ 已随 M23 L1 回填**：项目 D 核心已闭合（2×2 + generation consistency）；L2/L3/kernel profiling 明确不做。

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
- **当前下一步（2026-08-16 更新）**：第一阶段已收口。第二阶段 **M17 / M18 / M19 / M20 / M21 / M22(TP) / M23(L1) / M24(DCP) / M25(T) / M26** 主线已收口。
  **M23 L1 FROZEN**（不扩 L2/L3/kernel profiling）。**M25 T COMPLETE**（不做 E）。**M24 DCP E+O+H COMPLETE**（4×H20 / 1.5B）；**Ring Attention / P-D 仍 ⬜**。
  **M26**：A/C 完整；B 部分占位（PP/DP/EP + Ring/P-D）；D 核心已随 M23 L1 闭合。
  → 下一优先：M22 的 PP/DP/EP（仍需多卡窗口）；或 M13 maintainer review。
- **当前进度（2026-10-06）**：阶段 K / M27 **T+E+O+H COMPLETE**；8×H20 单节点同一新实例 fresh Gate0→E4 完整，正式实验最多 4 GPU（E4 每 arm 4 GPU）；E5 未执行、P1 未做，E0–E4 为最终范围；旧实例交接仅独立复现参考；post-hoc 审计 PASS_WITH_DOCUMENTATION_CORRECTION，修正已闭环。结果见 [`notes/m27/results.md`](notes/m27/results.md)。
- **当前进度（2026-10-12）**：阶段 L / M28 **DESIGN FROZEN v1 · CPU_VALIDATED · GPU NOT EXECUTED**；脚本、mock 演练、validator 自检完成，A/B 实验等待 GPU 与故障注入授权，暂无实测结论。见 [`notes/m28/execution_record.md`](notes/m28/execution_record.md)。
- **交接（2026-08-06）**：工作已交由 Cursor 继续。接手须先读 **[`HANDOFF.md`](HANDOFF.md)**（进度/下一步/已知坑）与 **[`AGENTS.md`](AGENTS.md)**（硬性规则：诚实性红线、Git 只署 CiaranCw、GPU 防 OOM、必需环境变量）。
