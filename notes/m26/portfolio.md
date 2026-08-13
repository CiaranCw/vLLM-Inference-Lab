# M26 · 项目组合 A–D（招聘者 5 分钟看懂）

> 本文件把两阶段的模块**重组成四个可讲的项目**，每个按同一模板写：
> **问题 → 源码路径 → 环境 → 基线 → 瓶颈 → 改动 → 性能变化 → 回归 → 复现**。
> 每条结论都能回溯到 `outputs/` 下的原始数据，索引见 [`../evidence_index.md`](../evidence_index.md)。
>
> **能力边界用三档标注**（这是本文件最重要的纪律）：
> 🟢 **已真实运行验证** ｜ 🟡 **只读源码，未获硬件验证** ｜ ⚪ **仅理论/设计，未实现**
> 把没做的事标清楚，比把它含糊过去更能说明工程严谨性 —— 这也是 ROADMAP 对 M26 的原话要求。

---

## 总览

| 项目 | 组成模块 | 完成度 | 阻塞于 |
|---|---|---|---|
| **A** vLLM V1 Trace & Optimization Lab | M0–M13、M16、M17 | 🟢 **完整** | — |
| **B** Distributed LLM Inference Lab | M22 | 🟢 TP ／ ⚪ PP·DP·EP | **多卡机时**（4×H20 窗口已结束） |
| **C** Production Serving Stack | M12、M20、M21 | 🟢 **完整**（四要素齐全） | — |
| **D** Cross-Engine Optimization | M10、M18、M19、M20 ／ **缺 TRT-LLM** | ⚪ **未完成** | **M23（TensorRT-LLM）** |

**一句话现状**：A 与 C 已经是可以直接拿去讲的完整项目；B 有真实多卡 TP 数据但并行策略只覆盖了 TP；
D 的量化与投机解码两条腿已就位，**跨引擎对照这条腿缺 M23，整项标为未完成**。

---

# 项目 A · vLLM V1 Trace & Optimization Lab　🟢 完整

**问题**：能不能把 vLLM V1 的推理主链路，用**一次真实运行的证据**串起来；并在这条链路上做一次
**可被验证**的性能改动？

**源码路径（带 `file:line`，pinned commit `ee0da84`）**：
`LLM.generate → Scheduler → GPUModelRunner → BlockTable → Attention(FlashAttention) → KV 写/读 → Sampling`。
用只读 trace 钩子采证，**trace 关闭时零副作用**（已回归验证）。

**环境**：单卡 RTX 5080(sm_120) / WSL2 / Qwen2.5-0.5B-Instruct / `enforce_eager` / `VLLM_USE_PRECOMPILED=1`。

**基线（M6）**：decode 单步延迟近平、吞吐近线性；compiled(CUDA Graph) 比 eager 快 **2.5–2.85×**；
KV **≈12 KiB/token**（实测 = 理论）。

**瓶颈定位（M7，本项目的技术核心）**：`torch.profiler` 证实
**decode 的 ~7ms 地板 = CPU kernel 提交开销** —— eager 下 GPU 仅忙 **23%**、每步 **257 次**提交；
compiled FULL 图把提交压到 **15**、wall **7.44→2.50ms**。而**长 prefill 是 GPU 计算受限**（gpu_frac **0.88**）。
统计口径经过审查修订（GPU 时间取 kernel 事件区间**并集**、提交计数覆盖 runtime+driver+graph 三类 API）。

**改动**：
- **M8 能写 kernel**：自写 Triton `slot_mapping` 生成 kernel，与 PyTorch 语义 oracle + vLLM 官方 kernel
  **三路交叉 22/22**，性能 ≈ 官方。
- **M9 能改（并诚实收尾）**：按 profiler 证据合并 decode 的两处 H2D。**机制成立、正确性完整，
  但 E2E 收益落在噪声内** —— isolated microbench 提前算出天花板仅 **0.13%**，故如实保留为**无收益报告**。
- **M17 能量化调度权衡**：APC 收益要等 prompt 长到穿透 M7 那条固定开销地板才显现
  （P=2048 省 **58.5%** 首 token；P=256/512 的 CI 含 0）；chunked prefill 把并发 decode 的**最坏 stall 砍半**
  （22.87→10.51ms）却**不改中位**，代价是长 prefill 自身 TTFT **+47%**。
  I5-b 进一步量化：固定前缀下相对收益随输出变长 **61.2%→22.9%→9.3%→3.4%** 单调下降。

**开源贡献（M13）**：[vllm-project/vllm#51017](https://github.com/vllm-project/vllm/pull/51017) ——
为 `BlockTable.compute_slot_mapping`（原本零覆盖）补单元测试，**18 passed / DCO pass / test-only**；
过程中用**变异测试**证明「参照实现必须独立于产品 helper」，否则测试会自证。
🟡 **状态：已提交，未合入**（卡在上游新贡献者门禁，需 maintainer 打 label）。

**回归**：`scripts/check_invariants.py`（M17 53 条不变量）、`scripts/reproduce.py`（四模式入口，
`--verify-only` 不跑实验也能核对产物与不变量）。不变量脚本本身经过**变异测试**验证有效性
（改文档数字 / 删样本 / 破坏守恒 → 3/3 均被捕获）。

**复现**：`python scripts/reproduce.py --module m17 --verify-only`

---

# 项目 B · Distributed LLM Inference Lab　🟢 TP ／ ⚪ PP·DP·EP

**问题**：多卡 Tensor Parallel 到底把什么变快了？「TP 加速」这个说法在什么口径下成立？

**环境**：**4×H20**（租用机时，2026-08-09）/ Qwen2.5-7B / `enforce_eager` / in-process；TP=1/2/4。

🟢 **已真实运行验证**：
- **权重与 KV**：每卡权重 ≈1/TP（0.501 / 0.254）；**KV 不复制**；KV 容量 1.22M→5.69M tokens。
- **核心反直觉结论**：**稳态 decode 单步 TP 显著更慢**（~**+19–24%**，四档 batch 的 Bootstrap CI 均排除 0）
  —— 每层 2 次 all-reduce 的通信税在小 batch 下盖过了算力增益。
- **长 prefill 才真受益**：P=4096 时 TP4 ≈ **3.39×**。
- **口径拆解（本项目最有价值的一条）**：bs=64 时 `speedup_vs_tp1 > 1`，但拆开看，
  **增益主要来自并发 prefill 的 first-token（804→230ms），不是 decode 单步变快**。
  → **口径不拆开，就会把通信税说成加速。**
- 不变量 **22 PASS**；跨 TP 输出一致性 hash 60/120（bs≥16 不一致，判为诊断项非 FAIL，
  因为 TP 改变 GEMM 切分与归约顺序、有限精度下贪心 argmax 可能翻转）。

🟡 **只读源码，未获硬件验证**：`MultiprocExecutor` / `InprocClient`、per-rank `collective_rpc`、
custom all-reduce 的优先路径选择（带 `file:line`，见 `notes/m22/tensor_parallel.md`）。

⚪ **仅设计，未实现未实测**：**PP / DP / EP / CP / P-D 分离**、多卡 serving、扩展效率曲线。
方案原文在 [`../../MULTI_GPU_EXPERIMENT_PLAN.md`](../../MULTI_GPU_EXPERIMENT_PLAN.md)。
**阻塞于多卡机时**（H20 窗口已结束），**不是**阻塞于其他模块。

**复现**：`outputs/m22/h20_formal_*.csv`、`scripts/m22_*.py`；`check_invariants.py --module m22`

---

# 项目 C · Production Serving Stack　🟢 完整

**问题**：把 vLLM 真正**部署成服务**，能不能回答运维必须回答的问题 ——
最大稳定 QPS 是多少、并发到哪 P99 失控、瓶颈是谁、一张卡承载多少用户、故障怎么定位？

**四要素齐全**（这是本项目的完整性所在）：

| 要素 | 模块 | 关键结论 |
|---|---|---|
| **在线服务 + SLO 容量** | M12 | 真实 `vllm serve` + 官方开环 Poisson 压测；**最大稳定 QPS≈40**、饱和 ~46.6；**过载是尾延迟劣化而非吞吐崩溃**；容量随负载 shape 变 |
| **Triton 部署对照** | M20 | **同镜像** C0=`vllm serve` vs C1=Triton+vLLM backend；QPS32 TTFT **24→719ms**、QPS40 **26→952ms**（CI 排除 0），而 **ITL 接近**（~9–12ms）→ 代价在**入口排队**而非逐步解码 |
| **监控与容量规划** | M21 | Prometheus(1s 抓取)+Grafana(provisioning)+DCGM+自写 exporter；**先证明指标可信**（21 个窗口服务端 counter 与客户端 bench **逐条精确相等**），再发现**服务端直方图 P99 ≠ 客户端 exact percentile**（中位 +9.7%，最大 +68%） |
| **故障处理** | M21 | 4 个演练全 PASS、**7 条告警规则全部真实 firing 过**；核心发现 **`SIGKILL` API server 不会杀死持有显存的 `EngineCore` 子进程** —— `up==0` 但显存 90s 不回收，下次拉起直接撞 `No available memory for the cache blocks` |

**瓶颈签名矩阵（M21 E4）**：四类压力源在面板上可区分 —— CPU 提交受限（GPU util 低 + 进程 CPU >100%）、
prefill 计算压力（粗粒度 GPU util 66%）、KV 受限（kv_usage→1 **且**抢占计数上升，两条正交证据）、
队列受限（waiting 堆到 1196 而 GPU 未满）。

**方法学亮点（面试可讲）**：
- **监控栈自身开销**用单变量 A/B 量化过：ON/OFF 差异 **≲1%**，小于 seed 噪声。
- **容量数字有保质期**：同配置下 M21 测得的稳定 QPS（**C ∈ [24,32)**）显著低于 M12 的 40，
  排除了监控开销、降频、机器整体变慢之后，差异定位到**高并发 TPOT +30%**顶到 `max_num_seqs=64` 硬闸的正反馈；
  **根因未定**（M12 当年没落环境指纹），因此如实记为「跨会话不稳定」，既不宣称复现 40 也不宣称 40 是错的。

⚪ **未做**：Kubernetes / 多副本 / 长时 soak / 跨机 SLA / runtime CUDA OOM 注入。
本环境 `DCGM_FI_PROF_*`（SM active）全为 N/A，**拿不到 SM 占用率**；WSL2 下无进程级显存归因。

**复现**：`monitoring/`（配置全部入库、面板由 provisioning 加载）、`scripts/m21_*.py`；
`check_invariants.py --module m21`（14 PASS）

---

# 项目 D · Cross-Engine Optimization　⚪ **未完成（阻塞于 M23）**

**问题**：同模型同硬件下，vLLM 与 TensorRT-LLM 的工程取舍差在哪？量化与投机解码在两个引擎上的收益是否一致？

**已就位的两条腿**（🟢 已真实运行验证）：

| 方向 | 模块 | 关键结论 |
|---|---|---|
| **量化** | M10（FP8 W8A8 + FP8 KV）、**M18（W4A16 AWQ/GPTQ）** | W4A16 整体权重 **−53.6%(0.5B) / −62.8%(1.5B)**，但 `embed`/`lm_head` **未被量化**，剔除后可量化部分 **−74.0%**（≈理论值）→ **模型越小稀释越狠**；速度上 **28 个对照点 0 个变快**。FP8 至少在长 prefill 换到 −27% 真加速，W4A16 连这段都没换到 |
| **投机解码** | M19（N-gram） | 优化的是**串行 execution rounds**（128→33）而非保证少算；完成延迟 −**72~75.5%**；Acceptance 重要但**非充分**指标 |

**⚪ 缺失的核心一条腿**：**TensorRT-LLM**（checkpoint 准备 → engine 构建 → 加载 → benchmark）、
**Triton + TRT-LLM Backend**、以及**同模型同硬件的严格跨引擎对照**（固定精度/长度/并发/warmup/采样/benchmark 方法）。

→ **本项目整体标为未完成，阻塞于 M23**。
**不把「量化 + spec decode 两条腿」包装成「跨引擎优化项目」** —— 缺的是这个项目的定义性内容。

**已有的相邻能力（可在面试中说明）**：M20 已经把 **Triton + vLLM backend** 跑通并做过同镜像对照，
所以 D 缺的不是 Triton 部署能力，而是 **TRT-LLM 引擎本身与跨引擎 benchmark 方法**。

---

## 核心图（≤5 张，讲的时候按这个顺序）

ROADMAP 原本设想的 5 张是「请求时序 / CB / KV block 生命周期 / profiler timeline / 优化前后」。
**如实说明**：前三类的证据在本 lab 里是 **trace 日志与不变量**，不是图（没有为了凑图而画图）；
profiler 的结论以 JSON 指标呈现。因此实际用于讲述的 5 张图按**结论价值**挑选：

| # | 图 | 讲什么 |
|---|---|---|
| 1 | `figures/fig1_m6_decode.png` | 能测量：decode 单步近平、吞吐近线性、compiled 2.5–2.85× |
| 2 | `figures/fig12_m21_bottleneck_signature.png` | 能归因：四类瓶颈在监控面板上的可区分签名 |
| 3 | `figures/fig11_m21_capacity.png` | 能做容量规划：拐点对到达时长极敏感；过载先堆队列 |
| 4 | `figures/fig13_m18_w4a16.png` | 能量化取舍：省显存但不省时间；分母决定「省了多少」 |
| 5 | `figures/fig9_m22_tp_scaling_h20_formal.png` | 能上多卡：TP 的加速要拆口径，否则把通信税说成加速 |

全部图均由**已落盘数据离线生成**（不需 GPU、不 import vllm）：
`scripts/m{6,16,17,18,19,21,22}_figures.py`。

---

## 能力边界总表（面试时主动说，不等对方问）

| 能力 | 档位 | 依据 |
|---|---|---|
| vLLM V1 主链路证明（Scheduler/KV/Attention/Sampling 闭环） | 🟢 已验证 | M1–M5 trace + 不变量 |
| 性能测量与瓶颈归因（CPU 提交 vs GPU 计算） | 🟢 已验证 | M6 / M7 profiler |
| 写 CUDA/Triton kernel | 🟢 已验证（**限 slot-mapping 类**） | M8 三路交叉 22/22 |
| 改产品代码并诚实评估 | 🟢 已验证（**结论是无收益**） | M9 |
| 量化（FP8 W8A8 / FP8 KV / W4A16 AWQ·GPTQ） | 🟢 已验证 | M10 / M18 |
| 投机解码（N-gram） | 🟢 已验证 | M19；draft-model / EAGLE / MTP ⚪ 未做 |
| 在线服务、SLO 容量、压测方法 | 🟢 已验证 | M12 |
| Triton Inference Server 部署与对照 | 🟢 已验证 | M20 |
| 监控栈与故障定位 | 🟢 已验证 | M21 |
| **多卡 Tensor Parallel** | 🟢 已验证（4×H20） | M22 |
| **PP / DP / EP / CP / P-D 分离** | ⚪ **仅设计** | 阻塞于多卡机时 |
| **TensorRT-LLM 与跨引擎对照** | ⚪ **未做** | 阻塞于 M23 |
| 长上下文（Ring Attention 等） | ⚪ **仅理论** | M24 未做 |
| 模型压缩（蒸馏/剪枝/低秩） | ⚪ **仅理论**（推理岗只需 T） | M25 未做 |
| Kubernetes / 多副本 / 跨机 SLA | ⚪ **未做** | 不在本 lab 范围 |

**统一边界声明**：除 M22 在 4×H20 外，所有绝对值仅限
**单卡 RTX 5080(sm_120) / WSL2 / Qwen2.5-0.5B 或 1.5B / `enforce_eager` / 单副本**，**不外推**。
