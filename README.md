# vLLM V1 Inference Trace & Optimization Lab

> 在**固定 vLLM commit**上，对 V1 单机推理主链路做**逐 step 结构化追踪 → 性能基线 → profiler 定位瓶颈 → 真实代码修改 → 正确性与性能回归**的长期工程项目。
> 不是"读过源码"，而是**观察真实运行、用证据验证理解、动手改并用数据说话**——包括一份**诚实的无收益报告**和一次**上游 PR**。

**环境**：单卡 RTX 5080（sm_120, consumer Blackwell）· WSL2 · Qwen2.5-0.5B-Instruct · vLLM `ee0da84`/v0.24.0（editable）· `enforce_eager` 基线。所有绝对值限本配置，不外推。

---

## 30 秒速览

- **主链路能证**：用只读 trace 钩子把 `LLM.generate → Scheduler → GPUModelRunner → BlockTable → Attention → KV 写/读 → Sampling` 串成**一次运行的闭环证据**（M1–M5）。
- **能测量**：进程内闭环 benchmark——decode 单步近平、吞吐近线性（compiled 比 eager 快 **2.5–2.85×**）；prefill 计算受限；KV≈**12 KiB/token**（实测=理论）（M6）。
- **能定位**：torch.profiler 证实 **decode 的 ~7ms 地板 = CPU kernel 提交开销**（eager GPU 仅忙 23%，257 次提交；compiled FULL 图把提交降到 15、wall 7.44→2.50ms），**长 prefill = GPU 计算受限**（M7）。
- **能写 kernel**：自写 Triton `slot_mapping` 生成 kernel，与 PyTorch 语义 oracle + vLLM 官方 kernel **三路交叉 22/22**，性能 ≈ 官方（M8）。
- **能改（诚实）**：profiler 驱动合并 decode 两处 H2D——机制成立、正确性完整，但 **E2E 收益落在噪声内**，如实保留为**无收益报告**（M9）。
- **量化 & 在线服务**：FP8 权重显存 −36% / KV 容量 ~2×，但 **decode 反而 +36%（诚实反常）**（M10）；真实 `vllm serve` + 开环 Poisson 测出**最大稳定 QPS ≈ 40**，且**过载是尾延迟劣化而非吞吐崩溃**（M12）。
- **上游 PR**：为 `compute_slot_mapping` 补单元测试（**本机 26/26 通过**），取代 stale PR #30683——研究/验证就绪，待提交（M13，见 §7）。

> 完整里程碑与后续扩展路线见 [`ROADMAP.md`](ROADMAP.md)（第一阶段 M0–M16 + 第二阶段 M17–M26）。每条结论 → 证据文件见 [`notes/evidence_index.md`](notes/evidence_index.md)。面试三档表达见 [`notes/m16/interview_and_presentation.md`](notes/m16/interview_and_presentation.md)。

---

## 能力阶梯（当前位置）

```
能读 ──→ 能串 ──→ 能证 ──→ 能改 ──→ 能优
                          ▲ 当前：能改（+完整能优方法论，待正空间目标）
```

| 能力域 | 层级 | 证据（模块） |
|---|---|---|
| Scheduler / Continuous Batching | 能证 ✅ | M2 |
| KV Cache / Block 管理 | 能证 ✅ | M3 |
| Attention Backend（闭环 slot_mapping/block_table） | 能证 ✅ | M4 |
| 执行模式 eager/compile/CUDA Graph | 能证 ✅ | M5 |
| 性能测量 latency/throughput/显存·KV | 能测量 ✅ | M6 |
| Profiler 瓶颈定位（CPU vs GPU / launch） | 能定位 ✅ | M7 |
| 写 Triton kernel | 能写 ✅ | M8 |
| profiler 驱动的真实改动 + 回归 | 能改 ✅ | M9 |
| 量化对比（显存/性能/质量/源码机制） | 能测量 ✅ | M10 |
| 在线服务容量 / SLO | 能测容量 ✅ | M12 |
| 上游贡献（测试补齐） | 🔄 就绪待提交 | M13 |

---

## 核心图（由既有实验 CSV 离线生成，复现：`python scripts/m16_figures.py`）

**M6 · Decode：eager vs compiled**
![M6 decode](figures/fig1_m6_decode.png)
> decode 单步延迟近平、吞吐近线性；compiled（CUDA Graph FULL）比 eager 快 2.5–2.85×；eager ~7ms 地板由 M7 证实是 CPU 提交开销。

**M6 · Prefill：延迟 vs 输入长度**
![M6 prefill](figures/fig2_m6_prefill.png)
> compiled 仅在 ≤128 提速（PIECEWISE 图）；≥256 超本配置捕获上限 → 回退 NONE ≈ eager。

**M10 · 量化随负载而异（诚实反常）**
![M10 quant](figures/fig3_m10_quant.png)
> FP8 W8A8 在**长 prefill 快 27%**（计算受限受益），却在**小 batch decode 慢 36%**（GEMV 访存受限 + 动态激活量化开销 + sm_120 未优化）。"量化提速"不是普适结论。

**M12 · 在线容量与 SLO 拐点**
![M12 capacity](figures/fig4_m12_capacity.png)
> QPS≤32 队列恒空、P99 TTFT 平稳 ~25–32ms；最大稳定 QPS≈40（3-seed worst 判 SLO）；44/48 时 P99 TTFT 破 300ms 线但吞吐仍跟随——**过载是尾延迟劣化，不是吞吐崩溃**。

---

## 逐模块成果（M0–M13，不遗漏信息）

| 模块 | 一句话成果 | 关键数据 | 源码改动 | 证据 |
|---|---|---|---|---|
| **M0** 环境/基础设施 | 固定版本 + 统一 trace 工具 + 零副作用回归 | commit ee0da84 / Qwen2.5-0.5B / sm_120；trace-off 归一化逐行一致 | 只读钩子（默认关） | `vllm_dev_environment.md`、`vllm-src/vllm/_lab_trace.py` |
| **M1** 主链路 Trace | 单请求从入口到输出的完整数据流 | backend=FLASH_ATTN；prefill/decode shape 对照 | 无 | `notes/day1/*` |
| **M2** Continuous Batching | 错峰混批、token 预算、chunked prefill | 同批 prefill+decode；max_num_seqs 排队；E5 预算共享/槽位争用 | 无 | `notes/day2/*` |
| **M3** KV / Block 管理 | 逻辑→物理块、prefix 命中、抢占重算 | hash/ref_cnt/共享/对齐；K5 尾部优先淘汰、前缀优先保留 | 无 | `notes/day3/*` |
| **M4** Attention Backend | **闭环**：SchedulerOutput→KV 写(slot_mapping)→读(block_table) | GQA 14/2/64；kv_cache shape (34801,2,16,2,64) | 2 只读钩子 | `notes/m4/*`、`outputs/m4/m4_closed_loop.jsonl` |
| **M5** 执行模式 | enforce_eager 实为一票否决；默认 O2 compiled 在 sm_120 未降级 | 捕获 51 PIECEWISE+35 FULL；compiled 与 eager 第 5 token 起分叉(数值差异非 bug) | 2 只读钩子 | `notes/m5/*`、`outputs/m5/*` |
| **M6** 引擎 Benchmark ★ | decode 近平/吞吐近线性、prefill 计算受限、KV 容量 | compiled 全 batch 快 2.5–2.85×；KV≈12KiB/token；最大并发 357× 是容量等价值(运行受 max_num_seqs=64 限) | 无 | `notes/m6/*`、`outputs/m6/baseline.csv` |
| **M7** Profiler ★ | decode 地板=CPU 提交；长 prefill=GPU 计算受限 | eager gpu_frac 0.23、257 提交→compiled 15、wall 7.44→2.50ms；prefill gpu_frac 0.88 | 无 | `notes/m7/*`、`outputs/m7/profile_*.json` |
| **M8** Triton Kernel ★ | 自写 slot_mapping 生成 kernel，三路交叉全绿 | **22/22**；microbench 自写≈vLLM 官方(28.4≈28.4us) | 无（官方作参照） | `notes/m8/*`、`outputs/m8/correctness.json` |
| **M9** profiler 驱动改动 ★ | 合并 decode H2D：机制成立但 E2E 无收益（诚实） | cudaMemcpyAsync/step 11→10；E2E 差异 ±0.01-0.02ms≪stdev；天花板 0.13% | 门控改动(默认关,可回退) | `notes/m9/*`、`outputs/m9/*` |
| **M10** 量化对比 ★ | FP8 显存收益明确、速度诚实含反常 | 权重 −36% / KV ~2×；PPL +0.51%~+2.17%；decode **+36% 变慢**、prefill −27% | 无 | `notes/m10/*`、`outputs/m10/*` |
| **M11** 双卡 TP | ⬜ 资源门控（本机单卡）——待多卡；无硬件仅 T+O | — | — | 见 ROADMAP M22 |
| **M12** 在线 Serving/SLO | 真实 serve + 开环 Poisson 的容量规划 | **最大稳定 QPS≈40**、饱和~46.6、过载=尾延迟劣化、容量随 shape 变 | 无 | `notes/m12/*`、`outputs/m12/*` |
| **M13** 首次上游 PR | 🔄 为 compute_slot_mapping 补单元测试，取代 stale #30683 | 本机 **26/26 通过**、ruff 全绿；待提交（见 §7 占位） | 纯加测试 | `notes/m13/*` |

★ = 第一阶段核心验收项。核心验收线 7/9（仅差 M13 上游 PR 落地、M11 多卡）。

---

## 亮点洞察（面试可讲的"非显然"结论）

1. **decode 慢不是 GPU 慢，是 CPU 喂不饱 GPU**：eager 下 GPU 仅忙 23%，每步 257 次 kernel 提交；CUDA Graph 把提交降到 15、wall 砍掉 2/3。这是 continuous batching/CUDA Graph 存在的根本原因。
2. **"量化一定提速"是错的**：同一 FP8 权重量化，长 prefill 快 27%、小 batch decode 反而慢 36%——收益取决于计算受限还是访存受限。
3. **过载 ≠ 吞吐崩溃**：开环 Poisson 下，超过 SLO 容量后吞吐仍跟随 offered QPS，先崩的是 **P99 尾延迟**；所以"最大稳定 QPS(40)" < "饱和吞吐(46)"。
4. **省计算 ≠ 省延迟**：prefix cache 命中省 94% prefill 计算 token，但首 token 延迟改善落在噪声内（短 prompt 下 ON 甚至略慢于 OFF）。
5. **不是所有 profiler 热点都值得改**：M9 的 H2D 合并机制成立、正确，但该路径已被上游高度优化（pinned+non_blocking+overlap），isolated microbench 提前算出天花板仅 0.13%——**能证明"为什么没收益"也是能力**。

> 完整证据链见 [`notes/evidence_index.md`](notes/evidence_index.md)；每个结论都可回溯到 trace/log/jsonl/csv。

---

## 复现

```bash
cd /home/ciaran/projects/vllm-inference-lab
source .venv/bin/activate

# 环境自检（不加载模型）
python scripts/smoke_import.py
# 确认实际加载的是 vllm-src（editable）
python -c "import os, vllm; print(os.path.realpath(vllm.__file__))"

# 各模块实验（示例；运行需本机 GPU，注意 GPU 占用）
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/m6_run_all.py      # 引擎 benchmark
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/m7_run_all.py      # profiler
python scripts/m8_test.py                                       # Triton kernel 三路交叉
VLLM_USE_FLASHINFER_SAMPLER=0 python scripts/m10_run_all.py     # 量化对比
python scripts/m12_run_sweep.py                                 # 在线开环 serving + SLO 扫

# 核心图（无需 GPU，仅由 CSV 生成）
python scripts/m16_figures.py    # -> figures/*.png
```

关键运行约定（本机 WSL2/sm_120）：`VLLM_USE_FLASHINFER_SAMPLER=0`（无 nvcc，FlashInfer sampler JIT 会失败）、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、`seed=0`、`temperature=0`。

---

## 工作区结构与约束

```text
/home/ciaran/projects/
├── vllm-src/           # vLLM 源码（唯一真源）。ee0da84=v0.24.0，含中文学习注释
├── vllm_playground/    # 干净基线 venv（只读，不得修改）
└── vllm-inference-lab/ # 本工作区
    ├── scripts/        # 各模块实验脚本 + m16_figures.py
    ├── notes/          # 逐模块源码学习笔记（含 evidence_index / m16 面试表达）
    ├── outputs/        # 实验输出（csv/json/jsonl；大件 raw/traces 走 gitignore、可复现）
    ├── figures/        # 核心图（由 CSV 离线生成）
    ├── logs/           # 实验日志、git 状态快照
    ├── ROADMAP.md      # 两阶段里程碑（M0–M26）
    └── vllm_dev_environment.md
```

- **源码修改只发生在 `../vllm-src`**；本目录 `.venv` 以 editable 安装 vllm-src，Python 改动重启进程即生效；`csrc`/CUDA kernel 用 `VLLM_USE_PRECOMPILED=1` 复用预编译，源码级改动不自动生效。
- 不修改 `../vllm_playground/.venv`（只读基线）；不对 `../vllm-src` 执行破坏性 git 操作；环境记录不写任何密钥/Token。
- 本目录**是 git 仓库**，已推送至 `github.com/CiaranCw/vLLM-Inference-Lab`（提交者仅 CiaranCw）。

---

## §7 M13 首次上游 PR ——（占位，待提交后补充）

> 研究与本机验证已完成，**尚未提交上游**，等提交后回填链接与 review 过程。当前快照见 [`notes/m13/STATUS.md`](notes/m13/STATUS.md)。

- **贡献**：为 vLLM `vllm/v1/worker/block_table.py` 的 `compute_slot_mapping` / `_compute_slot_mapping_kernel` 补单元测试（当前 `tests/` 零覆盖），取代 stale PR #30683（其死于 pre-commit 未过、非评审否决）。
- **已就绪**：草稿测试 `notes/m13/pr_draft/test_block_table.py` 在真实 RTX 5080 上 **26/26 通过**（2 手算 golden + 7 非 CP 边界 + 16 DCP4/PCP2 interleave×rank + 1 退化），vLLM 配置下 `ruff` 全绿。
- **待回填**：`[ ]` PR 链接　`[ ]` 最终 diff　`[ ]` CI/pre-commit 结果　`[ ]` review 往返与迭代　`[ ]` 是否合入。
- **待用户决策**：① go/no-go 进入提交准备；② 署名口径（vLLM 要求披露 AI 辅助 + `Co-authored-by:`，与本仓库 no-coauthor 约定不同）。
