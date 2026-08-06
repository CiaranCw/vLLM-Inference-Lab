# vLLM V1 Inference Trace & Optimization Lab

> 在**固定 vLLM commit**上，对 V1 单机推理主链路做**逐 step 结构化追踪 → 性能基线 → profiler 定位瓶颈 → 真实代码修改 → 正确性与性能回归**的长期工程项目。
> 不是"读过源码"，而是**观察真实运行、用证据验证理解、动手改并用数据说话**——包括一份**诚实的无收益报告**和一次**上游 PR**。

**环境**：单卡 RTX 5080（sm_120, consumer Blackwell）· WSL2 · Qwen2.5-0.5B-Instruct · vLLM `ee0da84`/v0.24.0（editable）· `enforce_eager` 基线。所有绝对值限本配置，不外推。
> M1–M12 与 M17 在上述 pinned commit 上完成；**M13 的上游 PR 另用独立干净 clone 跟最新 `main` 开发**（见 §7 与下方工作区结构）。

---

## 30 秒速览

- **主链路能证**：用只读 trace 钩子把 `LLM.generate → Scheduler → GPUModelRunner → BlockTable → Attention → KV 写/读 → Sampling` 串成**一次运行的闭环证据**（M1–M5）。
- **能测量**：进程内闭环 benchmark——decode 单步近平、吞吐近线性（compiled 比 eager 快 **2.5–2.85×**）；prefill 计算受限；KV≈**12 KiB/token**（实测=理论）（M6）。
- **能定位**：torch.profiler 证实 **decode 的 ~7ms 地板 = CPU kernel 提交开销**（eager GPU 仅忙 23%，257 次提交；compiled FULL 图把提交降到 15、wall 7.44→2.50ms），**长 prefill = GPU 计算受限**（M7）。
- **能写 kernel**：自写 Triton `slot_mapping` 生成 kernel，与 PyTorch 语义 oracle + vLLM 官方 kernel **三路交叉 22/22**，性能 ≈ 官方（M8）。
- **能改（诚实）**：profiler 驱动合并 decode 两处 H2D——机制成立、正确性完整，但 **E2E 收益落在噪声内**，如实保留为**无收益报告**（M9）。
- **量化 & 在线服务**：FP8 权重显存 −36% / KV 容量 ~2×，但 **decode 反而 +36%（诚实反常）**（M10）；真实 `vllm serve` + 开环 Poisson 测出**最大稳定 QPS ≈ 40**，且**过载是尾延迟劣化而非吞吐崩溃**（M12）。
- **上游 PR 已提交**：[vllm-project/vllm#51017](https://github.com/vllm-project/vllm/pull/51017) —— 为 `BlockTable.compute_slot_mapping` 补单元测试（该路径原本零覆盖），**18 passed / DCO pass / test-only**；过程中用变异测试证明「参照实现必须独立于产品 helper」（M13，见 §7）。
- **能量化调度权衡**：APC 收益要等 prompt 长到穿透 M7 那条固定开销地板才显现（P=2048 省 **58.5%** 首 token，P=256/512 的 CI 含 0）；chunked prefill 把并发 decode 的**最坏 stall 砍半**（22.87→10.51ms）却**不改中位**，代价是长 prefill 自身 TTFT **+47%**（M17，见 §8）。

> 完整里程碑与后续扩展路线见 [`ROADMAP.md`](ROADMAP.md)（第一阶段 M0–M16 + 第二阶段 M17–M26）。每条结论 → 证据文件见 [`notes/evidence_index.md`](notes/evidence_index.md)。面试三档表达见 [`notes/m16/interview_and_presentation.md`](notes/m16/interview_and_presentation.md)。
>
> **接手/协作者请先读**：[`HANDOFF.md`](HANDOFF.md)（当前进度、下一步选项、已知坑）与 [`AGENTS.md`](AGENTS.md)（硬性工作规则：诚实性红线 / Git 署名 / GPU 安全 / 必需环境变量）。

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
| 上游贡献（测试补齐） | **已提交** ✅（PR #51017，待审查） | M13 |
| 调度特性权衡（APC / chunked prefill） | 能证 + 能量化 ✅ | M17 |

---

## 核心图（由既有实验数据离线生成，复现：`python scripts/m16_figures.py` + `python scripts/m17_figures.py`）

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

**M17 · APC：收益必须长到穿透固定开销地板**
![M17 APC](figures/fig5_m17_apc.png)
> 左：`off` 与 `cold`（APC 开但未命中）几乎重合——两者都要算完整 P；`hot` 几乎水平，因为命中后只算 16 个尾 token。右：median 差的 **Bootstrap 95% CI**——**P=256/512 的 CI 含 0**（收益无法与噪声区分），**P=1024/2048 才显著**（−32.4% / −58.5%）。误差棒=15 次 measure 的 stdev / CI，均排除 warmup 样本。

**M17 · Chunked prefill：只改尾部，不改中位**
![M17 chunked](figures/fig6_m17_chunked.png)
> 左（两侧 `max_num_batched_tokens=4096` 严格单变量，chunked 侧靠 `long_prefill_token_threshold=512` 强制分成 4×512）：并发 decode 请求 D 的**最坏 stall 22.87→10.51ms（−54%，CI 显著）**，而 **ITL median/p95 的 CI 都含 0**；代价是长 prefill **自身 TTFT +47%（显著）**。右：单请求下每多一个 prefill step ≈ **+3.2ms**，量级与 M7 所证的每-step 固定 CPU 提交开销一致。

---

## 逐模块成果（M0–M13 + M17，不遗漏信息）

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
| **M13** 首次上游 PR ★ | ✅ **PR [#51017](https://github.com/vllm-project/vllm/pull/51017) 已提交**（Ready for review，未合入） | **18 passed**、DCO pass、1 commit/1 file/+384、零产品改动；变异测试证明 oracle 独立性；避开 #39591 同名冲突 | 纯加测试 | `notes/m13/*`（研究/审查/验证/PR 草稿全留痕） |
| **M17** APC & Chunked Prefill ★ | 量化 APC 的生效边界与 chunked prefill 的真实权衡 | APC：P=2048 **−58.5%**、P=1024 −32.4%（CI 排除 0），P=256/512 **CI 含 0**；chunked：D 最坏 ITL **−54%**、median/p95 CI 含 0、长 prefill TTFT **+47%**；I1–I4 全过；**I5-b/E2 未做** | 无 | `notes/m17/*`、`outputs/m17/*`、`logs/m17/*` |

★ = 第一阶段核心验收项（M17 属第二阶段阶段 E）。**第一阶段核心验收线 8/9**（仅差 M11 双卡 TP，属硬件资源门控）。

---

## 亮点洞察（面试可讲的"非显然"结论）

1. **decode 慢不是 GPU 慢，是 CPU 喂不饱 GPU**：eager 下 GPU 仅忙 23%，每步 257 次 kernel 提交；CUDA Graph 把提交降到 15、wall 砍掉 2/3。这是 continuous batching/CUDA Graph 存在的根本原因。
2. **"量化一定提速"是错的**：同一 FP8 权重量化，长 prefill 快 27%、小 batch decode 反而慢 36%——收益取决于计算受限还是访存受限。
3. **过载 ≠ 吞吐崩溃**：开环 Poisson 下，超过 SLO 容量后吞吐仍跟随 offered QPS，先崩的是 **P99 尾延迟**；所以"最大稳定 QPS(40)" < "饱和吞吐(46)"。
4. **省计算 ≠ 省延迟，但"长到一定程度"就等号了**：prefix cache 命中省掉 94% prefill 计算 token，短 prompt 下首 token 延迟改善却落在噪声内（M6）；M17 把前缀加长后发现这是**地板效应**——P=256/512 的 CI 仍含 0，而 P=1024/2048 省出 **32.4% / 58.5%**（CI 排除 0）。**同一份"省计算"，穿透那条 ~7ms 固定开销地板才变成"省延迟"。**
5. **chunked prefill 改的是尾部，不是平均**：它把并发 decode 的最坏 stall 砍半（22.87→10.51ms），但 ITL 的 median/p95 几乎不动（CI 含 0），且长 prefill 自己的 TTFT 变慢 47%——**这是一次公平交换，不是免费加速**。另外机制不是"长 prefill 把 decode 挤出去"：原始逐 step 记录显示 decode **仍被同批调度**（`d=1, p=2048`），只是它的 token 要等一个 2049-token 的长 step 才返回。
6. **不是所有 profiler 热点都值得改**：M9 的 H2D 合并机制成立、正确，但该路径已被上游高度优化（pinned+non_blocking+overlap），isolated microbench 提前算出天花板仅 0.13%——**能证明"为什么没收益"也是能力**。

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

# M17：一个进程只建一个引擎（同进程内多引擎无法归还 KV 显存），故必须分片跑
python scripts/m17_prefix_cache.py --shard e1-on
python scripts/m17_prefix_cache.py --shard e1-off
python scripts/m17_prefix_cache.py --shard e3 --e3-chunked on --e3-mbt 512   # 另跑 1024/2048/4096
python scripts/m17_prefix_cache.py --shard e3-conc --conc-mode chunked       # 另跑 nochunk
python scripts/m17_prefix_cache.py --shard summarize            # 汇总，不建引擎、不需 GPU

# 核心图（无需 GPU，仅由已落盘数据生成）
python scripts/m16_figures.py    # -> figures/fig1..4
python scripts/m17_figures.py    # -> figures/fig5..6
```

关键运行约定（本机 WSL2/sm_120）：`VLLM_USE_FLASHINFER_SAMPLER=0`（无 nvcc，FlashInfer sampler JIT 会失败）、`VLLM_ENABLE_V1_MULTIPROCESSING=0`、`seed=0`、`temperature=0`。

---

## 工作区结构与约束

```text
/home/ciaran/projects/
├── vllm-src/           # vLLM 源码（学习用真源）。ee0da84=v0.24.0，含中文学习注释
├── vllm-m13-upstream/  # M13 上游 PR 专用：干净 clone + 独立 .venv，跟最新 main（无学习注释）
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
- **上游贡献与学习树严格隔离**：`vllm-src` 为便于学习加了中文注释、且 pin 在 v0.24.0（脏树、shallow），**不可用于提 PR**；M13 因此另建 `vllm-m13-upstream`（干净 clone + `uv` 独立 venv + 跟最新 main），保证提交的 diff 只含目标文件。
- 不修改 `../vllm_playground/.venv`（只读基线）；不对 `../vllm-src` 执行破坏性 git 操作；环境记录不写任何密钥/Token。
- 本目录**是 git 仓库**，已推送至 `github.com/CiaranCw/vLLM-Inference-Lab`（提交者仅 CiaranCw）。

---

## §7 M13 首次上游 PR —— 已提交

> **[vllm-project/vllm#51017](https://github.com/vllm-project/vllm/pull/51017)** · `[Core] Add unit tests for BlockTable slot mapping` · **Ready for review**（2026-08-04 提交）
> 目标是**完整经历一次上游贡献流程**，非必须合入——截至本文更新时**尚未合入**，等待 maintainer 审查。

**贡献内容**：为 vLLM `vllm/v1/worker/block_table.py` 的 `BlockTable.compute_slot_mapping`（含其 Triton kernel）补**单元测试**——该路径此前在 `tests/` 中**零直接覆盖**，而它把逻辑 token 位置映射到物理 KV slot，错一位会**静默写坏 KV cache**（不报错），且含 hybrid 块 / DCP interleave / CUDA-graph padding / `SlotMappingMode.NONE` 等易错分支。

| 项 | 事实 |
|---|---|
| PR | [#51017](https://github.com/vllm-project/vllm/pull/51017)，base `24c939c`，head `4dd4fa7` |
| 规模 | **1 commit / 1 file / +384 / −0**，**零产品代码改动**（test-only） |
| 落点 | `tests/v1/worker/test_block_table_slot_mapping.py` |
| 测试 | **18 passed**（3 类手算 golden + 边界/ragged/prefix + 跨 1024 Triton tile padding + DCP 全 rank + hybrid 32/16 & 64/16 + `SlotMappingMode.NONE`） |
| CI | **DCO pass** ✅、mergify Summary pass、Meta Internal-Only pass；`pre-commit` job 在上游 runner 队列中（本地同 hook 集 13 Passed / 0 Failed） |
| 合规 | `git commit -s`（`Signed-off-by: CiaranCw`）+ `Assisted-by: Claude Code`（AGENTS.md 要求披露 AI 辅助）；提交前人工逐行审查、手算复核 golden、本人复跑测试 |

**工程判断（这次贡献里最有价值的部分）**：
1. **不盲信旧笔记**：候选研究基于 v0.24.0，动手前对**最新 main** 重新核实，发现 3 处语义已漂移——slot mapping **已改为只按 DCP 分片**（PCP 不再参与）、新增 **hybrid 块三分语义**（`kv_cache_block_size`/`kernel_block_size`/`blocks_per_kv_block`）、新增 **`SlotMappingMode`**。按旧假设写的测试根本跑不起来。
2. **Oracle 必须独立**：初版参照实现复用了产品的 `map_to_kernel_blocks`——用**变异测试**证明了风险真实：给该 helper 注入 `+1` 后，复用版本的 hybrid 用例**仍然通过**（漏检）；改为测试内自建展开后，3 个 hybrid 用例**全部失败**（正确捕获）。
3. **三方独立锚定**：手算 golden 与 numpy oracle 相互独立地钉住 kernel 输出，避免「参照实现与被测代码同错」。
4. **主动规避冲突**：发现 open PR #39591 会创建同名文件（但它测 row-tail 清零、完全不碰 slot mapping），故改用 `test_block_table_slot_mapping.py` 并在 PR 里说明、表示愿合并。
5. **前车教训**：stale PR #30683 曾提出同方向测试，**死于 pre-commit 未过后无人跟进**（非评审否决）——所以本次把 pre-commit 全绿当作硬门槛。

**完整过程留痕**：`notes/m13/`（[STATUS.md](notes/m13/STATUS.md) 断点快照 · [pr_slot_mapping_test.md](notes/m13/pr_slot_mapping_test.md) 研究与语义漂移 · `review/` 逐行审查+手算 golden+清单 · `validation/` 测试/pre-commit/变异回归日志 · `pr_draft/` 最终测试与 PR 正文）。
**待回填**：`[ ]` maintainer review 往返与迭代　`[ ]` 是否最终合入。

---

## §8 M17 APC & Chunked Prefill

> 第二阶段阶段 E 的首个模块。**T + O + E 已完成**（`notes/m17/`）。

**方法**：用 `LLMEngine` + `ScheduleSpy` 包裹 `scheduler.schedule` 抓 `SchedulerOutput`，逐 step 读**每请求**的 `num_scheduled_tokens`（只读 `total_` 无法分离 prefill 与 decode）。每配置 **5 warmup + 15 measure**，汇总只用 measure；统计口径为 median difference 的 **Bootstrap 95% CI**（10000 次，固定 seed=0，仅 stdlib）。

| 实验 | 构造 | 主结论 |
|---|---|---|
| **E1** APC | 共享前缀 P∈{256,512,1024,2048} + 独立 16-token 后缀；`off` / `cold`(未命中) / `hot`(命中) 三侧 | **P=1024/2048 收益显著**（−32.4% / −58.5%，CI 排除 0）；**P=256/512 的 CI 含 0** |
| **E3-single** 分块成本 | prompt=2048、只变 `max_num_batched_tokens` | Σchunks **恒 2048**、步数恰为 `ceil(2048/预算)`；单请求每多一步 ≈ **+3.0–3.2ms** |
| **E3-conc** ITL 抖动 ⭐ | decode 请求 D 稳定后插入 2048 长 prefill；**两侧 mbt=4096 严格单变量** | D 最坏 stall **−54%**、**median/p95 的 CI 含 0**、长 prefill 自身 TTFT **+47%** |

**两个值得单独说明的实验设计点**：
- **单变量对照不能靠"关开关"**：`enable_chunked_prefill=False` 时上游强制要求 `max_num_batched_tokens >= max_model_len`（`vllm/config/scheduler.py:268-279`），所以「off + 小预算」这个对照**无法构造**。改为两侧都 `mbt=4096`，chunked 侧单靠 `long_prefill_token_threshold=512` 强制分块（实测 4×512），两侧 D 的 ITL 样本数均为 92 → 可比。
- **一个进程只建一个引擎**：vLLM 引擎的 KV 显存 `del` + `empty_cache()` 也归还不掉（M6 已证），故脚本按 `--shard` 分片，每配置独立进程。

**机制（以原始逐 step 记录为准）**：nochunk 时长 prefill 那一步是 `d=1, p=2048, total=2049`——**D 仍被同批调度了 1 个 decode token**，只是该 token 的返回被这个长联合 step 整体推迟；chunked 侧每步 `d=1, p=512, total=513`，D 同样每步产出但等待更短。→ chunked prefill 改善的是「要等多长的一个 step」，不是「是否被调度」。

**未完成项（如实标注）**：`[ ]` **I5-b** 输出长度定量实验——冻结设计要求扫 `max_tokens` 验证「APC 相对收益随输出变长而下降」，本轮所有请求均 `max_tokens=1`（只测 prefill），**未执行**（I5-a 理论边界确认已完成）；`[ ]` **E2** block 生命周期 `ref_cnt`/驱逐内省（可选）。
**适用边界**：单卡 / eager / 0.5B / WSL2 / in-process，绝对值限本配置；**跑间有 1–3ms 漂移**，故只对「显著/不显著与量级」下结论。**首版（2026-08-04）结论因单样本且被首次 Triton JIT 编译污染已整体撤回并重跑**，撤回明细见 [`notes/m17/results.md`](notes/m17/results.md) §6。

**留痕与数据**：[`results.md`](notes/m17/results.md)（结果 + 撤回表）· [`prefix_cache_and_chunked_prefill.md`](notes/m17/prefix_cache_and_chunked_prefill.md)（T/O 源码定位 + 冻结设计 + 不变量 I1–I5）· `outputs/m17/*.csv` + `outputs/m17/raw/*.jsonl`（含 `phase`/`success`/`metric_scope`）· `logs/m17/*`。旧数据保留未删（`raw_old_20260805/`、`raw_pre_field_fix/`）。**零 vLLM 产品源码改动。**
