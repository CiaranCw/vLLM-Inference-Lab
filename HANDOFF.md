# HANDOFF.md —— 项目交接文档（交由 Cursor 继续）

> **2026-10-06 M27 当前状态**：阶段 K 的 M27 **COMPLETE**——新实例 `ins-a8qac2ve`（与上一台碰巧同 hostname `TENCENT64.site`）完成 fresh Gate0→E4，E0–E4 COMPLETE；E5 经 Gate8 可选跳过（SKIPPED），P1 profiler NOT RUN。规模边界：运行环境是单节点 8×H20，但正式实验只覆盖 1/2/4 GPU（E0–E3 按 arm 1–2 GPU，P:D ratio 的 E4 每 arm 4 GPU），最大正式并发 4 GPU，**没有 8-GPU P-D 实验**；M27 以 E0–E4 为最终范围，不再补 8 卡实验。独立 post-hoc 审计 `M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`，文档修正已闭环（[`notes/m27/FINAL_CLOSURE_ADDENDUM.md`](notes/m27/FINAL_CLOSURE_ADDENDUM.md)）。H20 原提交 `884503dc` / `c4db4561` 已规范化重放为 CiaranCw 提交，对应关系见 [`notes/m27/H20_COMMIT_PROVENANCE.md`](notes/m27/H20_COMMIT_PROVENANCE.md)；11 个 >100MB raw 不在 Git 中，身份见 [`notes/m27/H20_RAW_ARCHIVE_MANIFEST.tsv`](notes/m27/H20_RAW_ARCHIVE_MANIFEST.tsv)。RQ1–RQ4、负结果和计费见 [`notes/m27/results.md`](notes/m27/results.md) 与 [`notes/m27/execution_record.md`](notes/m27/execution_record.md)。下文早期交接中「M27 未执行 / 下一步上 H20」都是当时的历史状态，不代表当前；M24 冻结计划不改。

> **交接时间**：2026-08-06　**交接自**：Claude Code　**交接至**：Cursor
> **仓库**：`/home/ciaran/projects/vllm-inference-lab` → `github.com/CiaranCw/vLLM-Inference-Lab`
> **当前 HEAD**：以 `git log -1` 为准（本文件不写死 commit，避免过时）
>
> **读这份文档的顺序**：
> 1. **先读 [`AGENTS.md`](AGENTS.md)** —— 不可违反的硬性规则（诚实性红线、Git 署名、GPU 安全、必需环境变量）。**这些规则此前只存在于上一个助手的私有记忆里，本次交接已全部外化到该文件，务必遵守。**
> 2. 再读本文件 §1–§4（当前进度与环境）
> 3. 动手前读 §5（下一步该做什么）+ §6（已知坑）

---

## 1. 这个项目是什么

**一句话**：在**固定 vLLM commit**（`ee0da84` / v0.24.0）上，对 vLLM V1 单机推理主链路做「逐 step 结构化追踪 → 性能基线 → profiler 定位瓶颈 → 真实代码修改 → 正确性与性能回归」的长期工程项目，用于**求职**（推理系统 / ML Systems 岗）。

**核心卖点不是"读过源码"，而是**：观察真实运行、用证据验证理解、动手改并用数据说话——**包括一份诚实的无收益报告（M9）和一次真实的上游 PR（M13）**。

**因此**：项目的价值建立在**每条结论都能回溯到原始数据**之上。宁可结论少、边界清楚，也不要漂亮但不可追溯的数字。详见 `AGENTS.md` §1。

**总览入口**：[`README.md`](README.md)（招聘者视角，含 6 张核心图）· [`ROADMAP.md`](ROADMAP.md)（M0–M26 全部里程碑）· [`notes/evidence_index.md`](notes/evidence_index.md)（结论 → 证据文件的索引）。

---

## 2. 进度总表

### 第一阶段（M0–M16）：已收口 ✅
核心验收 **9/9**（原 M11 双卡 TP 属硬件门控，已由第二阶段 M22 在 4×H20 闭合）。

| 模块 | 主题 | 状态 | 一句话成果 |
|---|---|:--:|---|
| M0 | 环境/trace 基础设施 | ✅ | 固定版本 + 只读 trace 钩子（默认关，零副作用） |
| M1–M5 | 主链路 / CB / KV / Attention / 执行模式 | ✅ | 串成一次运行的闭环证据 |
| M6 | 引擎 benchmark ★ | ✅ | decode 近平、吞吐近线性、compiled 快 2.5–2.85×、KV≈12KiB/token |
| M7 | profiler ★ | ✅ | **decode 的 ~7ms 地板 = CPU kernel 提交开销**（GPU 仅忙 23%） |
| M8 | Triton kernel ★ | ✅ | 自写 slot_mapping kernel，三路交叉 22/22 |
| M9 | profiler 驱动改动 ★ | ✅ | 合并 decode H2D：机制成立但 **E2E 无收益**，如实保留 |
| M10 | 量化对比 ★ | ✅ | FP8 显存 −36%，但 **decode 反而 +36%**（诚实反常） |
| M11 | 双卡 TP | ⬜ | **硬件门控**（本机单卡）→ 并入 M22 |
| M12 | 在线 serving / SLO | ✅ | 最大稳定 QPS≈40；**过载是尾延迟劣化，非吞吐崩溃** |
| M13 | 首次上游 PR ★ | 🟡 | **PR #51017 已提交，等待 maintainer**（详见 §3） |
| M16 | 项目整理 + 面试表达 | ✅ | README/图/三档表达 |

### 第二阶段（M17–M26）：M17 / M18 / M19 / M20 / M21 / M22(TP) / M23(L1) / M24(DCP) / M25(T) / M26 已收口
| 模块 | 主题 | 状态 | 备注 |
|---|---|:--:|---|
| **M17** | Prefix Caching & Chunked Prefill | ✅ | **主线已收口**；三项可选扩展未执行、不阻塞后续（见 §4） |
| **M18** | 量化路径深化（W4A16/AWQ/GPTQ） | ✅ | **T+E+O COMPLETE**：省显存显著但**本机 28 个对照点 0 个变快**；见 `notes/m18/results.md` |
| **M19** | Speculative Decoding | ✅ | **N-gram T+E+O CLOSED**（见 §4.2）；draft/EAGLE/MTP/Medusa/`k` sweep = Optional |
| M20 | 服务化 + Triton | ✅ | **T+A+B+C COMPLETE**（same-runtime QPS 20/32/40；见 `notes/m20/results.md`） |
| M21 | 监控 + 容量规划（Prom/Grafana） | ✅ | **T+E+O COMPLETE**（栈+容量+瓶颈签名+故障演练；2026-08-12 已做 evidence audit，见 `notes/m21/results.md` §10） |
| **M22** | TP/PP/DP/EP | ✅/⬜ | **TP T+E+O+H COMPLETE**（4×H20，`notes/m22/results.md`）；**PP/DP/EP 未做** |
| M23 | TensorRT-LLM 跨引擎 | ✅ | **L1 FROZEN**：2×2 + 跨引擎 consistency（2/6、1/6，numerical divergence）；不继续 L2/L3；见 `notes/m23/results.md` |
| M24 | CP / Ring Attention / P-D 分离 | ✅/⬜ | **T + DCP E+O+H COMPLETE**（4×H20，`notes/m24/results.md`）；**Ring / P-D 未做** |
| M25 | 压缩（蒸馏/剪枝/低秩） | ✅ | **T COMPLETE**（`notes/m25/compression.md`）；本轮明确不做 E |
| M26 | 第二阶段收口 | ✅ | **完成（含明示占位）**：四项目 A–D README 化；**D 核心已随 M23 L1 闭合**；见 `notes/m26/portfolio.md` |
| **M27** | 阶段 K · P-D 分离 + NIXL KV 传输（M24 P-D 部分转入这里） | ✅ | **T+E+O+H COMPLETE**（8×H20 单节点环境，正式实验 E0–E4 覆盖 1/2/4 GPU、最大 4 GPU；E5 未执行、不在范围内，P1 未做；post-hoc 审计 PASS_WITH_DOCUMENTATION_CORRECTION）；见 `notes/m27/README.md` |

> 每个模块的完整定义（要做什么 / 记什么指标 / 产出什么文件 / 硬件门控）都在 [`ROADMAP.md`](ROADMAP.md) 对应小节，**动手前务必读该小节**，不要凭本表的一句话概括就开工。

---

## 3. M13 上游 PR 现状（🟡 阻塞中，非你能解决）

**PR**：[vllm-project/vllm#51017](https://github.com/vllm-project/vllm/pull/51017) —— `[Core] Add unit tests for BlockTable slot mapping`
**状态**（2026-08-06 核实）：`OPEN` / 非 draft / `MERGEABLE` / 无 label / 无人类 review。

| CI job | 结论 |
|---|---|
| DCO | ✅ pass |
| Meta Internal-Only / Summary | ✅ pass |
| **pre-run-check** | ❌ **fail** |
| pre-commit / approved | ⏭ skipped |

**`pre-run-check` 失败的原因不是代码问题**，而是上游对新贡献者的门禁：
> `PR must have the 'verified', 'ready', or 'ready-run-all-tests' label to run pre-commit, or the author must have at least 4 merged PRs (found 0)`

即：**需要有写权限的 maintainer 打 label 或评论 `/ci run`**。本地同一套 hook 已验证 **13 Passed / 0 Failed**，测试 **18 passed**。

**⚠ 你需要做的**：**什么都不做，也不要试图"修复"这个 CI**。
- **不得自行加 label**（没权限，且属于绕过门禁）。
- **不得 force-push / rebase / 改动这个 PR**，除非所有者明确要求。
- 若 maintainer 留下 review → **先把 review 内容报告给所有者**，由所有者决定如何回应。

**恢复上下文**：`notes/m13/pr_slot_mapping_test.md`（研究与上游语义漂移）· `notes/m13/review/`（逐行审查 / 手算 golden / 复现命令）· `notes/m13/validation/`（pytest / pre-commit / 变异回归）。
**上游工作区**：`/home/ciaran/projects/vllm-m13-upstream`（独立 clone + 独立 venv，跟最新 main，**与带学习注释的 `vllm-src` 严格隔离**）。本地分支 `m13/block-table-slot-mapping-tests` HEAD = `4dd4fa7`。

---

## 4. M17 刚完成的内容（最近一轮工作，建议先读懂再往下走）

**主题**：Automatic Prefix Caching（APC）与 Chunked Prefill 的定量对照 + 源码定位。
**产出**：`notes/m17/results.md`（结果 + 撤回表）、`notes/m17/prefix_cache_and_chunked_prefill.md`（T/O 源码定位 + 冻结设计 + 不变量 I1–I5）、`outputs/m17/*.csv` + `raw/*.jsonl`、`logs/m17/*`、`scripts/m17_prefix_cache.py`、`scripts/m17_figures.py`、`figures/fig5,fig6`。

**三个实验与结论**（全部以 Bootstrap 95% CI 判定）：

| 实验 | 结论 |
|---|---|
| **E1** APC（P∈{256,512,1024,2048}） | **P=1024 −32.4% / P=2048 −58.5%**（CI 排除 0）；**P=256/512 的 CI 含 0** → 未观察到超出噪声的稳定收益 |
| **E3-single** 分块成本 | Σchunks 恒 2048、步数 = `ceil(2048/预算)`；单请求每多一个 prefill step ≈ **+3.0–3.2ms** |
| **E3-conc** ITL 抖动 | 并发 decode 的**最坏 stall 22.87→10.51ms（−54%，显著）**，但 **median/p95 的 CI 含 0**；代价长 prefill 自身 TTFT **+47%** |

**关键洞察（可复用到后续模块）**：APC 的收益必须等 prompt 长到**穿透 M7 那条 ~7ms 固定开销地板**才显现——这解释了 M6"省计算≠省延迟"其实是**地板效应**而非普遍规律。

**⚠ M17 是一次教训密集的模块，请务必读 `notes/m17/results.md` §6「已撤回的结论」**。两轮审计共撤回 12 项表述，最值得记住的四条：

| 撤回的错误 | 根因 | 对你的启示 |
|---|---|---|
| E3「+5%/+15.6%/≈3×7ms」 | 每配置**仅 1 样本、零 warmup**，被首次 Triton JIT 污染（首样本 128–145ms vs 稳态 23–35ms，慢 **4–5×**） | **性能实验必须 warmup + 重复测量**，这台机器的 JIT 足以让单样本结论整体反向 |
| `chunked = scheduled > actual` | **判据方向写反**，把 **60/60 prefix 命中样本误标为"跨多 chunk"**，而且不报错 | **布尔判据必须有独立的第二证据源**（这里补了 `prefill_step_count`） |
| 「D 在长 prefill 那步拿不到输出」 | 与自己记录的原始 `steps_detail` 矛盾（实为 `d=1, p=2048, total=2049`，decode **仍被同批调度**） | **写机制解释前回去核对原始逐 step 记录** |
| 「4 档 cold 全部 ≤ off、cold=7.42」 | 事实错误（实为 8.0051 > 7.8465），且 7.42 不在任何数据里；暴露"自动核对只覆盖表格、没覆盖正文叙述句" | **核对数字时要覆盖正文，不只是表格** |

**M17 主线已收口（2026-08-06）**。核心实验（E1 / E3-single / E3-concurrency）完成并核验；
以下三项为**可选扩展，不阻塞后续模块**（manifest 标 `status: optional`，`reproduce.py --run` **不会**自动执行；
`check_invariants --module m17` 显示为 PENDING —— 既不计通过也不计失败）：
- `[ ]` **E3 no-chunk 生成正确性**：脚本已实现（`--shard e3-correctness`），**未执行**。
- `[ ]` **I5-b 输出长度定量实验**：脚本已实现（`--shard e1-output-len-*`），**未执行**。
- `[ ]` **E2 block 生命周期内省**（ref_cnt / 驱逐）：**脚本都未实现**，不为完成度强行做。

**⚠ 必须随 M17 结论一起呈现的限制**：当前模型**不正式支持关闭 chunked prefill**
（`vllm/engine/arg_utils.py:2476-2480` 明确告警 "may cause the engine to crash or produce
incorrect outputs"），因此 E3 的现有结果**主要支持调度形态与延迟观察，不外推为普适的生成正确性保证**。

- 已知**跑间波动**：同日两轮 P=1024/2048 的 median 差有 1–3ms 漂移（方向与显著性一致）；cold−off 的符号甚至在两轮间反转 → **亚毫秒级差异不作结论**。

---

## 4.2 M19 N-gram Speculative Decoding —— T+E+O CLOSED

**产出**：`notes/m19/{speculative_decoding.md,results.md}`、`scripts/m19_spec_decode.py`、
`scripts/m19_figures.py`、`outputs/m19/*.csv` + `raw/*.jsonl`、`logs/m19/*`、
`figures/{fig7_m19_acceptance,fig8_m19_speedup}.png`。**零 vLLM 产品源码改动。**
**2026-08-08**：语义收口（不改数据）——统一 Coverage+Acceptance+Round Cost 模型；
修正 overstated 表述（acceptance/length/`engine.step`/TPOT/`low_repeat`/correctness 等）。

**网格**：2×2×3，180 measure；ngram `{method:ngram, prompt_lookup_min/max:3/5, num_speculative_tokens:3}`。

| 结论 | 数据 / 证据强度 |
|---|---|
| Acceptance 随 workload/config 变化（**非**「完全由 workload」） | high（高初始 prompt 重叠）**1.000**；low（低初始重叠，**非**真正低输出重复）**0.75–0.85**；batch↔prompt confounding |
| 机制：串行 execution rounds 下降 | `engine.step` 强 proxy：**128→33 / 56–62**（非严格纯 Decode Target Forward） |
| 性能 | 完成延迟 −**72~75.5%** / −**54.8~57.4%**；**amortized TPOT**（≠ ITL）；吞吐大幅提升；12 组 CI 排除 0；Acceptance **非充分** |
| KV capacity | planning 口径约 **−1.9%**（≠ allocator peak） |
| 正确性 | 111/126 exact；batch-shape **可**致 divergence；未发现明确 Spec bug（**非**严格逐例排除） |

**隐藏变量已写死**：两侧 `async_scheduling=False`、`disable_log_stats=False`。

**Optional / Future（不阻塞）**：`k` sweep；true low-coverage/acceptance；profiler breakdown；
larger batch/model knee；compiled×Spec；small draft model；EAGLE/MTP/Medusa。

---

## 5. 下一步做什么（建议先和所有者确认）

> **2026-10-03 更新（历史：当时的下一步，已由 2026-10-05 的 H20 执行完成，见文首）**
> - **M27（阶段 K · P-D 分离 + NIXL）= T/O + DESIGN READY，E/H NOT EXECUTED**。下一步：**H20 M27 Gate 0 / E0**，按 [`notes/m27/M27_H20_RUNBOOK.md`](notes/m27/M27_H20_RUNBOOK.md) 执行（先 preflight，任一 STOP 条件就停下询问）。
> - 上机资源分三部分（见 [`notes/m27/offline_package.md`](notes/m27/offline_package.md) §3）：code = pre-H20 anchor commit（未 push）；
>   wheelhouse = `~/m27_offline/wheelhouse/`（NIXL，sha256 已核对）；model = 7B 身份由 `notes/m27/model_artifact/qwen2.5-7b-instruct.sha256` 冻结，
>   本地恢复副本在 `/mnt/f/wsl_backup/m22_offline/models/Qwen2.5-7B-Instruct`（[`model_artifact_manifest.md`](notes/m27/model_artifact_manifest.md)）。
>   H20 上仍需确认 `/root/.venv` 与 `/root/m22_offline` 是否还在。
> - 未验证的风险：UCX 在 driver 535 + cuda-compat 13 下能否走同机 GPU 直连；H20 实际可见 GPU 数与 NV 拓扑（M22 只看到 4 卡）。
>
> **2026-08-17 更新（历史）**
> - **命名统一已落地**：M1–M3 由 `dayN` 改为 `mN`（`notes/` / `outputs/` / `logs/` / `scripts/`）。原始 `.log` / `.jsonl` 内容未改；`backups/dayN/` 是历史备份目录名，未改。
> - **M24 DCP E+O+H COMPLETE**：4×H20 / 1.5B / TP=4；KV tokens 恰好 2.0×，六档 decode step DCP=2 更慢（CI 排除 0）；PCP 负例在 `world_size=8` 被拒。**Ring Attention / P-D 仍 ⬜**。见 `notes/m24/results.md`。
> - **M23 L1 已冻结**：不扩 L2/L3/kernel profiling。
> - **M25 T COMPLETE**：不做 E。
> - M13 PR 仍等 maintainer label。
> - 下一优先：M22 的 PP/DP/EP（仍需多卡窗口）；或 M13 review。
>
> 以下保留为背景（交接早期选项；M18/M19/M20/M21/M22/M23 均已收口）。

### 选项 A · M18 量化路径深化（承接 M10）—— ROADMAP 首选
- **要补的**：M10 已做 BF16 / FP8 W8A8 / FP8 KV；**M18 要补 W4A16（AWQ / GPTQ）**，并为 M23 的跨引擎对照做预备。
- **必须先解决的前置问题**：**本地没有 AWQ/GPTQ 量化模型**（HF 缓存里只有 `Qwen2.5-0.5B-Instruct`、`Qwen2.5-1.5B-Instruct`、`opt-125m`）。而 `AGENTS.md` §3 规定**不得自动下载模型** → **必须先问所有者**是否授权下载（例如 `Qwen/Qwen2.5-1.5B-Instruct-AWQ` 一类）。磁盘尚有 613G 可用，不是瓶颈。
- **强约束（写在 ROADMAP 里）**：**不得为了补全表格而伪造不支持的 FP8/FP4 路径**。sm_120 上哪条路径真的跑不起来，就如实记录为"不支持"并给出报错。
- **可复用**：`scripts/m10_quant.py` 的配置矩阵写法（COMMON 固定 + 每配置只改 3 个变量 + 配置落盘到 `outputs/m10/configs/*.json`）是本项目的范式，直接照搬。
- **注意 M10 的归因方法**：FP8 KV 必须用 `TRITON_ATTN`（FLASH_ATTN 不支持 fp8 KV，FLASHINFER 在 sm_120 JIT 失败），所以 M10 专门建了**同后端基线 C0T**，用 `C2−C0T` 归因而不是 `C2−C0`。做 M18 时同样要想清楚"基线是否同后端"。

### 选项 B · M19 Speculative Decoding —— 单卡最友好
- **优点**：**N-gram proposer 不需要额外模型**，纯单卡、无需下载任何东西，可以立刻开工；显存允许再上小 draft model。
- **顺序（ROADMAP 明确要求，别跳）**：普通 decode baseline → N-gram → 小 draft model →（显存允许）EAGLE/MTP。**不要一上来啃 Medusa 源码。**
- **要记的指标**：proposed / accepted 数、acceptance rate、target forward 次数、TPOT、吞吐、显存，**以及不同 batch 下的收益或退化**。
- **理论重点**：为什么中低 QPS / memory-bound decode 更受益，batch 增大后 verification 成本为何抵消收益。→ 这个正好能接上 **M7 的"decode 是 CPU 提交受限、GPU 只忙 23%"** 这条既有证据，是很自然的延续。
- **风险**：`enforce_eager` 下 spec decode 的收益可能被 M7 那条固定开销地板掩盖 → **预期要诚实处理"可能无收益"的结果**（M9 已有先例可参照）。

### 选项 C · 补完 M17 的 I5-b —— 最小、最快、收尾性质
- 只需扫 `max_tokens ∈ {1, 8, 32, 128}`，验证「固定前缀下 APC 的 E2E 相对收益随输出变长而下降」。
- **脚本骨架已就绪**：`scripts/m17_prefix_cache.py` 已有 `--shard` 机制、warmup/measure 分离、Bootstrap CI helper，加一个 shard 即可。
- 完成后把 `notes/m17/prefix_cache_and_chunked_prefill.md` §5 的 `[ ] I5-b` 勾上，并更新 results.md / ROADMAP / evidence_index。
- 适合作为**熟悉本项目工作方式的热身**（约需 1 次 GPU 运行）。

> **建议**：若想先熟悉项目 → 选 C；若想推进主线且不想等授权 → 选 B；若所有者愿意授权下载模型 → 选 A。

---

## 6. 环境与已知坑（实测，别踩第二遍）

### 6.1 三个目录的关系
```
/home/ciaran/projects/
├── vllm-src/            # vLLM 源码唯一真源。detached HEAD @ ee0da84 (v0.24.0)
│                        # ⚠ 工作树有约 1583 行未提交的中文学习注释（21 个已跟踪文件被改 + 1 个新增
│                        #   untracked 文件 vllm/_lab_trace.py = 共 22 项）——所有者的资产，禁止破坏性 git
├── vllm-m13-upstream/   # M13 上游 PR 专用：干净 clone + 独立 venv，跟最新 main（无学习注释）
├── vllm_playground/     # 干净基线 venv（vllm 作为普通 site-packages 安装，非 editable）
│                        # ⚠ 只读！不得 install/modify —— 它的作用是对照"我改了源码后原版是否还正常"
└── vllm-inference-lab/  # 本仓库。.venv 以 editable 方式装了 ../vllm-src
```
- **`vllm-inference-lab/.venv` 的 vllm 指向 `../vllm-src`**（editable）→ 改 `vllm-src` 的 Python 代码，**重启进程即生效**；改 `csrc/`（CUDA/C++）**不生效**（`VLLM_USE_PRECOMPILED=1`，需真重编译）。
- 自查命令：`python -c "import os,vllm;print(os.path.realpath(vllm.__file__))"` 应输出 `/home/ciaran/projects/vllm-src/vllm/__init__.py`。
- **判断"有没有动产品源码"的基线**：`git -C ../vllm-src status --short | wc -l` 应为 **22**（21 个被改的已跟踪文件 + 1 个 untracked 的 `vllm/_lab_trace.py`）。变多了就是你改了产品代码。

### 6.2 版本事实（2026-08-06 实测）
| 项 | 值 |
|---|---|
| GPU | RTX 5080（Blackwell **sm_120**），16303 MiB，driver 610.62 |
| Python / torch / triton / ruff | 3.12.13 / 2.11.0+cu130 / 3.6.0 / 0.16.0 |
| vllm（lab venv） | `0.24.1.dev0+gee0da84ab`（editable → vllm-src） |
| 平台 | WSL2（`Linux 6.18.33.2-microsoft-standard-WSL2`） |
| 基线模型 | `Qwen/Qwen2.5-0.5B-Instruct` |
| 磁盘 | 613G 可用；HF 缓存 5.6G；本仓库 8.8G |

### 6.3 必踩坑清单

1. **`VLLM_USE_FLASHINFER_SAMPLER=0` 是必需的**，否则 `profile_run` 的 dummy sampler 会 JIT flashinfer 并崩。根因：flashinfer 对 sm_120 的 `TARGET_CUDA_ARCHS` 返回空 → 报 "requires sm75 or higher"（**是 flashinfer 的探测缺口，不是 vLLM bug**，已在 `contribution_candidates.md` 排除区说明）。FlashAttention 后端不受影响。
2. **首次冷跑极慢**（Triton JIT，例如 `_compute_slot_mapping_kernel`，M17 实测首样本 128–145ms vs 稳态 23–35ms）→ **必须 warmup，且 warmup 与 measure 分开统计**。
3. **一个进程只能建一个引擎**：`del engine` + `torch.cuda.empty_cache()` **归还不掉 KV 显存**（M6 实证）。多配置对照 → 每配置独立进程（照抄 `m17_prefix_cache.py --shard`）。
4. **vLLM 会给 `add_request` 的 id 追加后缀**（`"p"` → `"p-8ef2061f"`）→ 用 `num_scheduled_tokens.get(rid)` 精确查找会**静默返回 0**，导致所有 per-request 计数失效。用前缀容错匹配（见 `m17_prefix_cache.py` 的 `_same_req` / `_sched_for`）。
5. **`num_scheduled_tokens` 是 per-request dict**；只读 `total_num_scheduled_tokens` **无法分离 prefill 与 decode**（M17 首版因此把 decode step 误计为 prompt chunk）。
6. **`max_num_partial_prefills` 在本版本不可调**：非默认值直接 `raise NotImplementedError("Concurrent Partial Prefill")`（`vllm/engine/arg_utils.py:2350`）。要强制长 prefill 分块，用 `long_prefill_token_threshold`（已实测有效）。
7. **`enable_chunked_prefill=False` 时上游强制 `max_num_batched_tokens >= max_model_len`**（`vllm/config/scheduler.py:268-279`）→ "关分块 + 小预算"这个对照**构造不出来**，别浪费时间。
8. **fp8 KV 必须配 `TRITON_ATTN`**（FLASH_ATTN 不支持，FLASHINFER 在 sm_120 JIT 失败）；且旧的 `VLLM_ATTENTION_BACKEND` 环境变量在本版本**已失效**，要通过 `LLM(attention_backend=...)` 传。
9. **本机 `tests/v1/worker/test_gpu_block_table.py` 有 3 个基线失败**（`RuntimeError: UVA is not available`，WSL2 限制）——**与本项目改动无关**，在 pristine main 上同样失败，不要试图"修好"。
10. **trace 钩子**：`vllm-src/vllm/_lab_trace.py`，开关 `VLLM_LAB_TRACE=1`（默认关）、`VLLM_LAB_TRACE_COMPONENTS=scheduler|kv|exec`、`VLLM_LAB_TRACE_JSONL=<path>`。被动钩子标 `# VLLM_INFERENCE_LAB_TRACE`；**会改变行为**的实验钩子标 `# VLLM_INFERENCE_LAB_EXPERIMENT`（如 `VLLM_LAB_BLOCK=1`），后者默认关。

### 6.4 标准运行姿势
```bash
cd /home/ciaran/projects/vllm-inference-lab
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0

nvidia-smi                       # ← 先确认 GPU 空闲！有别人的进程就停下来问所有者
python scripts/smoke_import.py   # 环境自检（不加载模型）

# 出图（不需 GPU / 不 import vllm，纯离线读 CSV）
python scripts/m16_figures.py    # -> figures/fig1..4
python scripts/m17_figures.py    # -> figures/fig5..6
```
各模块完整复现命令见 `README.md` §复现 与各模块笔记末尾。

---

## 7. 你接手后的第一批动作（建议）

1. **读 `AGENTS.md`**（5 分钟，硬规则）。
2. **读 `README.md`**（10 分钟，全局 + 6 张图）。
3. **读 `notes/m17/results.md`**（尤其 §6 撤回表与 §7 局限）—— 这是本项目"如何做诚实实验"的最佳范例，也是最近一轮工作的上下文。
4. **跑一次 `python scripts/m17_figures.py`** 验证环境可用（不需 GPU，30 秒内出图）。
5. **和所有者确认下一个模块**（§5 的 A/B/C）—— 特别是选 A 需要先获得**下载量化模型的授权**。
6. 开工前读 `ROADMAP.md` 里该模块的完整小节，并**先冻结实验设计与不变量再跑**（本项目的既有做法：设计先写进 `notes/mN/`，跑完逐条核对）。

---

## 8. 交接时的状态核实（2026-08-06 的历史快照，数字不再更新）

> 下表是交接当天的实测状态，**刻意保留原值不随后续进展刷新**。当前状态请自行核对
> （`git log -1` / `git status` / `ls figures/`）。

| 项 | 状态 |
|---|---|
| 本仓库 HEAD | `05e8e72`，已 push 到 `origin/main`，**工作树 clean** |
| 提交署名 | 全部 CiaranCw 单人，**无任何 AI trailer** |
| `../vllm-src` | HEAD `ee0da84`（v0.24.0）；脏文件 **22 = 基线**（本轮零产品源码改动） |
| M17 数据 | 已重跑落盘；旧数据全部保留（`outputs/m17/raw_old_20260805/`、`raw_pre_field_fix/`、`backups/`） |
| M17 文档数字 | 已对 CSV/JSONL 逐项核对（19/19 通过），含正文叙述句 |
| PR #51017 | OPEN / MERGEABLE / 无 label / `pre-run-check` 因**新贡献者门禁**失败（非代码问题）/ 无人类 review |
| GPU | 4132 MiB / 16303 MiB，3% 利用（交接时基本空闲，但**每次跑前仍需自查**） |
| figures | 交接时 6 张，均可由 `m16_figures.py` + `m17_figures.py` 离线重生成（现已 16 张，见 README 核心图） |

---

## 9. 一句话总结给接手方

**这个项目已经证明了"能读→能串→能证→能改→能测容量→能贡献上游"，也已经证明了"能撤回自己的错误结论"。你接手后最重要的不是快速产出新模块，而是维持同一条标准：每个数字可回溯、每个未完成项如实标注、每个结论带适用边界。** 具体规则见 `AGENTS.md`，最好的范例见 `notes/m17/`。
