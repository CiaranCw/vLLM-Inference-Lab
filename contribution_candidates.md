# contribution_candidates.md —— vLLM 上游贡献候选（持续维护）

> **用途**：ROADMAP「持续维护任务」的一部分。从现在起持续记录**可回馈上游的候选**——测试缺口、文档/行为不一致、明确 bug、benchmark/可观测性改进、有 profiler 证据的性能修复点。M13（★核心验收）只负责把其中**一个正式提交**（复现→定位→改→测试→benchmark→提交→回应 review→改 PR），而不是到 M13 才开始找。
>
> **本文件不追求数量，只收录有切实作用或技术价值、且对本机（单卡 RTX 5080 / sm_120 / WSL2）现实可行的候选。** 明确排除无意义、不切实际、根本不可能完成、或已被上游解决的项——被排除项连同「排除理由」一并记录，避免以后重复评估。

---

## 0. 关键前提：版本漂移与「跑前必做」核查（务必先读）

- **本仓库 pinned 在 `ee0da84` / v0.24.0**（`vllm-src` HEAD 实测正位于该 tag，0 commit 偏移）。**下面所有候选都是对 v0.24.0 源码直接核对后写下的**（不是仅凭实验期的运行时观测——实验期观测的是当时行为，代码可能已变）。
- **但上游贡献提交到 `main`，main 一直在动。** 因此每条候选都附带 **kill 标准**：M13 选定某条后，**动手前必须**：
  1. `git -C ../vllm-src fetch && git log` 或直接看 GitHub，确认该问题在**当前 main** 仍成立；
  2. GitHub 搜 issues / PRs（open + 近期 closed），确认**没有已在处理或已合入**的重复工作；
  3. 若已被解决或有人在做 → 标记该候选为「已失效/被抢先」，转下一条。
- **实验期的若干「sm_120 结论」在 v0.24.0 已过时**（见 §4 排除区），这正是设本节的原因。

---

## 1. 候选总览（按「作为第一次 PR 的适配度」排序）

| # | 候选 | 类别 | 来源模块 | 单卡 sm_120 可行 | 工作量 | 上游契合 | 推荐度 |
|---|---|---|---|:---:|:---:|:---:|:---:|
| **C1** | `_compute_slot_mapping_kernel` 通用单元测试（含 CP interleave / padding 哨兵 / 跨块 / ragged） | 测试缺口 | M8/M4 | ✅ | 中 | 高 | ★★★★★ **首选** |
| **C2** | `calculate_kv_scales` 弃用信息陈旧（写「v0.19 移除」但现为 v0.24.0） | 文档/信息一致性 bug | M10 | ✅ | 极小 | 中高 | ★★★★ 稳妥备选 |
| **C3** | 安装文档补充 consumer Blackwell / RTX 50 系（sm_120） | 文档缺口 | M0/M10 | ✅ | 小 | 中 | ★★★ 低风险备选 |
| **C4** | `quantized_kvcache.md` 明确「默认 scale=1.0＝未校准、质量影响与校准建议」 | 文档缺口 | M10 | ✅ | 小 | 中 | ★★★ 备选 |
| C5 | `vllm bench serve` 增加 SLO 驱动的容量扫/knee 检测 | 可观测性/工具 | M12 | ✅ | 大(需 RFC) | 中 | ★★ 观望 |
| C6 | 执行模式运行时诊断（逐 batch/长度 FULL/PIECEWISE/NONE） | 可观测性/工具 | M5/M6 | ✅ | 大(需 RFC) | 中低 | ★ 观望 |

> **强烈建议 M13 选 C1**：它同时满足「有真实技术价值（补真实测试缺口）」「与已证能力精确对口（M8 已建 22 例三路交叉 oracle）」「单卡 WSL2 完全可跑」「低风险、易被接受（纯加测试、不改行为）」四条。C2/C3/C4 作为「若 C1 被抢先」的稳妥备选。

---

## 2. Tier A — 推荐作为第一次 PR

### C1 · `_compute_slot_mapping_kernel` 缺少通用单元测试　★首选
- **类别**：测试缺口（纯加测试，不改产品行为——最适合第一次 PR）。
- **来源**：M8（自写 slot_mapping 生成 kernel + 三路交叉 oracle）、M4（trace 过 slot_mapping 进 KV cache 的闭环）。
- **上游现状（v0.24.0 已核对）**：
  - 生成 kernel 在 `vllm-src/vllm/v1/worker/block_table.py:337 _compute_slot_mapping_kernel`（另有多 KV-group 版 `vllm/v1/worker/gpu/block_table.py` `_compute_slot_mappings_kernel`）。
  - **`grep -rln compute_slot_mapping tests/` → 0 命中**：整个 `tests/` 无任何直接针对该生成 kernel 的单测。
  - 现有 `tests/v1/worker/test_gpu_block_table.py` 仅 2 个测试，且都只覆盖 `apply_staged_writes`（KV-group 融合/单组写入），**不覆盖 slot 生成逻辑**：block 边界、跨块、非连续物理块、乱序 block_table、ragged 多请求（`query_start_loc`）、CUDA-graph padding 尾部哨兵、context-parallel interleave（rank 不拥有→PAD）等分支**全部无测**。
- **为什么有价值**：slot_mapping 是「逻辑 token 位置 → KV cache 物理槽位」的核心映射，错一位即静默写坏 KV、污染输出且极难定位。它有 CP interleave / padding 哨兵等**易错分支**却零单测保护——是真实且被低估的回归风险面。
- **与已证能力对口**：M8 已实现并**三路交叉验证 22/22**（自写 / PyTorch 语义 oracle / vLLM 官方 kernel），覆盖随机/单 token/块内首末/跨块/非连续/乱序/尾 mask/ragged/padding 哨兵/CP interleave(world 2·4)。**把这套 oracle 改造成 pytest 直接测上游 kernel 即可**——现成的测试设计与参照实现，几乎为此候选量身定做。
- **单卡 sm_120 / WSL2 可行性**：✅ 完全可行。单测规模小、可 CPU/单 GPU 跑、无需多卡（CP interleave 是纯 index 数学，无需真多卡通信，M8 已证）。
- **工作量**：中。主要是把 M8 oracle 适配为上游测试风格（pytest fixture、参数化、与现有 `tests/v1/worker/` 约定一致），并对齐 kernel 当前签名。
- **验证/复现计划**：
  1. 对齐 `_compute_slot_mapping_kernel` 当前签名与语义（positions / block_table / query_start_loc / block_size / PAD 哨兵 / CP 参数）。
  2. 用 PyTorch 纯语义实现作 oracle（M8 已有），参数化覆盖上述全部分支。
  3. `pytest tests/v1/worker/test_block_table_slot_mapping.py`（新文件）本地全绿。
  4. 附一句 PR 说明：补齐此前无覆盖的生成路径分支。
- **kill 标准 / 需先确认**：动手前 `grep compute_slot_mapping tests/` 在**当前 main** 仍为空、且 GitHub 无「add slot mapping test」在途 PR。若 M8 kernel 签名与上游已分叉，以上游为准重写 oracle 断言（M8 版仅作设计参照）。
- **风险**：低。纯加测试；最坏是 review 要求调整测试组织/命名。

---

## 3. Tier B — 稳妥备选（若 C1 被抢先或想更小起步）

### C2 · `calculate_kv_scales` 弃用信息陈旧（自相矛盾的版本号）
- **类别**：明确的信息/行为不一致（doc-string + 运行时 warning 双处）。
- **来源**：M10（fp8 KV，C2dyn 用到 `calculate_kv_scales`）。
- **上游现状（v0.24.0 已核对）**：`vllm-src/vllm/config/cache.py:110-114` 的 doc-string 与 `:258-266` 的 `_warn_deprecated_calculate_kv_scales` warning **都写「will be removed in v0.19」**——但当前就是 **v0.24.0**，v0.19 早已过去、该选项仍在。信息自相矛盾、会误导用户。
- **为什么有价值**：小而明确的正确性/一致性修复；面向用户的弃用信息给出「已过去的移除版本」是明显纰漏。修法二选一（由上游意图决定）：① 把目标版本改成真实的未来版本；② 若确要移除则走正式移除流程。第一次 PR 宜取 ①（最小、无行为变更）。
- **单卡可行 / 工作量**：✅ / 极小（改文案，可能加一行测试断言 warning 文本）。
- **kill 标准**：确认 main 上该文案仍为「v0.19」；GitHub 无在途「deprecate/remove calculate_kv_scales」PR（若有正式移除在途，则本候选失效）。
- **风险**：低。但价值偏小，宜作 C1 的后备。

### C3 · 安装文档未覆盖 consumer Blackwell（RTX 50 系 / sm_120）
- **类别**：文档缺口。
- **来源**：M0（环境固定，本机 sm_120）、M10（sm_120 fp8 行为）。
- **上游现状（v0.24.0 已核对）**：`docs/getting_started/installation/gpu.cuda.inc.md` 的 requirements 仅列「compute capability 7.5 or higher (e.g., T4, RTX20xx, A100, L4, H100, **B200**, etc.)」，Blackwell 说明只提「**B200, GB200** require CUDA 12.8+」。**consumer Blackwell（RTX 50 系，sm_120）既不在示例 GPU 列表、也无对应 CUDA/已知限制说明**。
- **为什么有价值**：RTX 50 系是大量个人/研究者的实际推理卡，文档只字未提会让新用户不确定是否受支持、需要哪个 CUDA。补一句示例 + sm_120 的 CUDA 版本要求即可。
- **单卡可行 / 工作量**：✅ / 小（纯 docs）。
- **kill 标准**：确认 main 的该文档仍未提 RTX 50 / consumer Blackwell；GitHub 无在途同类 docs PR。**若维护者认为「7.5+ 已隐含覆盖、无需逐卡列举」则可能被拒**——提交前可先开 issue 轻量确认，避免无效 PR。
- **风险**：中（可能被视为「噪声级」补充）。故列为备选而非首选。

### C4 · fp8 KV cache 文档未说明「默认 scale=1.0＝未校准」的质量含义
- **类别**：文档缺口（用户使用陷阱）。
- **来源**：M10（C2 未校准 scale=1.0，PPL +2.17%；C2dyn 动态 scale 改善到 +1.72%）。
- **上游现状（v0.24.0 已核对）**：代码侧**已有** info 级提示——`vllm/config/cache.py:278-285` 在量化 KV 时 log「may cause accuracy drop without a proper scaling factor」。但**面向用户的 `docs/features/quantization/quantized_kvcache.md` 是否讲清「不提供 checkpoint scale 时默认 1.0、代表未校准、典型质量影响、以及如何校准」需在动手前核对**。
- **为什么有价值**：把 M10 的定量证据（未校准 +2.17% vs 动态 +1.72%）沉淀成用户文档，帮助用户理解 fp8 KV 的质量-显存权衡与校准路径。
- **单卡可行 / 工作量**：✅ / 小（docs；数值可引用 M10 结论，但需说明限 0.5B/eager/sm_120，不外推）。
- **kill 标准**：先读 `quantized_kvcache.md` 当前内容——**若已充分说明校准与默认 scale 语义，则本候选失效**（降级为「无需贡献」）。
- **风险**：中低。注意别把本机特定数值当作普适结论写进上游文档。

---

## 4. Tier C — 较大 / 需 RFC，先观望（非第一次 PR 之选）

### C5 · `vllm bench serve` 增加 SLO 驱动的容量扫 + knee 检测
- **来源**：M12（冻结 SLO→QPS 扫→动态 knee→3 seed worst-seed 判据→最大稳定 QPS 区间）。
- **现状**：官方 `vllm bench serve` **已有** `--goodput ttft:/tpot:/e2el:` 与 Poisson 到达（M12 正是用它），但**没有**「自动扫 QPS + knee 检测 + 多 seed 稳定性判据 → 直接给出最大稳定 QPS」的一键能力。M12 的 sweep/knee 逻辑目前是仓库内 driver 脚本。
- **为什么观望**：属功能增强、面较大、多半需先开 RFC/issue 讨论接口边界（且可能被判定为「留在用户脚本层即可」）。不适合作**第一次** PR，但作为「能力已具备、时机成熟再提」的储备项保留。

### C6 · 执行模式运行时诊断（逐 batch/长度确认 FULL/PIECEWISE/NONE）
- **来源**：M5/M6（`exec/dispatch`、`exec/capture` 钩子 + `diag_modes_*.jsonl`）。
- **现状**：M5/M6 的诊断依赖**自加的只读 trace 钩子**（学习期产物，不宜直接进上游）。要成为上游能力需重新设计为正式、低开销、默认关的可观测接口。
- **为什么观望**：需 RFC + 重新实现（不能照搬学习钩子）；面较大。储备项。

---

## 5. 明确排除区（记录理由，避免以后重复评估）

| 排除项 | 来源 | 排除理由 |
|---|---|---|
| **sm_120 上 FlashInfer sampler「必须关」当作 vLLM bug** | day1/day2 | 根因是 **flashinfer 包**在 WSL/无 nvcc 下 JIT 探测 sm_120 target arch 失败（`TARGET_CUDA_ARCHS` 空），**不是 vLLM 代码 bug**；且 v0.24.0 的 `vllm/v1/sample/ops/topk_topp_sampler.py:44-66` 已对不支持能力做**优雅 fallback + warning**（显式设 1 才 raise）。→ 非 vLLM 上游候选；若要报也是报 flashinfer，且新版 flashinfer 可能已修，需先核。 |
| **sm_120 FlashInfer attention/KV「不可用」** | m10 | v0.24.0 已把 FlashInfer 能力区间放到 `DeviceCapability(8,0)–(12,1)`（含 sm_120），并新增 `FLASHINFER_MLA_SPARSE_SM120` 后端（`registry.py`）与其测试。→ 实验期观测已**过时**，代码已支持。 |
| **fp8 在 sm_120 上 decode +36% 变慢** | m10 | 属 sm_120 fp8 **kernel 微架构/CUDA 版本**层面的性能特性（小 GEMV + 动态激活量化开销），修复需深 kernel 工作 + 特定硬件，且强依赖 CUDA/驱动版本——**不适合第一次 PR，且极可能是环境特定、非普适 bug**。最多作 §3-C4 文档里一句「decode 小 batch fp8 可能更慢」的提醒。 |
| **fp8 KV「无质量警告」** | m10 | v0.24.0 `cache.py:278-285` **已有** info 级质量提示。→ 该「缺口」已不成立（相关的用户文档补充另见 C4）。 |
| **M9 合并 decode 两处 H2D 拷贝** | m9 | 已实测：机制成立、正确性完整，但 **E2E 收益落在噪声内（~0.13%）**，根因是该路径已被上游高度优化（pinned+non_blocking+overlap）。→ 无正收益，不提交上游（保留为诚实无收益报告）。 |
| **实验期只读 trace 钩子 / `VLLM_LAB_*` 门控 / block 实验** | day1-m9 | 学习者自用诊断，非产品缺陷或改进；不进上游。 |
| **M8 自写 kernel 本身** | m8 | 官方 kernel 已生产就绪且更优；自写 kernel 的价值在**测试 oracle**（已转化为 C1），kernel 本体不提交。 |
| **WSL2 下 nsys/CUPTI device-trace 不可用** | m7 | 属 NVIDIA/WSL 平台限制，非 vLLM；最多在 vLLM profiling 文档加一句「WSL2 建议用 torch profiler」——价值很低，暂不列为正式候选。 |

---

## 6. M13 提交前流程清单（选定候选后逐项过）

1. **重验有效性**：对当前 `main`（非 pinned v0.24.0）复核该问题仍成立（见 §0 kill 标准）。
2. **查重**：GitHub 搜 open/近期 closed 的 issues+PRs，确认无重复/在途。
3. **CONTRIBUTING 合规**（据 `vllm-src/docs/contributing/README.md`）：
   - **DCO sign-off**：所有 commit `git commit -s`（`Signed-off-by:`），否则 CI 拒。
   - **pre-commit**：`pre-commit install`（需 `pre-commit>=4.5.1`），本地过 lint/format。
   - **测试**：`pytest` 跑相关子集本地绿；纯文档 PR 也要过 docs 构建/lint。
   - **RFC 门槛**：>500 LOC（不含 kernel/数据）才需先 RFC issue；C1–C4 均远低于此，通常无需 RFC（但 C3 建议先开轻量 issue 探口风）。
   - **kernel 类改动**（本轮候选均不涉及）：需 custom op 注册 + meta function + op 检查——记于此以备将来。
4. **⚠ AI 辅助声明与署名冲突（需用户决策）**：vLLM CONTRIBUTING 要求**披露 AI 辅助**并加 `Co-authored-by:` trailer；这与本仓库自身的提交约定（[[vllm-lab-no-coauthor-trailer]]：只出现 CiaranCw、不加 Co-Authored-By）**是不同仓库、不同规则**。上游 PR 须遵循 vLLM 政策——提交前请确认你对该 PR 的署名/披露口径（本仓库的 no-coauthor 约定不自动延伸到上游）。
5. **PR 内容**：复现步骤 → 定位 → 改动 → 测试 → （如涉性能）benchmark 前后对照 → 回应 review → 迭代。目标是**完整经历一次上游流程，非必须合入**。

---

## 7. 维护协议（本文件如何持续更新）

- **每完成一个后续模块**：若触及任何真实 vLLM 行为，回本文件登记「新种子」（来源模块 / 证据出处 / 类别 / 是否值得上游 / 是否已被 main 解决）。
- **候选状态流转**：`候选` → `已核实(main 仍成立)` → `进行中(M13)` → `已提交/已合入` 或 `已失效(被抢先/已修/被拒)`；失效项**保留并注明理由**，不删除。
- **与 ROADMAP 联动**：M13 选定后在 ROADMAP M13 处回填所选候选编号与 PR 链接。
- **诚实红线**：不把本机特定（sm_120/eager/0.5B/WSL2）绝对值当普适结论写进上游；文档类 PR 引用实验数据须标注适用边界。

---

*建立于 M12 收口后、M13 之前。首选目标：**C1（slot_mapping 生成 kernel 单元测试）**。*
