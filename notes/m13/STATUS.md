# M13 断点快照（RESUME / 交接）

> **状态：READY FOR HUMAN LINE-BY-LINE REVIEW**
> —— 技术修复与验证全部完成；**尚未创建 Commit、尚未 Push、尚未创建 PR**。
> **下一步是「你逐行审查」**，材料在 `notes/m13/review/`。
>
> **最后更新**：2026-08-04 06:40 UTC（本轮：修复独立审查发现的问题 + 重跑全部验证 + 生成逐行审查材料）

---

## 1. 最新上游状态
| 项 | 值 |
|---|---|
| 核查时间 | 2026-08-04 06:18–06:40 UTC |
| upstream/main | **`72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0`**（2026-08-04T06:15:02Z，`[Rust][Benchmark] Preserve UTF-8 across benchmark stream chunks (#50868)`） |
| 上轮基线 | `cb81048` → 本轮 upstream 前进 1 个 commit |
| 该 commit 影响面 | 只改 `rust/src/bench/src/backends/streaming.rs`；**未触及** `block_table.py` / `AGENTS.md` / `docs/contributing` / `tests/v1/worker` → 属「情况 B」，已安全 port |
| 本地 HEAD | `72cd542`（= origin/main，**无本地 commit**） |
| 分支 | `m13/block-table-slot-mapping-tests`（干净树，仅 1 个未跟踪测试文件） |
| 语义复核 | `block_table.py` 在 `cb81048..72cd542` **零改动**；DCP-only、hybrid 三分、`SlotMappingMode` 全部不变 |
| 规则复核 | `AGENTS.md`、`docs/contributing/README.md` **零改动**；`[Core]` 仍是合法前缀 |
| 测试缺口 | 仍成立：pristine main 上 `git grep compute_slot_mapping -- tests/` = **0**；`test_block_table.py` 仍不存在 |
| 重复 PR | 无等价 Open PR（查 `compute_slot_mapping` / `block table slot mapping` / `test_block_table` / `DCP slot mapping` / `hybrid block table`）；#30683 仍 `CLOSED`/未合并 |

## 2. 本轮已修复（独立审查发现的问题）
| # | 问题 | 修复 | 证据 |
|---|---|---|---|
| 1 | **Oracle 复用产品 helper**：`_reference_slot_mapping` 调 `BlockTable.map_to_kernel_blocks`，与 `add_row` 用同一函数 → helper 出 bug 时期望值与实际值**一起错**，hybrid 测试仍绿 | 新增独立 `_expand_kernel_block_ids`（公式 `kv_block_id * bpk + sub_block`），Oracle 不再调用任何产品 mapping helper、不读产品 `block_table.np` | 见 §3 变异测试；`grep map_to_kernel_blocks tests/...` 现**无输出** |
| 2 | Oracle docstring 声称「without reusing its indexing helpers」但实际在复用 | 重写 docstring，明确「从 KV manager id 出发、自行展开、不读产品表」 | 测试文件 L87-103 |
| 3 | 未使用的 `name` 参数（只用于 case 名） | 改用 `ids=[...]`，函数签名删掉 `name` | pytest 输出现为 `[single_token]`、`[prefix_offset]` 等 |
| 4 | `kernel_block_size or kv_cache_block_size` 把默认值与合法性混在布尔表达式 | 改为显式 `if kernel_block_size is None:` | 测试文件 L41-42 |
| 5 | **hybrid golden docstring 自相矛盾**（写「kernel block 19 … = 303」，但 303 = 18×16+15） | 经独立脚本复核确认 `position 47` 落在 kernel 块 **18**；期望值 303 本就正确，**修正注释块号** | 复核脚本输出见 review 材料 §3 |
| 6 | PR 标题分类 | `[CI/Build]` → **`[Core]`**（本 PR 不改 CI/Build/测试基础设施，保护的是 V1 核心 KV BlockTable 行为；`[Core]` 在合法列表内） | `docs/contributing/README.md:233-234` |
| 7 | PR Body **提前声称人工审查已完成** | 改为未来时占位：「the human submitter will review every changed line…」，并在 checklist 中标记该项待你定稿 | `pr_draft/PR_DESCRIPTION.md` |
| 8 | PR Body 点名 "AI assistance (Claude)" | 改为不点名的表述 + HTML 占位提示，由你按实际使用工具定稿 | 同上 |
| 9 | PR Body 混入无关基线失败（shellcheck SC2317、WSL2 UVA） | 从 PR Body 移除；**证据保留**在本文件 §5 与 `notes/m13/validation/` | 同上 |
| 10 | #30683 措辞可能暗示 maintainer 已接纳 | 改为纯事实：「previously explored… was not merged and targets an older API」，不声称被接纳 | 同上 |

## 3. Oracle 独立性的变异测试（关键证据）
**注入的错误**：给产品 `BlockTable.map_to_kernel_blocks` 的返回值 `+1`（`vllm/v1/worker/block_table.py:243-246`）。

| 用例 | 修复**前** | 修复**后** |
|---|:---:|:---:|
| `test_slot_mapping_with_hybrid_kernel_blocks[32-16]`（Oracle 驱动） | ✅ **passed（被同一 bug 带跑，漏检）** | ❌ **failed** |
| `test_slot_mapping_with_hybrid_kernel_blocks[64-16]`（Oracle 驱动） | ✅ **passed（漏检）** | ❌ **failed** |
| `test_slot_mapping_hybrid_matches_hand_computed_slots`（写死 golden） | ❌ failed | ❌ failed |

- **为什么这证明独立性**：修复前 Oracle 与产品共用展开逻辑，注入的 bug 同时污染期望值与实际值 → 断言仍成立；修复后 Oracle 自行展开，两条路径分叉 → 立刻抓住。
- **全量**：注入后 `3 failed / 15 passed`（仅 hybrid 相关失败，符合预期——非 hybrid 用例 `bpk=1`，不经过该 helper）。
- **恢复证据**：`git diff --name-only HEAD -- vllm/ csrc/` → **空**；恢复后重跑 **18 passed**。
- 另有上一轮的 kernel 级变异（`slot_offsets + 1` → 17/18 失败）证明测试整体非空转。

## 4. 当前实现（18 tests / 348 行 / 零产品改动）
`tests/v1/worker/test_block_table.py`，sha256 `3336963163439fd2531519f4640181444ec256c558c55c66f7226e2e934e2296`

| 组 | 数量 | 内容 |
|---|:--:|---|
| 普通手算 golden | 2 | `[80,95,96,127]`；两请求独立行 `[147,68]` |
| 边界/ragged/prefix | 6 | 单 token、块内首末、跨多块(64)、非连续物理块(96)、ragged 5 请求、prefix offset(70-89) |
| 跨 Triton tile padding | 1 | 1030 tokens（>1024 且非整数倍），buffer 2100 使尾部再跨一 tile |
| DCP 全 rank 分区 | 4 | world{2,4} × interleave{1,4}，遍历所有 rank + 「每 token 恰属一个 rank」守恒律 |
| DCP 手算 golden | 1 | world2/rank1 → `[-1,80,-1,81,-1,82]` |
| Hybrid 泛化 | 2 | (32,16)、(64,16)，跨 kernel 块与分配块边界 |
| Hybrid 手算 golden | 1 | `[160,175,176,191,288,303]` |
| `SlotMappingMode.NONE` | 1 | sentinel −12345（≠ PAD −1）保持不变 |

## 5. 验证结果（全部在干净树、仓库根目录、`.venv/bin/python`）
| 检查 | 命令 | 结果 |
|---|---|---|
| collect | `pytest ... --collect-only` | **18 collected** |
| 正式测试 | `pytest tests/v1/worker/test_block_table.py -v` | **18 passed, 0 failed, 0 skipped, 14 warnings, 5.06s** |
| ruff check | `python -m ruff check <file>` | All checks passed!（ruff 0.14.0，与 `.pre-commit-config.yaml` rev 一致） |
| ruff format | `python -m ruff format --check <file>` | 1 file already formatted |
| pre-commit（changed file） | `pre-commit run --files <file>` | **全部 Passed**（ruff/format/typos/**mypy**/SPDX/forbidden imports/…），0 个 Failed |
| git diff --check | `git diff --check HEAD` | 无输出 |
| 产品代码 diff | `git diff --name-only HEAD -- vllm/ csrc/` | **空** |
| 变异测试 | 见 §3 | 修复后 3/3 hybrid 被杀 |

**基线问题（与本改动无关，仅记录、不进 PR Body）**：
- `pre-commit run --all-files`：仅 `shellcheck` 失败 —— `docs/pre_run_check.sh` SC2317，upstream 既有文件，本 PR 未触碰。
- `tests/v1/worker/test_gpu_block_table.py`：本机 3 failed，根因 `RuntimeError: UVA is not available`（WSL2 无 UVA，MRV2 pinned-host 表）；**移除本文件后在 pristine main 上同样 3 failed**。本文件用 `pin_memory=False`，不受影响。

## 6. 逐行审查材料（`notes/m13/review/`）
| 文件 | 用途 |
|---|---|
| `LINE_BY_LINE_REVIEW.md` | 按行号分 16 节逐段讲解：目的/输入/输出/为什么需要/潜在错误/你要检查什么/面试怎么讲；Oracle 8 个公式逐条与产品 kernel 对照 |
| `MANUAL_GOLDEN_CASES.md` | 3 类 golden 的完整手算表（普通 / DCP world2 rank1 / hybrid 32-16），含退化检查与 3 道自测题 |
| `REVIEW_CHECKLIST.md` | 26 项复选框，**全部未勾选**，只能由你本人勾选 |
| `REPRO_COMMANDS.md` | 你应亲自执行的精确命令；commit/push/PR 命令标注「⛔ 仅在人工审查全部完成后执行」 |
| `REVIEW_DIFF.patch` | 最终 diff（354 行），仅供审查，不提交上游 |

## 7. 仍需你（提交人）完成
1. **逐行审查** 348 行测试文件（AGENTS.md 硬要求，我不能代做）。
2. **手算复核** 3 类 golden（材料已备）。
3. **亲自重跑** pytest 与 changed-file pre-commit。
4. **确认 DCO 身份**：当前 `CiaranCw <CiaranCw@users.noreply.github.com>` —— 需你确认（我不代确认）。
5. **确认 AI 工具与 trailer**：PR Body 中为不点名占位表述，需你按实际情况定稿。
6. **PR 创建前最后查重**。
7. **把 PR Body 的未来时改成完成时**（现为「will review」）。

## 8. 环境与隔离
- 干净工作区 `/home/ciaran/projects/vllm-m13-upstream`（独立 clone + 独立 `.venv`，`uv` 管理），与带中文学习注释的 `/home/ciaran/projects/vllm-src`（22 脏文件、shallow、pinned v0.24.0）**完全隔离**；后者不可用于 PR。
- 环境：Python 3.12.13 / torch 2.13.0+cu130 / triton 3.7.1 / ruff 0.14.0 / RTX 5080 sm_120 / driver 610.62。
- GPU：使用前后均查 `nvidia-smi`，**未 kill 任何进程、未 reset**。
- 本轮**未创建 commit、未 push、未建 PR、未添加任何 fork remote、未回复/关闭任何 Issue/PR**。
