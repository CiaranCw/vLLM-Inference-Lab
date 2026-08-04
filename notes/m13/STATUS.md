# M13 断点快照（RESUME / 交接）

> **状态：READY FOR FINAL HUMAN CONFIRMATION BEFORE COMMIT**
> —— 技术增强与验证全部完成；**尚未创建 Commit、尚未 Push、尚未创建 PR**（远程 CI 亦未运行）。
> **下一步**：你按 `notes/m13/review/FINAL_PRE_PR_CHECKLIST.md` 逐项确认。
>
> **最后更新**：2026-08-04 09:2x–09:5x UTC（本轮：按人工审查意见做 15 项低风险增强 + 全量重验 + 重生成审查材料）

---

## 1. 最新上游状态
| 项 | 值 |
|---|---|
| 核查时间 | 2026-08-04 09:22 UTC 起 |
| upstream/main | **`0b1c151cbcacd12e207f2a91e4ca9f5b6961ec8a`**（2026-08-04T08:03:46Z，`[Rust Frontend] Align tool rendering for Kimi K3 (#50540)`） |
| 上轮基线 | `72cd542` → 本轮 upstream 前进 1 commit |
| 该 commit 影响面 | 仅 `rust/src/chat/src/renderer/kimi_k3/*`（4 文件）；**监视清单全部零改动**：`vllm/v1/worker/block_table.py`、`AGENTS.md`、`docs/contributing/README.md`、`tests/v1/worker/`、`.pre-commit-config.yaml` |
| 移植 | 已按「暂存→`reset --hard origin/main`→恢复」安全 port，sha256 保持不变后再做本轮增强 |
| 本地 HEAD | `0b1c151`（= origin/main，**领先 0 commit，无本地 commit**） |
| 语义复核 | `compute_slot_mapping(num_reqs, query_start_loc, positions)` 未变；kernel 仍 `TOTAL_CP_WORLD_SIZE=dcp_world_size`（DCP-only）+ `KV_CACHE_BLOCK_SIZE`/`BLOCKS_PER_KV_BLOCK`；`SlotMappingMode` 仍为 `{TOKEN_TO_KV_SLOT, NONE}` |
| 测试缺口 | 仍成立：pristine main 上 `git grep compute_slot_mapping -- tests/` = **0**；`test_block_table.py` 仍不存在 |
| 重复 PR | `test_block_table` 搜索 → **0 open PR**；其余关键词命中均为 bugfix/perf/feature，无直接测试 PR；#30683 仍 `CLOSED`/未合并 |

## 2. 本轮人工审查增强（15 项，均为低风险、不扩大 scope）
| # | 项 | 内容 |
|---|---|---|
| 4.1 | 模块 docstring | 「through the public `BlockTable` API」→「through `BlockTable.compute_slot_mapping`」（测试确实直接设置已解析的 DCP 字段，原措辞过强） |
| 4.2 | Oracle docstring | 「from the public input semantics」→「from the `BlockTable` input semantics」 |
| 4.3 | `_compute()` | 新增 `assert len(kv_manager_block_ids) == len(positions_per_req)`（仅此一条，反映 helper 输入契约） |
| 4.4 | 普通 Golden ids | 加 `ids=["single_request_block_boundaries", "multiple_request_rows"]` |
| 4.5 | **三方锚定** | 普通 Golden 现同时断言 `Actual == Golden` **与** `Reference == Golden`（不用 `_check()`，以显式展示三条证据路径） |
| 4.6 | 输入最小化 | 普通 Golden `[[5,6,7,8]]`→`[[5,6,7]]`（块 8 未被引用）；ragged 末请求 `[40,41,42]`→`[40,41]`（17 token 只需 2 块）；多请求组保留 `[[9,2],[7,4]]` 以展示独立行 |
| 4.7 | Prefix 注释 | 改为「Tokens scheduled after a cached prefix use absolute, non-zero positions rather than indices into the current batch.」 |
| 4.8 | Tile 块数 | `num_tokens // 16 + 1` → 通用向上取整 `(num_tokens + block_size - 1) // block_size`，并显式传 `kv_cache_block_size=block_size` |
| 4.9 | Tile 手算锚点 | 删除恒真的 `assert len(actual) == num_tokens`；改为 `actual[1023:1030] == arange(1039, 1046)`（跨第一/第二 tile 的真实 token） |
| 4.10 | DCP 输入最小化 | 块 id 由 32 个 → `[[1, 2, 3]]`（world2 需 3 块 / world4 需 2 块），仍从 1 开始 |
| 4.11 | Ownership stack | `np.sum(list, axis=0)` → `np.stack(owning_ranks, axis=0).sum(axis=0)`；`owning_ranks: list[np.ndarray]` 类型标注 |
| 4.12 | **DCP Golden 跨虚拟块** | positions `[0..5]` → `[0,1,2,3,4,5,32,33]`；`position 33` 用到此前从未触及的 KV 块 6 → slot **96** |
| 4.13 | **Hybrid Golden 跨第二 kernel 块** | positions 加 `48`；此前 kernel 块 19 从未被触及；`position 48` → 块 19 offset 0 → slot **304**；同时修正 docstring（`position 47` 属块 **18**） |
| 4.14 | dtype 明确 | DCP / Hybrid 手算期望数组显式 `dtype=np.int64` |
| 4.15 | NONE 诊断 | `assert (actual == sentinel).all()` → `np.testing.assert_array_equal(actual, np.full_like(actual, sentinel))`（失败时显示差异位置） |

**明确保留（未按"顺手"改动）**：`MAX_NUM_BATCHED_TOKENS = 2048`、文件级 CUDA skip、直接设置已解析 `dcp_world_size/dcp_rank` 的策略、`_compute()` 走真实 `add_row → commit → compute_slot_mapping` 路径、独立 `_expand_kernel_block_ids`、三类手算 Golden、跨 1024 tile 测试、`SlotMappingMode.NONE`。**未新增** DCP world size / interleave / hybrid 尺寸 / 空请求用例；**未改产品代码**。

**新增 anchors 已独立复核**：用一份与测试无关的最小脚本重算 DCP `position 33 → 96`、Hybrid `position 48 → 304`、Tile `1023..1029 → 1039..1045`，三者全部 MATCH；并核对最小化后的块数恰好足够（3 / 2 / 3）。

## 3. 最终 Test Matrix（18 tests / 384 行）
| 组 | 数 | 内容 |
|---|:--:|---|
| 普通手算 Golden（三方锚定） | 2 | `single_request_block_boundaries` → `[80,95,96,127]`；`multiple_request_rows` → `[147,68]` |
| 边界 / Ragged / Prefix | 6 | single_token、first_and_last_position_in_block、spans_many_blocks(64)、non_contiguous_blocks(96)、ragged_multi_request(5 请求)、prefix_offset(70–89) |
| 跨 Triton Tile（真实 token + padding） | 1 | 1030 tokens（>1024 且非整数倍）；含 `[1023:1030] == 1039..1045` 手算锚点；尾部至 2100 全 PAD |
| DCP 分区 | 4 | world{2,4} × interleave{1,4}，遍历全 rank + 「每 token 恰属一个 rank」守恒律 |
| DCP 手算 Golden（跨虚拟块） | 1 | world2/rank1 → `[-1,80,-1,81,-1,82,-1,96]` |
| Hybrid 泛化 | 2 | (32,16)、(64,16)，跨 kernel 块与分配块边界 |
| Hybrid 手算 Golden（跨第二 kernel 块） | 1 | `[160,175,176,191,288,303,304]` |
| `SlotMappingMode.NONE` | 1 | sentinel −12345（≠ PAD −1）整段不变 |

## 4. 最终验证（干净树、仓库根目录、`.venv/bin/python`）
| 检查 | 结果 |
|---|---|
| `pytest ... --collect-only` | **18 collected** |
| `pytest ... -v` | **18 passed / 0 failed / 0 skipped / 14 warnings / 5.24s** |
| `ruff check` | All checks passed!（ruff 0.14.0，与 `.pre-commit-config.yaml` rev 一致） |
| `ruff format --check` | 1 file already formatted（**本轮曾需 reformat，已应用并复跑测试**） |
| `pre-commit run --files <file>` | **全部 Passed（0 Failed）**：ruff/format/typos/**mypy**/SPDX/forbidden-imports/… |
| `git diff --check HEAD` | 无输出 |
| `git diff --name-only HEAD -- vllm/ csrc/` | **空** |
| Oracle 独立性静态检查 | `grep "map_to_kernel_blocks("` → **无可执行调用**（仅 docstring 提名字） |
| **变异回归** | 注入产品 `map_to_kernel_blocks +1` → `hybrid[32-16]`、`hybrid[64-16]`、`hybrid_golden` **3/3 失败**；恢复后产品 `git diff` 为空且与 upstream 一致；重跑 **18 passed** |

**基线问题（与本改动无关，不进 PR Body）**：
- `pre-commit run --all-files` 的 `shellcheck`：`docs/pre_run_check.sh` SC2317（upstream 既有，未触碰）。
- `tests/v1/worker/test_gpu_block_table.py` 本机 3 failed：`RuntimeError: UVA is not available`（WSL2）；移除本文件后在 pristine main 上同样失败。本文件用 `pin_memory=False`，不受影响。

## 5. 当前 Diff
```
 tests/v1/worker/test_block_table.py | 384 ++++++++++++++++++++++++++++++++++++
 1 file changed, 384 insertions(+)
```
- **文件总行数**：384　**Tests**：18
- **SHA256**：`b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`（草稿与干净树**一致**）
- **产品代码改动**：**无**
- **是否存在 Commit**：**无**（`git rev-list --count origin/main..HEAD` = 0）

## 6. 文档同步状态
| 文件 | 状态 |
|---|---|
| `STATUS.md` | 本文件，已更新 |
| `pr_slot_mapping_test.md` | 已加「人工逐行审查后的最终增强」小节 |
| `pr_draft/test_block_table.py` | 与干净树 sha256 一致 |
| `pr_draft/PR_DESCRIPTION.md` | 标题 `[Core]`；Test Matrix / 命令 / 结果已更新；AI 披露仍为不点名未来时占位 |
| `review/LINE_BY_LINE_REVIEW.md` | **行号已按 384 行版本重生成**，含本轮 15 项修改说明 |
| `review/MANUAL_GOLDEN_CASES.md` | 含 DCP position 32/33、Hybrid position 48、Tile 1023–1029 完整手算 |
| `review/REVIEW_CHECKLIST.md` | 已更新，全部未勾选 |
| `review/FINAL_PRE_PR_CHECKLIST.md` | **本轮新建**，全部未勾选 |
| `review/REPRO_COMMANDS.md` | 已更新（含新 SHA / 行数 / 变异复跑步骤） |
| `review/REVIEW_DIFF.patch` | 已重新生成（390 行） |
| `validation/` | 日志已刷新 |

## 7. 仍需你（提交人）完成
1. 阅读**最终**变更后的 Diff（384 行，不是旧版本）
2. 亲自复跑 Pytest（期望 18 passed）
3. 亲自复跑 Changed-file Pre-commit（期望全绿）
4. 最后一次查重
5. **确认 DCO 身份**（当前 `CiaranCw <CiaranCw@users.noreply.github.com>` —— 我不代你确认）
6. **确认实际 AI 工具与 Commit Trailer**（PR Body 为不点名占位）
7. **把 PR Body 的人工审查表述改成真实完成时**
8. 之后才是 Commit / Push / 创建 PR（命令见 `review/REPRO_COMMANDS.md` §10）

## 8. 环境与隔离
- 干净工作区 `/home/ciaran/projects/vllm-m13-upstream`（独立 clone + 独立 `.venv`，`uv` 管理），与带中文学习注释的 `/home/ciaran/projects/vllm-src`（22 脏文件、shallow、pinned v0.24.0）**完全隔离**。
- 环境：Python 3.12.13 / torch 2.13.0+cu130 / triton 3.7.1 / ruff 0.14.0 / RTX 5080 sm_120 / driver 610.62。
- GPU：使用前查 `nvidia-smi`（3516MiB / 6%，无其他计算进程），**未 kill 任何进程、未 reset**。
- 本轮**未创建 commit、未 push、未建 PR、未加 fork remote、未动任何 Issue/PR、未改产品代码**。
