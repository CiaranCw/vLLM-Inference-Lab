# M13 断点快照（RESUME / 交接）

> **状态：DRAFT PR 已创建 —— https://github.com/vllm-project/vllm/pull/51017（等待 CI 排队完成 / maintainer review）**
> —— 提交人已完成人工审查（代码 / Golden / 18-18 pytest / pre-commit），已改名、signed-off commit、push 到 fork，并经你确认后创建 **Draft PR #51017**。
> **DCO 检查：pass** ✅　**mergeable: true**　1 commit / 1 file / +384 / 零产品改动。
> **下一步**：等 CI（当前 `queued`）→ 视结果决定是否 Ready for review；如需回应 review 再迭代。
>
> **最后更新**：2026-08-04 11:1x UTC（本轮：改名避开 #39591 路径冲突 + 复跑验证 + PR Body 定稿 + DCO + signed-off commit `e391a6b` + push 到 fork）

---

## 0. 本轮执行结果（提交人已完成人工审查后）
| 项 | 状态 |
|---|---|
| 人工审查 | ✅ 代码审查、Golden 理解、**人工 pytest 18/18**、**人工 pre-commit 通过**（由提交人本人完成） |
| **文件改名** | `test_block_table.py` → **`tests/v1/worker/test_block_table_slot_mapping.py`**，避开 open PR **#39591** 的同名路径冲突；内容未变（sha256 不变），改名后复跑 **18 passed**、pre-commit 13 Passed 0 Failed |
| **DCO 身份** | 按你提供的值设为 **repo-local**：`CiaranCw <1399538830@qq.com>`（未改动全局配置） |
| **Commit** | ✅ **`e391a6b`**，`git commit -s`；含 `Signed-off-by: CiaranCw <1399538830@qq.com>` 与 `Assisted-by: Claude Code`；仓库自带 `Sign-off Commit` hook 与全部 pre-commit hook 均 Passed |
| **Push** | ✅ 已创建 fork **`CiaranCw/vllm`** 并推送分支 `m13/block-table-slot-mapping-tests`；远程 commit = 本地 `e391a6b`；compare 显示 **ahead 1 / behind 0，仅 1 个新增文件** |
| **PR** | ✅ **Draft PR #51017** https://github.com/vllm-project/vllm/pull/51017 —— base `vllm-project:main`、head `CiaranCw:m13/block-table-slot-mapping-tests`、title `[Core] Add unit tests for BlockTable slot mapping`；正文取自 `pr_draft/PR_BODY_FINAL.md` |
| **正文换行修正** | 初版正文按 commit-message 习惯做了 ~79 列硬换行（**我的风格选择，非项目要求**）。核查确认：`AGENTS.md:120` 的 88 列**只约束 Python**、`.markdownlint.yaml` 明确 `MD013: false`、上游 PR 模板自身最长行 149 —— 故按你选择改为**不硬换行**（段落单行，靠 GitHub 自动折行；表格/代码块原样）。已用 **REST API** 更新（`gh pr edit` 因 GraphQL projectCards 弃用而未生效），远程正文与本地**逐字节一致**、语义等价 |
| **CI 状态** | **DCO pass** ✅、Meta Internal-Only Check pass、mergify Summary pass；`pre-commit` / `New PR Bot` 三个 job 状态 `queued`（排队，非失败、非等审批） |
| AI 披露 | 经你确认为 **Claude Code**；PR Body 与 commit trailer 均据实填写 |
| 产品代码 | 未修改（commit 内只有 1 个测试文件） |


## 1. 最新上游状态
| 项 | 值 |
|---|---|
| 核查时间 | **2026-08-04 10:52 UTC 起**（本轮） |
| upstream/main | **`a1657a0235a77ce85f8d348c83ff4adb29918b71`**（2026-08-04T10:06:04Z，`[Bugfix][Build] Fix DeepGEMM CUDA 12.9 FP8 header visibility (#51003)`） |
| 上轮基线 | `0b1c151` → 本轮 upstream 前进 **2 commits**（`e98a877` Rust gRPC 多模态、`a1657a0` DeepGEMM cmake） |
| 影响面 | 仅 `rust/**`、`cmake/external_projects/deepgemm.cmake`、`tools/install_deepgemm.sh`（14 文件）；**监视清单全部零改动**：`vllm/v1/worker/block_table.py`、`tests/v1/worker/`、`AGENTS.md`、`docs/contributing/README.md`、`.pre-commit-config.yaml`、`pyproject.toml` |
| 移植 | 已按「暂存→`reset --hard origin/main`→恢复」安全 port，sha256 保持不变后再做本轮增强 |
| 本地 HEAD | `a1657a0`（= origin/main，**领先 0 commit，无本地 commit**） |
| 语义复核 | `block_table.py` 相对上轮**逐字节未变**；`__init__`/`add_row`/`map_to_kernel_blocks`/`commit_block_table`/`compute_slot_mapping` 签名不变；kernel 仍 DCP-only + `KV_CACHE_BLOCK_SIZE`/`BLOCKS_PER_KV_BLOCK`；`SlotMappingMode` 不变 |
| 测试缺口 | 仍成立：pristine main 上 `git grep compute_slot_mapping -- tests/` = **0**；`test_block_table.py` 仍不存在 |
| 重复 PR | `test_block_table` 搜索 → **0 命中**；5 组关键词命中均为 bugfix/perf/feature。**逐 PR 文件级核查**后确认无等价工作（详见 §2b） |

## 2. 人工审查增强（上一轮完成的 15 项，本轮未再改测试代码）
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

## 2b. 重复 PR 逐项核查（本轮，文件级）
5 组关键词搜索命中的 PR 均**逐个核查**（不只看标题）：

| PR | 性质 | 是否重复 |
|---|---|:--:|
| #39936 | MultiGroupBlockTable CPU launch 开销优化（perf） | 否 |
| #47924 | RFC/prototype：decode step 整体编译 | 否 |
| #46446 / #45245 / #43879 | slot-mapping kernel **warmup 预编译**（bugfix） | 否 |
| #41124 | Mamba/SSM `None` block_size 处理（bugfix） | 否 |
| **#39591** | **`tests/v1/worker/test_block_table.py`**，但内容是 **row-tail 清零**修复 #39589 的 35 个测试（`append_row`/`add_row`/`move_row`/`clear_row`），文件内 `compute_slot_mapping`/`slot_mapping`/`dcp`/`SlotMappingMode` 引用次数**全为 0** | **否（路径重合，语义互补）** |
| #48970 | e2e hybrid-Mamba prefix-cache 回归（不在 `tests/v1/worker/`） | 否 |
| #30683（closed） | 旧 API + 已失效 `pcp × dcp` 假设 | 否（已关闭） |

- `gh pr list --search "test_block_table"` → **0 命中**。
- **⚠ 需提交人知情的现实风险**：**#39591 会创建同一路径的文件**。两者测的行为不同、内容互补，但**谁先合入，另一方需要 rebase 进同一文件**。已在 PR Body 中主动说明并表示愿意配合改名或合并到同一文件。

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

## 4. 本轮自动验证（Agent 执行；干净树、仓库根目录、`.venv/bin/python`）
> ⚠️ 这些是 **Agent 自动化结果**，**不等于**提交人亲自运行；提交人仍须自行复跑（见 §7）。
| 检查 | 结果 |
|---|---|
| `pytest ... --collect-only` | **18 collected** |
| `pytest ... -v` | **18 passed / 0 failed / 0 skipped / 14 warnings / 5.46s** |
| `ruff check` | All checks passed!（ruff 0.14.0，与 `.pre-commit-config.yaml` rev 一致） |
| `ruff format --check` | 1 file already formatted |
| `pre-commit run --files <file>` | **13 Passed / 0 Failed**：ruff/format/typos/**mypy**/SPDX/forbidden-imports/… |
| `git diff --check HEAD` | 无输出 |
| `git diff --name-only HEAD -- vllm/ csrc/` | **空** |
| `py_compile` | OK（语法有效） |
| Oracle 独立性静态检查 | `grep "map_to_kernel_blocks("` → **无可执行调用**（仅 L69 docstring 提名字）；且不读 `block_table.np/.gpu` 作期望来源 |
| 手算锚点独立复核 | 独立脚本重算：DCP `33→96`、Hybrid `48→304`、Tile `1023..1029→1039..1045`、三组 golden 数组 **全部 MATCH** |
| **变异回归** | 注入产品 `map_to_kernel_blocks +1` → `hybrid[32-16]`、`hybrid[64-16]`、`hybrid_golden` **3/3 失败**（日志 `validation/mutation_final.txt`，已标注 `EXPECTED FAILURE FROM TEMPORARY MUTATION`）；恢复后 `git diff -- vllm/ csrc/` 为空、与 upstream 逐字节一致；重跑 **18 passed** |

**基线问题（与本改动无关，不进 PR Body）**：
- `pre-commit run --all-files` 的 `shellcheck`：`docs/pre_run_check.sh` SC2317（upstream 既有，未触碰）。
- `tests/v1/worker/test_gpu_block_table.py` 本机 3 failed：`RuntimeError: UVA is not available`（WSL2）；移除本文件后在 pristine main 上同样失败。本文件用 `pin_memory=False`，不受影响。

## 5. 当前 Diff
```
 tests/v1/worker/test_block_table_slot_mapping.py | 384 ++++++++++++++++++++++++++++++++++++
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
| `pr_slot_mapping_test.md` | 已加最终增强小节；基线 SHA 已刷新 |
| `pr_draft/test_block_table_slot_mapping.py` | 与干净树 sha256 一致 |
| `pr_draft/PR_DESCRIPTION.md` | 标题 `[Core]`；**base SHA→`a1657a0`**、精确命令、结果标注为 *Automated preparation run*、缺口表述改为「on the base revision」、新增 #39591 路径重合说明；AI 披露仍为**不点名未来时占位** |
| `review/LINE_BY_LINE_REVIEW.md` | **行号已按 384 行版本重生成**，含本轮 15 项修改说明 |
| `review/MANUAL_GOLDEN_CASES.md` | 含 DCP position 32/33、Hybrid position 48、Tile 1023–1029 完整手算 |
| `review/REVIEW_CHECKLIST.md` | 已更新，**全部未勾选** |
| `review/FINAL_PRE_PR_CHECKLIST.md` | 已更新，**全部未勾选** |
| `review/REPRO_COMMANDS.md` | 已更新（含新 SHA / 行数 / 变异复跑步骤） |
| `review/REVIEW_DIFF.patch` | 已按最新基线重新生成 |
| `review/HUMAN_FINAL_COMMANDS.md` | **本轮新建**：提交人最小必做命令集 |
| `validation/mutation_final.txt` | **本轮新建**：变异回归日志（含 EXPECTED FAILURE 标注） |
| `validation/` | 日志已刷新 |

## 7. 仍需你（提交人）本人完成（Agent 不可代劳）
命令见 `review/HUMAN_FINAL_COMMANDS.md`。

1. **逐行阅读最终 Diff**（384 行，基线 `a1657a0`）
2. **亲自运行** `pytest tests/v1/worker/test_block_table_slot_mapping.py -v`
3. **亲自运行** `pre-commit run --files tests/v1/worker/test_block_table_slot_mapping.py`
4. **手算复核** 3 类 Golden（`MANUAL_GOLDEN_CASES.md`）
5. **确认 Git identity / DCO**（当前 `CiaranCw <CiaranCw@users.noreply.github.com>` —— 我不代确认）
6. **确认实际使用过的 AI 工具**并据此填写 `Assisted-by:` trailer（我不猜测工具名）
7. **把 PR Body 的 AI 披露段改成真实完成时**
8. 最后一次查重
9. `git commit -s` → push 到你的 fork → `gh pr create`（模板见 `REPRO_COMMANDS.md` §10）
10. 在 GitHub 页面复核最终 Diff 与 PR Body

## 8. 环境与隔离
- 干净工作区 `/home/ciaran/projects/vllm-m13-upstream`（独立 clone + 独立 `.venv`，`uv` 管理），与带中文学习注释的 `/home/ciaran/projects/vllm-src`（22 脏文件、shallow、pinned v0.24.0）**完全隔离**。
- 环境：Python 3.12.13 / torch 2.13.0+cu130 / triton 3.7.1 / ruff 0.14.0 / RTX 5080 sm_120 / driver 610.62。
- GPU：使用前查 `nvidia-smi`（4514MiB / 15%，无其他计算进程），**未 kill 任何进程、未 reset**。
- 本轮**未创建 commit、未 push、未建 PR、未加 fork remote、未动任何 Issue/PR、未改产品代码、未勾选任何人工 checkbox、未猜测任何 AI 工具名**。
