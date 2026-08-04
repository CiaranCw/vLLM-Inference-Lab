# M13 · 提交人审查清单（REVIEW_CHECKLIST）

> **规则**：全部复选框初始为**未勾选**。**只能由你（提交人）本人勾选** —— AI 不得代勾。
> vLLM `AGENTS.md`：「Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.」
>
> 对象：`tests/v1/worker/test_block_table.py`，**384 行 / 18 tests**
> SHA256：`b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`
> base：upstream/main **`0b1c151cbcacd12e207f2a91e4ca9f5b6961ec8a`**
>
> 提交前的最终确认另见 `FINAL_PRE_PR_CHECKLIST.md`。

## A. 理解与背景
- [ ] 我理解该 PR 为什么不是重复工作
- [ ] 我读过 current `BlockTable.compute_slot_mapping`（`vllm/v1/worker/block_table.py:182`）
- [ ] 我读过 current `_compute_slot_mapping_kernel`（同文件 `:379-442`）
- [ ] 我理解 KV allocation block 与 kernel block 的区别（`kv_cache_block_size` / `block_size` / `blocks_per_kv_block`）
- [ ] 我理解 DCP 为什么不包含 PCP（kernel 只接收 `dcp_world_size` / `dcp_rank`，见 `:206-207`）

## B. 手算验证（不看代码也能独立复核）
- [ ] 我能手算普通 Golden（`[80, 95, 96, 127]`）
- [ ] 我能手算多请求 Golden（`[147, 68]`，理解每请求独立行）
- [ ] 我能手算 DCP Golden，含**跨虚拟块**的 `position 33 → slot 96`
- [ ] 我能手算 Hybrid Golden，含**跨第二个 kernel 子块**的 `position 48 → slot 304`
- [ ] 我能手算 Tile 边界锚点 `position 1023 → 1039`、`1024 → 1040`
- [ ] 我确认 hybrid `position 47` 落在 kernel 块 **18**（而非 19）

## C. 测试设计正确性
- [ ] 我确认 Oracle 不调用产品 Mapping Helper（`grep -n "map_to_kernel_blocks(" …` 无输出）
- [ ] 我确认 Oracle 不读取 `block_table.np` / `block_table.gpu` 作为期望值
- [ ] 我理解普通 Golden 为什么要**同时**锚定 Actual 与 Reference（三方证据）
- [ ] 我理解 `_expand_kernel_block_ids` 独立存在的必要性（避免两边一起错）
- [ ] 我认可变异测试证据：注入 `map_to_kernel_blocks +1` → 3 个 hybrid 测试全部失败
- [ ] 我理解 Padding Program 为什么使用 `num_reqs + 1`（最后一个 program 专门填 padding）
- [ ] 我理解 1024 Tile Case（1030 tokens 跨 tile 且非整数倍，buffer 2100 使尾部再跨一 tile）
- [ ] 我理解 `SlotMappingMode.NONE` 为什么不写 Buffer（`:189-192` 直接 return）
- [ ] 我理解为什么必须 `commit_block_table`（`add_row` 只写 CPU 侧）
- [ ] 我认可 `_compute()` 中请求行数一致性 assert 的必要性
- [ ] 我理解 DCP ownership 用 `np.stack(..., axis=0)` 的意图（rank 为第 0 维、shape 一致）

## D. 逐行审查与复现
- [ ] 我逐行读完最终测试文件（384 行，不是旧版本）
- [ ] 我亲自运行目标 Pytest 并看到 18 passed
- [ ] 我亲自运行 Changed-file Pre-commit 并看到全绿
- [ ] 我确认产品代码 Diff 为空（`git diff --name-only HEAD -- vllm/ csrc/`）

## E. 提交合规（PR 创建前最后确认）
- [ ] 我确认 DCO 姓名和邮箱（`git config user.name` / `user.email`）
- [ ] 我确认 PR 中披露的 AI 工具真实（草稿为不点名占位，需我定稿）
- [ ] 我确认 Commit Trailer（`Signed-off-by:` 必需；AI 归属 trailer 按 AGENTS.md）
- [ ] 我已把 PR Body 中「submitter will review…」的未来时改为真实完成时
- [ ] 我最后一次搜索重复 Open PR
- [ ] 我确认 PR Body 中所有陈述均为当前事实（SHA、测试数、结果）

---

## 若任一项无法勾选
不要创建 PR。把疑问记在此处，我们先解决：

```
（在此记录你的疑问 / 发现的问题）
```
