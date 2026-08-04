# M13 · 提交人审查清单（REVIEW_CHECKLIST）

> **规则**：全部复选框初始为**未勾选**。**只能由你（提交人）本人勾选**——AI 不得代勾。
> vLLM `AGENTS.md` 明确：「Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.」
> 只有本清单全部勾选后，才可进入创建 PR 的步骤。

## A. 理解与背景
- [ ] 我理解该 PR 为什么不是重复工作
- [ ] 我读过 current `BlockTable.compute_slot_mapping`（`vllm/v1/worker/block_table.py:182`）
- [ ] 我读过 current `_compute_slot_mapping_kernel`（同文件 `:379-442`）
- [ ] 我理解 KV allocation block 与 kernel block 的区别（`kv_cache_block_size` / `block_size` / `blocks_per_kv_block`）
- [ ] 我理解 DCP 为什么不包含 PCP（kernel 只接收 `dcp_world_size` / `dcp_rank`，见 `:206-207`）

## B. 手算验证（不看代码也能独立复核）
- [ ] 我能手算普通 Golden（`[80, 95, 96, 127]`）
- [ ] 我能手算 DCP Golden（`[-1, 80, -1, 81, -1, 82]`）
- [ ] 我能手算 Hybrid Golden（`[160, 175, 176, 191, 288, 303]`）
- [ ] 我确认 `position 47` 在 hybrid 用例中落在 kernel 块 **18**（而非 19）

## C. 测试设计正确性
- [ ] 我确认 Oracle 不调用产品 Mapping Helper（`grep -n "map_to_kernel_blocks" tests/v1/worker/test_block_table.py` 无输出）
- [ ] 我确认 Oracle 不读取 `block_table.np` / `block_table.gpu` 作为期望值
- [ ] 我理解 Padding Program 为什么使用 `num_reqs + 1`（最后一个 program 专门填 padding）
- [ ] 我理解 1024 Tile Case（1030 tokens 跨 tile 且非整数倍，buffer 2100 让尾部再跨一 tile）
- [ ] 我理解 `SlotMappingMode.NONE` 为什么不写 Buffer（`:189-192` 直接 return）
- [ ] 我理解为什么必须 `commit_block_table`（`add_row` 只写 CPU 侧）
- [ ] 我认可「注入 `map_to_kernel_blocks` 错误后 3 个 hybrid 测试全部失败」这一独立性证据

## D. 逐行审查与复现
- [ ] 我逐行读完测试文件（347 行）
- [ ] 我亲自运行目标 Pytest 并看到全部通过
- [ ] 我亲自运行 Changed-file Pre-commit 并看到全绿
- [ ] 我确认产品代码 Diff 为空（`git diff --name-only HEAD -- vllm/ csrc/`）

## E. 提交合规（PR 创建前最后确认）
- [ ] 我确认 DCO 姓名和邮箱（`git config user.name` / `user.email`）
- [ ] 我确认 PR 中披露的 AI 工具真实（草稿中为占位表述，需我按实际情况定稿）
- [ ] 我确认 Commit Trailer（`Signed-off-by:` 必需；AI 归属 trailer 按 AGENTS.md）
- [ ] 我已把 PR Body 中「submitter will review…」的未来时改为真实完成时
- [ ] 我最后一次搜索重复 Open PR（见 `REPRO_COMMANDS.md` §7）
- [ ] 我确认 PR Body 中所有陈述均为当前事实（SHA、测试数、结果）

---

## 若任一项无法勾选
不要创建 PR。把疑问记在此处，我们先解决：

```
（在此记录你的疑问 / 发现的问题）
```
