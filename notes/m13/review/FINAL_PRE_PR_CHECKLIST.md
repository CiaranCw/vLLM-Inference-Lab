# M13 · 提交前最终人工确认清单（FINAL_PRE_PR_CHECKLIST）

> **这是创建 PR 之前的最后一道关卡。全部复选框初始未勾选，只能由你（提交人）本人勾选；AI 不得代勾。**
>
> 最终对象：`tests/v1/worker/test_block_table.py`
> **行数**：384　**Tests**：18
> **SHA256**：`b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`
> **base**：upstream/main `a1657a0235a77ce85f8d348c83ff4adb29918b71`（记录于 2026-08-04 09:2x UTC）

> **Agent 已自动验证 vs 你需人工确认**：Agent 已自动跑过 pytest / pre-commit / ruff / diff /
> 变异回归，并独立复核过三个手算锚点——但这些**只是 Agent 结果**（`Agent verified`），
> **不能替代**你的人工确认（`Human pending`）。AGENTS.md 要求提交人端到端负责。
> 因此下列每一项仍需**你本人**执行并勾选；**Agent 未勾选任何一项**。

## 基线与内容
- [ ] 我确认 upstream/main SHA 仍是提交前最新基线（已重新 `git fetch origin main` 核对）
- [ ] 我逐行读过**最终**测试 Diff，而不是旧版本
- [ ] 我确认最终测试 SHA256 与本文件顶部一致
- [ ] 我确认 upstream diff 只包含 `tests/v1/worker/test_block_table.py`

## 设计理解
- [ ] 我能解释 Actual、Reference 和 Golden 为什么必须独立
- [ ] 我能解释为什么 Reference 不得调用 `BlockTable.map_to_kernel_blocks`

## 手算复核
- [ ] 我手算了普通 Golden（`[80, 95, 96, 127]` 与 `[147, 68]`）
- [ ] 我手算了 DCP `position 33 → slot 96`
- [ ] 我手算了 Hybrid `position 48 → slot 304`
- [ ] 我手算了 Tile boundary `position 1023 → 1039`

## 亲自复现
- [ ] 我亲自运行目标 Pytest（看到 18 passed）
- [ ] 我亲自运行 Changed-file Pre-commit（看到全绿）
- [ ] 我确认产品代码没有修改（`git diff --name-only HEAD -- vllm/ csrc/` 无输出）

## 提交合规
- [ ] 我完成最后一次重复 PR 搜索
- [ ] 我确认 DCO 姓名和邮箱
- [ ] 我确认实际使用的 AI 工具（并据此改写 PR Body 的披露段）
- [ ] 我确认 Commit Trailer
- [ ] 我把 PR Body 的人工审查表述改成真实完成时
- [ ] 我确认 PR Body 中的测试结果与本地真实输出一致

---

## 全部勾选后才执行（命令见 `REPRO_COMMANDS.md` §9）
```
git commit -s → push 到你的 fork → gh pr create
```
**在此之前，我（AI）不会执行任何 commit / push / PR 操作。**
