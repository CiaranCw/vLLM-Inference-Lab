# PR 草稿：Add unit tests for BlockTable slot mapping

> **状态：READY FOR FINAL HUMAN CONFIRMATION BEFORE COMMIT** —— 尚未创建 Commit、尚未 Push、尚未创建 PR（远程 CI 亦未运行）。
> **提交前你必须做两件事**：① 逐行审查测试文件（AGENTS.md：纯 code-agent PR 不允许，提交人须端到端负责）；② 把下面 Body 中 **AI 披露段的未来时改成真实完成时**，并按实际使用的工具填名。
> 审查材料见 `notes/m13/review/`。

---

## 建议标题
```
[Core] Add unit tests for BlockTable slot mapping
```
> 分类依据 `docs/contributing/README.md:233-234`：`[Core]` = 「changes in the core vLLM logic」。
> 不用 `[Test]`（**不在合法列表内**）；不用 `[CI/Build]`——本 PR 不改 CI、不改 Build、不改测试基础设施，保护的是 V1 核心 KV BlockTable 行为。

---

## PR Body（英文，可直接粘贴；AI 披露段需你定稿）

### Purpose

`BlockTable.compute_slot_mapping` maps logical token positions to physical KV
cache slots. An off-by-one in this path silently corrupts KV cache contents
instead of raising, which is hard to trace back from garbled output. It also
carries several easy-to-break branches: hybrid KV/kernel block sizes, decode
context parallel (DCP) interleaving, CUDA graph padding, and the
`SlotMappingMode.NONE` short circuit used by Mamba-style state caches.

Today this path has no direct test coverage:

```console
$ git grep -l "compute_slot_mapping" -- tests/
$ # (no matches)
```

`tests/v1/worker/test_gpu_block_table.py` covers the MRV2 `BlockTables`
staged-write and `move_row` behaviour, and
`tests/v1/worker/test_gpu_model_runner.py::test_hybrid_block_table_initialization`
covers hybrid block table *initialization* (asserting the expanded row against
`map_to_kernel_blocks` itself). Neither computes a slot mapping.

This PR adds `tests/v1/worker/test_block_table.py`, exercising
`BlockTable.compute_slot_mapping`. It is test-only: no production code is
modified.

### Test plan

```bash
python -m pytest tests/v1/worker/test_block_table.py -v
pre-commit run --files tests/v1/worker/test_block_table.py
```

### Test result

```
18 passed, 14 warnings in 5.24s
```

`pre-commit run --files tests/v1/worker/test_block_table.py`: all applicable
hooks pass (`ruff check`, `ruff format`, `typos`, `mypy`, SPDX headers, and the
rest).

Environment: RTX 5080 (sm_120), driver 610.62, Python 3.12.13,
torch 2.13.0+cu130, triton 3.7.1, against
`0b1c151cbcacd12e207f2a91e4ca9f5b6961ec8a`.

### What is covered

| Area | Cases |
| --- | --- |
| Hand-computed golden slots, anchoring both the kernel output and the reference | plain mapping (2: single request block boundaries, multiple request rows) |
| Block boundaries, ragged batches, prefix offsets | single token, first/last position in a block, spanning many blocks, non-contiguous block ids, ragged multi-request, absolute non-zero positions after a cached prefix (6) |
| CUDA graph padding across a Triton tile | 1030 tokens crossing the kernel's 1024-token tile with a hand-computed anchor on positions 1023-1029, padded tail crossing a second tile |
| Decode context parallel partitioning | world size {2, 4} × interleave {1, 4}, every rank, plus "each token is owned by exactly one rank" |
| Hand-computed DCP golden crossing a virtual block | world size 2, rank 1, positions up to 33 so the second KV block is used |
| Hybrid blocks | KV/kernel sizes (32, 16) and (64, 16), end to end through `compute_slot_mapping` |
| Hand-computed hybrid golden crossing the second kernel block | 32/16 blocks, positions up to 48 so both kernel sub-blocks of a KV block are used |
| `SlotMappingMode.NONE` | buffer is left untouched (checked with a sentinel distinct from `PAD_SLOT_ID`) |

The reference implementation independently expands KV manager block IDs into
kernel block IDs and rebuilds the expected mapping from the inputs passed to
`BlockTable`. It does not call `BlockTable.map_to_kernel_blocks`, does not read
the block table that `add_row` produced, and does not reuse the kernel's
indexing implementation. This matters in practice: with a deliberate off-by-one
injected into `map_to_kernel_blocks`, an earlier draft whose reference reused
that helper still passed the hybrid cases, while the current version fails all
of them. The plain hand-computed cases assert the expected slots against both
the kernel output and the reference, so neither can drift unnoticed.

### Why this is not a duplicate

- No open or merged PR adds direct tests for this path (searched
  `compute_slot_mapping`, `block table slot mapping`, `test_block_table`,
  `DCP slot mapping`, `hybrid block table`).
- The open slot-mapping PRs are unrelated in scope: #41270 / #41272 remove
  slot mapping from attention *metadata*; #46446 / #45245 / #43879 pre-compile
  the kernel during warmup; #47924 is an RFC/prototype.
- The closed #30683 by @pisceskkk previously explored direct tests for this
  area, and credit for identifying the testing direction goes there. It was not
  merged and targets an older API: it used
  `compute_slot_mapping(req_indices, positions)` and assumed slot mapping was
  sharded by `pcp_world_size * dcp_world_size`. On current main the signature is
  `(num_reqs, query_start_loc, positions)` and the kernel is sharded by DCP
  only, so these tests were written against the current API and additionally
  cover hybrid blocks, `SlotMappingMode`, and tile-crossing padding.

### AI assistance disclosure

<!-- Before submission: name the actual AI tools used, and change the sentence
     below to the past tense once the review has actually been completed. -->

AI assistance was used to research the current implementation, draft the tests,
and prepare validation materials. Before submission, the human submitter will
review every changed line, reproduce the test results locally, and take
responsibility for the change.

---

## 本仓库对应关系
- 已验证测试文件：`notes/m13/pr_draft/test_block_table.py`，与干净树 `tests/v1/worker/test_block_table.py` **sha256 一致**：`b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`
- 逐行审查材料：`notes/m13/review/{LINE_BY_LINE_REVIEW,MANUAL_GOLDEN_CASES,REVIEW_CHECKLIST,FINAL_PRE_PR_CHECKLIST,REPRO_COMMANDS}.md`、`REVIEW_DIFF.patch`
- 断点与验证证据：`notes/m13/STATUS.md`、`notes/m13/validation/`
- Commit 需 `git commit -s`（DCO）并按 AGENTS.md 加 AI 归属 trailer。**当前尚未创建 commit**（待你确认 sign-off 身份）。

## 刻意从正式 PR Body 中移除的无关基线问题（证据保留在 STATUS.md §5）
- `pre-commit run --all-files` 的 `shellcheck` 失败：`docs/pre_run_check.sh` SC2317，upstream 既有、本 PR 未触碰。
- 本机 `test_gpu_block_table.py` 3 failed：WSL2 `UVA is not available`，移除本 PR 文件后同样失败。
