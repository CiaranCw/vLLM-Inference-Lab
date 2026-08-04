# PR 草稿：Add unit tests for `BlockTable.compute_slot_mapping`

> **状态：READY FOR HUMAN REVIEW BEFORE PR** —— 尚未 push、尚未创建 PR。
> 以下英文正文可直接作为 PR body；标题前缀 `[CI/Build]` 取自 `docs/contributing/README.md` 的合法分类（**`[Test]` 不在合法列表内**）。
> 提交前请**逐行通读** `tests/v1/worker/test_block_table.py`：AGENTS.md 明确「纯 code-agent PR 不允许，提交人须能端到端理解并为改动负责」。

---

## 建议标题
```
[CI/Build] Add unit tests for BlockTable.compute_slot_mapping
```

---

## PR Body（英文，可直接粘贴）

### Purpose

`BlockTable.compute_slot_mapping` maps logical token positions to physical KV
cache slots. An off-by-one in this path silently corrupts KV cache contents
instead of raising, which is hard to trace back from garbled output. It also
carries several easy-to-break branches: hybrid KV/kernel block sizes, decode
context parallel (DCP) interleaving, CUDA graph padding, and the
`SlotMappingMode.NONE` short circuit used by Mamba-style state caches.

Today this path has no direct test coverage:

```console
$ grep -rln "compute_slot_mapping\b" tests/
$ # (no matches)
```

`tests/v1/worker/test_gpu_block_table.py` covers the MRV2 `BlockTables`
staged-write and `move_row` behaviour, and
`tests/v1/worker/test_gpu_model_runner.py::test_hybrid_block_table_initialization`
covers hybrid block table *initialization* (asserting the expanded row against
`map_to_kernel_blocks` itself). Neither computes a slot mapping.

This PR adds `tests/v1/worker/test_block_table.py`, testing through the public
`BlockTable` API. It is test-only: no production code is modified.

### Test plan

```bash
python -m pytest tests/v1/worker/test_block_table.py -v
```

### Test result

```
18 passed, 14 warnings in 5.34s
```

Environment: RTX 5080 (sm_120), driver 610.62, Python 3.12.13,
torch 2.13.0+cu130, triton 3.7.1, built with `VLLM_USE_PRECOMPILED=1`
against `cb8104839c141609d99f1254459ef3a4f1bd4263`.

Pre-commit on the changed file passes (`ruff check`, `ruff format`, `typos`,
`mypy`, SPDX and the remaining applicable hooks). `pre-commit run --all-files`
is clean except a pre-existing `shellcheck` finding in
`docs/pre_run_check.sh` (SC2317), untouched by this PR.

Note on the neighbouring suite: `tests/v1/worker/test_gpu_block_table.py`
reports 3 failures on this machine with `RuntimeError: UVA is not available`
(WSL2). These reproduce on an unmodified checkout with this PR's file removed,
so they are environmental and unrelated. The new tests use `pin_memory=False`
and are unaffected.

### What is covered

| Area | Cases |
| --- | --- |
| Hand-computed golden slots (independent of the reference implementation) | plain mapping (2), DCP world size 2 (1), hybrid 32/16 (1) |
| Block boundaries | first/last position in a block, spanning many blocks |
| Physical layout | non-contiguous block ids |
| Batching | single token, ragged multi-request, non-zero prefix offset |
| CUDA graph padding | 1030 tokens crossing the kernel's 1024-token tile, padded tail crossing a second tile |
| Decode context parallel | world size {2, 4} × interleave {1, 4}, every rank, plus "each token is owned by exactly one rank" |
| Hybrid blocks | KV/kernel sizes (32, 16) and (64, 16), end to end through `compute_slot_mapping` |
| `SlotMappingMode.NONE` | buffer is left untouched (checked with a sentinel distinct from `PAD_SLOT_ID`) |

The reference implementation rebuilds the expected mapping step by step from the
public input semantics (virtual block → DCP ownership → local offset → KV to
kernel block expansion → physical block → slot → `PAD_SLOT_ID`) rather than
mirroring the kernel's index arithmetic. The hand-computed golden cases guard
against the reference and the kernel being wrong in the same way; both golden
groups and the reference caught a real arithmetic mistake in an earlier draft of
these tests.

### Why this is not a duplicate

- No open or merged PR adds direct tests for this path (searched `slot mapping`,
  `block_table test`, `test_block_table`).
- The open slot-mapping PRs are unrelated in scope: #41270 / #41272 remove
  slot mapping from attention *metadata*; #46446 / #45245 / #43879 pre-compile
  the kernel during warmup; #47924 is an RFC/prototype.
- The closed, stale #30683 by @pisceskkk proposed unit tests for
  `block_table.py` in the same spirit, and credit for the direction goes there.
  That patch cannot run on current main: it targeted the older
  `compute_slot_mapping(req_indices, positions)` signature and assumed slot
  mapping was sharded by `pcp_world_size * dcp_world_size`. On current main the
  signature is `(num_reqs, query_start_loc, positions)` and the kernel is
  sharded by DCP only, so these tests were written against the current API and
  additionally cover hybrid blocks, `SlotMappingMode`, and tile-crossing
  padding.

### AI assistance disclosure

AI assistance (Claude) was used to research the current implementation, draft
these tests, and run the validation described above. The submitter reviewed
every line and reproduced the test results locally.

---

## 本仓库对应关系
- 已验证测试文件：`notes/m13/pr_draft/test_block_table.py`（与干净树中 `tests/v1/worker/test_block_table.py` **sha256 完全一致**：`829c0e37...`）。
- 研究与设计留痕：`notes/m13/pr_slot_mapping_test.md`；断点与验证：`notes/m13/STATUS.md`、`notes/m13/validation/`。
- Commit 需 `git commit -s`（DCO），并按 AGENTS.md 加 AI 归属 trailer（如 `Co-authored-by:` / `Assisted-by:`）。**当前尚未创建 commit**，等确认 sign-off 身份。
