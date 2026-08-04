### Purpose

`BlockTable.compute_slot_mapping` maps logical token positions to physical KV cache slots. An off-by-one in this path silently corrupts KV cache contents instead of raising, which is hard to trace back from garbled output. It also carries several easy-to-break branches: hybrid KV/kernel block sizes, decode context parallel (DCP) interleaving, CUDA graph padding, and the `SlotMappingMode.NONE` short circuit used by Mamba-style state caches.

On the base revision before this change, the tests directory had no direct coverage of `BlockTable.compute_slot_mapping`:

```console
$ git grep -l "compute_slot_mapping" 24c939c47df59d0d3e9af2fe63046e21e65b5924 -- tests/
$ # (no matches)
```

`tests/v1/worker/test_gpu_block_table.py` covers the MRV2 `BlockTables` staged-write and `move_row` behaviour, and `tests/v1/worker/test_gpu_model_runner.py::test_hybrid_block_table_initialization` covers hybrid block table *initialization* (asserting the expanded row against `map_to_kernel_blocks` itself). Neither computes a slot mapping.

This PR adds `tests/v1/worker/test_block_table_slot_mapping.py`, exercising `BlockTable.compute_slot_mapping`. It is test-only: no production code is modified.

### Test plan

```bash
.venv/bin/python -m pytest tests/v1/worker/test_block_table_slot_mapping.py -v
.venv/bin/pre-commit run --files tests/v1/worker/test_block_table_slot_mapping.py
```

### Test result

```
18 passed, 14 warnings in 5.34s
```

`.venv/bin/pre-commit run --files tests/v1/worker/test_block_table_slot_mapping.py`: all applicable hooks pass (`ruff check`, `ruff format`, `typos`, `mypy`, SPDX headers, and the rest).

Environment: RTX 5080 (sm_120), driver 610.62, Python 3.12.13, torch 2.13.0+cu130, triton 3.7.1, against `24c939c47df59d0d3e9af2fe63046e21e65b5924`.

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

The reference implementation independently expands KV manager block IDs into kernel block IDs and rebuilds the expected mapping from the inputs passed to `BlockTable`. It does not call `BlockTable.map_to_kernel_blocks`, does not read the block table that `add_row` produced, and does not reuse the kernel's indexing implementation. This matters in practice: with a deliberate off-by-one injected into `map_to_kernel_blocks`, an earlier draft whose reference reused that helper still passed the hybrid cases, while the current version fails all of them. The plain hand-computed cases assert the expected slots against both the kernel output and the reference, so neither can drift unnoticed.

### Why this is not a duplicate

- No open or merged PR adds direct tests for `BlockTable.compute_slot_mapping` (searched `compute_slot_mapping`, `BlockTable slot mapping`, `test_block_table`, `DCP slot mapping`, `hybrid block table`).
- The open slot-mapping PRs are unrelated in scope: #41270 / #41272 remove slot mapping from attention *metadata*; #46446 / #45245 / #43879 pre-compile the kernel during warmup; #47924 is an RFC/prototype.
- Open PR #39591 adds a `tests/v1/worker/test_block_table.py` covering a different behaviour: zeroing the block table row tail to fix concurrent variable-length prefill non-determinism (#39589). Its tests exercise `append_row` / `add_row` / `move_row` / `clear_row` and never call `compute_slot_mapping`. To keep the two independent, this PR uses the separate file name `test_block_table_slot_mapping.py`; happy to merge the files if maintainers prefer a single module.
- The closed #30683 by @pisceskkk previously explored direct tests for this area, and credit for identifying the testing direction goes there. It was not merged and targets an older API: it used `compute_slot_mapping(req_indices, positions)` and assumed slot mapping was sharded by `pcp_world_size * dcp_world_size`. On current main the signature is `(num_reqs, query_start_loc, positions)` and the kernel is sharded by DCP only, so these tests were written against the current API and additionally cover hybrid blocks, `SlotMappingMode`, and tile-crossing padding.

### AI assistance disclosure

AI assistance (Claude Code) was used to research the current implementation, draft and refine the tests, and prepare validation materials. I reviewed every changed line, verified the hand-computed golden cases by hand, and reproduced the test results locally (18 passed; changed-file pre-commit clean). I take responsibility for this change.
