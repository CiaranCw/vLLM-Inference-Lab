# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``BlockTable.compute_slot_mapping``.

These tests cover the ``_compute_slot_mapping_kernel`` reachable through the
public ``BlockTable.compute_slot_mapping`` entry point, which maps logical
token positions to physical KV-cache slots.

Coverage:
  * Non-CP correctness edge cases: block boundaries, cross-block sequences,
    non-contiguous / shuffled physical block numbers, ragged multi-request
    batches, and non-zero starting positions (prefix).
  * CUDA-graph padding: the region ``[num_tokens, max_num_batched_tokens)``
    must be filled with ``PAD_SLOT_ID``.
  * Context-Parallel (DCP/PCP) interleave, including the DCP=4/PCP=2 cases
    with interleave sizes 1 and 16 (superseding stale PR #30683, which was
    written against an older ``compute_slot_mapping(req_indices, positions)``
    signature; the current signature is ``(num_reqs, query_start_loc,
    positions)``).

The reference (`_reference_slot_mapping`) is an independent, plain-numpy
re-implementation of the kernel semantics, used as an oracle.

NOTE: The CP interleave logic only exists in the GPU Triton kernel (the CPU
fallback in ``vllm/utils/cpu_triton_utils.py`` asserts world_size == 1), so
these tests require CUDA.
"""

import numpy as np
import pytest
import torch

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable

BLOCK_SIZE = 16
MAX_NUM_REQS = 8
MAX_NUM_BLOCKS_PER_REQ = 64
MAX_NUM_BATCHED_TOKENS = 2048

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="slot mapping Triton kernel requires CUDA"
)


def _reference_slot_mapping(
    positions: np.ndarray,
    block_table: np.ndarray,
    query_start_loc: np.ndarray,
    block_size: int,
    max_num_tokens: int,
    *,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    interleave: int = 1,
) -> np.ndarray:
    """Plain-numpy oracle mirroring ``_compute_slot_mapping_kernel``.

    slot = block_table[req, pos // vbs] * block_size + local_offset
      where vbs = block_size * cp_world_size, and, under context parallelism,
      a token is PAD_SLOT_ID on ranks that do not own it.
    """
    slot = np.full((max_num_tokens,), PAD_SLOT_ID, dtype=np.int64)
    vbs = block_size * cp_world_size
    for r in range(len(query_start_loc) - 1):
        s, e = int(query_start_loc[r]), int(query_start_loc[r + 1])
        if e <= s:
            continue
        pos = positions[s:e].astype(np.int64)
        blk_idx = pos // vbs
        blk_num = block_table[r, blk_idx].astype(np.int64)
        voff = pos - blk_idx * vbs
        is_local = (voff // interleave) % cp_world_size == cp_rank
        loff = (voff // (cp_world_size * interleave)) * interleave + (voff % interleave)
        sl = blk_num * block_size + loff
        slot[s:e] = np.where(is_local, sl, PAD_SLOT_ID)
    return slot


def _make_block_table(
    interleave: int = 1,
    pcp_world_size: int = 1,
    pcp_rank: int = 0,
    dcp_world_size: int = 1,
    dcp_rank: int = 0,
) -> BlockTable:
    bt = BlockTable(
        block_size=BLOCK_SIZE,
        max_num_reqs=MAX_NUM_REQS,
        max_num_blocks_per_req=MAX_NUM_BLOCKS_PER_REQ,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        pin_memory=False,
        device=torch.device("cuda"),
        kernel_block_size=BLOCK_SIZE,
        cp_kv_cache_interleave_size=interleave,
    )
    # compute_slot_mapping reads these attributes directly to derive
    # TOTAL_CP_WORLD_SIZE / TOTAL_CP_RANK; set them explicitly instead of
    # initializing real distributed groups.
    bt.pcp_world_size = pcp_world_size
    bt.pcp_rank = pcp_rank
    bt.dcp_world_size = dcp_world_size
    bt.dcp_rank = dcp_rank
    return bt


def _run(
    bt: BlockTable,
    block_id_rows: list[list[int]],
    positions_per_req: list[list[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Populate the block table, run the kernel, and return
    (actual_full, reference_full, query_start_loc) over the padded length."""
    num_reqs = len(block_id_rows)
    for r, block_ids in enumerate(block_id_rows):
        bt.add_row(block_ids, r)
    bt.commit_block_table(num_reqs)

    lens = [len(p) for p in positions_per_req]
    query_start_loc = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    positions = np.concatenate(positions_per_req).astype(np.int64)

    bt.compute_slot_mapping(
        num_reqs,
        torch.from_numpy(query_start_loc).cuda(),
        torch.from_numpy(positions).cuda(),
    )
    actual = bt.slot_mapping.gpu[:MAX_NUM_BATCHED_TOKENS].cpu().numpy()

    # Build a padded block_table array for the oracle.
    max_cols = max(len(r) for r in block_id_rows)
    bt_np = np.zeros((num_reqs, max_cols), dtype=np.int64)
    for r, block_ids in enumerate(block_id_rows):
        bt_np[r, : len(block_ids)] = block_ids
    ref = _reference_slot_mapping(
        positions,
        bt_np,
        query_start_loc,
        BLOCK_SIZE,
        MAX_NUM_BATCHED_TOKENS,
        cp_world_size=bt.pcp_world_size * bt.dcp_world_size,
        cp_rank=bt.pcp_rank * bt.dcp_world_size + bt.dcp_rank,
        interleave=bt.cp_kv_cache_interleave_size,
    )
    return actual, ref, query_start_loc


# --------------------------------------------------------------------------- #
# Non-CP correctness edge cases
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize(
    "name,block_id_rows,positions_per_req",
    [
        # basic multi-request, distinct contiguous blocks
        ("basic_multi", [[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2], [0, 1]]),
        # single token
        ("single_token", [[9]], [[0]]),
        # single request spanning multiple blocks
        ("single_request_cross_block", [[3, 7, 1, 5]], [list(range(50))]),
        # first / last position within a block
        (
            "block_boundaries",
            [[8, 2, 5]],
            [[0, BLOCK_SIZE - 1, BLOCK_SIZE, 2 * BLOCK_SIZE - 1, 2 * BLOCK_SIZE]],
        ),
        # non-contiguous physical block numbers
        (
            "non_contiguous_physical",
            [[100, 3, 57, 12, 88, 1, 44, 9]],
            [list(range(120))],
        ),
        # ragged multi-request with varying lengths
        (
            "ragged_multi",
            [[1, 2, 3, 4], [10, 11], [20, 21, 22, 23, 24], [30], [40, 41, 42]],
            [list(range(60)), list(range(20)), list(range(70)), [0], list(range(33))],
        ),
        # non-zero starting positions (prefix already cached)
        ("prefix_offset", [[5, 6, 7, 8, 9, 10]], [list(range(40, 90))]),
    ],
)
def test_compute_slot_mapping_non_cp(name, block_id_rows, positions_per_req):
    bt = _make_block_table()
    actual, ref, qsl = _run(bt, block_id_rows, positions_per_req)
    num_tokens = int(qsl[-1])
    np.testing.assert_array_equal(actual, ref, err_msg=name)
    # CUDA-graph padding: everything past the real tokens must be PAD_SLOT_ID.
    assert (actual[num_tokens:] == PAD_SLOT_ID).all(), f"{name}: padding region"


@requires_cuda
@pytest.mark.parametrize(
    "block_id_rows,positions_per_req,expected",
    [
        # Two requests, contiguous blocks. block_size=16.
        #   req0 pos {0,1,2} → block 0 → slots {0,1,2}
        #   req1 pos {0,1}   → block 4 → slots {64,65}
        ([[0, 1, 2, 3], [4, 5, 6, 7]], [[0, 1, 2], [0, 1]], [0, 1, 2, 64, 65]),
        # Cross-block with non-contiguous physical blocks [3,7,1,5].
        #   pos 0,15  → block 3 → 48,63     pos 16,31 → block 7 → 112,127
        #   pos 32,47 → block 1 → 16,31     pos 48    → block 5 → 80
        (
            [[3, 7, 1, 5]],
            [[0, 15, 16, 31, 32, 47, 48]],
            [48, 63, 112, 127, 16, 31, 80],
        ),
    ],
)
def test_compute_slot_mapping_golden(block_id_rows, positions_per_req, expected):
    """Anchor against hand-computed slot values (independent of the oracle)."""
    bt = _make_block_table()
    actual, _, qsl = _run(bt, block_id_rows, positions_per_req)
    num_tokens = int(qsl[-1])
    np.testing.assert_array_equal(
        actual[:num_tokens], np.array(expected, dtype=np.int64)
    )
    assert (actual[num_tokens:] == PAD_SLOT_ID).all()


# --------------------------------------------------------------------------- #
# Context-Parallel interleave (supersedes PR #30683 under the current API)
# --------------------------------------------------------------------------- #
@requires_cuda
@pytest.mark.parametrize("pcp_rank", [0, 1])
@pytest.mark.parametrize("dcp_rank", [0, 1, 2, 3])
@pytest.mark.parametrize("interleave", [1, 16])
def test_compute_slot_mapping_dcp4_pcp2(pcp_rank, dcp_rank, interleave):
    bt = _make_block_table(
        interleave=interleave,
        pcp_world_size=2,
        pcp_rank=pcp_rank,
        dcp_world_size=4,
        dcp_rank=dcp_rank,
    )
    # One request, enough positions to exercise several interleave chunks.
    block_id_rows = [list(range(1, MAX_NUM_BLOCKS_PER_REQ + 1))]
    positions_per_req = [list(range(96))]
    actual, ref, qsl = _run(bt, block_id_rows, positions_per_req)
    num_tokens = int(qsl[-1])
    np.testing.assert_array_equal(
        actual,
        ref,
        err_msg=f"dcp4/pcp2 interleave={interleave} "
        f"pcp_rank={pcp_rank} dcp_rank={dcp_rank}",
    )
    assert (actual[num_tokens:] == PAD_SLOT_ID).all()


@requires_cuda
def test_compute_slot_mapping_cp_degenerates_to_non_cp():
    """world_size == 1 must reproduce the plain slot formula."""
    bt = _make_block_table(pcp_world_size=1, pcp_rank=0, dcp_world_size=1, dcp_rank=0)
    actual, ref, _ = _run(bt, [[7, 3, 9, 2]], [list(range(60))])
    np.testing.assert_array_equal(actual, ref)
