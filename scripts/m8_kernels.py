"""M8 · Triton kernels + PyTorch references。

包含：
  基础练习（达概念即止）：vector_add / act_and_mul(SiLU-gate) / row_wise_softmax
  主验收：slot_mapping 生成 kernel（simplified `_compute_slot_mapping_kernel`）
          —— 消费 positions + block_table(+query_start_loc) 生成 slot_mapping，
             支持多请求、padding、Context Parallel interleave（参数化）。

术语（勿混淆，详见 notes/m8）：
  生成侧（本文件主对象）：slot = block_table[req, pos//block_size]*block_size + offset
  消费侧（reshape_and_cache，M4 已 trace，本模块不实现）：按 slot_mapping 把 K/V 散写进 KV cache

所有 @triton.jit 必须定义在 .py 文件里（Triton 需要读取源码）。直接运行本文件会对每个 kernel 做一次 launch 自测。
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton  # 复用 vllm 的 triton 封装，避免版本差异

PAD_ID = -1  # 与 vLLM PAD_SLOT_ID 一致


# ============================================================================
# 基础练习 1：vector add
# ============================================================================
@triton.jit # 第一次调用时，Triton 根据参数和 GPU 架构将它 JIT 编译成 GPU 代码
def vector_add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0) # 获取当前 Program 是第几个并行实例
    offs = pid * BLOCK + tl.arange(0, BLOCK) # 计算当前 Program 负责的数据下标
    mask = offs < n # 防止越界
    x = tl.load(x_ptr + offs, mask=mask) # 从 GPU 加载对应位置的数据
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask) # 加法 + 写回


# Kernel 需要 Python Wrapper 才能使用
def vector_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    n = x.numel()
    vector_add_kernel[(triton.cdiv(n, 1024),)](x, y, out, n, BLOCK=1024) # 使用向上取整除法计算需要的 Program 数量
    return out


# PyTorch 版本，定义正确结果
def vector_add_ref(x, y):
    return x + y


# ============================================================================
# 基础练习 2：fused activation = SiLU-gate（act_and_mul）
#   x: [n_rows, 2*d] -> out[:, :] = silu(x[:, :d]) * x[:, d:]
#   将 x 的最后一维一分为二
# ============================================================================
@triton.jit
def act_and_mul_kernel(x_ptr, out_ptr, d, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < d
    a = tl.load(x_ptr + row * (2 * d) + offs, mask=mask) # 每一行的长度为 2 * d，那么起始位置就是 row * (2 * d)，这里取到的是前半部分
    b = tl.load(x_ptr + row * (2 * d) + d + offs, mask=mask) # 添加偏移 d，从后半部分取
    silu = a * tl.sigmoid(a)
    tl.store(out_ptr + row * d + offs, silu * b, mask=mask)


def act_and_mul(x: torch.Tensor) -> torch.Tensor:
    n_rows, two_d = x.shape
    d = two_d // 2
    out = torch.empty((n_rows, d), dtype=x.dtype, device=x.device)
    act_and_mul_kernel[(n_rows,)](x, out, d, BLOCK=triton.next_power_of_2(d))
    return out


def act_and_mul_ref(x):
    d = x.shape[1] // 2
    return torch.nn.functional.silu(x[:, :d]) * x[:, d:]


# ============================================================================
# 基础练习 3：row-wise softmax（数值稳定，减 max；支持非 2 次幂列宽）
#            为了数值稳定，指数都减去一个最大值
# ============================================================================
@triton.jit
def softmax_kernel(x_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(x_ptr + row * n_cols + offs, mask=mask, other=-float("inf")) # Mask 外的使用负无穷使得取指数后为 0，避免影响结果
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    tl.store(out_ptr + row * n_cols + offs, e / s, mask=mask)


def softmax(x: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = x.shape
    out = torch.empty_like(x)
    softmax_kernel[(n_rows,)](x, out, n_cols, BLOCK=triton.next_power_of_2(n_cols))
    return out


def softmax_ref(x):
    return torch.softmax(x, dim=-1)


# ============================================================================
# 主验收：slot_mapping 生成 kernel（simplified _compute_slot_mapping_kernel）
#   grid=(num_reqs+1,)：每 program 处理一个请求的 token 区间；末 program 填 padding。
#   公式：slot = block_table[req, pos//vbs]*block_size + local_offset
#   CP：vbs=block_size*CP_WORLD_SIZE；world=1 时退化为 slot=blk*bs+pos%bs。
# ============================================================================
@triton.jit
def slot_mapping_gen_kernel(
    num_tokens,
    max_num_tokens,
    query_start_loc_ptr,  # [num_reqs+1] int32
    positions_ptr,  # [num_tokens] int64
    block_table_ptr,  # [max_reqs, max_blocks] int32 (flat)
    block_table_stride,
    block_size,
    slot_mapping_ptr,  # [max_num_tokens] int64 (输出)
    CP_WORLD_SIZE: tl.constexpr,
    CP_RANK: tl.constexpr,
    INTERLEAVE: tl.constexpr,
    PAD: tl.constexpr,
    BLOCK: tl.constexpr,  # token 维 tile
):
    req = tl.program_id(0)
    # 末 program：把 [num_tokens, max_num_tokens) 填 PAD（CUDA graph 兼容 / 显式 padding）
    if req == tl.num_programs(0) - 1:
        for i in range(num_tokens, max_num_tokens, BLOCK):
            offs = i + tl.arange(0, BLOCK)
            tl.store(slot_mapping_ptr + offs, PAD, mask=offs < max_num_tokens)
        return

    start = tl.load(query_start_loc_ptr + req).to(tl.int64)
    end = tl.load(query_start_loc_ptr + req + 1).to(tl.int64)
    vbs = block_size * CP_WORLD_SIZE
    row = req * block_table_stride
    for i in range(start, end, BLOCK):
        offs = i + tl.arange(0, BLOCK)
        mask = offs < end
        pos = tl.load(positions_ptr + offs, mask=mask, other=0)
        blk_idx = pos // vbs
        # 掩蔽 lane 的 pos=0 → blk_idx=0，加载 block_table[req,0]（合法），最终由 store mask 丢弃
        blk_num = tl.load(block_table_ptr + row + blk_idx).to(tl.int64)
        voff = pos - blk_idx * vbs
        is_local = (voff // INTERLEAVE) % CP_WORLD_SIZE == CP_RANK
        loff = (voff // (CP_WORLD_SIZE * INTERLEAVE)) * INTERLEAVE + (voff % INTERLEAVE)
        slot = blk_num * block_size + loff
        slot = tl.where(is_local, slot, PAD)
        tl.store(slot_mapping_ptr + offs, slot, mask=mask)


def slot_mapping_gen(
    positions: torch.Tensor,  # [num_tokens] int64
    block_table: torch.Tensor,  # [num_reqs, max_blocks] int32
    query_start_loc: torch.Tensor,  # [num_reqs+1] int32
    block_size: int,
    max_num_tokens: int | None = None,
    *,
    cp_world_size: int = 1,
    cp_rank: int = 0,
    interleave: int = 1,
    tile: int = 1024,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    num_tokens = positions.shape[0]
    num_reqs = block_table.shape[0]
    if max_num_tokens is None:
        max_num_tokens = num_tokens
    assert positions.is_contiguous() and positions.dtype == torch.int64
    # out 可传入预填哨兵值的缓冲（测试用，检测漏写）；默认填 PAD_ID
    slot_mapping = out if out is not None else torch.full(
        (max_num_tokens,), PAD_ID, dtype=torch.int64, device=positions.device)
    slot_mapping_gen_kernel[(num_reqs + 1,)]( # 多出来的一个负责处理 padding，[num_tokens, max_num_tokens) 填 -1
        num_tokens,
        max_num_tokens,
        query_start_loc,
        positions,
        block_table,
        block_table.stride(0),
        block_size,
        slot_mapping,
        CP_WORLD_SIZE=cp_world_size,
        CP_RANK=cp_rank,
        INTERLEAVE=interleave,
        PAD=PAD_ID,
        BLOCK=tile,
    )
    return slot_mapping


def slot_mapping_gen_ref(
    positions, block_table, query_start_loc, block_size, max_num_tokens=None,
    *, cp_world_size=1, cp_rank=0, interleave=1,
) -> torch.Tensor:
    """PyTorch 语义 Oracle（清晰、逐请求）。"""
    num_tokens = positions.shape[0]
    if max_num_tokens is None:
        max_num_tokens = num_tokens
    slot = torch.full((max_num_tokens,), PAD_ID, dtype=torch.int64, device=positions.device)
    qsl = query_start_loc.tolist()
    vbs = block_size * cp_world_size
    for r in range(len(qsl) - 1):
        s, e = qsl[r], qsl[r + 1]
        if e <= s:
            continue
        pos = positions[s:e].to(torch.int64)
        blk_idx = pos // vbs
        blk_num = block_table[r, blk_idx].to(torch.int64)
        voff = pos - blk_idx * vbs
        is_local = (voff // interleave) % cp_world_size == cp_rank
        loff = (voff // (cp_world_size * interleave)) * interleave + (voff % interleave)
        sl = blk_num * block_size + loff
        slot[s:e] = torch.where(is_local, sl, torch.full_like(sl, PAD_ID))
    return slot


# ============================================================================
# 自测：直接运行本文件时，对每个 kernel 各 launch 一次（验证 Triton JIT 真跑）
# ============================================================================
if __name__ == "__main__":
    dev = "cuda"
    # vector add
    x = torch.randn(12345, device=dev); y = torch.randn(12345, device=dev)
    assert torch.allclose(vector_add(x, y), vector_add_ref(x, y)); print("[selftest] vector_add OK")
    # act_and_mul
    x = torch.randn(64, 2 * 100, device=dev)
    assert torch.allclose(act_and_mul(x), act_and_mul_ref(x), atol=1e-5); print("[selftest] act_and_mul OK")
    # softmax（非 2 次幂宽度）
    for w in (127, 128, 255, 512, 1024):
        x = torch.randn(33, w, device=dev)
        assert torch.allclose(softmax(x), softmax_ref(x), atol=1e-6), f"softmax w={w}"
    print("[selftest] softmax OK (127/128/255/512/1024)")
    # slot_mapping 生成（2 请求 ragged + padding）
    bs = 16
    bt = torch.tensor([[5, 6, 7, 8], [13, 2, 9, 11]], dtype=torch.int32, device=dev)
    qsl = torch.tensor([0, 20, 55], dtype=torch.int32, device=dev)
    positions = torch.cat([torch.arange(20), torch.arange(35)]).to(torch.int64).to(dev)
    got = slot_mapping_gen(positions, bt, qsl, bs, max_num_tokens=63, tile=8)
    ref = slot_mapping_gen_ref(positions, bt, qsl, bs, max_num_tokens=63)
    assert torch.equal(got, ref), (got, ref)
    assert (got[55:] == PAD_ID).all(), got[55:]
    print("[selftest] slot_mapping_gen OK (ragged+padding, equal to PyTorch ref)")
    print("[selftest] ALL PASS")
