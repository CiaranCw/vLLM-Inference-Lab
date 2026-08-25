"""M8 · correctness 套件。

- 基础练习：vector_add / act_and_mul / softmax（含非 2 次幂列宽）与 PyTorch 对照 allclose。
- 主验收 slot_mapping 生成：全测试矩阵 + 三路功能交叉验证
    自写 Triton（slot_mapping_gen） vs PyTorch 语义 Oracle（slot_mapping_gen_ref） vs vLLM 官方 `_compute_slot_mapping_kernel`。
  生成侧输出为 int64 精确值 → 断言 torch.equal（不是 allclose）。

结果写 outputs/m8/correctness.json（每用例 pass + 关键数值）；任一 fail 立即可见。

说明：
- 有效 slot 唯一性是消费侧（散写）才关心的写竞争问题；生成侧只做 index 计算，允许重复（如前缀共享）。
  为保持 ref 比较干净，各请求分配互不重叠的物理块。
- 非连续/乱序物理块用例名为 non_contiguous_physical_blocks / shuffled_slot_assignment，
  仅验证 Block Pool 任意物理分配下的 index 计算，不声称完整模拟 prefix cache 生命周期。
"""

from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m8_kernels as K

from vllm.v1.worker.block_table import _compute_slot_mapping_kernel

DEV = "cuda"
PAD = K.PAD_ID
results = []


def _record(name, ok, detail=""):
    results.append({"name": name, "pass": bool(ok), "detail": detail})
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def vllm_gen(positions, block_table, qsl, block_size, max_num_tokens,
             cp_world=1, cp_rank=0, interleave=1):
    """启动 vLLM 官方生成 kernel（一致性对照）。"""
    n = positions.shape[0]
    num_reqs = block_table.shape[0]
    slot = torch.full((max_num_tokens,), PAD, dtype=torch.int64, device=positions.device)
    _compute_slot_mapping_kernel[(num_reqs + 1,)](
        n, max_num_tokens, qsl, positions, block_table, block_table.stride(0),
        block_size, slot,
        TOTAL_CP_WORLD_SIZE=cp_world, TOTAL_CP_RANK=cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=interleave, PAD_ID=PAD, BLOCK_SIZE=1024)
    return slot


# ----------------------------- 基础练习 -----------------------------
def test_basics():
    x = torch.randn(50000, device=DEV); y = torch.randn(50000, device=DEV)
    _record("basic/vector_add", torch.allclose(K.vector_add(x, y), x + y))
    x = torch.randn(128, 2 * 137, device=DEV)  # d 非 2 次幂
    _record("basic/act_and_mul", torch.allclose(K.act_and_mul(x), K.act_and_mul_ref(x), atol=1e-5))
    for w in (127, 128, 255, 512, 1024):
        x = torch.randn(37, w, device=DEV)
        ok = torch.allclose(K.softmax(x), K.softmax_ref(x), atol=1e-6)
        _record(f"basic/softmax_w{w}", ok)


# ----------------------------- 生成侧：构造 + 三路交叉 -----------------------------
def _distinct_block_table(num_reqs, max_blocks, seed=0):
    """给各请求分配互不重叠的物理块号（乱序），返回 int32 block_table。"""
    g = torch.Generator().manual_seed(seed)
    total = num_reqs * max_blocks
    perm = torch.randperm(total * 2, generator=g)[:total] + 1  # 从 1 起，避开 0(null)
    return perm.reshape(num_reqs, max_blocks).to(torch.int32).to(DEV)


def check_generate(name, positions, block_table, qsl, block_size, max_num_tokens,
                   cp_world=1, cp_rank=0, interleave=1, tile=1024, sentinel=False):
    """三路交叉：mine vs ref vs vLLM，全等 + pad 区 == PAD + （可选）哨兵无残留。"""
    n = positions.shape[0]
    ref = K.slot_mapping_gen_ref(positions, block_table, qsl, block_size, max_num_tokens,
                                 cp_world_size=cp_world, cp_rank=cp_rank, interleave=interleave)
    out = None
    if sentinel:
        out = torch.full((max_num_tokens,), -999, dtype=torch.int64, device=DEV)
    mine = K.slot_mapping_gen(positions, block_table, qsl, block_size, max_num_tokens,
                              cp_world_size=cp_world, cp_rank=cp_rank, interleave=interleave,
                              tile=tile, out=out)
    vllm = vllm_gen(positions, block_table, qsl, block_size, max_num_tokens,
                    cp_world=cp_world, cp_rank=cp_rank, interleave=interleave)
    eq_ref = torch.equal(mine, ref)
    eq_vllm = torch.equal(mine, vllm)
    pad_ok = bool((mine[n:max_num_tokens] == PAD).all()) if max_num_tokens > n else True
    sentinel_ok = bool((mine != -999).all()) if sentinel else True
    ok = eq_ref and eq_vllm and pad_ok and sentinel_ok
    detail = f"n={n} eq_ref={eq_ref} eq_vllm={eq_vllm} pad_ok={pad_ok} sentinel_ok={sentinel_ok}"
    _record(name, ok, detail)
    return ok


def test_generate():
    bs = 16
    # 1) 随机合法映射（多请求，各自不重叠物理块）
    lens = [23, 40, 7, 64]
    bt = _distinct_block_table(len(lens), 8, seed=1)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    positions = torch.cat([torch.arange(L) for L in lens]).to(torch.int64).to(DEV)
    check_generate("gen/random_multi", positions, bt, qsl, bs, positions.shape[0])
    # 2) 单 token
    bt1 = torch.tensor([[9]], dtype=torch.int32, device=DEV)
    check_generate("gen/single_token", torch.tensor([0], dtype=torch.int64, device=DEV),
                   bt1, torch.tensor([0, 1], dtype=torch.int32, device=DEV), bs, 1)
    # 3) 单请求
    bt1 = _distinct_block_table(1, 8, seed=2)
    check_generate("gen/single_request", torch.arange(50, dtype=torch.int64, device=DEV),
                   bt1, torch.tensor([0, 50], dtype=torch.int32, device=DEV), bs, 50)
    # 4/5) 块内首/末位置（pos=0 与 pos=block_size-1）
    bt1 = _distinct_block_table(1, 8, seed=3)
    check_generate("gen/block_first", torch.tensor([0, bs, 2 * bs], dtype=torch.int64, device=DEV),
                   bt1, torch.tensor([0, 3], dtype=torch.int32, device=DEV), bs, 3)
    check_generate("gen/block_last",
                   torch.tensor([bs - 1, 2 * bs - 1, 3 * bs - 1], dtype=torch.int64, device=DEV),
                   bt1, torch.tensor([0, 3], dtype=torch.int32, device=DEV), bs, 3)
    # 6) 跨块（seqlen > block_size 多次）
    check_generate("gen/cross_block", torch.arange(70, dtype=torch.int64, device=DEV),
                   bt1, torch.tensor([0, 70], dtype=torch.int32, device=DEV), bs, 70)
    # 7) 非连续物理块（block_table 物理号非连续）
    bt_nc = torch.tensor([[100, 3, 57, 12, 88, 1, 44, 9]], dtype=torch.int32, device=DEV)
    check_generate("gen/non_contiguous_physical_blocks", torch.arange(120, dtype=torch.int64, device=DEV),
                   bt_nc, torch.tensor([0, 120], dtype=torch.int32, device=DEV), bs, 120)
    # 8) 乱序 slot 分配（block_table 打乱）
    g = torch.Generator().manual_seed(7)
    bt_sh = (torch.randperm(200, generator=g)[:8] + 1).reshape(1, 8).to(torch.int32).to(DEV)
    check_generate("gen/shuffled_slot_assignment", torch.arange(120, dtype=torch.int64, device=DEV),
                   bt_sh, torch.tensor([0, 120], dtype=torch.int32, device=DEV), bs, 120)
    # 9) 请求长度不整除小 tile（触发尾部 mask）
    lens = [7, 20, 35]
    bt = _distinct_block_table(len(lens), 8, seed=4)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    positions = torch.cat([torch.arange(L) for L in lens]).to(torch.int64).to(DEV)
    check_generate("gen/tile_tail_mask(tile=8)", positions, bt, qsl, bs, positions.shape[0], tile=8)
    # 10) ragged 多请求
    lens = [5, 33, 16, 48, 1]
    bt = _distinct_block_table(len(lens), 8, seed=5)
    qsl = torch.tensor([0] + list(torch.tensor(lens).cumsum(0).tolist()), dtype=torch.int32, device=DEV)
    positions = torch.cat([torch.arange(L) for L in lens]).to(torch.int64).to(DEV)
    check_generate("gen/ragged_multi_request", positions, bt, qsl, bs, positions.shape[0])
    # 11) padding（num_tokens < max_num_tokens，哨兵 -999 检测漏写）
    check_generate("gen/padding_sentinel", positions, bt, qsl, bs, positions.shape[0] + 13, sentinel=True)
    # 12) CP interleave（world=2, rank=0/1, interleave=1 与 2），与官方对齐
    bt1 = _distinct_block_table(1, 16, seed=6)
    for cw, cr, il in [(2, 0, 1), (2, 1, 1), (2, 0, 2), (4, 2, 1)]:
        check_generate(f"gen/cp_world{cw}_rank{cr}_il{il}",
                       torch.arange(96, dtype=torch.int64, device=DEV), bt1,
                       torch.tensor([0, 96], dtype=torch.int32, device=DEV), bs, 96,
                       cp_world=cw, cp_rank=cr, interleave=il)


def main():
    test_basics()
    test_generate()
    n_pass = sum(r["pass"] for r in results)
    n_tot = len(results)
    os.makedirs("outputs/m8", exist_ok=True)
    with open("outputs/m8/correctness.json", "w", encoding="utf-8") as f:
        json.dump({"passed": n_pass, "total": n_tot, "all_green": n_pass == n_tot,
                   "cases": results}, f, ensure_ascii=False, indent=2)
    print(f"\n[m8_test] {n_pass}/{n_tot} passed  all_green={n_pass == n_tot}")
    print("[m8_test] -> outputs/m8/correctness.json")
    sys.exit(0 if n_pass == n_tot else 1)


if __name__ == "__main__":
    main()
