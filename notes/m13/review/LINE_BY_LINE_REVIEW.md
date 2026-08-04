# M13 · 逐行人工审查主文档（LINE_BY_LINE_REVIEW）

> **对象**：`tests/v1/worker/test_block_table.py`（347 行，新增文件）
> **干净工作树**：`/home/ciaran/projects/vllm-m13-upstream`，分支 `m13/block-table-slot-mapping-tests`，base = upstream/main **`72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0`**
> **sha256**：`24fdf0a04bd58702595dd67add4f1888fa14757e065c32f50702aa2016ed1afc`
> **本文件用途**：供**你（提交人）**逐行审查。AGENTS.md 明确「纯 code-agent PR 不允许，提交人须端到端理解并为改动负责」。**我不能代你完成审查。**
>
> 建议边读边对照产品源码：`vllm/v1/worker/block_table.py`（`compute_slot_mapping` 在 `:182`，kernel 在 `:379`）。

---

## 0. 先建立全局认知（读代码前 3 分钟）

被测函数做一件事：**把「逻辑 token 位置」翻译成「KV cache 里的物理槽位」**。

```
基本公式（无 DCP、无 hybrid）：
  slot = block_table[req][pos // block_size] * block_size + (pos % block_size)
```

三个必须分清的块概念（当前 main 才有）：

| 概念 | 代码字段 | 含义 |
|---|---|---|
| KV 分配块大小 | `kv_cache_block_size`（构造参数 `block_size`） | KV manager 分配粒度；决定 **virtual block 跨度** |
| kernel 块大小 | `block_size`（= `kernel_block_size`） | attention kernel 粒度；**决定最终 slot 算术** |
| 展开倍数 | `blocks_per_kv_block` | = 前者 ÷ 后者；一个分配块含几个 kernel 块 |

三个易错分支：**DCP 分片**（token 归哪个 rank）、**hybrid 块展开**（1 个 KV 块 → N 个 kernel 块）、**CUDA graph padding**（尾部必须是 `PAD_SLOT_ID`）。

---

## 1. L1–L9：SPDX、模块 Docstring

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``BlockTable.compute_slot_mapping``.

``compute_slot_mapping`` maps logical token positions to physical KV cache
slots. A silent off-by-one here corrupts KV cache contents rather than raising,
so the mapping is covered here through the public ``BlockTable`` API across
plain, hybrid-block, and decode-context-parallel configurations.
"""
```

- **目的**：SPDX 头（pre-commit 的 `Check SPDX headers` 强制）；docstring 说明**为什么值得测**——错了不报错、只静默写坏 KV。
- **输入/输出**：无。
- **潜在错误**：SPDX 缺失会被 CI 拒。
- **你要检查**：这两行是否与仓库其它测试文件完全一致（`head -2 tests/v1/worker/test_gpu_block_table.py` 对比）。
- **面试怎么讲**：「这个路径的失败模式是静默数据损坏，不是异常，所以必须有单测护住。」

## 2. L11–L22：Imports 与 CUDA Skip Marker

```python
import numpy as np
import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.worker.block_table import BlockTable, SlotMappingMode

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)
```

- **目的**：`pytestmark` 让**整个模块**在非 CUDA 环境 skip（而非 fail）。
- **为什么需要**：slot mapping 是 Triton GPU kernel；CPU fallback（`vllm/utils/cpu_triton_utils.py`）走 C++ op 且 **`assert TOTAL_CP_WORLD_SIZE == 1`**，无法覆盖 DCP，所以本套测试定位为 CUDA-only。
- **潜在错误**：若用 `torch.cuda.is_available()` 而非 `current_platform.is_cuda()`，会偏离仓库惯例。
- **你要检查**：① `current_platform.is_cuda()` 是否为本仓库现行惯例（对比 `test_gpu_block_table.py:10-13`，它用同一写法）；② **collect 阶段不会因无 CUDA 崩溃**（`--collect-only` 已验证）。
- **面试怎么讲**：「CP 语义只在 GPU kernel 里，CPU 后端明确不支持，所以 skip 而不是假装能跑。」

## 3. L24–L28：测试常量

```python
MAX_NUM_REQS = 8
MAX_NUM_BLOCKS_PER_REQ = 32
# The kernel pads slots for CUDA graph compatibility in tiles of 1024, so keep
# the default buffer larger than one tile to exercise the padding loop.
MAX_NUM_BATCHED_TOKENS = 2048
```

- **目的**：默认容量。`2048 > 1024`（kernel 的 `BLOCK_SIZE=1024` tile）**故意**如此，让 padding 循环至少跑两次。
- **潜在错误**：若设 < 1024，padding 的 `for` 循环只走一趟，跨 tile 的 bug 测不出来（这是旧版本的缺陷之一）。
- **你要检查**：`MAX_NUM_BLOCKS_PER_REQ=32` 是否够用——32 块 × 16 token = 512 token；用例里超过 512 token 的都单独传 `max_num_blocks_per_req`（见 §11）。
- **面试怎么讲**：「buffer 故意大于一个 Triton tile，否则 padding 的循环边界测不到。」

## 4. L31–L58：`_make_block_table`

```python
def _make_block_table(
    kv_cache_block_size: int = 16,
    kernel_block_size: int | None = None,
    dcp_world_size: int = 1,
    ...
) -> BlockTable:
    if kernel_block_size is None:
        kernel_block_size = kv_cache_block_size
    block_table = BlockTable(...)
    # Decode context parallelism is derived from the DCP process group, which
    # is not initialized in unit tests; set the resolved values directly.
    block_table.dcp_world_size = dcp_world_size
    block_table.dcp_rank = dcp_rank
    return block_table
```

- **目的**：构造真实 `BlockTable`，并把 DCP 参数注入。
- **`if kernel_block_size is None`（L41-42）**：本轮修复项——原写法 `kernel_block_size or kv_cache_block_size` 会把 `0` 也当成缺省（虽然 0 非法，但把「合法性」与「默认值」混在布尔表达式里不好审）。现在语义显式。
- **为什么直接设 `dcp_world_size`（L56-57）**：`BlockTable.__init__`（`block_table.py:121-134`）用 `get_dcp_group()`/`get_pcp_group()` 取值，**未初始化分布式时 catch `AssertionError` 回落 1**。单测不起进程组，故构造后直接赋值——这正是 `compute_slot_mapping`（`:206-207`）读取的两个字段。
- **潜在错误**：**过度 mock 风险**。这里没有 mock 任何函数，只是覆盖两个已解析的实例属性；若上游把这两个字段改成 property/只读，本测试会立刻 AttributeError（属于好的失败）。
- **你要检查**：① 打开 `block_table.py:121-134` 确认字段名与回落逻辑；② 确认 `compute_slot_mapping` 真的只读 `dcp_*` 而**不读** `pcp_*`（`:206-207`）——这是本 PR 最重要的语义前提；③ `pin_memory=False` 是否合理（是：避免 WSL2 上 `UVA is not available`）。
- **面试怎么讲**：「DCP 的 world size/rank 在真实运行时来自进程组；单测不起 NCCL，就直接注入已解析值，因为被测 kernel 只消费这两个整数。」

## 5. L61–L79：独立展开 `_expand_kernel_block_ids`（**本轮核心修复**）

```python
def _expand_kernel_block_ids(
    kv_manager_block_ids: list[int],
    blocks_per_kv_block: int,
) -> np.ndarray:
    """... kernel_block_id = kv_block_id * blocks_per_kv_block + sub_block ...
    Kept independent of ``BlockTable.map_to_kernel_blocks`` so a
    bug there cannot cancel out in both the expected and actual values.
    """
    return np.array(
        [
            kv_block_id * blocks_per_kv_block + sub_block
            for kv_block_id in kv_manager_block_ids
            for sub_block in range(blocks_per_kv_block)
        ],
        dtype=np.int64,
    )
```

- **目的**：**测试自己**把 KV manager 块 id 展开成 kernel 块 id。
- **为什么必须存在（重要）**：产品 `add_row`/`append_row`（`block_table.py:146-149`）用 `map_to_kernel_blocks` 展开写进表。旧版 Oracle **也调用同一个** `map_to_kernel_blocks` → 若该 helper 有 bug，**期望值和实际值一起错、测试照样绿**。
- **本轮已实测证明该风险真实**：给产品 `map_to_kernel_blocks` 注入 `+1` 后——
  - 修复**前**：`hybrid[32-16]`、`hybrid[64-16]` **仍然 PASSED**（Oracle 被同一 bug 带跑），只有手算 golden 失败；
  - 修复**后**：3 个 hybrid 测试**全部 FAILED**。
- **输入/输出**：`[5, 9]`, `bpk=2` → `[10, 11, 18, 19]`。
- **潜在错误**：若把公式写成 `kv_block_id + sub_block * n` 之类，hybrid 手算 golden 会立刻抓住（golden 是写死的常量）。
- **你要检查**：① 手算 `[5,9]`,bpk=2 是否真等于 `[10,11,18,19]`；② `grep -n "map_to_kernel_blocks" tests/v1/worker/test_block_table.py` **必须无输出**（Oracle 不再依赖产品 helper）；③ 该公式与产品 docstring（`block_table.py:226-238` 的示例 `[0,1,2] → [0,1,2,3,4,5]`）语义一致但**实现独立**。
- **面试怎么讲**：「Oracle 绝不能调用被测代码的 helper，否则两边一起错、测试变成空转。我用变异测试证明了修复前后的差别。」

## 6. L82–L137：`_reference_slot_mapping`（Oracle，逐公式）

对每个 token，Oracle 依次算：

| 步骤 | 代码 | 含义 | 对照产品 kernel |
|---|---|---|---|
| ① | `virtual_block_size = kv_cache_block_size * world_size`（L118） | 一个「虚拟块」跨多少 token：**用分配块大小**，因为一个 KV 块被切给各 DCP rank | kernel L413 `virtual_block_size = KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE` |
| ② | `virtual_block_index = position // virtual_block_size`（L121） | 这是第几个虚拟块 | kernel L419 |
| ③ | `virtual_block_offset = position % virtual_block_size`（L122） | 虚拟块内偏移 | kernel L420（写成 `pos - idx*vbs`，等价） |
| ④ | `is_local = (offset // interleave) % world_size == rank`（L123） | 以 interleave 为粒度 round-robin 分给各 rank；**不是我的就 PAD** | kernel L421-423 |
| ⑤ | `local_offset = (offset // (world*interleave))*interleave + offset % interleave`（L125-127） | 本 rank 视角下的偏移（把「别人的」挤掉后重新编号） | kernel L424-428 |
| ⑥ | `block_index = virtual_block_index * bpk + local_offset // kernel_block_size`（L128-131） | 落到第几个 **kernel** 块 | kernel L430-433 |
| ⑦ | `block_id = kernel_block_ids[req][block_index]`（L132） | 取物理 kernel 块号——**来自测试自己展开的列表**（§5），不是产品表 | kernel L434-438 从 `block_table_ptr` 读 |
| ⑧ | `slot = block_id * kernel_block_size + local_offset % kernel_block_size`（L133-135） | 最终槽位：**用 kernel 块大小** | kernel L439-440 |
| ⑨ | 非 local 与尾部保持 `PAD_SLOT_ID`（L111 预填） | padding 契约 | kernel L399-408 / L441 |

- **⑦ 与 kernel 的关键差异（务必理解）**：kernel 从**已被 `add_row` 展开过的表**里读；Oracle 从**原始 KV manager id 自己展开**。两条路径独立汇合到同一个期望值——这正是能抓住 `map_to_kernel_blocks` bug 的原因。
- **潜在错误**：⑥ 若误用 `kv_cache_block_size` 代替 `kernel_block_size` 做除法，hybrid 用例会挂；④/⑤ 若 world/interleave 顺序写反，DCP 用例会挂。
- **你要检查**：把上表右列逐条与 `vllm/v1/worker/block_table.py:413-441` 对读，确认「语义等价但写法独立」；**特别确认第 ① 步用的是 `KV_CACHE_BLOCK_SIZE` 而不是 kernel block size**（这是当前 main 的新语义，旧版本没有）。
- **面试怎么讲**：「Oracle 是一份慢速、逐 token 的 Python 重写，从公共输入语义出发，而不是把 Triton 索引式抄一遍。」

## 7. L140–L161：`_compute`（驱动公共 API）

```python
    for row_idx, block_ids in enumerate(kv_manager_block_ids):
        block_table.add_row(block_ids, row_idx)
    block_table.commit_block_table(num_reqs)
    ...
    query_start_loc = np.concatenate([[0], np.cumsum(query_lens)]).astype(np.int32)
    ...
    block_table.compute_slot_mapping(num_reqs, ...cuda(), ...cuda())
    slot_mapping = block_table.slot_mapping.gpu.cpu().numpy()
```

- **目的**：走**完整公共路径** `add_row → commit_block_table → compute_slot_mapping`，再把结果读回 numpy。
- **`commit_block_table` 为什么必须（易漏）**：`add_row` 只写 CPU 侧 `block_table.np`；kernel 读的是 `block_table.gpu`。**不 commit 就会读到全 0 的表**，测试会以诡异方式失败。
- **`query_start_loc` 语义**：累积偏移（ragged batch 的标准表达），`qsl[i+1]-qsl[i]` = 第 i 个请求本轮 token 数。
- **潜在错误**：dtype 必须匹配 kernel 注释——`query_start_loc` int32、`positions` int64（见 kernel 签名 L383-384）。
- **你要检查**：① dtype 是否与 kernel 期望一致；② 是否测的是**公共入口**而非私有 kernel（是：调 `compute_slot_mapping`）。
- **面试怎么讲**：「必须 commit 把 block table 从 CPU 拷到 GPU，否则 kernel 读到的是空表——这是这个 API 的真实使用契约。」

## 8. L164–L176：`_check`（通用断言）

```python
    np.testing.assert_array_equal(actual[:num_tokens], expected[:num_tokens])
    assert (actual[num_tokens:] == PAD_SLOT_ID).all()
```

- **目的**：真实 token 段与 Oracle 比对；**尾部整段**必须是 `PAD_SLOT_ID`。
- **潜在错误**：只比对前 N 个而不查尾部，会漏掉 CUDA graph padding 的 bug。
- **你要检查**：尾部断言覆盖到 `max_num_batched_tokens` 全长（`actual` 是整段 buffer）。

## 9. L179–L207：普通映射手算 Golden（2 例）

期望值**写死**，与 Oracle 无关（防 Oracle 与 kernel 同错）。手算见 `MANUAL_GOLDEN_CASES.md` §1。
- **你要检查**：`position 47 // 16 = 2` → 用 `[5,6,7,8]` 的第 2 个元素 = **7** → `7*16+15 = 127`。（本轮之前这里曾错写成 `block 8 → 143`，被这个 golden 抓到——**这正是 golden 存在的价值**。）

## 10. L210–L238：边界 / Ragged / Prefix 参数化（6 例）

- **本轮修复**：原来把 case 名当成未使用的 `name` 参数传进函数；现改用 `ids=[...]`，函数签名不再收 `name`，pytest 输出直接显示 `[single_token]`、`[prefix_offset]` 等。
- **覆盖**：单 token / 块内首末 / 跨多块(64) / 非连续物理块(96) / ragged 5 请求变长 / prefix offset（`range(70,90)`，模拟前缀已缓存的 decode）。
- **你要检查**：`ragged_multi_request` 里 5 个请求的块数是否够放各自 token 数（如 60 token 需 ≥4 块，给了 `[1,2,3,4]` 正好 4 块 = 64 token 容量）。

## 11. L241–L253：跨 Triton Tile Padding

```python
    num_tokens = 1030
    num_blocks = num_tokens // 16 + 1
    block_table = _make_block_table(
        max_num_batched_tokens=2100, max_num_blocks_per_req=num_blocks
    )
```

- **目的**：`1030 > 1024` 且**不是** 1024 的整数倍；buffer 2100 让 padding 尾部再跨一个 tile。
- **为什么这么选**：kernel 的 padding 是 `for i in range(num_tokens, max_num_tokens, 1024)` + `mask`（L401-407）。只有跨 tile 且非整数倍，才能同时测到「循环多趟」与「mask 边界」。
- **你要检查**：`num_blocks = 1030//16+1 = 65` 是否足够（65×16 = 1040 ≥ 1030 ✓）；以及为何这里要单独传 `max_num_blocks_per_req`（默认 32 块只够 512 token，会 broadcast 报错——本轮之前确实踩到过）。
- **面试怎么讲**：「padding 是给 CUDA graph 用的定长 buffer，尾部必须是明确哨兵；我特意让 token 数跨过 tile 边界且非整数倍。」

## 12. L256–L278：DCP 全 Rank 分区

```python
    owning_ranks.append(actual != PAD_SLOT_ID)
    ...
    np.testing.assert_array_equal(
        np.sum(owning_ranks, axis=0), np.ones(len(positions_per_req[0]), dtype=int)
    )
```

- **目的**：world{2,4} × interleave{1,4}，**遍历所有 rank**，断言「每个 token 恰被 1 个 rank 拥有」——这是一个**不变量**（守恒律），比逐值比对更能抓分片逻辑错误。
- **潜在错误**：若分片公式让某 token 被 2 个 rank 都认领（重复写 KV）或没人认领（KV 丢失），这里立刻挂。
- **你要检查**：`np.sum(..., axis=0)` 是对 rank 维求和；`np.ones(96)` 长度与 token 数一致。
- **面试怎么讲**：「除了值对不对，我还测了一条守恒律：每个 token 在 DCP 下必须恰好归属一个 rank，不重不漏。」

## 13. L281–L296：DCP 手算 Golden

world=2、rank=1、interleave=1 → 虚拟块 = 16×2 = 32；rank1 拿奇数 offset。手算见 `MANUAL_GOLDEN_CASES.md` §2。
- **你要检查**：`[PAD, 80, PAD, 81, PAD, 82]` —— 偶数位是别人的（PAD），奇数位 local_offset 依次 0,1,2 → 80,81,82。

## 14. L299–L316：Hybrid 泛化（32/16 与 64/16）

- **先断言配置生效**：`use_hybrid_blocks` 为真、`blocks_per_kv_block == kv/kernel`。
- **再跑 `3 * kv_cache_block_size` 个 token**：同时跨 kernel 块边界与 KV 分配块边界。
- **你要检查**：`[[5, 9, 2]]` 3 个 KV 块 × bpk 后是否够 `3*kv_cache_block_size` 个 token（32/16：3 块 = 6 kernel 块 = 96 token = 3×32 ✓；64/16：3 块 = 12 kernel 块 = 192 = 3×64 ✓）。
- **注意**：这两个 case **依赖 Oracle**（非写死），所以 §5 的独立性修复对它们至关重要。

## 15. L319–L335：Hybrid 手算 Golden

KV 块 `[5,9]`、bpk=2 → kernel 块 `[10,11,18,19]`；slot 用 kernel 块大小 16。手算见 `MANUAL_GOLDEN_CASES.md` §3。
- **你要检查**：`position 32` 为何落到 kernel 块 **18**（= KV 块 9 的第一个子块 `9*2+0`），而不是 12——这是 hybrid 展开的核心。

## 16. L338–L347：`SlotMappingMode.NONE`

```python
    sentinel = -12345
    assert sentinel != PAD_SLOT_ID
    block_table.slot_mapping.gpu.fill_(sentinel)
    actual, _ = _compute(block_table, [[5, 6]], [[0, 1, 2]])
    assert (actual == sentinel).all()
```

- **目的**：Mamba/GDN 类状态缓存把 block table 当**状态索引**用，不需要 per-token slot；`compute_slot_mapping` 在该模式下**直接 return**（`block_table.py:189-192`）。
- **为什么用 sentinel**：不能依赖未初始化显存；且 sentinel **必须区别于** `PAD_SLOT_ID(-1)`，否则「没写」和「写了 PAD」无法区分——`assert sentinel != PAD_SLOT_ID` 把这个前提固化。
- **潜在错误**：若上游改成「NONE 模式也填 PAD」，本测试会失败——**这是期望行为**（契约变更应当被发现）。
- **你要检查**：打开 `block_table.py:189-192` 确认「直接 return、不触碰 buffer」的契约。
- **面试怎么讲**：「NONE 模式的契约是完全不碰 buffer，我用一个区别于 PAD 的哨兵证明它真的没写，而不是碰巧写了 PAD。」

---

## 17. 审查时建议实际执行的对读

```bash
cd /home/ciaran/projects/vllm-m13-upstream
sed -n '182,215p' vllm/v1/worker/block_table.py    # compute_slot_mapping 调用侧
sed -n '379,443p' vllm/v1/worker/block_table.py    # kernel 全文
grep -n "map_to_kernel_blocks" tests/v1/worker/test_block_table.py   # 必须无输出
```

## 18. 已知限制（审查时需知情）
- 未真起多卡：DCP 通过注入 `dcp_world_size/dcp_rank`（CP 是纯 index 数学；CPU fallback 明确不支持 CP）。
- 未覆盖 `num_reqs=0` / 长度 0 请求 / `num_tokens=0`：MRV1 调用侧不产生此类输入，且 `np.concatenate` 空批本身非法；**仅记录不写期望**。
- 未测 MRV2 的 `compute_slot_mappings`（复数，`vllm/v1/worker/gpu/block_table.py`）——另一路径，不扩大 scope。
- 本机 `test_gpu_block_table.py` 有 3 个失败，为**基线环境问题**（WSL2 `UVA is not available`），移除本文件同样失败，与本改动无关。
