# M13 · 逐行人工审查主文档（LINE_BY_LINE_REVIEW）

> **对象**：`tests/v1/worker/test_block_table_slot_mapping.py`（**384 行**，新增文件，18 tests）
> **干净工作树**：`/home/ciaran/projects/vllm-m13-upstream`，分支 `m13/block-table-slot-mapping-tests`
> **base = upstream/main `a1657a0235a77ce85f8d348c83ff4adb29918b71`**
> **SHA256**：`b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`
> **本文件用途**：供**你（提交人）**逐行审查。AGENTS.md 明确「纯 code-agent PR 不允许，提交人须端到端理解并为改动负责」。**我不能代你完成审查。**
>
> 建议边读边对照产品源码：`vllm/v1/worker/block_table.py`（`compute_slot_mapping` 在 `:182`，kernel 在 `:379`）。
> 手算细节见 `MANUAL_GOLDEN_CASES.md`。

---

## 0. 先建立全局认知（读代码前 3 分钟）

被测函数把**逻辑 token 位置**翻译成 **KV cache 物理槽位**：

```
最朴素情形（无 DCP、无 hybrid）：
  slot = block_table[req][pos // block_size] * block_size + (pos % block_size)
```

三个必须分清的块概念（当前 main 才有）：

| 概念 | 字段 | 作用 |
|---|---|---|
| KV 分配块大小 | `kv_cache_block_size`（构造参数 `block_size`） | KV manager 分配粒度，决定 **virtual block 跨度** |
| kernel 块大小 | `block_size`（= `kernel_block_size`） | attention kernel 粒度，**决定最终 slot 算术** |
| 展开倍数 | `blocks_per_kv_block` | 前者 ÷ 后者；一个分配块含几个 kernel 块 |

三个易错分支：**DCP 分片**（token 归哪个 rank）、**hybrid 展开**（1 KV 块 → N kernel 块）、**CUDA graph padding**（尾部必须 `PAD_SLOT_ID`）。

**三条独立证据路径**（本轮强化的核心）：
```
Actual（真实 kernel 输出）  ┐
Reference（测试自建 oracle）├─ 三者必须一致
Golden（手写死的期望值）    ┘
```

---

## 1. L1–L9：SPDX 与模块 Docstring

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for ``BlockTable.compute_slot_mapping``.

``compute_slot_mapping`` maps logical token positions to physical KV cache
slots. A silent off-by-one here corrupts KV cache contents rather than raising,
so the mapping is covered here through ``BlockTable.compute_slot_mapping``
across plain, hybrid-block, and decode-context-parallel configurations.
"""
```

- **目的**：SPDX 头（pre-commit `Check SPDX headers` 强制）；docstring 说明为什么值得测——错了不报错、只静默写坏 KV。
- **本轮修正（4.1）**：原写「through the public `BlockTable` API」**过强**——测试确实会直接设置内部已解析的 `dcp_world_size`/`dcp_rank`（L56-57）。现改为「through `BlockTable.compute_slot_mapping`」，只声称测的是这个入口，不再声称全程只用 public API。
- **你要检查**：措辞是否与实际做法一致（不再有"public API"的过度声明）。
- **面试怎么讲**：「这个路径的失败模式是静默数据损坏而不是异常，所以必须单测护住。」

## 2. L11–L22：Imports 与 CUDA Skip

```python
pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(),
    reason="requires CUDA",
)
```

- **目的**：整个模块在非 CUDA 环境 **skip 而非 fail**；`--collect-only` 在无 GPU 时也不崩。
- **为什么 CUDA-only**：slot mapping 是 Triton GPU kernel；CPU fallback（`vllm/utils/cpu_triton_utils.py`）走 C++ op 且 **`assert TOTAL_CP_WORLD_SIZE == 1`**，无法覆盖 DCP。
- **你要检查**：`current_platform.is_cuda()` 是仓库惯例（对比 `tests/v1/worker/test_gpu_block_table.py:10-13` 同写法）。

## 3. L24–L28：测试常量

```python
MAX_NUM_REQS = 8
MAX_NUM_BLOCKS_PER_REQ = 32
# The kernel pads slots for CUDA graph compatibility in tiles of 1024, so keep
# the default buffer larger than one tile to exercise the padding loop.
MAX_NUM_BATCHED_TOKENS = 2048
```

- **2048 是刻意的**：> kernel 的 `BLOCK_SIZE=1024` tile，使 padding 循环至少跑两趟。**本轮明确不下调**。
- **你要检查**：若改小于 1024，跨 tile 的 padding bug 就测不到。

## 4. L31–L58：`_make_block_table`

```python
    if kernel_block_size is None:
        kernel_block_size = kv_cache_block_size
    block_table = BlockTable(...)
    # Decode context parallelism is derived from the DCP process group, which
    # is not initialized in unit tests; set the resolved values directly.
    block_table.dcp_world_size = dcp_world_size
    block_table.dcp_rank = dcp_rank
```

- **L41-42（本轮 4.2 修复）**：原 `kernel_block_size or kv_cache_block_size` 把「默认值解析」与「真值判断」混在布尔表达式（`0` 也会走默认支路）。现显式 `is None`。
- **L56-57**：`BlockTable.__init__`（`block_table.py:121-134`）从 `get_dcp_group()` 取值，未初始化分布式时 catch `AssertionError` 回落 1；单测不起进程组，故直接注入**已解析值**——这正是 `compute_slot_mapping`（`:206-207`）消费的两个整数。**本轮明确保留此策略**（不引入多进程/多 GPU 依赖）。
- **你要检查**：① 打开 `block_table.py:121-134` 确认字段名与回落；② 确认 kernel 只读 `dcp_*`、**不读** `pcp_*`；③ `pin_memory=False` 合理（避免 WSL2 `UVA is not available`）。

## 5. L61–L79：独立展开 `_expand_kernel_block_ids`（**关键**）

```python
    return np.array(
        [
            kv_block_id * blocks_per_kv_block + sub_block
            for kv_block_id in kv_manager_block_ids
            for sub_block in range(blocks_per_kv_block)
        ],
        dtype=np.int64,
    )
```

- **为什么必须存在**：产品 `add_row`（`block_table.py:146-149`）用 `map_to_kernel_blocks` 展开写表。若 oracle **也调**同一函数，该 helper 出 bug 时期望值与实际值**一起错**，测试照样绿。
- **实测证据（上一轮）**：注入 `map_to_kernel_blocks +1` 后，复用版本的 `hybrid[32-16]`/`[64-16]` **仍 passed（漏检）**；独立版本 **3/3 失败**。本轮已复跑确认仍然 3/3 失败。
- **输入/输出**：`[5, 9]`, bpk=2 → `[10, 11, 18, 19]`。
- **你要检查**：`grep -n "map_to_kernel_blocks(" tests/v1/worker/test_block_table_slot_mapping.py` → **必须无输出**（仅 L69 docstring 提及名字，无可执行调用）。

## 6. L82–L137：`_reference_slot_mapping`（Oracle，逐公式）

**本轮 4.2 修正**：docstring 首行由「from the public input semantics」改为「from the ``BlockTable`` input semantics」——避免"public"过度声明。

| 步 | 代码行 | 含义 | 产品 kernel 对照 |
|---|---|---|---|
| ① | L118 `virtual_block_size = kv_cache_block_size * world_size` | 虚拟块跨度：**用分配块大小**（KV 块被切给各 rank） | kernel L413 `KV_CACHE_BLOCK_SIZE * TOTAL_CP_WORLD_SIZE` |
| ② | L121 `virtual_block_index = position // virtual_block_size` | 第几个虚拟块 | kernel L419 |
| ③ | L122 `virtual_block_offset = position % virtual_block_size` | 虚拟块内偏移 | kernel L420（写作 `pos - idx*vbs`，等价） |
| ④ | L123 `is_local = (off // interleave) % world_size == rank` | 以 interleave 为粒度 round-robin；不是我的 → PAD | kernel L421-423 |
| ⑤ | L125-127 `local_offset = (off // (world*il))*il + off % il` | 本 rank 视角的紧凑偏移 | kernel L424-428 |
| ⑥ | L128-131 `block_index = v_index * bpk + local_offset // kernel_block_size` | 落到第几个 **kernel** 块（hybrid 在此体现） | kernel L430-433 |
| ⑦ | L132 `block_id = kernel_block_ids[req][block_index]` | 物理 kernel 块号 —— **来自测试自建展开**（§5），非产品表 | kernel L434-438 从 `block_table_ptr` 读 |
| ⑧ | L133-135 `slot = block_id * kernel_block_size + local_offset % kernel_block_size` | 最终槽位：**用 kernel 块大小** | kernel L439-440 |
| ⑨ | L111 预填 `PAD_SLOT_ID` | 非 local 与尾部 padding | kernel L399-408 / L441 |

- **⑦ 是与 kernel 的关键差异**：kernel 读**已被 `add_row` 展开过的表**；oracle **自行展开**。两条独立路径汇合到同一期望值——这才能抓住展开 bug。
- **你要检查**：把右列逐条与 `block_table.py:413-441` 对读；**特别确认第 ① 步用 `KV_CACHE_BLOCK_SIZE` 而非 kernel block size**（当前 main 的新语义）。

## 7. L140–L162：`_compute`（驱动真实路径）

```python
    assert len(kv_manager_block_ids) == len(positions_per_req)
    num_reqs = len(kv_manager_block_ids)
    for row_idx, block_ids in enumerate(kv_manager_block_ids):
        block_table.add_row(block_ids, row_idx)
    block_table.commit_block_table(num_reqs)
    ...
    block_table.compute_slot_mapping(num_reqs, ...cuda(), ...cuda())
```

- **L146（本轮 4.3 新增）**：断言两个入参的请求行数一致 —— 把 helper 的输入契约在 Python 层显式化，避免不一致输入进到 GPU kernel 后产生难诊断失败。**只加这一条**，不做过度防御。
- **`commit_block_table` 为什么必须**：`add_row` 只写 CPU 侧 `block_table.np`；kernel 读 `block_table.gpu`。**不 commit 就读到全 0 表**。
- **dtype**：`query_start_loc` int32、`positions` int64（对照 kernel 签名 L383-384）。
- **本轮明确保留**：走真实 `add_row → commit → compute_slot_mapping` 路径。

## 8. L165–L177：`_check`（通用断言）

```python
    np.testing.assert_array_equal(actual[:num_tokens], expected[:num_tokens])
    assert (actual[num_tokens:] == PAD_SLOT_ID).all()
    return actual[:num_tokens]
```

- 真实 token 段与 oracle 比对；**尾部整段**必须 `PAD_SLOT_ID`。
- **注意**：返回值固定是 `actual[:num_tokens]` —— 所以在调用方写 `assert len(actual) == num_tokens` 是**恒真**的（本轮 4.9 已删除该无效断言，见 §11）。

## 9. L180–L219：普通手算 Golden（2 例，**本轮三方锚定**）

```python
    ids=[
        "single_request_block_boundaries",
        "multiple_request_rows",
    ],
...
    expected_array = np.array(expected, dtype=np.int64)
    actual, num_tokens = _compute(...)
    np.testing.assert_array_equal(actual[:num_tokens], expected_array)      # Actual == Golden
    reference = _reference_slot_mapping(...)
    np.testing.assert_array_equal(reference[:num_tokens], expected_array)   # Reference == Golden
    assert (actual[num_tokens:] == PAD_SLOT_ID).all()
```

- **本轮 4.4**：加可读 `ids=`（输出为 `[single_request_block_boundaries]` 等）。
- **本轮 4.5（重要）**：原来只验证 `Actual == Golden`；现**同时**验证 `Reference == Golden`。这把 oracle 本身也钉在手算值上——若 oracle 漂移，即使 kernel 正确也会被发现。**刻意不调用 `_check()`**，因为这里要显式展示三条证据路径。
- **本轮 4.6**：第一组输入由 `[[5,6,7,8]]` 精简为 `[[5,6,7]]`（positions 只用到逻辑块 0/1/2；块 8 从未被引用）。第二组保留 `[[9,2],[7,4]]` 以展示**每个请求有独立行**。
- **你要检查**：`position 47 // 16 = 2` → 取 `[5,6,7]` 的第 2 个 = **7** → `7*16+15 = 127`。（早前草稿曾误写 `block 8 → 143`，被 golden 抓到。）

## 10. L221–L249：边界 / Ragged / Prefix 参数化（6 例）

- **本轮 4.6**：ragged 最后一个请求由 `[40,41,42]` 精简为 `[40,41]`（17 token 只需 2 块）。
- **本轮 4.7**：prefix 注释改为
  ```python
  # Tokens scheduled after a cached prefix use absolute, non-zero
  # positions rather than indices into the current batch.
  ```
  原「Decode steps of a request…」易被误解为「单 token decode」；新措辞准确表达「position 是绝对位置、不从本 batch 的 0 开始」，该用例正是防止把 batch 内下标当成模型 position。
- **覆盖**：单 token / 块内首末 / 跨多块(64) / 非连续物理块(96) / ragged 5 请求 / prefix offset(70–89)。
- **你要检查**：各请求块数是否仍够用（`60 token → 4 块`✓、`17 token → 2 块`✓）。

## 11. L252–L272：跨 Triton Tile Padding（**本轮强化**）

```python
    num_tokens = 1030
    block_size = 16
    num_blocks = (num_tokens + block_size - 1) // block_size
    block_table = _make_block_table(
        kv_cache_block_size=block_size,
        max_num_batched_tokens=2100,
        max_num_blocks_per_req=num_blocks,
    )
    actual = _check(...)
    # Anchor the real tokens that cross from the first tile into the second:
    # block ids start at 1, so position 1023 is block 64 offset 15 (slot 1039)
    # and position 1024 is block 65 offset 0 (slot 1040).
    np.testing.assert_array_equal(
        actual[1023:1030], np.arange(1039, 1046, dtype=np.int64)
    )
```

- **本轮 4.8**：`num_blocks` 改为**通用向上取整**（原 `num_tokens // 16 + 1` 硬编码 16，且整除时会多分一块）；并显式传 `kv_cache_block_size=block_size` 让依赖关系可见。
- **本轮 4.9**：删除恒真的 `assert len(actual) == num_tokens`（只验证 helper 切片），换成**手算 tile 边界锚点**：物理块从 1 开始 → `position 1023` = 块 64 offset 15 = **1039**；`position 1024` = 块 65 offset 0 = **1040**；…`1029` → **1045**。
- **为什么 1030**：`> 1024` 且非 1024 整数倍；buffer 2100 让 padding 尾部再跨一个 tile。
- **你要检查**：`1023 // 16 = 63` → 第 63 个块 = `list(range(1,66))[63]` = **64** → `64*16+15 = 1039` ✓。

## 12. L275–L298：DCP 全 Rank 分区（**本轮精简 + 显式 stack**）

```python
    kv_manager_block_ids = [[1, 2, 3]]
    ...
    owning_ranks: list[np.ndarray] = []
    ...
    ownership = np.stack(owning_ranks, axis=0)
    np.testing.assert_array_equal(
        ownership.sum(axis=0),
        np.ones(len(positions_per_req[0]), dtype=np.int64),
    )
```

- **本轮 4.10**：块 id 由 32 个精简为 `[[1, 2, 3]]`（96 positions 在 world=2 最多需 3 KV 块、world=4 需 2 块）。仍从 **1** 开始，避免默认 0 值掩盖错误。
- **本轮 4.11**：`np.sum(list)` 改为 `np.stack(..., axis=0)` 再 `.sum(axis=0)` —— 明确 rank 是第 0 维、隐含要求各 rank mask 同 shape、避免对 Python list 的隐式转换；并给 `owning_ranks` 加类型标注。
- **不变量**：每个 token **恰被一个 rank 拥有**（不重不漏）。这比逐值比对更能抓分片逻辑错误。

## 13. L301–L330：DCP 手算 Golden（**本轮跨虚拟块**）

```python
    actual, num_tokens = _compute(block_table, [[5, 6]], [[0, 1, 2, 3, 4, 5, 32, 33]])
    ... np.array([PAD, 80, PAD, 81, PAD, 82, PAD, 96], dtype=np.int64)
```

- **本轮 4.12**：positions 由 `[0..5]` 扩展为 `[0,1,2,3,4,5,32,33]`。原用例中 **KV 块 6 从未被使用**；现在 `position 33` 落入**第 2 个虚拟块**并用到块 6 → 覆盖「跨虚拟块 + 第二个 KV 块」。
- 配置：block 16、world 2、rank 1、interleave 1 → 虚拟块 = 32；rank 1 拿**奇数** offset。
- `position 32`（偶）→ 属 rank 0 → PAD；`position 33`（奇）→ 虚拟块 1、local offset 0 → 块 **6** → `6*16+0 = 96`。
- **本轮 4.14**：期望数组显式 `dtype=np.int64`。
- **本轮明确保留**：本用例直接调 `_compute()`（golden 路径），**不**替换为 reference。

## 14. L333–L350：Hybrid 泛化（32/16 与 64/16）

```python
    assert block_table.use_hybrid_blocks
    assert block_table.blocks_per_kv_block == kv_cache_block_size // kernel_block_size
    num_tokens = 3 * kv_cache_block_size
    _check(block_table, [[5, 9, 2]], [list(range(num_tokens))])
```

- 先确认 hybrid 配置生效，再跑 `3 × 分配块` 个 token，**同时跨 kernel 块与分配块边界**。
- **注意**：这两例**依赖 oracle**，所以 §5 的独立性修复对它们至关重要（变异测试正是杀在这两例 + golden）。

## 15. L353–L372：Hybrid 手算 Golden（**本轮扩到 position 48**）

```python
    actual, num_tokens = _compute(block_table, [[5, 9]], [[0, 15, 16, 31, 32, 47, 48]])
    ... np.array([160, 175, 176, 191, 288, 303, 304], dtype=np.int64)
```

- **本轮 4.13**：新增 `position 48`。KV 块 9 展开为 kernel 块 **18、19**；原用例最大到 47，**只用到块 18**，块 19 从未被触及。现在 `position 48`（分配块内 offset 16）进入 kernel 块 **19**、offset 0 → `19*16 = 304`，覆盖「同一分配块内跨到第二个 kernel 子块」。
- **本轮同时修正 docstring**：明确 `position 47 → kernel block 18, slot 303`（早前误写块 19，与 303 自相矛盾）。
- **你要检查**：`47 % 32 = 15`、`15 // 16 = 0` → 仍在块 18；`48 % 32 = 16`、`16 // 16 = 1` → 进块 19。

## 16. L375–L384：`SlotMappingMode.NONE`

```python
    sentinel = -12345
    assert sentinel != PAD_SLOT_ID
    block_table.slot_mapping.gpu.fill_(sentinel)
    actual, _ = _compute(block_table, [[5, 6]], [[0, 1, 2]])
    np.testing.assert_array_equal(actual, np.full_like(actual, sentinel))
```

- **契约**：Mamba/GDN 类把 block table 当**状态索引**，`compute_slot_mapping` 在该模式下**直接 return**（`block_table.py:189-192`），完全不碰 buffer。
- **sentinel 设计**：不依赖未初始化显存；且**必须区别于** `PAD_SLOT_ID(-1)`，否则「没写」与「写了 PAD」无法区分 —— `assert sentinel != PAD_SLOT_ID` 把该前提固化。
- **本轮 4.15**：`assert (actual == sentinel).all()` 改为 `np.testing.assert_array_equal(actual, np.full_like(actual, sentinel))` —— 失败时能显示**具体差异位置**，而不是只报 False。语义不变。

---

## 17. 建议实际执行的对读
```bash
cd /home/ciaran/projects/vllm-m13-upstream
sed -n '182,215p' vllm/v1/worker/block_table.py    # compute_slot_mapping 调用侧
sed -n '379,443p' vllm/v1/worker/block_table.py    # kernel 全文
grep -n "map_to_kernel_blocks(" tests/v1/worker/test_block_table_slot_mapping.py   # 必须无输出
```

## 18. 已知限制（审查时需知情）
- 未真起多卡：DCP 通过注入已解析的 `dcp_world_size/dcp_rank`（CP 是纯 index 数学；CPU fallback 明确不支持 CP）。**本轮明确不改为多进程**。
- 未覆盖 `num_reqs=0` / 长度 0 请求 / `num_tokens=0`：MRV1 调用侧不产生此类输入，`np.concatenate` 空批本身非法；仅记录不写期望。
- 未测 MRV2 的 `compute_slot_mappings`（复数，`vllm/v1/worker/gpu/block_table.py`）——另一路径，不扩大 scope。
- 本机 `test_gpu_block_table.py` 有 3 个失败，为**基线环境问题**（WSL2 `UVA is not available`），移除本文件同样失败，与本改动无关。
