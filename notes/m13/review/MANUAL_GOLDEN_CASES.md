# M13 · 手算 Golden 完整推导（MANUAL_GOLDEN_CASES）

> **用途**：让你**不运行任何代码**就能独立验证测试里写死的期望值。
> 若你的手算与下面任何一步不符，**先怀疑测试、不要怀疑自己**，然后我们一起查。
>
> 公式基准：`vllm/v1/worker/block_table.py` kernel（`:379-442`），upstream/main **`0b1c151`**。
> 对应测试文件 SHA256：`b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`（384 行）。

---

## 0. 公式速查

```
virtual_block_size   = kv_cache_block_size * dcp_world_size
virtual_block_index  = position // virtual_block_size
virtual_block_offset = position %  virtual_block_size

is_local             = (virtual_block_offset // interleave) % dcp_world_size == dcp_rank
local_offset         = (virtual_block_offset // (dcp_world_size * interleave)) * interleave
                       + (virtual_block_offset % interleave)

kernel_block_ids     = 每个 kv_block_id 展开为: kv_block_id * blocks_per_kv_block + [0..bpk-1]
block_index          = virtual_block_index * blocks_per_kv_block
                       + local_offset // kernel_block_size
block_id             = kernel_block_ids[block_index]

slot                 = block_id * kernel_block_size + (local_offset % kernel_block_size)
非 local / 尾部 padding = PAD_SLOT_ID = -1
```

**退化检查**：`world=1, interleave=1, bpk=1` → `virtual_block_size = kv_cache_block_size`、`is_local` 恒真、`local_offset = offset`、`block_index = position // block_size` → 回到最朴素公式。✓

---

## 1. 普通 Golden（测试 L180–L219，2 例，**三方锚定**）

> 本轮起，这两例同时断言 **Actual == Golden** 与 **Reference == Golden**。

### 1-A `single_request_block_boundaries`：`([[5, 6, 7]], [[0, 15, 16, 47]])`
配置：`kv=kernel=16`、`bpk=1`、`world=1`、`interleave=1` → `virtual_block_size = 16`，`kernel_block_ids = [5,6,7]`，`is_local` 恒真。

| position | v_index = pos//16 | v_offset = pos%16 | local_offset | block_index | block_id | slot |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | `[5,6,7][0]` = **5** | 5×16+0 = **80** |
| 15 | 0 | 15 | 15 | 0 + 15//16 = 0 | **5** | 5×16+15 = **95** |
| 16 | 1 | 0 | 0 | 1 | `[5,6,7][1]` = **6** | 6×16+0 = **96** |
| 47 | **2** | 15 | 15 | 2 | `[5,6,7][2]` = **7** | 7×16+15 = **127** |

**期望 = `[80, 95, 96, 127]`** ✓
> ⚠️ `47 // 16 = 2`（不是 3）→ 取第 3 个元素 **7**。早前草稿误写 `block 8 → 143`，被此 golden 抓出。
> **本轮 4.6**：输入由 `[[5,6,7,8]]` 精简为 `[[5,6,7]]`（块 8 从未被引用）。

### 1-B `multiple_request_rows`：`([[9, 2], [7, 4]], [[3], [20]])`
| req | position | v_index | v_offset | block_index | 该 req 的行 | block_id | slot |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 3 | 0 | 3 | 0 | `[9, 2]` | **9** | 9×16+3 = **147** |
| 1 | 20 | 1 | 4 | 1 | `[7, 4]` | **4** | 4×16+4 = **68** |

**期望 = `[147, 68]`** ✓
> 关键：req 1 的 `position 20 // 16 = 1` → 取**自己那一行**的第 1 个元素（`[7,4][1] = 4`），不是 req 0 的行。

---

## 2. Tile 边界锚点（测试 L252–L272，**本轮 4.9 新增**）

配置：`kv=kernel=16`、`bpk=1`、`world=1`；`num_tokens = 1030`；`num_blocks = ceil(1030/16) = 65`；块 id = `list(range(1, 66))` = `[1,2,…,65]`（**从 1 开始**，避免默认 0 掩盖错误）；buffer = 2100。

| position | v_index = pos//16 | v_offset | block_index | `[1..65][v_index]` | slot |
|---:|---:|---:|---:|---:|---:|
| 1023 | 1023//16 = **63** | 15 | 63 | `1 + 63` = **64** | 64×16+15 = **1039** |
| 1024 | 1024//16 = **64** | 0 | 64 | `1 + 64` = **65** | 65×16+0 = **1040** |
| 1025 | 64 | 1 | 64 | **65** | **1041** |
| 1026 | 64 | 2 | 64 | **65** | **1042** |
| 1027 | 64 | 3 | 64 | **65** | **1043** |
| 1028 | 64 | 4 | 64 | **65** | **1044** |
| 1029 | 64 | 5 | 64 | **65** | **1045** |

**期望 `actual[1023:1030] = [1039, 1040, 1041, 1042, 1043, 1044, 1045]`**（即 `np.arange(1039, 1046)`）✓

> **为什么锚在这里**：`position 1023` 是**第一个 Triton tile 的最后一个** token（tile = 1024），`position 1024` 进入**第二个 tile**。这条断言证明跨 tile 的真实 token 映射连续正确。
> 尾部 `[1030, 2100)` 由 `_check()` 断言全为 `PAD_SLOT_ID`，且该尾部本身又跨过第二个 tile 边界。
> **本轮同时**：`num_blocks` 改用通用向上取整 `(n + bs - 1) // bs`（原 `n//16+1` 硬编码且整除时多分一块）；删除恒真的 `assert len(actual) == num_tokens`。

---

## 3. DCP Golden：world=2, rank=1, interleave=1（测试 L301–L330，**本轮 4.12 扩展**）

配置：`kv=kernel=16`、`bpk=1`、`world=2`、`rank=1`、`interleave=1`、KV 行 `[5, 6]`。
- `virtual_block_size = 16 × 2 = **32**`（一个 KV 块被 2 rank 分摊，虚拟跨度翻倍）
- `is_local = (offset // 1) % 2 == 1` → **奇数 offset 属 rank 1**
- `local_offset = (offset // 2) * 1 + (offset % 1) = offset // 2`

| position | v_index | v_offset | offset%2 | is_local(rank1) | local_offset | block_index | block_id | slot |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | ❌ rank0 | — | — | — | **-1** |
| 1 | 0 | 1 | 1 | ✅ | 0 | 0 | `[5,6][0]`=**5** | 5×16+0 = **80** |
| 2 | 0 | 2 | 0 | ❌ | — | — | — | **-1** |
| 3 | 0 | 3 | 1 | ✅ | 1 | 0 | **5** | 5×16+1 = **81** |
| 4 | 0 | 4 | 0 | ❌ | — | — | — | **-1** |
| 5 | 0 | 5 | 1 | ✅ | 2 | 0 | **5** | 5×16+2 = **82** |
| **32** | **1** | 0 | 0 | ❌ rank0 | — | — | — | **-1** |
| **33** | **1** | 1 | 1 | ✅ | 0 | 1 + 0//16 = **1** | `[5,6][1]`=**6** | 6×16+0 = **96** |

**期望 = `[-1, 80, -1, 81, -1, 82, -1, 96]`**（dtype `int64`）✓

> **本轮为何扩展到 32/33**：原用例 positions 只到 5，**KV 块 6 从未被使用**。加入 `position 33` 后覆盖「跨虚拟块 + 用到第二个 KV 块」。
> **直觉**：rank 1 只存奇数位 token 的 KV，并**紧凑**放在自己块里（本地偏移 0,1,2…），不留空洞。
> **交叉检查**：同输入下 rank 0 应得 `[80, -1, 81, -1, 82, -1, 96, -1]`（偶数位）；两 rank 合起来恰好覆盖每个 token 各一次 → 正是 §12 分区测试的守恒律。

---

## 4. Hybrid Golden：kv=32 / kernel=16（测试 L353–L372，**本轮 4.13 扩展**）

配置：`kv_cache_block_size = 32`、`kernel_block_size = 16` → `bpk = 2`；`world=1`、`interleave=1`；KV 行 `[5, 9]`。

**第 1 步：展开 kernel 块 id**（`kv_id * 2 + sub`）
- KV 块 5 → `10, 11`；KV 块 9 → `18, 19` ⇒ `kernel_block_ids = [10, 11, 18, 19]`

**第 2 步**：`virtual_block_size = 32 × 1 = 32`（**用分配块 32，不是 kernel 块 16**）

| position | v_index = pos//32 | v_offset = pos%32 | local_offset | block_index = v_index*2 + off//16 | block_id | slot = id*16 + off%16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 + 0 = **0** | **10** | 10×16+0 = **160** |
| 15 | 0 | 15 | 15 | 0 + 0 = **0** | **10** | 10×16+15 = **175** |
| 16 | 0 | 16 | 16 | 0 + 1 = **1** | **11** | 11×16+0 = **176** |
| 31 | 0 | 31 | 31 | 0 + 1 = **1** | **11** | 11×16+15 = **191** |
| 32 | **1** | 0 | 0 | 2 + 0 = **2** | **18** | 18×16+0 = **288** |
| 47 | 1 | 15 | 15 | 2 + 15//16 = **2** | **18** | 18×16+15 = **303** |
| **48** | 1 | **16** | 16 | 2 + 16//16 = **3** | **19** | 19×16+0 = **304** |

**期望 = `[160, 175, 176, 191, 288, 303, 304]`**（dtype `int64`）✓

> **本轮为何加 position 48**：原用例最大到 47，**kernel 块 19 从未被触及**。`position 48` 在同一分配块内 offset 16 → 跨入**第二个 kernel 子块 19**。
> ⚠️ **易错点**：`position 47` 落在块 **18**（`47%32=15`、`15//16=0`，仍在第一个 kernel 子块），不是 19。若误算成 19 会得 `19*16+15 = 319`（错）。
> **核心直觉**：一个 32-token 分配块 = 两个 16-token kernel 块；`position 32` 进入第 2 个**分配**块（KV 块 9），其首个 kernel 子块是 `9*2 = 18`；再往后 16 个 token 才进 19。

---

## 5. 自测题（不看代码，凭手算回答）
1. `kv=kernel=16`、块 `[3,7,1,5]`、`position 33` 的 slot？ → `33//16=2` → 块 1 → `1*16+1 = **17**`
2. `world=4, rank=2, interleave=1`、块 `[5,…]`、`position 6` 属谁？ → `6%64=6`、`6%4=2` → **属 rank 2**；`local_offset = 6//4 = 1` → `5*16+1 = **81**`
3. `kv=64/kernel=16`、KV 块 `[3]`、`position 20`？ → `bpk=4` → 展开 `[12,13,14,15]`；`vbs=64`、`v_index=0`、`off=20` → `block_index = 0 + 20//16 = 1` → 块 13 → `13*16 + 4 = **212**`
