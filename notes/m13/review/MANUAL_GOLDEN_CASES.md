# M13 · 手算 Golden 完整推导（MANUAL_GOLDEN_CASES）

> **用途**：让你**不运行任何代码**就能独立验证测试里写死的期望值。
> 若你的手算与下面任何一步不符，**先怀疑测试、不要怀疑自己**——然后我们一起查。
>
> 公式基准：`vllm/v1/worker/block_table.py` 的 kernel（`:379-442`），upstream/main `72cd542`。

---

## 0. 公式速查（三合一）

```
virtual_block_size   = kv_cache_block_size * dcp_world_size
virtual_block_index  = position // virtual_block_size
virtual_block_offset = position %  virtual_block_size

is_local             = (virtual_block_offset // interleave) % dcp_world_size == dcp_rank
local_offset         = (virtual_block_offset // (dcp_world_size * interleave)) * interleave
                       + (virtual_block_offset % interleave)

kernel_block_ids     = 对每个 kv_block_id 展开: kv_block_id * blocks_per_kv_block + [0..bpk-1]
block_index          = virtual_block_index * blocks_per_kv_block
                       + local_offset // kernel_block_size
block_id             = kernel_block_ids[block_index]

slot                 = block_id * kernel_block_size + (local_offset % kernel_block_size)
非 local 或尾部 padding = PAD_SLOT_ID = -1
```

**退化检查**：`dcp_world_size=1, interleave=1, bpk=1` 时 →
`virtual_block_size = kv_cache_block_size`、`is_local` 恒真、`local_offset = offset`、
`block_index = position // block_size` → 回到最朴素公式。✓

---

## 1. 普通 Block Mapping（对应测试 L179–L207，2 个 case）

### Case 1-A：`([[5, 6, 7, 8]], [[0, 15, 16, 47]])`
配置：`kv_cache_block_size = kernel_block_size = 16`，`bpk = 1`，`dcp_world_size = 1`，`interleave = 1`。
- `virtual_block_size = 16 * 1 = 16`
- `bpk = 1` → `kernel_block_ids = [5, 6, 7, 8]`（不展开）
- `is_local` 恒为真（world=1，`0 % 1 == 0`）

| position | v_index = pos//16 | v_offset = pos%16 | local_offset | block_index | block_id | slot = id*16 + off%16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 + 0//16 = **0** | `[5,6,7,8][0]` = **5** | 5×16 + 0 = **80** |
| 15 | 0 | 15 | 15 | 0 + 15//16 = **0** | **5** | 5×16 + 15 = **95** |
| 16 | 1 | 0 | 0 | 1 + 0 = **1** | `[..][1]` = **6** | 6×16 + 0 = **96** |
| 47 | **2** | 15 | 15 | 2 + 0 = **2** | `[..][2]` = **7** | 7×16 + 15 = **127** |

**期望数组 = `[80, 95, 96, 127]`** ✓（与测试 L187 一致）

> ⚠️ **易错点**：`47 // 16 = 2`（不是 3），所以取第 3 个元素 **7**，不是 8。
> 本轮之前这里曾误写 `block 8 → 143`，被这个 golden 抓出——golden 的价值就在此。

### Case 1-B：`([[9, 2], [7, 4]], [[3], [20]])`（两请求各有独立行）
| req | position | v_index | v_offset | block_index | 该 req 的行 | block_id | slot |
|---:|---:|---:|---:|---:|---|---:|---:|
| 0 | 3 | 0 | 3 | 0 | `[9, 2]` | **9** | 9×16 + 3 = **147** |
| 1 | 20 | 1 | 4 | 1 | `[7, 4]` | **4** | 4×16 + 4 = **68** |

**期望数组 = `[147, 68]`** ✓（与测试 L193 一致）
> 关键：req 1 的 `position 20` → `20//16 = 1` → 取**自己那一行**的第 1 个元素（`[7,4][1] = 4`），不是 req 0 的行。

---

## 2. DCP Golden：world_size = 2, rank = 1, interleave = 1（对应测试 L281–L296）

配置：`kv_cache_block_size = kernel_block_size = 16`，`bpk = 1`，`dcp_world_size = 2`，`dcp_rank = 1`，`interleave = 1`，KV 块行 `[5, 6]`。
- `virtual_block_size = 16 × 2 = 32` ← **一个 KV 块被 2 个 rank 分摊，虚拟跨度翻倍**
- `is_local = (offset // 1) % 2 == 1` → **offset 为奇数才属于 rank 1**
- `local_offset = (offset // 2) * 1 + (offset % 1) = offset // 2`

| position | v_index | v_offset | `offset % 2` | is_local(rank1) | local_offset = off//2 | block_index | block_id | slot |
|---:|---:|---:|---:|:---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | ❌ 属 rank0 | — | — | — | **-1 (PAD)** |
| 1 | 0 | 1 | 1 | ✅ | 0 | 0 + 0//16 = 0 | **5** | 5×16 + 0 = **80** |
| 2 | 0 | 2 | 0 | ❌ | — | — | — | **-1 (PAD)** |
| 3 | 0 | 3 | 1 | ✅ | 1 | 0 + 1//16 = 0 | **5** | 5×16 + 1 = **81** |
| 4 | 0 | 4 | 0 | ❌ | — | — | — | **-1 (PAD)** |
| 5 | 0 | 5 | 1 | ✅ | 2 | 0 | **5** | 5×16 + 2 = **82** |

**期望数组 = `[-1, 80, -1, 81, -1, 82]`** ✓（与测试 L293-296 一致）

**直觉**：rank 1 只存「奇数位」token 的 KV，并把它们**紧凑地**放在自己的块里（本地偏移 0,1,2…），而不是留空洞。
**交叉检查**：同样输入下 rank 0 应得 `[80, -1, 81, -1, 82, -1]`（偶数位）；两者「按位或」覆盖全部 6 个 token，恰好各一次 → 这正是测试 §12 的守恒律。

---

## 3. Hybrid Golden：kv=32 / kernel=16（对应测试 L319–L335）

配置：`kv_cache_block_size = 32`，`kernel_block_size = 16` → `bpk = 32/16 = 2`；`dcp_world_size = 1`，`interleave = 1`；KV 块行 `[5, 9]`。

**第 1 步：展开 kernel 块 id**（`kv_block_id * 2 + sub_block`）
- KV 块 5 → `5*2+0 = 10`, `5*2+1 = 11`
- KV 块 9 → `9*2+0 = 18`, `9*2+1 = 19`
- ⇒ `kernel_block_ids = [10, 11, 18, 19]`

**第 2 步**：`virtual_block_size = kv_cache_block_size * world = 32 * 1 = 32`（**注意用分配块 32，不是 kernel 块 16**）

| position | v_index = pos//32 | v_offset = pos%32 | local_offset | block_index = v_index*2 + off//16 | block_id | slot = id*16 + off%16 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0*2 + 0//16 = **0** | `[10,11,18,19][0]` = **10** | 10×16 + 0 = **160** |
| 15 | 0 | 15 | 15 | 0 + 15//16 = **0** | **10** | 10×16 + 15 = **175** |
| 16 | 0 | 16 | 16 | 0 + 16//16 = **1** | **11** | 11×16 + 0 = **176** |
| 31 | 0 | 31 | 31 | 0 + 31//16 = **1** | **11** | 11×16 + 15 = **191** |
| 32 | **1** | 0 | 0 | 1*2 + 0 = **2** | `[..][2]` = **18** | 18×16 + 0 = **288** |
| 47 | 1 | 15 | 15 | 1*2 + 15//16 = **2** | **18** | 18×16 + 15 = **303** |

> ⚠️ **易错点**：`position 47` 落在 kernel 块 **18**（不是 19）。因为 `47 % 32 = 15`、`15 // 16 = 0`，
> 仍在该分配块的**第一个** kernel 子块内。若误以为是块 19，会算出 `19*16+15 = 319`（错）。
> **本轮已修**：测试注释原写「kernel block 19 … = 303」自相矛盾（303 = 18×16+15，对应块 18），
> 现已改为块 18；**期望值 303 自始正确**，已用独立脚本复核。

**期望数组 = `[160, 175, 176, 191, 288, 303]`** ✓（与测试一致）

**核心直觉**：一个 32-token 的分配块 = 两个 16-token 的 kernel 块；`position 32` 进入第 2 个**分配**块（KV 块 9），其首个 kernel 子块是 `9*2 = 18`。

---

## 4. 自测清单（不看代码，凭手算回答）
1. `kv=16/kernel=16`、块 `[3,7,1,5]`、`position 33` 的 slot 是多少？（提示：`33//16=2` → 块 1 → `1*16+1 = 17`）
2. world=4、rank=2、interleave=1、块 `[5,...]`、`position 6` 属于谁？（`6%64=6`，`6%4=2` → **属 rank 2**，`local_offset = 6//4 = 1` → `5*16+1 = 81`）
3. kv=64/kernel=16、KV 块 `[3]`、`position 20` 的 slot？（`bpk=4` → 展开 `[12,13,14,15]`；`vbs=64`，`v_index=0`,`off=20` → `block_index = 0 + 20//16 = 1` → 块 13 → `13*16 + 4 = 212`）
