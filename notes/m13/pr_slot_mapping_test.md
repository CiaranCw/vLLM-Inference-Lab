# M13 · 首次开源 PR — 研究笔记（候选 C1：`compute_slot_mapping` 单元测试）

> **阅读顺序**：本文件分两大部分。
> - **【第一部分】当前 main（`a1657a0`，2026-08-04 09:xx UTC 实测）** —— 权威、以此为准。
> - **【第二部分】历史快照（v0.24.0 / `ee0da84`，2026-07-28）** —— 仅存档研究过程，**其中多条结论已被推翻**（逐条见 §A.4）。
>
> **当前状态**：已在干净工作树基于最新 main 完成实现与验证（**18 passed**、pre-commit 绿、零产品改动、Oracle 独立性经变异测试证明），**尚未 commit / 尚未 push / 尚未建 PR**。断点见 `STATUS.md`，逐行审查材料见 `review/`。

---

# 【第一部分】当前 main（权威）

## A.1 事实基线
| 项 | 值 |
|---|---|
| 核查时间 | **2026-08-04 09:22 UTC 起**（本轮；前两轮 06:18 / 03:21） |
| upstream/main | **`a1657a0235a77ce85f8d348c83ff4adb29918b71`**（2026-08-04T08:03:46Z）——上轮 `72cd542`，前进 1 commit（仅 `rust/src/chat/renderer/kimi_k3/`，与本 PR 无关，已安全 port） |
| 最新 release | v0.26.0（2026-07-27） |
| 干净工作区 | `/home/ciaran/projects/vllm-m13-upstream`（独立 clone + 独立 `.venv`），分支 `m13/block-table-slot-mapping-tests`，HEAD = `a1657a0`（**无本地 commit**） |
| 目标文件 | `tests/v1/worker/test_block_table.py`（新增，**384 行 / 18 tests**，sha256 `b0a15b41…`；main 上此文件仍不存在） |
| 语义复核 | `block_table.py` 在 `cb81048..0b1c151` **零改动**；`AGENTS.md` / `docs/contributing/README.md` / `.pre-commit-config.yaml` / `tests/v1/worker/` 亦零改动 |
| 当前状态 | **READY FOR FINAL HUMAN CONFIRMATION BEFORE COMMIT**（未 commit / 未 push / 未建 PR；远程 CI 未运行） |

## A.2 MRV1 / MRV2 与「测哪个对象」
两条 block_table 并存，**必须分清**：
- `vllm/v1/worker/block_table.py` —— `BlockTable` / `MultiGroupBlockTable` / `SlotMappingMode` / `get_block_table_width`。被 **`gpu_input_batch.py`、`gpu_model_runner.py`（MRV1）、`tpu_input_batch.py`** 及 KV connector（mooncake/nixl）使用 → **活跃路径，本 PR 的目标**。
- `vllm/v1/worker/gpu/block_table.py` —— `BlockTables`（复数）+ `_compute_slot_mappings_kernel`（复数），被 **MRV2（`gpu/model_runner.py`）**、pcp_manager、spec-decode 使用。MRV2 仍从 V1 模块 import `get_block_table_width`。**已有测试 `test_gpu_block_table.py` 测的是它（staged writes / move_row），不测 slot mapping。**
- 另有 CPU fallback：`vllm/utils/cpu_triton_utils.py:_compute_slot_mapping_kernel_impl`（C++ op，**断言 `TOTAL_CP_WORLD_SIZE == 1`，即 CPU 不支持 CP**）→ CP 语义只能在 GPU 测。

## A.3 当前 kernel 的逐步公式（`block_table.py:379-442`，已实测验证）
调用侧 `compute_slot_mapping`（`:182-211`）关键点：
- `SlotMappingMode.NONE` → **直接 return，不触碰 slot_mapping buffer**（`:189-192`）。
- 传入 kernel 的 constexpr：`KV_CACHE_BLOCK_SIZE=self.kv_cache_block_size`、`BLOCKS_PER_KV_BLOCK=self.blocks_per_kv_block`、**`TOTAL_CP_WORLD_SIZE=self.dcp_world_size`**、**`TOTAL_CP_RANK=self.dcp_rank`**、`CP_KV_CACHE_INTERLEAVE_SIZE`、`PAD_ID=PAD_SLOT_ID(-1)`、`BLOCK_SIZE=1024`（Triton tile）。
- 传入的 `block_size` 参数是 **kernel block size**（`self.block_size`，混合时 = kernel_block_size）。

kernel 内（每 program 处理一个 req；最后一个 program 填 padding）：
```
virtual_block_size   = KV_CACHE_BLOCK_SIZE * dcp_world_size      # 注意用“分配块大小”
virtual_block_index  = pos // virtual_block_size
virtual_block_offset = pos %  virtual_block_size
is_local             = (virtual_block_offset // INTERLEAVE) % world == rank
local_block_offsets  = (virtual_block_offset // (world*INTERLEAVE))*INTERLEAVE
                       + (virtual_block_offset % INTERLEAVE)
block_indices        = virtual_block_index * BLOCKS_PER_KV_BLOCK
                       + local_block_offsets // kernel_block_size      # 混合块在此体现
block_numbers        = block_table[req, block_indices]                 # 已是 kernel block id
slot                 = block_numbers * kernel_block_size + local_block_offsets % kernel_block_size
slot                 = slot if is_local else PAD_ID
```
padding：最后一个 program 把 `[num_tokens, max_num_batched_tokens)` 按 1024 tile 填 `PAD_ID`。

**三块语义（务必分清）**：
- `kv_cache_block_size`（= 构造参数 `block_size`）：KV manager 分配粒度，决定 virtual block 跨度。
- `block_size`（= `kernel_block_size`）：attention kernel 粒度，**决定最终 slot 算术**。
- `blocks_per_kv_block = kv_cache_block_size // kernel_block_size`：一个分配块含几个 kernel 块；`add_row`/`append_row` **已自动**用 `map_to_kernel_blocks` 把 KV manager id 展开成 kernel id 写入表（`:146-149`）。故 oracle 必须从「公共 API 输入语义」重建，而不是偷读 kernel 输出。

**实测确认（`/tmp/m13_probe.py`，已删）**：
- 非混合 kv=kernel=16、blocks `[5,6,7,8]`、pos `0,15,16,20` → `[80,95,96,100]` ✓
- 混合 kv=32/kernel=16、KV blocks `[5,9]` → 表内 kernel `[10,11,18,19]`；pos `0,15,16,20,31,32,47` → `[160,175,176,180,191,288,303]` ✓
- DCP world2：rank0 拿偶 offset、rank1 拿奇 offset，非本 rank = `-1` ✓
- `SlotMappingMode.NONE`：预填 `-12345` 后调用，buffer **保持不变** ✓

## A.4 相对旧快照的漂移（旧结论已推翻）
1. **PCP 已不参与 slot mapping**：现在只按 **DCP** 分片（旧假设 `pcp × dcp` 错误）。`pcp_world_size/pcp_rank` 仍在 `__init__` 读取，但**不传入 kernel**。
2. **新增混合块语义**：`kv_cache_block_size` / `kernel_block_size` / `blocks_per_kv_block` 三分（旧只有单一 block_size）。
3. **新增 `SlotMappingMode`**：`TOKEN_TO_KV_SLOT`（默认）与 `NONE`（Mamba/GDN 状态缓存，跳过）。
4. **`[Test]` 不是合法 PR 前缀**：`docs/contributing/README.md:223+` 列表为 Bugfix/CI-Build/Doc/Model/Frontend/Kernel/Core/Hardware/Misc → **本轮定为 `[Core]`**（本 PR 不改 CI/Build/测试基础设施，保护的是 V1 核心 KV BlockTable 行为；上轮曾暂定 `[CI/Build]`，已改）。
5. **新增强制规则 `AGENTS.md`（根目录）**：禁裸 pip（必须 `uv`）、88 字符行宽、PR 须含查重说明+测试命令与结果+AI 披露、**纯 code-agent PR 不允许**（提交人须能端到端负责）、违规可**直接封禁**。
6. **测试环境不再需要 /tmp 绕过**：装 `requirements/test/cuda.in` 后真实 `tests/conftest.py` 可导入，正式从仓库根目录运行。

## A.4b 独立 Oracle 的必要性（本轮新增，含实证）

旧版 Oracle 调用产品 `BlockTable.map_to_kernel_blocks` 展开 hybrid 块——而产品 `add_row` 用的**是同一个函数**。后果：该 helper 若有 bug，**期望值与实际值一起错**，hybrid 测试照样绿。

**实证（注入 `map_to_kernel_blocks` 返回值 +1）**：

| 用例 | 修复前 | 修复后 |
|---|:--:|:--:|
| `hybrid_kernel_blocks[32-16]`（Oracle 驱动） | ✅ passed（**漏检**） | ❌ failed |
| `hybrid_kernel_blocks[64-16]`（Oracle 驱动） | ✅ passed（**漏检**） | ❌ failed |
| `hybrid_matches_hand_computed_slots`（写死 golden） | ❌ failed | ❌ failed |

**修复**：测试内自建 `_expand_kernel_block_ids`（`kernel_block_id = kv_block_id * bpk + sub_block`），Oracle 从原始 KV manager id 自行展开，**不调用任何产品 mapping helper、不读产品 `block_table.np`**。恢复后 `git diff -- vllm/ csrc/` 为空、18 tests 全绿。

**另一条已有证据**：kernel 级变异（`slot_offsets + 1`）→ 17/18 失败（仅 `SlotMappingMode.NONE` 用例应存活，因其不跑 kernel），证明整套测试非空转。

## A.4c 人工逐行审查后的最终增强（本轮）

一轮人工逐行技术审查未发现 slot mapping 数学 / DCP 语义 / hybrid 块语义 / `SlotMappingMode.NONE` 行为上的功能性阻塞问题。本轮只做**少量、明确、低风险**的增强，**不改产品语义、不扩大 PR scope**：

1. **三方锚定**：普通手算 Golden 现同时断言 `Actual == Golden` 与 `Reference == Golden`——把 oracle 本身也钉在手算值上（此前只验 Actual）。
2. **DCP Golden 跨虚拟块**：positions 扩到 `[0..5, 32, 33]`；`position 33` 用到此前从未被引用的 KV 块 6（虚拟块 1）→ slot 96。
3. **Hybrid Golden 跨第二个 kernel 块**：positions 加 `48`；此前 kernel 块 19 从未被触及；`position 48` → 块 19 offset 0 → slot 304。同时修正 docstring（`position 47` 属块 **18**，原误写 19）。
4. **Triton tile 边界手算断言**：删除恒真的 `assert len(actual) == num_tokens`（只验 helper 切片），改为 `actual[1023:1030] == 1039..1045`——锚定真实 token 从第一个 tile 跨入第二个；tile 块数改用通用向上取整。
5. **测试输入最小化**：普通 Golden `[[5,6,7,8]]`→`[[5,6,7]]`、ragged 末请求 `[40,41,42]`→`[40,41]`、DCP 分区 32 块→`[[1,2,3]]`（均为「恰好足够」，且块 id 仍从 1 开始以免默认 0 掩盖错误）。
6. **可读性与诊断**：普通 Golden 加 `ids=`；`_compute()` 加请求行数一致性 assert；ownership 显式 `np.stack(axis=0)`；手算期望数组显式 `dtype=np.int64`；NONE 断言改用 `assert_array_equal` 以显示差异位置；两处 docstring 措辞由「public API / public input semantics」收紧为「`BlockTable.compute_slot_mapping` / `BlockTable` input semantics」。

**新增 anchors 均已用独立脚本复核**（DCP 33→96、Hybrid 48→304、Tile 1023–1029→1039–1045 全部 MATCH）。变异回归复跑：注入产品 `map_to_kernel_blocks +1` → 3 个 hybrid 测试全失败，恢复后产品 diff 为空、18 tests 全绿。

## A.5 当前测试矩阵（18 tests，`tests/v1/worker/test_block_table.py`）
| 组 | 内容 |
|---|---|
| 手算 golden ×2（参数化，**三方锚定** Actual+Reference+Golden） | `single_request_block_boundaries`：blocks `[5,6,7]` pos `0,15,16,47` → `[80,95,96,127]`；`multiple_request_rows` → `[147,68]` |
| 手算 golden（DCP，**跨虚拟块**） | world2/rank1、pos `0..5,32,33` → `[-1,80,-1,81,-1,82,-1,96]` |
| 手算 golden（Hybrid，**跨第二 kernel 块**） | kv32/kernel16、KV blocks `[5,9]`、pos 含 48 → `[160,175,176,191,288,303,304]` |
| 边界 ×6 | 单 token / 块内首末 / 跨多块(64) / 非连续物理块(96) / ragged 5 请求 / prefix offset(70..89) |
| 跨 Triton tile | 1030 tokens（非 1024 倍数）+ padding 再跨一 tile（buffer 2100）；含手算锚点 `[1023:1030] == 1039..1045` |
| DCP ×4 | world{2,4} × interleave{1,4} 全 rank（块 id 精简为 `[[1,2,3]]`）；`np.stack` 断言「每 token 恰被一个 rank 拥有」 |
| Hybrid ×2 | (32,16) 与 (64,16) 端到端跨 kernel/分配块边界 |
| SlotMappingMode.NONE | sentinel −12345（≠ PAD −1）保持不变 |

**删除**旧草稿的 PCP×DCP 16 例矩阵（语义不成立，保留会误导）。**变异测试**：给 kernel `slot_offsets` 注入 +1 → **17/18 失败**（仅 NONE 用例应当存活），证明非空转；随后已还原。

## A.6 已知限制
- 未真实起多卡：DCP 通过设置 `block_table.dcp_world_size/dcp_rank` 覆盖（CP 分片是纯 index 数学，且 CPU fallback 明确不支持 CP，无法在 CPU 侧验）。
- 未覆盖 `num_reqs=0` / 某请求长度 0 / `num_tokens=0`：调用侧（MRV1）不会产生这类输入，且 `np.concatenate` 空批本身非法契约；如需覆盖应先确认上游契约，故仅记录不写期望。
- 未测 MRV2 的 `compute_slot_mappings`（复数，`gpu/block_table.py`）——属另一路径，本 PR 不扩大 scope。
- 本机 3 个邻近测试失败为**基线环境问题**（WSL2 `UVA is not available`），与本改动无关（移除本文件后同样失败）。

## A.7 与 PR #30683 的关系
`CLOSED`/未合并/`stale`（作者 @pisceskkk，closed 2026-04-09）。方向相同（为 block_table 加 UT），**故 PR 描述中致谢其提出该测试方向**；但**不夸大**：
- **不声称** maintainer 已接纳该方向（无明确 maintainer 评论证据；PR 描述措辞已相应收紧）；
- 其代码基于旧 API（`compute_slot_mapping(req_indices, positions)`）与已失效的 `pcp × dcp` 分片假设，**无法在当前 main 运行**。

本实现按当前 API 重写，并新增 hybrid / `SlotMappingMode` / 跨 tile padding 覆盖。

## A.8 下一步（PR 流程）
见 `STATUS.md` §7（命令已备好）。阻塞项：DCO 身份确认、AI 署名 trailer 口径、人工逐行审阅（AGENTS.md 硬要求）。

---
---

# 【第二部分】历史快照（v0.24.0 / `ee0da84`，2026-07-28 记录）

> ⚠️ **以下内容仅为历史存档**，其关于 kernel 语义（`pcp × dcp`）、无混合块、无 `SlotMappingMode`、`[Test]` 前缀、26/26 通过、/tmp 运行方式等结论**均已被 §A 推翻或取代**。保留以呈现研究演进过程。


---

## 0. TL;DR（先看结论）

- **选定候选**：C1 = 为 `vllm/v1/worker/block_table.py` 的 `compute_slot_mapping` / `_compute_slot_mapping_kernel` 补**单元测试**（纯加测试，不改产品行为，最适合第一次 PR）。
- **缺口真实、且在当前 `main` 仍成立**：`gh` 代码搜索 `compute_slot_mapping path:tests` → **0 命中**（§2）。
- **上游有前车 PR #30683**（`[UT][PCP&DCP] UT for block_table.py`）：**已被 stale 自动关闭**，但**不是因想法被否，而是卡在 pre-commit 检查失败后作者未跟进**（§3）。这既证明「维护者欢迎该测试」，又给出**最关键教训：pre-commit 必须过**。
- **最优策略**：**复活 #30683（rebase 到 main + 修 pre-commit）并用 M8 的互补边界用例扩展它**——#30683 强在 CP（dcp/pcp/interleave）维度，M8 强在非 CP 正确性边界（跨块 / 非连续物理块 / 乱序 / ragged / padding 哨兵 / tile 尾 mask）。二者合并 = 全面覆盖（§5、§6）。
- **本机可行性**：✅ 单卡 sm_120 / WSL2 完全够跑（单测规模小；CP interleave 是纯 index 数学，无需真多卡——M8 已证）。

---

## 1. 背景：M13 要什么，为什么是 C1

ROADMAP 的 M13 是第一阶段★核心验收之一：**完整走一次上游贡献流程**（复现→定位→改→测试→benchmark→提交→回应 review→改 PR），目标是**经历完整流程，非必须合入**。候选来自 `contribution_candidates.md`。

选 C1 的四条理由（详见 `contribution_candidates.md`）：
1. **有真实技术价值**：`slot_mapping` 是「逻辑 token 位置 → KV cache 物理槽位」的核心映射，错一位就静默写坏 KV、污染输出、极难定位；它有 CP interleave / padding 哨兵等易错分支却**零单测保护**。
2. **与已证能力精确对口**：M8 已实现并**三路交叉验证 22/22**（自写 Triton / PyTorch 语义 oracle / vLLM 官方 kernel），现成的 oracle + 用例矩阵。
3. **单卡可行、低风险**：纯加测试，不改行为，最坏是 review 让调整组织/命名。
4. **易被接受**：见 §3，同类 PR 曾被维护者欢迎。

---

## 2. Kill 标准核查①：缺口在当前 main 仍成立吗？

> 原则：候选是对 pinned v0.24.0 写的，但 PR 提到 `main`，**动手前必须对 main 重验**。

用 `gh`（登录态，代码搜索默认搜默认分支＝main）：

```bash
gh search code "_compute_slot_mapping_kernel" --repo vllm-project/vllm     # 只在产品代码出现
gh api "search/code?q=compute_slot_mapping+repo:vllm-project/vllm+path:tests"  # total_count = 0
```

**结果**：
- `_compute_slot_mapping_kernel` 只出现在**产品代码**：
  - `vllm/v1/worker/block_table.py`（Triton kernel 定义）
  - `vllm/utils/cpu_triton_utils.py`（CPU 版 `_compute_slot_mapping_kernel_impl`）
  - `vllm/v1/worker/cpu_model_runner.py`（为 CPU 路径 monkeypatch 替换）
  - `vllm/model_executor/warmup/qwen_triton_warmup.py`（**竟有专门 warmup**）
- `compute_slot_mapping` 在 `path:tests` → **0 命中**。

**结论**：**测试缺口在 main 上依然成立**。且该 kernel 跨 GPU/CPU 路径复用、还专门 warmup，**佐证其重要性**（越重要越该有测试）。

---

## 3. Kill 标准核查②：上游有没有人在做 / 做过？（关键发现）

`gh search prs` 查在途/历史 PR，命中一条决定性前车：

### PR #30683 `[UT][PCP&DCP] UT for block_table.py`（作者 pisceskkk）
- **它做了什么**：为 `compute_slot_mapping` 加单元测试，参数化覆盖 `dcp_world_size × dcp_rank × pcp_world_size × pcp_rank × cp_kv_cache_interleave_size`；文件 `tests/v1/worker/test_block_table.py`（+254 行）；自述 `17 passed`。
- **状态**：**CLOSED，标签 `stale`**。创建 2025-12-15，关闭 2026-04-09。
- **为什么流产**（看评论链，重要）：
  - `mergify` 机器人提示 **pre-commit 检查失败**，要求作者 `pre-commit install && pre-commit run --all-files` 后重推；
  - 作者**未跟进** → 90 天无活动被 `github-actions` 自动标 stale → 再 30 天自动关闭；
  - 还 CC 了维护者（`@LucasWilkinson` 等）——**说明想法被接纳，死于流程而非评审否决**。

### 其它相关 PR（非直接重复，但说明该区域活跃）
- `#46446 / #45245 / #42165`：都在给 slot-mapping kernel 加 **warmup**（预编译）——关注的是「kernel 首次 JIT 慢」，不是测试。
- `#47013`：bugfix，处理 KV-cache 消费侧 list slot_mapping。
- `#41270 / #41272`：从 attention metadata 移除未用的 slot_mapping。
- 均**不与「为生成 kernel 加单元测试」重复**。

**结论**：**无在途重复 PR**；存在一个**被欢迎但流产**的前车 #30683。→ 这不是障碍，而是**机会**：复活它、补齐流程、扩展覆盖。

---

## 4. Kernel 语义完整拆解（易懂版）

> 目标：任何人读完这节都能手算出 slot_mapping，从而能判断测试断言对不对。

### 4.1 它解决什么
KV cache 按 **block（页）** 分块存储。每个请求有一张 `block_table[row]`：第 `k` 个**逻辑块**存在哪个**物理块**。给定某 token 的**逻辑位置 `pos`**，要算它在 KV cache 里的**物理槽位 `slot`**（后续 `reshape_and_cache` 按此散写 K/V）。

### 4.2 基本公式（无 Context Parallel，即 world=1）
```
logical_block_idx = pos // block_size
offset            = pos %  block_size
physical_block    = block_table[row, logical_block_idx]
slot              = physical_block * block_size + offset
```
例：`block_size=16`，某请求 `block_table[row]=[5,6,7,8]`，`pos=20` →
逻辑块 `20//16=1` → 物理块 `block_table[row,1]=6` → 偏移 `20%16=4` → `slot=6*16+4=100`。

### 4.3 上游 kernel 的实际实现（`block_table.py:337-393`）
grid = `(num_reqs + 1,)`：**每个 program 处理一个请求的 token 区间**，最后多出的一个 program 负责**填 padding**。

- **前 `num_reqs` 个 program**（`req_idx` 对应第 req 个请求）：
  - 用 `query_start_loc[req_idx]..query_start_loc[req_idx+1]` 定位该请求 token 在扁平数组中的 `[start,end)` 区间（这就是 **ragged batch**：各请求长度不同、首尾相接）。
  - 以 `BLOCK_SIZE=1024` 为 tile 遍历该区间，`mask = offsets < end` 处理**尾部不整除**。
  - 按 §4.4 的 CP 公式算 `slot`，写回。
- **最后一个 program**（`req_idx == num_programs-1`）：
  ```python
  for i in range(num_tokens, max_num_tokens, BLOCK_SIZE):
      store(slot_mapping[i:], PAD_ID, mask=offsets < max_num_tokens)
  ```
  把 `[num_tokens, max_num_tokens)` 全填 `PAD_ID`（=`PAD_SLOT_ID`，值 -1）。**这是为 CUDA graph 定长 buffer 兼容**：真实 token 之外的槽位必须是明确的哨兵，不能是脏值。

### 4.4 Context Parallel（CP）interleave —— 最微妙的分支
CP 把一个序列的 KV **切给多个 rank** 存。引入三个 constexpr：
- `TOTAL_CP_WORLD_SIZE`（= `pcp_world_size * dcp_world_size`）：一共几个 rank 分。
- `TOTAL_CP_RANK`（= `pcp_rank * dcp_world_size + dcp_rank`）：当前是第几个 rank。
- `CP_KV_CACHE_INTERLEAVE_SIZE`：以多大**块**为单位做 round-robin 分配。

```python
virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE          # 逻辑块在“全局”视角更大
block_idx  = pos // virtual_block_size
voff       = pos %  virtual_block_size                          # 全局块内偏移
is_local   = (voff // INTERLEAVE) % WORLD == RANK               # 这个 token 归我吗？
loff       = (voff // (WORLD*INTERLEAVE)) * INTERLEAVE + (voff % INTERLEAVE)   # 我本地的偏移
slot       = block_table[row, block_idx] * block_size + loff
slot       = slot if is_local else PAD_ID                       # 不归我 → PAD
```

**worked example**：`block_size=16, WORLD=2, INTERLEAVE=1` → `vbs=32`。token 按 `voff` 奇偶分给两个 rank：
| voff | is_local(rank0) | loff(rank0) | is_local(rank1) | loff(rank1) |
|---|---|---|---|---|
| 0 | ✅ | 0 | ❌(PAD) | — |
| 1 | ❌(PAD) | — | ✅ | 0 |
| 2 | ✅ | 1 | ❌ | — |
| 3 | ❌ | — | ✅ | 1 |
→ rank0 拿到全部偶数 voff、本地偏移 `voff//2`；rank1 拿奇数、本地偏移 `voff//2`。**world=1 时该式退化为 §4.2 基本公式**（`is_local` 恒真、`loff=offset`）——这是测试必须覆盖的退化一致性。

### 4.5 PyTorch 语义 Oracle（测试参照，来自 M8 `scripts/m8_kernels.py:slot_mapping_gen_ref`）
逐请求、逐 token 复刻上述公式的**清晰慢速版**，作为断言基准（与 kernel 独立实现 → 交叉验证）：
```python
vbs = block_size * cp_world_size
for r in range(num_reqs):
    s, e = qsl[r], qsl[r+1]
    pos = positions[s:e]
    blk_idx = pos // vbs
    blk_num = block_table[r, blk_idx]
    voff    = pos - blk_idx * vbs
    is_local = (voff // interleave) % cp_world_size == cp_rank
    loff = (voff // (cp_world_size*interleave)) * interleave + (voff % interleave)
    slot[s:e] = where(is_local, blk_num*block_size + loff, PAD_ID)
```

---

## 5. 现有覆盖 vs 缺口（覆盖增量分析）

| 覆盖维度 | #30683（前车 PR） | M8 实验（本仓库） | 合并后 C1 目标 |
|---|:---:|:---:|:---:|
| DCP/PCP/interleave × rank 组合 | ✅ 强（参数化 17 例） | ✅（world 2/4、interleave 1/2） | ✅ |
| world=1 退化一致性 | ✅ | ✅ | ✅ |
| 跨块（seqlen ≫ block_size） | ？未见 | ✅ | ✅ |
| 非连续物理块号 | ？未见 | ✅ | ✅ |
| 乱序 block_table | ？未见 | ✅ | ✅ |
| ragged 多请求（变长首尾相接） | 部分 | ✅ | ✅ |
| 块内首/末位置（pos=0 / bs-1） | ？未见 | ✅ | ✅ |
| **padding 区 = PAD_ID**（CUDA graph 尾） | ？未见 | ✅（哨兵 -999 检测漏写） | ✅ |
| tile 尾 mask（长度不整除 tile） | ？未见 | ✅（tile=8） | ✅ |

> `？未见`：从 #30683 diff 的 grep 看未出现 `non_contig/shuffl/cross/sentinel` 等关键字，**推断未覆盖**；动手前应拉取 #30683 完整 diff 逐条确认，避免重复或遗漏。

**增量结论**：#30683 与 M8 **互补**。C1 = #30683 的 CP 矩阵 + M8 的非 CP 边界 + padding 哨兵，形成对该 kernel 的**全分支覆盖**。

---

## 6. 测试设计（待实现的 pytest 蓝本）

- **文件**：`tests/v1/worker/test_block_table.py`（沿用 #30683 的路径与命名，便于「复活/取代」）。
- **被测对象**：优先测**公开入口** `BlockTable.compute_slot_mapping(...)`（真实构造 `BlockTable`、设 CP 参数），而非直接 call 私有 kernel——这样测试更稳、贴近真实调用（#30683 即如此）。
- **参照**：把 §4.5 oracle **内联进测试文件**（上游不能依赖本仓库 `scripts/m8_kernels.py`）。
- **断言**：`np.testing.assert_array_equal(got, ref)`（含 padding 区 == PAD_ID；可选哨兵预填检测漏写）。
- **参数化矩阵**（合并 §5）：
  1. `dcp1_pcp1_interleave1`（退化基线）
  2. `dcp/pcp/interleave × rank` 组合（来自 #30683）
  3. 跨块 / 非连续物理块 / 乱序 / 块内首末（来自 M8）
  4. ragged 多请求（变长）
  5. padding：`num_tokens < max_num_tokens`，尾部全 PAD_ID
  6. tile 尾 mask（长度不整除）
- **设备/跳过**：kernel 走 Triton（GPU）。测试加 `@pytest.mark.skipif(not torch.cuda.is_available())`；对齐 `tests/v1/worker/` 现有约定（看 `test_gpu_block_table.py` 的 fixture 风格）。
- **CPU 路径**（可选加分）：`vllm/utils/cpu_triton_utils.py` 有 `_compute_slot_mapping_kernel_impl`，`cpu_model_runner` 会 monkeypatch——可加一组 CPU 用例覆盖，但**首版可先聚焦 GPU 主路径**，避免范围膨胀。

---

## 7. 提交流程合规（从 #30683 的死因学到的）

> #30683 死于 **pre-commit 未过 + 未跟进**。C1 必须在这两点上不犯错。

据 `vllm-src/docs/contributing/README.md`：
1. **DCO sign-off**：每个 commit `git commit -s`（`Signed-off-by:`），否则 CI 拒。
2. **pre-commit**（#30683 的死因）：`uv pip install pre-commit`（需 `>=4.5.1`）→ `pre-commit install` → `pre-commit run --all-files` 全绿再推。**这是硬门槛**。
3. **测试**：本地 `pytest tests/v1/worker/test_block_table.py` 全绿；附 Test Plan / Test Result 到 PR 描述（照 #30683 格式）。
4. **RFC 门槛**：>500 LOC（不含 kernel/数据）才需先 RFC；C1 仅一个测试文件（~300 行），无需 RFC。
5. **⚠ AI 辅助署名（需你决策）**：vLLM 要求**披露 AI 辅助**并加 `Co-authored-by:` trailer；这与本仓库 no-coauthor 约定（[[vllm-lab-no-coauthor-trailer]]）是**不同仓库、不同规则**。上游 PR 须遵 vLLM 政策——署名口径由你定。

### ⚠ 重大发现：#30683 的 API 已与当前 main 漂移（改变了策略）
拉取 #30683 完整 diff 后发现：它调用 **`block_table.compute_slot_mapping(req_indices, positions)`**（逐 token 的 `req_indices` 数组）、并在构造时 mock `get_dcp_group`/`get_pcp_group`。但**当前签名是 `compute_slot_mapping(num_reqs, query_start_loc, positions)`**（累积偏移 `query_start_loc`，非 req_indices）——API 在 #30683(2025-12) 之后变了。
- **含义**：#30683 的测试**无法在当前 main 直接跑**（会因签名不符失败）。
- **反而 M8 的 harness 恰好对齐当前 kernel 签名**（M8 是近期对当前 vllm-src 写的，用 `query_start_loc` + `TOTAL_CP_WORLD_SIZE`/`TOTAL_CP_RANK`/`CP_KV_CACHE_INTERLEAVE_SIZE`）。
- **策略定为 B（全新，但吸收两者）**：用**当前 API**新写 `tests/v1/worker/test_block_table.py`，采 M8 的 oracle 口径 + #30683 的 DCP4/PCP2 参数化风格，PR 描述里**引用并取代 #30683**、credit 原作者。

### 构造机制（读 `BlockTable.__init__` 与 CPU fallback 确认）
- `__init__(block_size, max_num_reqs, max_num_blocks_per_req, max_num_batched_tokens, pin_memory, device, kernel_block_size, cp_kv_cache_interleave_size)`。
- CP 参数：`__init__` 尝试 `get_pcp_group()/get_dcp_group()`，**未初始化分布式时捕获 `AssertionError` 回退 world=1**。→ 非 CP 用例无需 mock；CP 用例**最简做法＝构造后直接设 `bt.pcp_world_size/pcp_rank/dcp_world_size/dcp_rank`**（`compute_slot_mapping` 直接读这些属性算 total），比 mock import 更干净。
- kernel 读 `self.block_table.gpu`，而 `add_row` 只写 `.np` → **cuda 上必须 `commit_block_table(num_reqs)` 把 np→gpu**（#30683 用 `device=cpu`（np≡gpu）才免此步）。
- **设备**：CP interleave 逻辑**只在 GPU Triton kernel**（`vllm/utils/cpu_triton_utils.py` 的 CPU fallback 走 C++ op 且 `assert TOTAL_CP_WORLD_SIZE==1`）。→ 测试须 CUDA + `skipif(not cuda)`，且这为「CUDA-only 测试」提供正当理由。

### 当前状态（本轮止步于此）
- ✅ 草稿测试已写：`notes/m13/pr_draft/test_block_table.py`（当前 API + oracle + 非CP边界 + padding + DCP4/PCP2 interleave{1,16} + world=1 退化）。
- ✅ PR 描述草稿：`notes/m13/pr_draft/PR_DESCRIPTION.md`。
- ✅ **GPU 验证已通过**（GPU 空闲后运行）：real RTX 5080/sm_120 上对真实上游 kernel，**26 passed in 6.03s**（2 手算 golden 锚点 + 7 非CP边界 + 16 DCP4/PCP2 interleave{1,16}×8rank + 1 退化）。standalone 运行（绕开需额外依赖的 `tests/conftest`；测试不依赖任何 conftest fixture）。
- ✅ **审查补强**：加了 2 例**手算 golden 值**（独立于 numpy oracle），消除「oracle 镜像 kernel＝循环论证」的审稿隐患。
- ✅ **规范检查**：在 **vLLM `pyproject.toml` 配置下** `ruff check` 与 `ruff format --check` 均通过（注意：从仓库外用 ruff 默认配置会误报 I001，须在 vllm-src 内跑）。
- ⏸ **待干净环境补跑**：完整 `pre-commit run --all-files`（mypy/codespell 等其余 hook）与 CI 下 `pytest tests/`（需 `requirements/test.txt`，如 tblib）。
- ⏸ **未提交 PR**：待用户审查草稿 + 备干净工作树 + DCO sign-off 后再提上游。

---

## 8. 下一步（动手清单）

1. **准备干净工作树**（重要）：当前 `vllm-src` **工作树有大量本地学习标注**（~22 文件加中文注释 + `_lab_trace.py`；`git status` 见 §9）。**不能从这个脏树直接切分支提 PR**。→ 另做**干净 checkout**（或 fork + clone），或把标注 stash/隔离到独立分支。
2. 拉取 #30683 完整 diff，逐条确认其覆盖，敲定「复活(A) 或全新(B)」。
3. 按 §6 写 `tests/v1/worker/test_block_table.py`，内联 oracle。
4. `pre-commit run --all-files` + `pytest` 全绿（§7 硬门槛）。
5. `git commit -s`（DCO），推 fork，开 PR（描述含 Purpose / Test Plan / Test Result / 引用 #30683）。
6. 回应 review、迭代。**目标是完整经历流程，非必须合入。**

---

## 9. 研究留痕（可复查）

**核查命令（都在登录态 `gh` / 本地 vllm-src 上跑）**：
```bash
# 缺口重验（main）
gh search code "_compute_slot_mapping_kernel" --repo vllm-project/vllm
gh api "search/code?q=compute_slot_mapping+repo:vllm-project/vllm+path:tests"   # total_count=0

# 前车 PR
gh pr view 30683 --repo vllm-project/vllm --json ...        # stale，死于 pre-commit
gh pr diff 30683 --repo vllm-project/vllm                   # +254, tests/v1/worker/test_block_table.py，CP 参数化

# 本机 pinned 源码核对
grep -rln compute_slot_mapping /home/ciaran/projects/vllm-src/tests/    # 0 命中
```

**关键事实快照（截至研究时）**：
- pinned：`vllm-src` HEAD = v0.24.0 / `ee0da84`（committed）；**但工作树脏**（本地学习标注，见下）。
- kernel 位置：`vllm/v1/worker/block_table.py:337 _compute_slot_mapping_kernel`；入口 `BlockTable.compute_slot_mapping()`（`:145`）。
- oracle：`scripts/m8_kernels.py:slot_mapping_gen_ref`；用例矩阵：`scripts/m8_test.py:test_generate`（12 组 / 展开 22 例）。
- `vllm-src` 本地修改文件（`git status --short`，均为学习标注/M9 门控，与 C1 无关但影响「干净分支」）：
  `vllm/v1/worker/{block_table,gpu_model_runner,gpu_input_batch,utils}.py`、`vllm/v1/attention/*`、`vllm/v1/core/*`、`vllm/entrypoints/*` 等 ~22 文件 + 未跟踪 `vllm/_lab_trace.py`。

---

*研究阶段结束。首选路径：复活/取代 #30683 + 合并 M8 边界矩阵，全分支覆盖 `compute_slot_mapping`。动手前先备干净工作树、拉 #30683 完整 diff、过 pre-commit。*
