# M13 · 首次开源 PR — 研究笔记（候选 C1：`compute_slot_mapping` 单元测试）

> **本笔记目的**：记录 M13「首次上游贡献」的**研究阶段**全过程与结论——为什么选 C1、缺口是否真实、上游有无前车、kernel 到底做什么、测试怎么设计、提交怎么合规。保留研究痕迹，供后续动手与复查。
> **状态**：研究完成，**尚未动手写测试/提 PR**。下一步见 §8。
> **边界**：本阶段只做「复现→定位→设计」，不改上游代码、不提交。

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

### 复活 vs 全新，二选一（动手前定）
- **A. 复活 #30683**（推荐）：基于其分支 rebase 到 main、修 pre-commit、补 M8 增量用例。礼貌起见可在 PR 里 credit 原作者 pisceskkk（`Co-authored-by:` 或描述致谢）。**风险**：拿不到原分支写权限 → 退回 B。
- **B. 全新 PR**：新写 `test_block_table.py`，**在描述里引用 #30683 为前车**（说明「复活其思路 + 扩展覆盖」）。更简单、干净。

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
