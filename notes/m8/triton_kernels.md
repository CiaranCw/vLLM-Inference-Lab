# M8 · Triton Kernel 能力建设（★核心验收 #5：slot-mapping 类 kernel）

> 目标：从"看懂 kernel"迈到"能写 kernel"。主验收是一个**与 vLLM 真实主链相关的 Triton kernel**——**slot_mapping 生成**（simplified `_compute_slot_mapping_kernel`），配 correctness 与 microbenchmark。外加三个基础练习打底。
> 关键纪律：**写对**（三路交叉验证）、性能**无数量级退化**、指标口径公平、差距可解释；**不追生产级实现、不扩展到未用分支、不伪造结果**。本模块**零 vLLM 源码改动**（把官方 kernel 作参照调用，不修改）。
>
> 证据：`outputs/m8/correctness.json`（22/22 全绿）、`outputs/m8/microbench.csv` + `bench_summary.json`、`logs/m8/*.log`。脚本：`scripts/m8_{kernels,test,bench}.py`。环境：Triton 3.6.0 / RTX 5080(sm_120, cap 12.0) / torch 2.11+cu130。复现见文末。

---

## 0. 先厘清一个最容易混的概念：slot_mapping 的「生成」与「消费」

`slot_mapping` 在 vLLM 里由**两个不同的 kernel** 前后接力，二者都属"slot-mapping 类"，但职责相反，**绝不能混为一谈**：

| kernel | 输入 → 输出 | 一句话 | 本模块 |
|---|---|---|---|
| **生成**：`_compute_slot_mapping_kernel`（`block_table.py:338`） | `positions + block_table + query_start_loc` → `slot_mapping` | 算出"每个 token 该写 KV cache 的哪个物理槽位" | ✅ **本模块实现** |
| **消费**：`reshape_and_cache_flash`（`_custom_ops.py` / attention backend） | `key/value + slot_mapping + kv_cache` → 写入 cache | 按算好的槽位把 K/V **散写**进分页 KV cache | ❌ 不实现（M4 已 trace） |

本模块做**生成侧**。它是**纯整数 index 运算**（int64 进、int64 出），所以消费侧那些 bf16/FP8/head-major 布局/量化 scale 的问题在这里**统统不适用**——这让 kernel 干净且易验证。公式（vLLM 源码 docstring 原文）：

```
逻辑块号 logical = position // block_size
块内偏移 offset  = position %  block_size
物理块号 phys    = block_table[req, logical]
slot            = phys * block_size + offset
```
> 直觉：`block_table` 是"逻辑块→物理块"的地址簿（M3 讲过）；生成侧就是拿每个 token 的绝对 `position` 查这本地址簿，换算成"物理槽位编号" `slot = 物理块×块大小 + 块内偏移"。消费侧再拿这个 slot 去写 K/V。

---

## 1. Triton 编程模型速讲（读懂后面 kernel 必备）

Triton 让你用 Python 写"**一个 program 处理一块数据**"的 GPU kernel，编译器负责向量化/调度：
- `tl.program_id(0)`：当前 program 的编号（第几个并行实例）。`grid=(N,)` 就启动 N 个 program。
- `tl.arange(0, BLOCK)`：一个 program 内处理的一段下标（向量化，`BLOCK` 是 `constexpr` 编译期常量）。
- `mask = offs < n`：**边界掩码**——尾部不足一整块时，用 mask 防越界读写（这是最常见的坑）。
- `tl.load(ptr + offs, mask=..., other=...)` / `tl.store(ptr + offs, val, mask=...)`：带掩码的读/写。
- `constexpr`：编译期常量（如 `BLOCK`、`CP_WORLD_SIZE`），不同取值会生成不同 kernel。
- **reduction**（softmax 用）：`tl.max(x, axis=0)` / `tl.sum(...)`，在一个 program 的向量内做归约。

> 重要教训：`@triton.jit` 函数**必须写在 .py 文件里**（Triton 要读源码做 JIT），heredoc/交互式会报 "could not get source code"。而且 `py_compile` **只验 Python 语法、不验 Triton JIT**——必须**真正 launch 一次**才算验证通过（本模块每个 kernel 都有实跑自测）。

---

## 2. 三个基础练习（达概念即止）
每个都有 PyTorch reference + correctness（`scripts/m8_kernels.py` 自测 + `m8_test.py`）：
1. **vector_add**：1D grid + mask。教 `program_id / arange / mask / load / store`。
2. **act_and_mul（SiLU-gate）**：`out = silu(x[:,:d]) * x[:,d:]`（正是 M7 profiler 里见到的 `vllm::act_and_mul_kernel`）。教**逐元素融合**（一个 kernel 里算 sigmoid、乘、再乘门控，省去中间张量与多次 launch）。对照 `F.silu(a)*b`。
3. **row-wise softmax**：每行一个 program，`x-=max` 数值稳定，`exp`、`sum`、除。教 **reduction + 数值稳定**。**特意测非 2 次幂列宽 127/128/255/512/1024**（`next_power_of_2`+mask），确保尾部掩码真的被走到。

---

## 3. 主验收：slot_mapping 生成 kernel（设计与地址计算）

`scripts/m8_kernels.py::slot_mapping_gen_kernel`，对齐 vLLM 官方结构的**简化版**：

- **grid = `(num_reqs + 1,)`**：每个 program 负责**一个请求**的 token 区间 `[query_start_loc[r], query_start_loc[r+1])`；**最后一个 program 负责 padding** —— 把 `[num_tokens, max_num_tokens)` 填 `PAD_ID=-1`（对齐 vLLM，为 CUDA graph 固定形状留位）。
- **请求内按 token tile 循环**：`for i in range(start, end, BLOCK)`，`mask = offs < end` 处理尾部不整除。
- **地址计算**：`blk_idx = pos // vbs`；`blk_num = block_table[req, blk_idx]`；`voff = pos - blk_idx*vbs`；`slot = blk_num*block_size + local_offset`。其中 `vbs = block_size * CP_WORLD_SIZE`。
- **Context Parallel interleave**（`CP_WORLD_SIZE>1` 时启用；纯 index 数学，单卡即可验证）：按 `is_local = (voff//INTERLEAVE) % CP_WORLD_SIZE == CP_RANK` 决定该 token 是否归本 rank，`local_offset` 做交织重排；**`CP_WORLD_SIZE=1` 时 `is_local` 恒真、退化为基础公式**。
- **stride 处理**：生成侧数据是 1D 整型数组（`positions`/`slot_mapping` 断言 **contiguous**）+ 2D `block_table`（用**传入的行 stride**，不假设列连续）。无 H 维、无多 dtype，故不存在"任意 stride vs contiguous"的纠结。
- **安全性**：token 侧 `store(mask=offs<end)`；被掩蔽 lane 的 `pos` 取 `other=0` → `blk_idx=0` → 读 `block_table[req,0]`（合法），其结果被 store mask 丢弃，**无越界/危险地址**（这与消费侧"`slot=-1` 要 `where(valid, slot, 0)` 防危险地址"是不同层面的问题）。

---

## 4. Correctness：三路功能交叉验证（22/22 全绿）

对同一输入跑**三种实现**并逐元素比较（int64 精确 → `torch.equal`，非 allclose）：
1. **自写 Triton**（`slot_mapping_gen`）
2. **PyTorch 语义 Oracle**（`slot_mapping_gen_ref`，逐请求、清晰可读——最重要的语义基准）
3. **vLLM 官方 `_compute_slot_mapping_kernel`**（确认与真实主链一致；已实测可 standalone import 调用）

> 这是"**功能交叉验证**"，不是形式证明，也不是三套完全独立的算法——PyTorch ref 定义语义，官方 kernel 锚定与 vLLM 的一致性。

覆盖用例（全 PASS，见 `outputs/m8/correctness.json`）：随机多请求、单 token、单请求、块内首/末位置、跨块、`non_contiguous_physical_blocks`、`shuffled_slot_assignment`、**请求长度不整除小 tile(=8) 触发尾部 mask**、ragged 多请求、**padding（输出缓冲预填哨兵 -999 检测漏写；pad 区必须全 -1）**、**CP interleave（world=2 rank0/rank1、interleave=2、world=4）与官方对齐**。基础三 kernel 亦全 PASS（含非 2 次幂 softmax）。

**两点如实说明**：
- 生成侧**允许重复 slot**（如前缀共享时不同请求映射到同一物理块）——这只是 index 计算，合法；"有效 slot 唯一/写竞争"是**消费侧**（散写）才关心的问题。为使 ref 比较干净，测试给各请求分配互不重叠的物理块。
- `non_contiguous_physical_blocks` / `shuffled_slot_assignment` 只验证"Block Pool 任意物理分配下 index 算得对"，**不声称完整模拟 prefix cache 生命周期**。

---

## 5. Microbenchmark（规模曲线）

`triton.testing.do_bench`（内部 CUDA event + warmup）计时，**计时区间仅含 kernel/op 调用**；输入张量与输出缓冲在计时区外分配（两个 kernel 复用预分配 out；torch 版天然分配中间张量，如实标注）。固定 `block_size=16`、单请求、各 impl 共享同一输入。

| num_tokens | 自写 Triton (us) | vLLM Triton (us) | PyTorch ref (us) |
|---:|---:|---:|---:|
| 128   | 7.33  | 7.20  | 26.40 |
| 512   | 7.70  | 7.81  | 30.56 |
| 1024  | 7.87  | 7.84  | 29.92 |
| 2048  | 8.35  | 8.35  | 30.05 |
| 4096  | 11.97 | 12.00 | 30.14 |
| 8192  | 16.67 | 16.70 | 30.18 |
| 16384 | 28.38 | 28.35 | 30.43 |

**观察与解释**：
- **自写 Triton ≈ vLLM 官方 kernel**（全尺寸几乎重合，差异在测量噪声内）→ 说明本 kernel 不仅**写对**、性能也**无数量级退化**，达到 M8 目标。
- **规模趋势**：两个 Triton kernel 在 ≤2048 近乎平坦（~7–8us，**单 kernel launch 的固定地板**，呼应 M7"小任务被 launch 开销主导"），之后随 num_tokens 增长（转为访存主导，搬运 int64 越来越多）。
- **PyTorch ref 小尺寸慢 ~3–4×**（~26–30us 平台期）：因为它是**多个独立 op**（`//`、`%`、`gather`、`*`、`+`）各自 launch，小数据下被**多次 launch 开销**主导；到 n=16384 计算量上来后与单 kernel 收敛（30.4 vs 28.4us）。这正是"**把多步融进一个 kernel 省 launch**"的价值（也是 M7 的核心结论在 kernel 编写侧的印证）。
- 指标 `effective_payload_bandwidth_GBps` = `(读 positions 8B + 写 slot 8B)×num_tokens / time`，是**逻辑有效载荷带宽，仅供实现间相对比较**，不代表硬件 DRAM 带宽（未计 block_table 读、事务放大、L2、地址计算）；本 kernel 逻辑载荷本就很小（int64 索引），故绝对 GB/s 不高属正常。

---

## 6. 局限（明确未覆盖的分支）
- **只实现生成侧**；消费侧散写（`reshape_and_cache_flash`）不实现（M4 已 trace，另属一支）。
- CP interleave 仅作**单卡纯 index 数学**验证（与官方 kernel 对齐），**未做真实多卡 dcp/pcp**。
- 未覆盖：多 KV-group / hybrid(mamba) 布局、消费侧的 FP8/head-major(5D)/量化 scale、TMA、autotune 搜索、生产 runtime 集成。
- 单卡 / 单 GPU 型号（sm_120）；microbench 的绝对数字仅本机本配置成立。

## 7. 承上启下
- 呼应 M3（block_table 逻辑→物理）、M4（slot_mapping 进入 KV 写入的闭环）：M8 亲手写出了"生成 slot_mapping"这一环，并与 vLLM 真实 kernel 对齐。
- 呼应 M7：microbench 再次量化了"多 op → 多次 launch"的开销，以及"融进单 kernel"的收益。
- 与 M9 并行：M9 将由 M7 的 profiler 证据驱动做一次真实 vLLM 改动。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
python scripts/m8_kernels.py     # 各 kernel launch 自测（JIT 真跑）
python scripts/m8_test.py        # correctness 三路交叉 -> outputs/m8/correctness.json（应 22/22）
python scripts/m8_bench.py       # 规模曲线 -> outputs/m8/microbench.csv + bench_summary.json
```
