# M22 · Tensor Parallel —— T + O（理论 + 源码定位）

> **状态**：**T + O 完成**；对应的 **E + H 实测结论见 [`results.md`](results.md)**（4×H20，2026-08-09）。
> **本文件仍只放理论与源码定位，不含把 H20 数字写进 T+O 段**——避免「源码笔记」与「实测报告」口径混写。
>
> **版本基准（T+O 阅读）**：`../vllm-src` @ `ee0da84ab9e04ac7610e28580af62c365e898389` = tag **v0.24.0**。
> H20 上实际加载的是同主版本 **0.24.0 wheel**（见 `logs/m22/environment.txt`）。
> 下文所有 `file:line` 均为上机前逐条实读核对。
>
> 本文件原用途：把多卡机时从「现场探索」变成「已验证脚本的数据采集」。
> 每一条 R 结论都对应实验脚本里的一个设计决定；实测是否支持假见 `results.md`。

---

## 0. 结论速览：三处会让原计划当场失败的事实

| # | 原计划的假设 | 真实源码 | 后果 |
|---|---|---|---|
| **A** | `max_model_len=4096` 且 prefill 扫到 `prompt_len=4096` | `v1/engine/input_processor.py:423-432`：`prompt_len == max_model_len` 且 runner 是 generate 时**直接 `ValueError`** | 4096 档 **100% 崩**，且是加请求时才崩 |
| **B** | 用 `VLLM_TORCH_PROFILER_DIR` 开 profiler | 该环境变量在 v0.24.0 **已不存在**（全仓零命中），改为 `ProfilerConfig` / `--profiler-config` | P1 通信证据**完全采不到** |
| **C** | 通信证据 = 「NCCL/allreduce」 | 单节点 TP 默认**优先走 vLLM custom all-reduce**（`cuda_communicator.py:254-311`），NCCL 只是 fallback 之一 | 按 NCCL 找 kernel 会**什么都找不到**，进而误判为「没有通信开销」 |

以上三条均已在修订后的方案中修复，详见 `MULTI_GPU_EXPERIMENT_PLAN.md`。

---

## 1. T · 理论（TP 在推理侧到底做了什么）

### 1.1 切分方式

Transformer 层里两处大 GEMM 被按不同维度切开（Megatron-LM 的经典切法）：

- **Attention**：QKV 投影按 **head 维**切（Column Parallel），每卡拿 `total_heads / tp` 个 head 独立算 attention；
  输出投影 `o_proj` 按 **输入维**切（Row Parallel），算完需要一次 **all-reduce** 把部分和加起来。
- **MLP**：`gate_proj` / `up_proj` 按输出维切（Column Parallel），`down_proj` 按输入维切（Row Parallel），
  同样以一次 **all-reduce** 收尾。

→ **每个 decoder layer 有 2 次 all-reduce**（attention 后 1 次、MLP 后 1 次）。
L 层模型每次前向就是 **2L 次 all-reduce**。这是 TP 通信开销的主体。

### 1.2 为什么小 batch 时 TP 可能更慢（ROADMAP M22 明确要回答的问题）

decode 阶段每步只算 1 个 token/请求，GEMM 退化成 GEMV：

- **计算量**按 `1/tp` 下降，但 GEMV 本来就是**访存受限**，切开后每卡的权重读取量减半，收益是实打实的；
- **然而**每层仍要付 2 次 all-reduce 的**固定延迟**（launch + 同步 + 传输），
  而 all-reduce 的 payload 只有 `batch × hidden` 这么小 —— 典型是几十 KB 到几 MB，
  **完全由延迟而非带宽主导**；
- 本项目 M7 已实证：单卡 eager decode 每步有 ~7ms 的 **CPU kernel 提交开销地板**（GPU 仅忙 23%）。
  TP 会把每步的 kernel 数量再乘上通信 kernel，**固定开销进一步抬高**。

→ 所以「小 batch 时 TP 更慢」的机制假说是：**省下的计算 < 新增的 2L 次通信固定开销**。
batch 增大后，计算量线性增长而 all-reduce 的次数不变（payload 变大但仍在带宽富余区），
所以收益应当随 batch 上升。**这是本轮要用数据检验的假说，不是预设结论。**

### 1.3 TP 扩的是什么

TP 扩的是**单卡放不下的模型容量**（权重被切开）+ 每卡腾出的显存可以给 KV。
**它不是用来扩吞吐的**（那是 DP 的活）。这条边界必须写进结论，避免把 TP 说成万能加速。

---

## 2. O · 源码定位：R1–R7 逐条核查

### R1 · TP>1 时的 executor 与进程拓扑 ✅ 主路径可用

**executor 选择**：`v1/executor/abstract.py:47-92` `Executor.get_class()`。
后端未指定且 `world_size_across_dp > 1` 时，`config/parallel.py:871-913` 默认设为 `"mp"`
→ **单节点 TP=2/4 无 Ray ⇒ `MultiprocExecutor`**。

**`VLLM_ENABLE_V1_MULTIPROCESSING`**：定义在 `envs.py:139`（默认 `True`）与 `envs.py:1269-1271`。
它控制的是 **EngineCore 是否跑在独立进程**，**不是**选哪个 executor。

> ⚠ **这是一个必须写进 preflight 的硬门禁**（`v1/engine/llm_engine.py:224-226`）：
> ```python
> if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
>     logger.debug("Enabling multiprocessing for LLMEngine.")
>     enable_multiprocessing = True
> ```
> 也就是说，**代码里传 `enable_multiprocessing=False` 会被环境变量直接覆盖**。
> 本项目一贯 `export VLLM_ENABLE_V1_MULTIPROCESSING=0`，所以是对的——
> 但它从「习惯」升级成了「不设就会静默改变进程拓扑、让所有内省失效」的关键前提。

**进程拓扑（`multiprocess_mode=False` + TP=4）**：

```
Driver 进程（调用 engine.step()）
├── LLMEngine
│   └── InprocClient            (core_client.py:105-133 / :304-315)
│       └── EngineCore          ← 同进程
│           ├── scheduler       ← Python 对象，在本进程  (v1/engine/core.py:269-306)
│           └── MultiprocExecutor
│               └── WorkerProc × 4  ← 独立进程，各持 1 张卡的权重与 KV
│                                     (v1/executor/multiproc_executor.py:176-201)
```

`schedule()` 在 driver 进程内被调用（`v1/engine/core.py:642`）。

→ **结论：M6/M17/M19 用的 `engine.engine_core.engine_core.scheduler` 内省与
`ScheduleSpy` 猴补丁在 TP>1 下依然成立**，前提是 `VLLM_ENABLE_V1_MULTIPROCESSING=0`。
**这是本轮最重要的一条好消息**：既有实验骨架可以直接复用，不需要重写观测方式。

**driver 进程的显存无意义**：模型与 KV 全在 worker 进程。
`MultiprocExecutor._init_executor` 不做 `init_device`；CUDA 初始化只发生在
`WorkerProc`（`multiproc_executor.py:625-634`）。
→ **driver 里的 `torch.cuda.memory_allocated()` 不能用来代表模型显存**（M6 的 `mem_snapshot()` 在 TP>1 下失真）。

---

### R2 / R6 · per-rank 显存与 KV 观测 ✅ 有稳定主路径

**主路径：`collective_rpc` + 模块级可调用对象。**

`LLMEngine.collective_rpc`（`v1/engine/llm_engine.py:492-499`）：

```python
def collective_rpc(
    self,
    method: str | Callable[[WorkerBase], _R],
    timeout: float | None = None,
    args: tuple = (),
    kwargs: dict[str, Any] | None = None,
) -> list[_R]:
    return self.engine_core.collective_rpc(method, timeout, args, kwargs)
```

语义见 `v1/executor/abstract.py:152-184`：**可以传方法名，也可以传可调用对象**；
传可调用对象时它会收到 **worker 作为第一个参数**。
`MultiprocExecutor` 用 **cloudpickle** 序列化（`multiproc_executor.py:340-402`），
worker 侧 `partial(cloudpickle.loads(method), self.worker)`（`:987-992`）。
**返回 `list`，每 rank 一项。**

> 工程约束：可调用对象必须能被 cloudpickle 序列化 → **写成模块级函数**，
> 不要用捕获了不可序列化状态的闭包。

**worker 上可读的字段**（`v1/worker/gpu_worker.py`）：

| 字段 | 位置 | 含义 |
|---|---|---|
| `worker.device` | `:313` | `cuda:N` |
| `worker.init_snapshot` | `:341` | init 时的 `MemorySnapshot` |
| `worker.available_kv_cache_memory_bytes` | `:489` | **该 rank** profile 出的可用 KV 字节 |
| `worker.non_torch_memory` / `peak_activation_memory` | `:473-475` | 非 torch 占用 / 峰值激活 |
| `worker.model_runner.model_memory_usage` | `gpu_model_runner.py:5449` | **权重字节数** |
| `worker.cache_config.num_gpu_blocks` | `:591-596` | 分配的 KV block 数 |

**⚠ 日志里的数字不是 per-rank**，这点必须写清楚，否则会重犯 M6「KV 容量等价值」的混淆：

- `"Available KV cache memory: %s GiB"`（`gpu_worker.py:508-511`）是**该 rank 的值**，
  但用 `logger.info_once` 且默认 `scope="local"` → **只有 local rank 0 会打印**，
  看到一行不代表其他 rank 相同。
- `"GPU KV cache size: %s tokens"` / `"Maximum concurrency ... %.2fx"`
  （`v1/core/kv_cache_utils.py:2135-2163`）是**取过 `min(num_blocks)` 之后的调度器统一容量**，
  **不是各 rank 之和**，也**不是**运行时真实并发。
  `get_kv_cache_capacity()`（`:1800-1810`）的返回值是 `max_concurrency * max_model_len`
  —— 即**「容量等价并发」**，与 `max_num_seqs` 限制下的实际并发是两码事。

→ 因此实验必须**分开记录**七类量，不得混谈（见方案 §D1 的 schema）。

**fallback**：若 `collective_rpc` 传可调用对象失败（cloudpickle 问题），退化到
`collective_rpc("determine_available_memory")`（`abstract.py:146-147`，但会重跑 profile，有副作用），
再退化到 `nvidia-smi --query-gpu=memory.used` 逐卡采样（**精度差、含其他进程占用，只能当 sanity check**）。
三级 fallback 的精度差异必须落盘标注。

---

### R3 · TP>1 的 profiler ⚠ 原计划的 API 不存在，已修正

**`VLLM_TORCH_PROFILER_DIR` 在 v0.24.0 全仓零命中。** 正确入口是 `ProfilerConfig`
（`config/profiler.py:34-67`）经 `EngineArgs.profiler_config`（`engine/arg_utils.py:662`）传入。

```python
profiler: ProfilerKind | None = None          # 需设为 "torch"
torch_profiler_dir: str = ""                  # 必须是绝对路径
torch_profiler_with_stack: bool = True        # 默认开（trace 会更大）
torch_profiler_record_shapes: bool = False
torch_profiler_use_gzip: bool = True
```

未设置就调用 `profile()` 会**抛 RuntimeError 而不是静默跳过**（`gpu_worker.py:929-937`），
`profiler="torch"` 但目录为空会在配置校验期 `ValueError`（`config/profiler.py:139-140`）——
两处都是显式失败，**不会出现「以为在采集其实没采」的情况**，这对本项目有利。

**调用链**：`LLMEngine.start_profile/stop_profile`（`v1/engine/llm_engine.py:409-413`）
→ `EngineCore.profile`（`v1/engine/core.py:824-825`）
→ `Executor.profile` = `collective_rpc("profile", ...)`（`v1/executor/abstract.py:256-257`）
→ **每个 worker 各自建 profiler**（`gpu_worker.py:940-980`）。

**每 rank 一份 trace，文件名可映射回 rank**：
rank 后缀由 `distributed/utils.py:664-696` `get_worker_rank_suffix()` 生成，形如
`dp0_pp0_tp{0..3}_dcp0_ep0_rank{0..3}`，再交给 PyTorch 的 `tensorboard_trace_handler`
拼成 `{worker_name}.{time_ns}.pt.trace.json[.gz]`。
→ 例：`dp0_pp0_tp2_dcp0_ep0_rank2.<ns>.pt.trace.json.gz` 就是 TP rank 2。

**沿用 M7 的 profiler 纪律**（本项目既有标准，不重新发明）：
GPU 时间取 `cat=="kernel"` 事件的**区间并集**，不对 `key_averages` 求和（避免高层算子与底层 kernel 双计）。
并且——**通信 kernel 的 GPU 时间占比 ≠ 通信造成的净 wall-time 损失**，因为通信与计算可能重叠。
本轮只报「通信 kernel 占 GPU busy 的比例」与「每步通信 kernel 次数」，
**不把它换算成「TP 慢了多少毫秒是因为通信」**。

---

### R4 · TP 档之间会静默改变的配置 ⚠ 至少两处

| 配置 | 风险 | 处置 |
|---|---|---|
| **`async_scheduling`** | 默认 `None`（`config/scheduler.py:158`）。`config/vllm.py:936-1005` 的自动决策里有 `executor_supports_async_sched = executor_class.supports_async_scheduling()`（`:938`），而 **executor 类在 TP=1（UniProc）与 TP>1（MultiProc）下不同** → **三档 TP 的自动决策可能不一致，直接毁掉单变量** | **三档一律显式 `async_scheduling=False`** |
| **`distributed_executor_backend`** | TP=1 → `uni`；TP>1 → `mp`（`config/parallel.py:871-913`）。这是 TP 的**固有伴随变化，无法消除** | **不可消除 → 必须在结论里显式声明**：TP=1 与 TP>1 的差异里天然包含「单进程 vs 多进程 executor」这一项 |
| `disable_custom_all_reduce` | 默认 `False`（`config/parallel.py:202-203`）；`nnodes>1` 时被强制 `True`（`:963-995`） | 单节点保持默认，但**把实际生效的 all-reduce 后端落盘**（见 R5） |
| `max_num_batched_tokens` / `max_num_seqs` | 可能按 world size 派生 | **全部显式写死并 readback** |

→ 方案要求三层证据：**传入配置 → 从引擎 readback 的 effective config → 不变量核对**。

---

### R5 · 单节点 TP 实际走哪条 collective ⚠ 默认不是 NCCL

**TP 通信组**：`distributed/parallel_state.py:1786-1792` 建 `_TP`，
`:1279-1294` → `GroupCoordinator`，`:424-479` 里当 `world_size>1` 时构造平台 communicator（CUDA → `CudaCommunicator`）。
入口 API：`distributed/communication_op.py:12-14` `tensor_model_parallel_all_reduce()` → `get_tp_group().all_reduce()`。

**分发顺序**（`distributed/device_communicators/cuda_communicator.py:254-311`）：

```
NCCL symm mem (env 默认关)
  → quick reduce (ROCm only)
  → flashinfer (env 默认关)
  → ★ vLLM custom all-reduce
  → torch SymmMem
  → PyNCCL (ncclAllReduce)
  → torch.distributed.all_reduce
```

**custom all-reduce 的启用条件**（`custom_all_reduce.py:75-169`）：自定义 op 可加载、
同节点、`world_size ∈ {2,4,6,8}`、`world_size>2` 时要求 `is_fully_connected`（NVLink 一跳）、
非 ROCm 需 `_can_p2p`。**逐次调用还有大小门槛**（`:230-243`）：
16 字节对齐、弱连续、且 `inp_size < max_size`（默认 **8 MiB**）。

`config/parallel.py:202-203` `disable_custom_all_reduce: bool = False`（默认允许）；
CUDA 平台 `platforms/cuda.py:563-564` `use_custom_allreduce() -> True`。

> **对实验的影响（这条最关键）**：
> 若按 `ncclDevKernel_*` 去 trace 里找，很可能**一无所获**，
> 进而误判成「TP 没有通信开销」——这正是原计划埋的雷。
>
> ⚠ **但表达强度必须收住**：`_log_all_reduce_backend_selection` 打印的是
> **该 group「可能被选中」的后端集合**，不是「每次调用都走同一个」。
> 真正的选择是 **per-call** 的：`should_custom_ar()`（`custom_all_reduce.py:230-243`）
> 每次都要看**这一次**的 payload 大小、16 字节对齐、弱连续性；
> 不满足就返回 `None` 落到下一个后端（SymmMem → PyNCCL → torch.distributed）。
>
> 所以**不得**写成「本引擎走 custom all-reduce」这种引擎级不可变事实。
> 正确的分析单位是：
> ```
> 具体某个 workload → 这一轮有哪些 collective → 每次实际走了什么 kernel → 出现在哪
> ```
> 同一个引擎里，小 payload 的 decode all-reduce 与大 payload 的 prefill all-reduce
> **完全可能走不同后端**。P1 必须按 trace 里的**真实 kernel 名**分类统计，
> 而不是拿后端选择日志一锤定音。

**trace 里该找什么**：

| 路径 | 符号 |
|---|---|
| custom all-reduce | `cross_device_reduce_1stage` / `cross_device_reduce_2stage`（`csrc/custom_all_reduce.cuh:299-323`；ws=4 全连接时 <512KiB 走 1stage，否则 2stage，见 `:591-596`） |
| torch SymmMem | `symm_mem::multimem_all_reduce_` / `two_shot_all_reduce_` |
| PyNCCL | `ncclAllReduce` 主机侧调用（`pynccl.py:188-196`）+ NCCL 设备 kernel（名字随 NCCL 版本变，**vLLM 源码里没有硬编码**） |

**运行时自证据**：`cuda_communicator.py:191` `_log_all_reduce_backend_selection()` 会打印
该 group 实际启用了哪些 all-reduce 后端。**preflight 必须抓这行日志并落盘**——
这是判断「到底走了哪条路」的第一手证据，比猜 kernel 名可靠。

**静态无法确定、必须运行时确认的**：NVLink 全连接、P2P 可用性、自定义 op 是否真的编进了 wheel、
SymmMem rendezvous 是否成功、每次调用的实际 payload 大小。

---

### R7 · 模型能否切 TP=1/2/4 ⚠ 本地保底模型有坑

**唯一硬约束**（`config/model.py:1163-1170`）：

```python
if total_num_attention_heads % tensor_parallel_size != 0:
    raise ValueError(...)
```

**KV head 不足时不报错，而是复制**（`config/model.py:1259-1270`）：

```python
# We will replicate the KV heads in the case where the number of KV heads is
# smaller than the tensor parallel size so each GPU has at least one KV head.
return max(1, total_num_kv_heads // parallel_config.tensor_parallel_size)
```

**本地已有模型（实读 config.json）**：

| 模型 | attn heads | KV heads | TP=2 | TP=4 |
|---|---:|---:|---|---|
| Qwen2.5-0.5B-Instruct | **14** | 2 | ✅ 14%2=0；KV 2→1/卡 | ❌ **14%4≠0 → ValueError，跑不了** |
| Qwen2.5-1.5B-Instruct | **12** | 2 | ✅ 12%2=0；KV 2→1/卡 | ⚠ 12%4=0 可跑，但 **KV heads 2<4 → 每卡仍 1 个 KV head，等于复制 2×** |

→ **两条直接影响方案的推论**：
1. **0.5B 只能用于 TP=1 本地预演，不能作为 TP=4 的对象**；
2. **保底模型 1.5B 在 TP=4 下 KV 被复制**（`max(1, 2//4) = 1`，4 张卡共 4 个 KV head vs 实际 2 个）
   → 「每卡 KV 容量随 TP 上升」的朴素预期**不成立**，
   若真用到保底方案，必须按此解释，**不得当成异常**。

**Qwen2.5-7B-Instruct（2026-08-08 已用公开 metadata 核实，不再是未知项）**

来源：`https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/config.json`
（663 字节的公开 config metadata，**未下载任何模型权重**）。实测字段：

```json
{"num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584,
 "intermediate_size": 18944, "num_hidden_layers": 28, "vocab_size": 152064,
 "tie_word_embeddings": false, "max_position_embeddings": 32768,
 "torch_dtype": "bfloat16"}
```

对照 `config/model.py:1163-1170` 的硬约束与 `:1259-1270` 的 KV 复制规则：

| TP | `28 % tp` | attn heads/rank | KV heads/rank | Σ KV heads | KV 是否被复制 | 结论 |
|---:|---:|---:|---:|---:|---|---|
| 1 | 0 ✅ | 28 | 4 | 4 | 否 | ✅ 可跑 |
| 2 | 0 ✅ | 14 | 2 | 4 | **否** | ✅ 可跑 |
| 4 | 0 ✅ | 7 | 1 | 4 | **否** | ✅ 可跑 |

→ **7B 在 TP=1/2/4 三档均满足结构约束，且 KV head 不会被复制**（`4 // 4 = 1 ≥ 1`，
未触发 `max(1, ...)` 的兜底分支）。这比保底模型 1.5B 好——后者在 TP=4 下 KV 会被复制 2×。

其余切分维度也都整除：`intermediate_size 18944 / 4 = 4736`、
`head_dim = 3584/28 = 128`、`vocab 152064 / 4 = 38016`。

**embedding / lm_head 的切分行为**（影响「每卡权重是否≈1/k」的解释）：
vLLM 的 Qwen2 实现用 `VocabParallelEmbedding`（`models/qwen2.py:358`）与
`ParallelLMHead`（`:513`）——**两者都按 vocab 维切分**
（`layers/vocab_parallel_embedding.py:302` `divide(...)`）。
且 7B 的 `tie_word_embeddings=false`，故 lm_head 是**独立且被切分**的权重矩阵。
→ 因此 7B 的每卡权重**预期相当接近 1/k**（残留的复制部分只有 layernorm 权重与 QKV bias，量级可忽略）。
**但这仍是 expected observation 而非 invariant**：实测偏离时按结构事实解释，不判失败。

> **远端仍要读一次实际下载模型的 `config.json` 做一致性校验**，但这只是 sanity check
> （防止拿到不同 revision），**不再是 blocker 探索**。

---

## 3. 这些结论如何落到实验设计

| 源码事实 | 实验决定 |
|---|---|
| R1：InprocClient 下 scheduler 仍在本进程 | **直接复用 M6/M17/M19 的 `ScheduleSpy`**，逐 step 读 `num_scheduled_tokens` → 用于精确隔离 steady-state decode |
| R1：driver 显存无意义 | **删掉 driver 侧 `mem_snapshot()` 作为模型显存的用法**，改用 R2 的 per-rank RPC |
| R2：七类量语义不同 | D1 的 schema **逐项分开落盘**，杜绝 M6「容量等价值 vs 实际并发」的混淆 |
| R3：`ProfilerConfig` + 每 rank 一份 trace | profiler 走独立 shard（避免给主实验加开销），trace 按 rank 后缀归档 |
| R4：async_scheduling 会因 executor 不同而异 | 三档**显式 False**；executor backend 差异**如实声明为不可消除的伴随变量** |
| R5：默认走 custom all-reduce | P1 改名为 **collective communication evidence**；先抓后端选择日志，再按实际路径找 kernel |
| R7：整除性 + KV 复制 | preflight 门禁；1.5B fallback 的 KV 行为提前写好解释 |

---

## 4. 本文件的边界（诚实性声明）

- 以上正文为**静态源码核查 + 本地 config 实读**时的设计依据。
- **多卡实测数字、显著性、通信 trace 解析一律以 [`results.md`](results.md) 为准**（已在 4×H20 采集）。
- 「custom all-reduce 会被优先选中」在 T+O 阶段是代码路径推断；P1 实测见 results §5
  （TP≥2 观察到 custom AR / NCCL / symm_mem；TP=1 无跨卡通信 kernel）。
- **PP / DP / EP / CP / P-D**：本文件与本模块均无结论，仍为 ⬜。
