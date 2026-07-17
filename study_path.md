## 先给结论

你下一步**不应该继续横向阅读更多 vLLM 文件**，也不应该立刻转去 TensorRT-LLM、SGLang 或 speculative decoding。

你现在的问题不是“代码覆盖得不够”，而是：

> 已经知道很多函数分别做什么，但还没有亲眼观察它们在同一轮推理中如何共同变化，也没有经历“提出假设—运行实验—定位瓶颈—修改代码—性能验证”的完整工程闭环。

你现有笔记已经覆盖了从请求、Scheduler、KV Cache、InputBatch、BlockTable 到 Attention backend 的主链路，知识范围实际上已经足够。
目前应当从“源码阅读阶段”切换到“推理引擎实验与修改阶段”。

按照你的就业目标，接下来半年的投入比例建议是：

```text
45%  vLLM 主链路实验、修改和开源贡献
25%  CUDA / Triton Kernel
15%  Benchmark、Profiler、性能分析
10%  分布式推理与量化
 5%  项目文档、面试表达和简历成果
```

当前推理岗位对 GPU 架构、CUDA、系统级性能调优、推理框架以及 CUDA C++/Triton kernel 能力的要求非常直接；量化、TensorRT、分布式执行通常属于进一步加分项。([NVIDIA招聘][1])

---

# 一、你现在处于哪一个阶段

推理引擎学习可以分为五个能力层级：

| 层级    | 能力                              |
| ----- | ------------------------------- |
| 1. 能读 | 知道函数大致做什么                       |
| 2. 能串 | 能讲清一个请求的完整数据流                   |
| 3. 能证 | 能通过日志、Tensor 和 profiler 证明自己的理解 |
| 4. 能改 | 能修改某一模块，并保证正确性                  |
| 5. 能优 | 能用基准测试证明修改带来了收益                 |

你目前大约位于 **2 和 3 之间**：

* 已经能解释 `SchedulerOutput`、`InputBatch`、`block_table`、`slot_mapping`、Attention metadata；
* 但这些概念主要来自静态阅读；
* 还没有看到它们在连续多个 engine step 中怎样变化；
* 还没有用 profiler 证明时间花在了哪里；
* 还没有修改过一条主路径并做回归测试。

因此后续学习必须遵循：

```text
观察真实运行
  ↓
验证源码理解
  ↓
建立性能基线
  ↓
定位瓶颈
  ↓
修改一处
  ↓
验证正确性与性能
  ↓
形成项目和 PR
```

---

# 二、首先建立一个贯穿后续学习的主项目

建议你不要做十几个零散 Demo，而是建立一个长期项目：

# `vLLM V1 Inference Trace & Optimization Lab`

它最终应包含五部分：

```text
1. 推理主链路结构化 Trace
2. Continuous Batching 可视化
3. KV Cache / PagedAttention 状态可视化
4. Benchmark 与 Profiler 报告
5. 至少一项有数据支撑的代码修改
```

建议目录：

```text
vllm-inference-lab/
├── README.md
├── environment/
│   ├── hardware.md
│   ├── software.md
│   └── vllm_commit.txt
├── workloads/
│   ├── single_prefill.py
│   ├── single_decode.py
│   ├── mixed_batch.py
│   ├── prefix_cache.py
│   └── preemption.py
├── tracing/
│   ├── trace_schema.py
│   ├── trace_collector.py
│   └── invariants.py
├── results/
│   ├── traces/
│   ├── benchmarks/
│   ├── profiles/
│   └── figures/
├── patches/
├── tests/
└── reports/
    ├── 01_request_lifecycle.md
    ├── 02_continuous_batching.md
    ├── 03_kv_cache.md
    ├── 04_profiling.md
    └── 05_optimization.md
```

这个项目会成为你后续简历和面试的核心材料。

---

# 三、先固定代码版本，不要追着最新 main 分支跑

vLLM 迭代速度非常快，近期版本已经持续引入 Model Runner V2、Hybrid KV Cache、不同 CUDA Graph 机制和新的 backend。当前官方 release 已经继续演进到 v0.25.0。([GitHub][2])

你的做法应该是：

1. 找到当前笔记对应的 commit。
2. 用这个 commit 完成前 6 周实验。
3. 所有代码修改单独开分支。
4. 第 7 周以后，再比较最新稳定版本有哪些结构变化。

记录：

```bash
git rev-parse HEAD > environment/vllm_commit.txt
git status > environment/git_status.txt
nvidia-smi > environment/nvidia_smi.txt
python -m torch.utils.collect_env > environment/torch_env.txt
pip freeze > environment/requirements.txt
```

不要一边学习一边随时 `git pull`，否则函数和数据结构不断变化，你很难形成稳定认知。

---

# 四、第一周：把静态源码变成动态执行过程

这一周不学习新模块，只做一件事：

> 让一个请求从 Scheduler 到 Attention 的关键状态全部可见。

## 第 1 天：建立可重复运行的最小基线

选择一个能稳定放进显存的 0.5B～1.5B decoder-only 模型。

先完成：

```text
单请求
短 prompt
生成 4～8 个 token
temperature=0
禁用不必要的高级功能
```

记录：

```text
模型名称
vLLM commit
GPU 型号
dtype
block size
max model length
gpu memory utilization
输入 prompt
最终输出
运行命令
```

验收标准：

```text
同一命令可以连续运行三次；
结果一致；
环境能够从源码启动；
你知道实际走的是哪一个 Attention backend。
```

---

## 第 2 天：给 Scheduler 加结构化 Trace

不要用散乱的 `print()`，定义统一事件：

```python
{
    "step": 0,
    "event": "scheduler_output",
    "req_id": "request-0",
    "status": "running",
    "num_tokens": 12,
    "num_computed_tokens": 8,
    "num_scheduled_tokens": 4,
    "new_block_ids": [7],
}
```

插入点：

```text
Scheduler.schedule() 返回前
Scheduler.update_from_output() 更新前后
```

重点记录：

```text
waiting / running
token_budget
num_scheduled_tokens
num_computed_tokens
new_block_ids
preempted requests
finished requests
```

验收标准：

```text
你能够仅看日志指出：
哪一步是 prefill；
哪一步开始 decode；
每轮为请求安排了多少 token。
```

---

## 第 3 天：Trace GPUModelRunner 和 InputBatch

插入点：

```text
GPUModelRunner._update_states()
InputBatch.add_request()
BlockTable.add_row() / append_row()
GPUModelRunner._prepare_inputs()
```

记录：

```text
req_id_to_index
每个请求所在 row
num_tokens_no_spec
num_computed_tokens_cpu
num_scheduled_tokens
positions
query_start_loc
seq_lens
block table
```

必须验证：

```text
query_start_loc[i + 1] - query_start_loc[i]
=
第 i 个请求本轮的 num_scheduled_tokens
```

以及：

```text
seq_len
=
本轮开始前 num_computed_tokens
+
本轮 num_scheduled_tokens
```

---

## 第 4 天：验证 slot mapping

对每个真实 token 输出：

```text
req_id
position
logical_block_idx
physical_block_id
block_offset
slot_id
```

普通非 CP 路径下，写一个 Python reference：

```python
def reference_slot(
    position: int,
    block_ids: list[int],
    block_size: int,
) -> int:
    logical_block = position // block_size
    block_offset = position % block_size
    physical_block = block_ids[logical_block]
    return physical_block * block_size + block_offset
```

将 Python 结果与 Triton kernel 结果逐个比较。

验收标准：

```text
所有真实 token 完全一致；
padding 位置均为 PAD_ID；
你能手算至少两个跨 block 的 token。
```

---

## 第 5 天：Trace Attention 入口

记录：

```text
layer_name
query.shape
key.shape
value.shape
output.shape
slot_mapping
query_start_loc
seq_lens
block_table.shape
backend name
forward_includes_kv_cache_update
```

不要给所有层打印全部 Tensor。只选择：

```text
第 0 层
中间一层
最后一层
```

验收标准：

```text
能证明 slot_mapping 最终进入 KV cache update；
能证明 block_table 最终进入 paged attention；
能说明为什么 forward 参数中没有 block_table。
```

---

## 第 6 天：运行四种场景

必须分别运行：

```text
S1：单请求完整 prefill
S2：同一个请求连续 decode
S3：一个长 prefill + 一个 decode 混合
S4：第二个请求在第一个请求 decode 期间到达
```

每个场景绘制一张 step 表：

| Step | Request | 状态 | scheduled tokens | computed before | seq len after | blocks |
| ---: | ------- | -- | ---------------: | --------------: | ------------: | ------ |

---

## 第 7 天：写第一份报告

题目：

```text
《vLLM V1 中一个请求的动态生命周期》
```

报告必须包含：

1. 一张时序图；
2. 一个请求连续多个 step 的状态表；
3. 一次 slot mapping 手算；
4. 一次 Attention metadata 对照；
5. 三个你原先理解错误或不完整的地方。

到这里，之前分散的函数才会第一次真正连起来。

---

# 五、第 2～4 周：彻底打通 Continuous Batching 和 KV Cache

## 第 2 周：Continuous Batching 实验

目标不是再读 `schedule()`，而是证明它的行为。

### 任务

构造请求：

```text
A：长 prompt，早到达
B：短 prompt，晚 2 个 step 到达
C：短 prompt，晚 4 个 step 到达
```

分别改变：

```text
max_num_batched_tokens
max_num_seqs
长 prefill threshold
chunked prefill 设置
```

每轮记录：

```text
token budget 被谁消耗
同一 batch 中哪些请求是 prefill
哪些请求是 decode
是否发生请求重排
每个请求何时获得首 token
```

产出：

```text
continuous_batching_timeline.png
scheduler_steps.jsonl
02_continuous_batching.md
```

验收标准：

> 不看源码，只看 Trace，就能解释这一轮 Scheduler 为什么这样选请求。

---

## 第 3 周：PagedAttention 与 Prefix Cache

### 任务 1：KV block 生命周期

跟踪每个 physical block：

```text
分配
ref_cnt 增加
写入 KV
登记 block hash
请求释放
进入 cached-free 状态
被重新命中
被驱逐复用
```

建立表：

| Block ID | Request | Ref count | Hash | 状态 | Step |
| -------: | ------- | --------: | ---- | -- | ---: |

### 任务 2：Prefix Cache

运行两个共享长前缀的请求：

```text
请求 A：prefix + question A
请求 B：prefix + question B
```

记录：

```text
A 产生哪些 block hash
A 完成后哪些 blocks 仍在 cache
B 命中多少完整 blocks
B 少计算了多少 prompt tokens
```

### 任务 3：无前缀缓存对照

关闭 prefix caching，重复同样 workload。

对比：

```text
prefill scheduled tokens
TTFT
GPU kernel 时间
KV block 复用情况
```

验收标准：

> 能用 block ID、hash、ref count 和 scheduled token 数证明 prefix cache 的作用，而不只是口头解释。

---

## 第 4 周：KV 不足与 Preemption

通过降低可用 KV Cache 或增加上下文长度制造压力。

记录：

```text
哪个请求申请 block 失败
Scheduler 选择了哪个 victim
释放了哪些 blocks
请求如何从 running 回到 waiting
恢复后 block table 是否被整体替换
重算了多少 token
```

必须验证两个不变量：

```text
活跃请求引用的 block 不能出现在普通 free 状态；
被恢复请求不能盲目 append 已经失效的旧 block table。
```

产出：

```text
03_kv_cache_lifecycle.md
preemption_timeline.png
kv_block_state.csv
```

---

# 六、第 5～6 周：从“能解释”进入“能测量”

vLLM 官方提供了 Benchmark CLI、参数 sweep 和 serving benchmark；官方 profiling 指南则区分了 Nsight Systems 和 PyTorch Profiler：前者适合低开销性能分析，后者适合观察 shape、内存和调用栈。([vLLM][3])

## 第 5 周：建立 Benchmark 方法论

至少测量：

```text
TTFT：Time To First Token
TPOT：Time Per Output Token
ITL：Inter-token Latency
Request latency
Output token throughput
Request throughput
GPU memory usage
KV cache utilization
```

建立 workload 矩阵：

| Workload  |      Prompt | Output | Concurrency | QPS |
| --------- | ----------: | -----: | ----------: | --: |
| 短对话       |         128 |    128 |      1/8/32 |  多档 |
| 长上下文      |        4096 |    128 |       1/4/8 |  多档 |
| 输出密集      |         128 |   1024 |         1/8 |  多档 |
| Prefix 共享 | 2048 shared |    128 |        8/32 |  多档 |

规则：

```text
预热和正式测量分开；
每组至少重复 3 次；
固定模型、commit、dtype 和随机种子；
Benchmark 时关闭 profiler；
使用中位数和 P95，不只报平均值。
```

产出：

```text
baseline.csv
benchmark_config.yaml
baseline_report.md
```

---

## 第 6 周：Profiler 定位瓶颈

### 第一步：PyTorch Profiler

观察：

```text
CPU 调用栈
Tensor shape
H2D copy
模型算子
Attention kernel
采样与 metadata 构造
```

### 第二步：Nsight Systems

重点看：

```text
GPU kernel 时间线
kernel launch gap
CPU 是否及时提交任务
H2D 与 kernel 是否重叠
NCCL 通信
CUDA Graph replay
```

### 第三步：形成 Bottleneck Sheet

| 现象          | 证据               | 初步原因            | 验证实验         |
| ----------- | ---------------- | --------------- | ------------ |
| GPU 间歇空闲    | Nsight gap       | CPU metadata 开销 | 增大 batch 对照  |
| TTFT 较高     | prefill kernel 长 | 长 prompt        | 缩短 prompt 对照 |
| 高并发 TPOT 恶化 | attention 时间增长   | KV 读取压力         | 固定上下文对照      |

验收标准：

> 所有“这里可能慢”的判断都必须附带 profiler 证据。

---

# 七、第 7～8 周：补齐 CUDA / Triton，不再只停留在 Python

## 第 7 周：Triton 基础实战

按顺序实现：

```text
1. Vector Add
2. Fused activation
3. Row-wise Softmax
4. Ragged sequence boundary 操作
5. 简化版 slot mapping kernel
```

简化版 slot mapping 先只支持：

```text
单 KV group
无 CP
无 padding
固定 block size
```

然后逐步增加：

```text
padding
多请求 query_start_loc
Context Parallel interleave
```

每个 kernel 必须有：

```text
PyTorch reference
随机 correctness test
边界条件 test
microbenchmark
不同输入规模曲线
```

不要仅仅把源码抄一遍。

---

## 第 8 周：做第一次真实代码修改

此时不要预先决定优化什么，遵循：

```text
Profiler 发现热点
  ↓
提出原因假设
  ↓
写 microbenchmark
  ↓
确认热点可复现
  ↓
修改代码
  ↓
correctness test
  ↓
E2E benchmark
```

可能的候选区域：

```text
Attention metadata CPU 构造
sampling metadata 构造
BlockTable CPU→GPU 提交
小 batch kernel launch 开销
某个重复 Tensor 分配
某条 H2D 同步路径
```

只有 profiler 指向它时才动手。

修改报告必须写：

```text
修改前火焰图或 timeline
根因
代码差异
单元测试
Microbenchmark
E2E benchmark
收益区间
退化场景
```

若没有稳定收益，也保留失败报告。能够证明“为什么没有收益”同样是推理优化能力。

---

# 八、第 9～12 周：补齐岗位需要的第二层能力

## 第 9 周：量化

不要从所有量化算法理论开始，先做执行路径实验：

```text
BF16 baseline
权重量化模型
量化 KV Cache
```

对比：

```text
模型显存
KV Cache 显存
吞吐
TTFT
TPOT
输出一致性或质量变化
```

源码阅读聚焦：

```text
权重在哪里反量化
Q/K/V 使用什么 dtype
KV Cache 使用什么 dtype
scale 存在哪里
Attention backend 如何消费 scale
```

硬件不支持 FP8 时，不要强行做虚假对比，改用硬件支持的 INT4/AWQ/GPTQ 路径。

---

## 第 10 周：双 GPU 分布式推理

优先学习 Tensor Parallel，不要一次进入多机。

实验：

```text
单 GPU
TP=2
```

观察：

```text
每张卡保存哪些权重
QKV projection 如何切分
Attention 后哪里发生 collective
NCCL 时间占比
单卡 batch 太小时为何 TP 可能更慢
```

产出：

```text
tp1_vs_tp2.csv
nsight_nccl_timeline.png
distributed_inference.md
```

没有双 GPU 时，建议短时租用云端双卡机器；不要为了环境问题停滞数周。

---

## 第 11 周：Serving 与负载调优

启动 OpenAI-compatible server，然后做固定 QPS 和并发测试。

实验变量：

```text
请求到达分布
并发数
QPS
prompt/output 长度
max_num_seqs
max_num_batched_tokens
prefix caching
```

回答：

```text
吞吐何时饱和？
P95/P99 延迟何时失控？
GPU 饱和还是 CPU 饱和？
KV Cache 是否成为容量瓶颈？
服务发生 OOM 时如何定位？
```

这一步才是真正把“推理内核”连接到“在线服务”。

---

## 第 12 周：第一次开源贡献

第一次 PR 不应追求大功能，优先顺序：

```text
测试缺口
明确 bug
Benchmark 改进
Profiler / trace 工具
文档与代码行为不一致
小型性能修复
```

正式贡献必须包含测试、文档、lint；涉及 kernel 时还要满足 custom op 注册、meta function 和相应 op 检查要求。([vLLM][4])

目标不是“PR 一定合入”，而是完整经历：

```text
复现 issue
定位代码
设计修改
测试
Benchmark
提交
回应 review
修改 PR
```

---

# 九、第 13～16 周：形成求职竞争力

## 第 13 周：再看第二个推理引擎

此时再选择一个：

```text
SGLang
或
TensorRT-LLM
```

不要完整读第二遍源码，只做同一问题的横向比较：

```text
请求调度方式
KV Cache 管理
Attention backend
CUDA Graph
Prefix caching
Serving benchmark
```

输出：

```text
《vLLM 与另一推理引擎的核心路径对比》
```

---

## 第 14 周：选择一个高级专题

只选一个：

```text
Speculative Decoding
MoE 推理
Context Parallel / Ring Attention
KV Cache Offloading
Disaggregated Prefill/Decode
```

选择原则：

* 有相关硬件和实验条件；
* 可以产生数据；
* 与导师项目或招聘方向有关；
* 不是只读论文和源码。

---

## 第 15 周：整理最终项目

README 必须让招聘者五分钟内看到：

```text
问题是什么
你分析了哪条源码路径
实验环境是什么
基线是多少
你发现了什么瓶颈
修改了什么
性能变化是多少
有没有回归
代码如何复现
```

核心图表控制在 5 张以内：

```text
请求时序图
Continuous Batching 图
KV block 生命周期图
Profiler timeline
优化前后性能图
```

---

## 第 16 周：面试化表达

每个主题准备三种版本：

```text
30 秒：一句话概括
3 分钟：完整原理和数据流
10 分钟：结合源码、实验和性能数据深入讲
```

重点题目：

```text
Continuous Batching 为什么提高吞吐？
PagedAttention 解决了什么内存问题？
block_table 和 slot_mapping 有什么区别？
Prefill 和 Decode 的瓶颈分别是什么？
KV Cache 为什么影响最大并发数？
如何确定一次优化真的有效？
如何用 profiler 区分 CPU 和 GPU 瓶颈？
为什么 TP=2 不一定比单卡快？
```

---

# 十、当前阶段暂时不要做的事

未来 6～8 周先停止：

```text
继续无目标地通读所有 vLLM 文件
同时学习 vLLM、SGLang、TensorRT-LLM 三套源码
深入多模态、Mamba、MLA、DCP 等所有分支
没有 benchmark 就修改性能代码
看到一个复杂 kernel 就逐行死磕
只做笔记而不运行实验
追随 main 分支每次结构变化
```

当前应该只锁定：

```text
decoder-only
单机单卡
普通 FlashAttention backend
Scheduler
KV Cache
GPUModelRunner
BlockTable
Attention
Profiler
```

---

# 十一、每周固定工作方式

每周都按照同一闭环：

```text
周一：
提出一个具体问题和可验证不变量

周二：
阅读与问题直接相关的源码，不扩散

周三：
增加 Trace 或编写实验脚本

周四：
运行实验并保存原始数据

周五：
Profiler 或 correctness 验证

周末：
修改代码、做对照测试、写报告
```

每周结束必须产生至少一种可检查成果：

```text
代码 commit
实验脚本
JSON/CSV 数据
Profiler trace
图表
技术报告
单元测试
PR
```

不能再用“本周看完了几个文件”作为进度指标。

---

# 十二、三个月后的最低验收线

完成前 12 周后，你应当具备以下成果：

```text
1 个完整的 vLLM Trace & Optimization 项目
5 个以上可复现实验 workload
1 套 Benchmark 数据
2 份以上 Profiler 分析
1 个 Triton slot-mapping 类 kernel
1 次有测试和性能数据的 vLLM 修改
1 次开源 PR 经历
1 份分布式推理对比报告
1 份量化推理对比报告
```

那时你对就业方向的表述就不再是：

```text
“我读过 vLLM，了解 PagedAttention。”
```

而可以变成：

```text
“我基于固定 vLLM commit 对 V1 推理主链路进行了逐 step
结构化追踪，验证了 continuous batching、KV block 分配、
slot mapping 和 paged attention 的动态行为；随后使用
PyTorch Profiler 与 Nsight Systems 建立基线，定位并修改了
具体瓶颈，并通过 microbenchmark、E2E throughput、TTFT 和
TPOT 进行了正确性与性能回归。”
```

这才是推理引擎开发岗位真正有辨识度的经历。

[1]: https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/Manager--Large-Language-Model-Inference_JR2006769?utm_source=chatgpt.com "Manager, Large Language Model Inference"
[2]: https://github.com/vllm-project/vllm/releases "Releases · vllm-project/vllm · GitHub"
[3]: https://docs.vllm.ai/en/latest/benchmarking/?utm_source=chatgpt.com "Benchmark Suites"
[4]: https://docs.vllm.ai/en/stable/contributing/ "Contributing to vLLM - vLLM"
