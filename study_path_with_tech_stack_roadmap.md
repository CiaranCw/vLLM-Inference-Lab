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

---

# 十三、面向推理系统岗位的技术栈扩展路线

前面的 16 周路线，主要解决的是：

```text
把 vLLM V1 的单机推理主链路真正学懂
→ 能观察
→ 能验证
→ 能修改
→ 能分析性能
→ 能形成一次开源贡献
```

它可以建立推理系统岗位最重要的基础，但还不能自动等价于以下招聘要求：

```text
分布式推理与多种并行策略
Triton Inference Server 生产部署
模型量化与其他压缩技术
投机解码、Medusa、EAGLE
Context Parallel / Ring Attention
百亿、千亿模型高并发服务
TensorRT-LLM 工程实践
```

这些能力需要在 vLLM 主线之后继续分层扩展。为了避免“只学理论却把它写成落地经验”，后续所有技术统一使用下面四种学习等级标记。

## 13.1 学习等级定义

| 标记 | 含义 | 达标标准 |
|---|---|---|
| **T：理论理解** | 理解论文、算法和系统设计 | 能画结构图、解释动机、推导开销并回答面试问题 |
| **E：可复现实验** | 必须亲自运行实验 | 有脚本、配置、原始数据、对照结果和结论 |
| **O：开源项目实践** | 必须依托真实开源项目 | 能定位源码路径、修改配置或代码、处理版本与运行问题 |
| **H：特定硬件/集群实践** | 必须使用多卡、多节点或指定硬件 | 有真实通信、显存、吞吐、延迟和故障数据 |

必须注意：

```text
T 只能证明“理解过”
E 才能证明“验证过”
O 才能证明“做过真实系统”
H 才能证明“具备相应规模的落地经验”
```

对于简历中的“熟悉”，通常至少需要达到 `T + E`；  
对于“有实践经验”，通常至少需要达到 `E + O`；  
对于“有大模型分布式部署经验”，必须达到 `E + O + H`。

---

## 13.2 技术栈覆盖总表

| 技术栈 | 理论是否足够 | 推荐最低等级 | 是否需要开源项目 | 是否需要额外硬件 | 推荐支撑项目 |
|---|---|---:|---:|---:|---|
| Continuous Batching、PagedAttention、KV Cache | 否 | T+E+O | 必须 | 单卡即可起步 | vLLM |
| 张量并行 TP | 否 | T+E+O+H | 必须 | 至少双 GPU | vLLM、PyTorch Distributed |
| 流水线并行 PP | 否 | T+E+O+H | 必须 | 多 GPU，最好多节点 | vLLM |
| 数据并行 DP | 否 | T+E+O+H | 必须 | 至少双 GPU/双实例 | vLLM、负载均衡组件 |
| Expert Parallel / MoE | 否 | T+E+O+H | 必须 | 多 GPU | vLLM、DeepEP/NCCL 相关实现 |
| Triton Inference Server | 否 | T+E+O | 必须 | 单 GPU 可入门 | Triton、vLLM Backend |
| 权重量化、KV Cache 量化 | 否 | T+E+O | 必须 | 当前单卡可完成部分实验 | vLLM、TensorRT-LLM、llm-compressor |
| 知识蒸馏 | 基础岗位可先学理论 | T；压缩岗需 T+E+O | 视岗位而定 | 单 GPU 可做小模型 | Transformers、TRL、自建训练项目 |
| 结构化剪枝 | 基础岗位可先学理论 | T；压缩岗需 T+E+O | 视岗位而定 | 单 GPU 可做小模型 | PyTorch、SparseML 等 |
| 低秩分解 | 基础岗位可先学理论 | T；算法岗需 T+E | 不一定 | 单 GPU 即可 | PyTorch 自建实验 |
| Speculative Decoding | 否 | T+E+O | 必须 | 单 GPU 可做 N-Gram；双模型方式看显存 | vLLM |
| Medusa / EAGLE / MTP | 否 | T+E+O | 必须 | 取决于模型与 draft 方案 | vLLM、speculators |
| Context Parallel | 否 | T+E+O+H | 必须 | 多 GPU | vLLM |
| Ring Attention | 否 | T+E+O+H | 必须 | 多 GPU | Ring Attention/长上下文开源实现 |
| 百亿模型服务 | 否 | T+E+O+H | 必须 | 多 GPU 服务器 | vLLM、Triton、监控与压测组件 |
| 千亿/MoE 模型服务 | 否 | T+E+O+H | 必须 | 多节点集群 | vLLM、TensorRT-LLM、Ray/Kubernetes |
| TensorRT-LLM | 否 | T+E+O | 必须 | NVIDIA GPU | TensorRT-LLM |
| P/D 分离、KV 传输 | 否 | T+E+O+H | 必须 | 至少双实例，最好多 GPU | vLLM、LMCache/NIXL 等 |
| 生产可观测性与容量规划 | 否 | T+E+O | 必须 | 单机可入门，集群更完整 | Prometheus、Grafana、压测工具 |

vLLM 当前官方能力覆盖 TP、PP、DP 和 EP 等并行方式；分离式 Prefill/Decode 会将两个阶段放到不同实例中，以便分别调节 TTFT 和 ITL。[5][6]  
Triton 官方提供 vLLM Backend 和 TensorRT-LLM Backend，因此 Triton 最适合作为引擎学习后的服务编排与生产部署层，而不是替代对引擎内部机制的理解。[7][8]

---

# 十四、第 17～20 周：vLLM 高级推理优化

这一阶段仍以 vLLM 为主，不要立刻切换第二个引擎。

## 14.1 Prefix Caching 与 Chunked Prefill

### 学习等级

```text
T + E + O
```

理论知识不够。必须通过共享前缀 workload 证明：

```text
哪些 KV blocks 被缓存
第二个请求命中了哪些 blocks
减少了多少 prefill token
TTFT 有什么变化
什么情况下没有收益
```

vLLM 的 Automatic Prefix Caching 会复用共享前缀对应的 KV Cache，从而避免重复进行相同前缀的 Prefill 计算。[9]

### 必须产出

```text
workloads/prefix_cache_comparison.py
results/prefix_cache_on_off.csv
reports/06_prefix_cache.md
```

### 是否需要额外硬件

单张 RTX 5080 足够。

---

## 14.2 量化执行路径

### 学习等级

```text
T + E + O
```

需要先掌握的理论：

```text
W8A8、W4A16 的含义
对称/非对称量化
per-tensor、per-channel、per-group
scale 与 zero point
权重量化和 KV Cache 量化的区别
量化误差、显存、带宽和计算吞吐之间的关系
```

但不能停在公式上。必须选取当前硬件实际支持的方案，至少完成：

```text
BF16 baseline
vs
一种权重量化模型
vs
可行时的 FP8 KV Cache
```

对比：

```text
模型权重显存
KV Cache 容量
启动时间
TTFT
TPOT
吞吐
输出质量的基本变化
```

### 是否必须使用开源项目

必须。建议先在 vLLM 中观察：

```text
QuantizationConfig
量化 Linear 层替换
权重加载
scale 的保存和传递
实际使用的 GEMM / dequant kernel
Attention backend 如何消费 KV scale
```

随后可用 TensorRT-LLM 做同类路径对照。TensorRT-LLM 官方量化能力覆盖多种低精度配方，包括 FP8、FP4 和量化 KV Cache，但具体可用性取决于 GPU 架构。[10]

### 是否需要额外硬件

当前 RTX 5080 可完成一部分低精度推理实验，但必须按实际框架和硬件支持情况选择方案，不得为了补齐表格而伪造不支持的 FP8/FP4 路径。

---

## 14.3 Speculative Decoding

### 学习等级

```text
T + E + O
```

必须先理解：

```text
target model
draft/proposer
一次提出 K 个 token
target verification
accepted/rejected tokens
acceptance rate
为何低 QPS、memory-bound decode 更容易受益
为什么 batch 增大后 verification 成本可能抵消收益
```

vLLM 官方将 Speculative Decoding 用于降低中低 QPS、内存受限场景下的 inter-token latency，并提供 N-Gram、draft model、EAGLE、MTP 等路径。[11]

### 推荐实验顺序

```text
普通 Decode baseline
→ N-Gram speculation
→ 小型 draft model
→ EAGLE / MTP / Medusa 类方案
```

不要一开始就直接进入 Medusa 源码。

### 必须记录

```text
proposed token 数
accepted token 数
acceptance rate
target forward 次数
TPOT
吞吐
显存
不同 batch size 下收益或退化
```

### 是否必须使用开源项目

必须依托 vLLM 的真实 speculative decoding 路径。  
仅阅读论文只能达到 `T`，不能写成“掌握投机解码工程实践”。

### 是否需要额外硬件

N-Gram 单卡即可；draft model 和 target model 是否能单卡共存取决于模型规模和显存。

---

# 十五、第 21～24 周：在线服务、Triton 与生产调优

## 15.1 vLLM OpenAI-Compatible Server

### 学习等级

```text
T + E + O
```

先把现有离线实验迁移为在线服务：

```text
客户端
→ API Server
→ 请求队列
→ Engine
→ Scheduler
→ Model Runner
→ 流式响应
```

实验变量：

```text
并发数
固定 QPS
Poisson 到达
prompt/output 长度
max_num_seqs
max_num_batched_tokens
streaming / non-streaming
prefix caching
```

必须测量：

```text
TTFT
TPOT / ITL
E2E latency
P50/P95/P99
request throughput
output token throughput
错误率
GPU、CPU、显存和 KV Cache 使用率
```

### 必须产出

```text
serving/load_generator.py
serving/workload_config.yaml
results/online_serving_baseline.csv
reports/07_online_serving.md
```

---

## 15.2 Triton Inference Server

### 学习等级

```text
T + E + O
```

Triton 不适合只学理论。招聘要求中的“部署与调优经验”至少应包含一次真实部署。

NVIDIA 官方的 Triton vLLM Backend 用于在 Triton 中运行 vLLM Engine；Triton 还可以通过 TensorRT-LLM Backend 服务 TensorRT-LLM 模型。[7][8]

### 第一阶段：单模型部署

完成：

```text
Triton Server 容器启动
模型仓库结构
vLLM Backend 配置
客户端请求
流式响应
健康检查
日志与指标
```

### 第二阶段：对照实验

使用同一个模型，对比：

```text
vLLM 原生 Server
vs
Triton + vLLM Backend
```

比较：

```text
部署复杂度
启动时间
吞吐
TTFT
P95/P99
指标暴露
模型管理
故障恢复
```

### 第三阶段：动态配置和服务治理

学习：

```text
model repository
instance/config 管理
HTTP/gRPC 客户端
健康检查
Prometheus metrics
请求超时
服务更新和回滚
```

### 是否需要额外硬件

单 GPU 可以完成部署入门和性能对照。  
要形成高可用、多副本和路由经验，则需要多实例或 Kubernetes 环境。

---

## 15.3 监控与容量规划

### 学习等级

```text
T + E + O
```

必须搭建：

```text
Prometheus
Grafana
GPU 指标
vLLM/Triton 服务指标
自定义压测结果
```

回答以下工程问题：

```text
最大稳定 QPS 是多少？
并发增加到什么位置后 P99 开始失控？
瓶颈是 GPU 计算、显存、KV Cache、CPU 还是请求队列？
怎样估算一张卡能够承载多少用户？
OOM、请求超时、worker 退出时怎样定位？
```

理论只能帮助你定义指标，不能替代真实压测和监控面板。

---

# 十六、第 25～30 周：分布式推理与并行策略

分布式推理是图片中与当前 vLLM 学习最直接相关、同时又最需要额外硬件的一项。

vLLM 官方支持多种并行与扩展方式，包括 TP、PP、DP 和面向 MoE 的 EP。[5]

## 16.1 第 25～26 周：Tensor Parallel

### 学习等级

```text
T + E + O + H
```

### 理论必须掌握

```text
Column Parallel Linear
Row Parallel Linear
QKV projection 如何切分
Attention head 如何切分
MLP 如何切分
AllReduce / ReduceScatter / AllGather 的位置
通信量与计算量
为何小 batch 下 TP 可能更慢
```

### 实践要求

至少双 GPU，运行：

```text
TP=1
vs
TP=2
```

记录：

```text
每卡权重和显存
每卡 KV Cache
NCCL collective
kernel 与通信重叠
TTFT
TPOT
吞吐
不同 batch size 下 scaling efficiency
```

### 必须依托

```text
vLLM
torch.distributed
NCCL
Nsight Systems
```

仅在单卡上阅读源码，最多只能写“理解 TP 原理”，不能写“具有 TP 部署经验”。

---

## 16.2 第 27 周：Pipeline Parallel

### 学习等级

```text
T + E + O + H
```

理论重点：

```text
按 Transformer layers 切分 stage
stage 间 activation 传输
pipeline bubble
microbatch
PP 与 TP 的组合
不同 stage 负载不均衡
```

实践至少需要多 GPU。模型应大到具有切分意义，否则实验只能证明功能，不能代表真实性能。

必须回答：

```text
为什么 PP 可以降低单卡权重显存？
为什么它会增加流水线等待？
推理中的 microbatch 如何影响 bubble？
何时应使用 PP 而不是继续扩大 TP？
```

---

## 16.3 第 28 周：Data Parallel 与请求路由

### 学习等级

```text
T + E + O + H
```

DP 在推理服务中主要用于复制模型实例、处理不同请求批次，从而扩展吞吐，而不是切分单个模型。[12]

实践内容：

```text
两个 DP replica
请求负载均衡
不同到达率
会话黏性
故障 replica 摘除
吞吐扩展
尾延迟
```

项目支撑：

```text
vLLM Data Parallel
或多个 vLLM Server + Nginx/Envoy/自定义 Router
```

---

## 16.4 第 29 周：Expert Parallel 与 MoE

### 学习等级

```text
T + E + O + H
```

理论重点：

```text
Router / Top-K experts
token dispatch
All-to-All
expert load imbalance
capacity 与 dropped tokens
TP、DP、EP 的组合
DeepEP 类通信优化
```

必须使用真实 MoE 模型和多 GPU 才能形成有效实践。

如果暂时没有硬件，本阶段先完成：

```text
T：论文和源码调用链
O：静态定位 vLLM EP group、dispatch 和通信路径
```

但不得把它表述为集群落地经验。

---

## 16.5 第 30 周：分布式总结项目

最终产出：

```text
reports/08_distributed_inference.md
results/tp1_tp2.csv
results/dp_scaling.csv
profiles/nccl_timeline/
figures/parallelism_topology.png
```

报告必须区分：

```text
扩展模型容量：TP / PP
扩展服务吞吐：DP
扩展 MoE 专家：EP
扩展长上下文：Context Parallel
```

---

# 十七、第 31～34 周：TensorRT-LLM 与跨引擎对比

## 17.1 TensorRT-LLM 基础

### 学习等级

```text
T + E + O
```

不能只阅读 TensorRT-LLM 文档。必须完成：

```text
模型 checkpoint 准备
Engine 构建
Engine 加载
离线推理
在线服务
Benchmark
```

TensorRT-LLM 提供单 GPU、多 GPU和多节点推理能力，并包含量化、KV Cache 管理、Chunked Prefill、Speculative Decoding 和分离式服务等高级能力。[13]

### 重点理解

```text
vLLM 的动态 Python/PyTorch 执行路径
vs
TensorRT-LLM 的 Engine 构建、tactic 和 runtime 路径
```

---

## 17.2 同模型同硬件对照

选择一个双方均支持的模型：

```text
Transformers
vs
vLLM
vs
TensorRT-LLM
```

严格固定：

```text
模型和精度
GPU
输入输出长度
并发
warmup
采样参数
Benchmark 方法
```

比较：

```text
安装与转换复杂度
模型准备时间
启动时间
TTFT
TPOT
吞吐
显存
量化能力
动态请求适配
调试难度
扩展性
```

### 必须产出

```text
reports/09_vllm_vs_tensorrt_llm.md
results/cross_engine_benchmark.csv
scripts/reproduce_cross_engine.sh
```

---

## 17.3 Triton + TensorRT-LLM Backend

完成：

```text
TensorRT-LLM Engine
→ Triton TensorRT-LLM Backend
→ OpenAI/HTTP/gRPC 客户端
→ 并发压测
→ 指标与故障处理
```

这样才能把：

```text
vLLM
TensorRT-LLM
Triton
```

串成完整技术体系，而不是三个相互独立的名词。

---

# 十八、第 35～38 周：长上下文、Ring Attention 与 P/D 分离

## 18.1 Context Parallel

### 学习等级

```text
T + E + O + H
```

理论重点：

```text
Sequence/Context 维度切分
每张 GPU 持有哪部分 token
Attention 为何需要跨卡交换 K/V
通信量随上下文长度如何变化
Prefill 与 Decode 的并行策略为什么不同
```

需要多 GPU 才能验证真实通信和扩展效果。

---

## 18.2 Ring Attention

### 学习等级

```text
T + E + O + H
```

必须先掌握：

```text
标准 Attention 的数据依赖
序列切分
环形传输 K/V block
online softmax
通信与计算重叠
causal mask 在分块中的处理
```

学习顺序：

```text
论文与简化推导
→ PyTorch 小规模 reference
→ 阅读一个 Ring Attention 开源实现
→ 双卡/多卡运行
→ Nsight 查看 P2P/NCCL 与 Attention 计算重叠
```

### 理论是否足够

如果只是面试中的“了解 Ring Attention”，`T` 可以作为起点。  
如果岗位明确要求“Ring Attention 技术实践”，必须达到 `T + E + O + H`。

不要把 vLLM 中看到 Context Parallel 配置等同于已经掌握 Ring Attention。

---

## 18.3 Disaggregated Prefill/Decode

### 学习等级

```text
T + E + O + H
```

vLLM 的 Disaggregated Prefilling 将 Prefill 和 Decode 放在不同实例中，使两者可以采用不同并行配置，分别优化 TTFT 与 ITL。[6]

理论重点：

```text
Prefill compute-bound
Decode memory-bandwidth-bound
为何两者适合不同 GPU 配置
KV Cache 怎样在实例间传输
传输延迟何时抵消分离收益
routing、backpressure、容错
```

项目应包括：

```text
Prefill worker
Decode worker
Connector / KV transfer
Router
在线 workload
TTFT、ITL、throughput@latency 对照
```

最低需要两个实例；要体现实际资源隔离，最好使用多 GPU。

---

# 十九、第 39～42 周：模型压缩专题的边界

图片中的“知识蒸馏、结构化剪枝、低秩分解”不属于 vLLM 主体，但应根据求职岗位决定学习深度。

## 19.1 知识蒸馏

### 推理系统岗位

推荐等级：

```text
T
```

掌握：

```text
teacher/student
logit distillation
feature distillation
sequence-level distillation
温度系数
蒸馏模型如何影响推理成本
```

能解释压缩后的模型为何更容易部署即可。

### 模型压缩/算法岗位

推荐等级：

```text
T + E + O
```

必须训练一个小型 student，比较：

```text
精度或任务指标
参数量
训练成本
推理延迟
吞吐
```

然后把 student 放入 vLLM 或其他引擎完成端到端推理测试。

---

## 19.2 结构化剪枝

### 推理系统岗位

推荐等级：

```text
T
```

重点理解：

```text
非结构化稀疏为何不一定带来真实加速
结构化剪掉 head、channel、layer 的差别
硬件和 kernel 是否能利用稀疏结构
参数减少为何不必然等于延迟下降
```

### 压缩或稀疏计算岗位

推荐等级：

```text
T + E + O
```

必须完成：

```text
剪枝
微调恢复
导出模型
在真实推理引擎中 Benchmark
验证是否获得 wall-clock 加速
```

只有模型文件变小而推理不加速，不应声称完成了推理优化。

---

## 19.3 低秩分解

### 推理系统岗位

推荐等级：

```text
T
```

掌握矩阵近似、SVD、秩与误差，以及低秩分解可能把一个大 GEMM 变成两个小 GEMM。

### 压缩算法岗位

推荐等级：

```text
T + E
```

用 PyTorch 完成：

```text
Linear 权重 SVD
选择不同 rank
重构误差
参数量
FLOPs
实际 latency
```

注意：

```text
理论 FLOPs 下降
≠
GPU 实际延迟一定下降
```

因为 kernel 数量、矩阵形状、访存和并行度都可能导致反直觉结果。

### 与 LoRA 的区别

LoRA 使用低秩更新矩阵进行参数高效微调或 Adapter 服务，不应直接等同于对原模型权重进行低秩分解压缩。

---

# 二十、百亿与千亿模型落地经验的真实门槛

## 20.1 可以在单卡上完成的部分

当前 RTX 5080 可以积累：

```text
Scheduler 与 Continuous Batching
PagedAttention 与 KV Cache
小模型高并发 workload
服务压测方法
Profiler
量化路径
Speculative Decoding
Triton 单机部署
TensorRT-LLM 单机实践
故障定位和监控
```

这些是大模型服务中的可迁移基础。

## 20.2 必须使用多 GPU/多节点的部分

以下内容不能仅靠单卡模拟后就写成“落地经验”：

```text
百亿模型权重跨卡切分
TP / PP 实际通信
MoE Expert Parallel
多节点 NCCL
跨节点 KV Cache 传输
千亿模型容量规划
节点故障恢复
集群级负载均衡
大规模 P/D 分离
```

至少应短期租用双卡或多卡云服务器完成阶段性实验。

## 20.3 百亿模型最低实践标准

最低建议环境：

```text
双卡或四卡 GPU
一个无法舒适放入单卡的模型
TP 或 PP
在线 Serving
真实并发压测
NCCL/Nsight 数据
Prometheus 指标
故障恢复记录
```

最低产出：

```text
模型切分拓扑
每卡显存
通信 timeline
吞吐与尾延迟
扩展效率
OOM 和通信故障排查
```

## 20.4 千亿/MoE 模型标准

千亿级或大型 MoE 服务通常需要多节点资源。学生阶段不必长期持有集群，但至少应做到：

```text
理解系统设计
阅读真实开源实现
完成缩小规模的同构实验
在短期集群环境中完成一次真实运行
明确说明实验规模和边界
```

简历应如实写：

```text
“在 4×GPU 环境完成 TP/EP 缩小规模验证”
```

而不是扩大表述为：

```text
“拥有千亿模型生产集群落地经验”
```

---

# 二十一、最终项目组合与求职表达

完成扩展路线后，不应呈现十几个互不关联的小 Demo，而应整理为四个项目。

## 项目 A：vLLM V1 Trace & Optimization Lab

覆盖：

```text
Scheduler
Continuous Batching
KV Cache
PagedAttention
Attention Backend
Profiler
CUDA/Triton
一次代码修改或 PR
```

要求等级：

```text
T + E + O
```

---

## 项目 B：Distributed LLM Inference Lab

覆盖：

```text
TP
PP
DP
EP（可选）
NCCL
多 GPU Serving
扩展效率
```

要求等级：

```text
T + E + O + H
```

---

## 项目 C：Production Serving Stack

覆盖：

```text
vLLM Server
Triton vLLM Backend
Prometheus/Grafana
压测
P95/P99
容量规划
故障处理
```

要求等级：

```text
T + E + O
```

---

## 项目 D：Cross-Engine Optimization

覆盖：

```text
vLLM
TensorRT-LLM
Triton TensorRT-LLM Backend
量化
Speculative Decoding
同模型同硬件 Benchmark
```

要求等级：

```text
T + E + O
```

---

# 二十二、各技术栈的最终掌握标准

| 技术 | “了解”标准 | “熟悉”标准 | “有实践经验”标准 |
|---|---|---|---|
| vLLM | 能解释架构 | 能追踪并修改主链路 | 有实验、Profiler、修改和 PR |
| TP/PP/DP | 能解释切分与通信 | 能定位 vLLM 分布式源码 | 在多 GPU 上有 Benchmark 与通信数据 |
| Triton | 知道 Backend 架构 | 能配置模型仓库和客户端 | 部署、压测、监控并排查故障 |
| 量化 | 理解误差和低精度类型 | 能追踪量化执行路径 | 有 BF16/量化模型对照数据 |
| Speculative Decoding | 理解 proposer/verification | 能解释 vLLM 路径 | 有 acceptance rate 与 TPOT 数据 |
| Medusa/EAGLE/MTP | 理解多 token 预测 | 能运行框架支持方案 | 有真实配置、收益和退化分析 |
| Ring Attention | 理解环形 K/V 交换 | 能读开源实现 | 有多卡通信与性能数据 |
| TensorRT-LLM | 知道 Engine 工作流 | 能构建并运行 Engine | 有跨引擎 Benchmark 和服务部署 |
| 百亿模型服务 | 能做容量估算 | 能设计 TP/PP/DP 拓扑 | 有多卡模型运行、压测和故障数据 |
| 千亿/MoE 服务 | 理解 EP 与多节点架构 | 能阅读真实实现并做缩小实验 | 有多节点/集群实践，且明确实验规模 |
| 蒸馏/剪枝/低秩 | 能解释算法与部署影响 | 能完成小模型实验 | 有压缩、恢复、部署和 wall-clock 数据 |

---

# 二十三、建议的总时间安排

在前 16 周 vLLM 主路线完成后，后续可以按以下顺序推进：

| 时间 | 主题 | 主要等级 | 硬件要求 |
|---|---|---|---|
| 第 17～20 周 | Prefix Cache、量化、Speculative Decoding | T+E+O | 单卡 |
| 第 21～24 周 | vLLM Server、Triton、监控与容量规划 | T+E+O | 单卡/多实例 |
| 第 25～30 周 | TP、PP、DP、EP | T+E+O+H | 双卡到多卡 |
| 第 31～34 周 | TensorRT-LLM 与跨引擎对比 | T+E+O | NVIDIA GPU |
| 第 35～38 周 | Context Parallel、Ring Attention、P/D 分离 | T+E+O+H | 多 GPU |
| 第 39～42 周 | 蒸馏、剪枝、低秩的岗位定向补充 | T 或 T+E+O | 单卡起步 |
| 第 43～44 周 | 项目整理、简历与面试表达 | 项目收敛 | 无新增要求 |

优先级必须保持：

```text
vLLM 单卡核心
> 服务化与性能分析
> 多 GPU 分布式
> TensorRT-LLM / Triton
> 高级解码和长上下文
> 非推理主线的模型压缩算法
```

如果求职时间不足，不要平均分配时间。最低优先完成：

```text
1. vLLM Trace & Optimization Lab
2. 一次多 GPU Tensor Parallel 实验
3. 一次 Triton + vLLM 部署
4. 一次量化或 Speculative Decoding 对照
5. 一次开源贡献
```

这五项比“每个名词都看过一点”更能形成可验证的求职竞争力。

---

# 二十四、后续路线的总验收线

完成整个路线后，至少应拥有：

```text
1 个 vLLM 源码 Trace 与优化项目
1 个 Triton 在线服务项目
1 个 TensorRT-LLM 跨引擎对照项目
1 套多 GPU TP/DP 实验
1 个量化推理对照
1 个 Speculative Decoding 实验
1 份长上下文或 P/D 分离设计与实践报告
1 次真实开源贡献经历
1 套 Prometheus/Grafana 监控与容量报告
```

同时必须能够清楚标记自己的能力边界：

```text
已经真实运行和验证的内容
已经读过源码但尚未获得硬件验证的内容
只掌握理论、尚未进入工程实践的内容
```

这种区分不会削弱简历，反而能证明你具备严谨的系统工程意识。

[5]: https://docs.vllm.ai/en/stable/serving/parallelism_scaling/ "Parallelism and Scaling - vLLM"
[6]: https://docs.vllm.ai/en/latest/features/disagg_prefill/ "Disaggregated Prefilling - vLLM"
[7]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/vllm_backend/README.html "vLLM Backend - Triton Inference Server"
[8]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/tensorrtllm_backend/README.html "TensorRT-LLM Backend - Triton Inference Server"
[9]: https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/ "Automatic Prefix Caching - vLLM"
[10]: https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html "Quantization - TensorRT-LLM"
[11]: https://docs.vllm.ai/en/latest/features/speculative_decoding/ "Speculative Decoding - vLLM"
[12]: https://docs.vllm.ai/en/stable/configuration/optimization/ "Optimization and Tuning - vLLM"
[13]: https://nvidia.github.io/TensorRT-LLM/overview.html "TensorRT-LLM Overview"

