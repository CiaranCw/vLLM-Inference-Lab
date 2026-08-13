# M26 · 第二阶段面试化表达（30 秒 / 3 分钟 / 10 分钟 + 7 道重点题）

> 本文件**扩展** [`../m16/interview_and_presentation.md`](../m16/interview_and_presentation.md)（第一阶段的三档表达 + 9 道题），
> 只补第二阶段（M17–M22）的内容。两份合起来覆盖全部模块。
> 每个数字都能回溯到 `outputs/`，索引见 [`../evidence_index.md`](../evidence_index.md)。
> 项目组合与能力边界见 [`portfolio.md`](portfolio.md)。

---

## 一、整个 lab 的三档陈述（两阶段合并版）

### 30 秒版

> 我在固定的 vLLM commit 上做了一个可追溯的推理系统实验室。先用只读 trace 把 V1 主链路串成一次运行的闭环证据，
> 再用 profiler 证明 decode 的 7ms 地板是 CPU kernel 提交而不是 GPU 慢；然后往三个方向做了真实实验：
> **服务化**（真实压测 + Triton 对照 + Prometheus 监控与故障演练）、**推理优化**（prefix caching、chunked prefill、
> 投机解码、FP8 与 4bit 量化）、**多卡**（4×H20 上的 Tensor Parallel）。
> 全程有不变量脚本做回归，**所有结论都标了适用边界，没做的事我会直接说没做**。

### 3 分钟版

> 第一阶段解决「能不能证」：trace 把 Scheduler → KV → Attention → Sampling 串起来，
> profiler 定位出 decode 是 CPU 提交受限（GPU 只忙 23%、每步 257 次提交），长 prefill 才是计算受限。
> 这条结论成了后面所有优化的判断依据。
>
> 第二阶段解决「能不能用」。**服务化**上我做完了一整条栈：真实 `vllm serve` 开环压测测出最大稳定 QPS，
> Triton+vLLM backend 做同镜像对照发现代价在入口排队而非逐步解码，然后搭 Prometheus+Grafana+DCGM 监控栈，
> 做了四类瓶颈的指标签名和四个故障演练。**优化**上做了 prefix caching、chunked prefill、N-gram 投机解码、
> FP8 与 W4A16 量化，每个都是单变量对照 + Bootstrap CI。**多卡**上租了 4×H20 做 Tensor Parallel。
>
> 我认为更值得讲的是**几个反直觉结论**：投机解码优化的是串行轮数而不是保证少算；TP 的「加速」拆开看
> 主要来自并发 prefill，decode 单步其实更慢；4bit 量化省了 6 成显存但在我这台机器上 28 个对照点没有一个变快。
> 还有一条方法学教训：容量数字有保质期 —— 同样的配置隔了几周再测就明显不同，而当初没留环境指纹导致无法归因。

### 10 分钟版（配 [`portfolio.md`](portfolio.md) 的 5 张核心图）

按项目 A → C → B → D 讲，每个项目用「问题 / 证据 / 反直觉结论 / 边界」四段。
D 讲到一半要主动说明**它没完成、缺 TRT-LLM**，并说清缺的是什么、为什么不把已有的两条腿包装成完整项目。

---

## 二、7 道重点题（锚定第二阶段证据）

### Q10. Prefix caching 一定能降延迟吗？

**不一定，要看 prompt 长到有没有穿透固定开销的地板。**
M17 实测：命中前缀省掉 94% 的 prefill 计算 token，但 **P=256/512 时首 token 延迟改善的 CI 含 0**；
只有 P=1024/2048 才显出 **−32.4% / −58.5%**（CI 排除 0）。
原因就是 M7 那条 ~7ms 的 CPU 提交地板 —— **省下的计算量必须大过地板，才会变成省下的时间**。

进一步（I5-b）：固定前缀、加长输出时，相对收益 **61.2% → 22.9% → 9.3% → 3.4%** 单调下降，
到 `max_tokens=128` 时完成延迟的 CI 已经含 0。因为 **APC 只省 prefill、不省 decode**，
输出越长，那一次省下的 prefill 被摊得越薄。恒定的是**首 token** 的绝对收益（~15ms，四档 CI 全排除 0）。

### Q11. Chunked prefill 是「加速」吗？

**不是加速，是一次公平交换，而且只改尾部不改中位。**
M17 单变量对照（两侧 `max_num_batched_tokens` 都是 4096，chunked 侧靠 `long_prefill_token_threshold=512`
强制切成 4×512）：并发 decode 请求的**最坏 stall 从 22.87ms 降到 10.51ms（−54%，CI 显著）**，
但 **ITL 的 median 和 p95 的 CI 都含 0**；代价是长 prefill 自己的 TTFT **+47%**（显著）。

机制上还纠正过一个常见误解：**不是「长 prefill 把 decode 挤出去了」**。
原始逐 step 记录显示 decode **仍被同批调度**（`d=1, p=2048`），只是它的 token 要等一个 2049-token 的长 step 才返回。

### Q12. 投机解码到底优化了什么？Acceptance 高就一定快吗？

**优化的是串行 execution rounds，不是保证少算；Acceptance 是重要但不充分的指标。**
M19 直接计数 engine execution rounds：高初始 prompt 重叠的 workload 从 **128 → 33**，
完成延迟降 **72–75.5%**，amortized TPOT 从 7.6ms 降到 1.8ms。

但 acceptance 恒为 1.000 的那组，降幅却在 75.5%→72% 之间波动 —— 所以我建了
**Coverage + Acceptance + Round Cost** 三要素模型，而不是只看 acceptance。
另外教科书说「batch 增大收益消失」，**在我这台机器 batch≤16 没触及** —— 因为还在 M7 的 CPU 提交受限区。
**这不等于规律不成立**，只能说这个网格没测到。

### Q13. 把 vLLM 放到 Triton Inference Server 后面，代价是什么？

**代价主要在入口排队，不在逐步解码。**
M20 用**同一个镜像**做串行对照（C0=`vllm serve`，C1=Triton+vLLM backend，避免跨版本混淆）：
QPS32 时 TTFT median 从 **24ms 涨到 719ms**，QPS40 从 **26ms 涨到 952ms**（Bootstrap CI 全程排除 0），
而 **ITL 仍在 ~9–12ms 量级、两侧接近**。

方法学上有个坑值得说：容器里的 vLLM 版本和 host 上的不一样，所以**必须同镜像对照**，
不能拿 host 的历史压测结果去减容器的结果。

### Q14. 面板上的 P99 能直接当 SLA 验收数字吗？

**不能。** M21 先用守恒律验证服务端指标可信 —— 21 个压测窗口里，服务端 counter 增量与客户端 bench 汇总
（完成数 / prompt tokens / generation tokens）**逐条精确相等**。这一关过了才谈结论。

然后发现：**服务端 `histogram_quantile` 的近似分位数 ≠ 客户端 exact percentile**，
TTFT P99 中位偏高 **+9.7%**，最大 **+68%**。差异同时来自三件事：histogram 只存 bucket 计数所以是近似估计、
server 与 client 的观测边界不同、以及 scrape 与窗口聚合的近似 —— **这个实验无法把偏差归因到其中任何单一项**。
能下的结论只有「**两者不可互换**」：面板 P99 适合看趋势和告警，**SLO 验收要用客户端 exact percentile**。

还有一条同源的教训：DCGM 报 GPU util 2% 而 NVML 同期 33%，根因是 `dcgm-exporter` 默认 **30 秒**采集一次。
**只有一个数据源时，你不会知道它是错的。**

### Q15. 服务进程 `up == 0` 了，是不是就代表资源已经释放？

**不是。** M21 的故障演练实测：`SIGKILL` 打在 vLLM API server 父进程上，
**不会杀死持有显存的 `EngineCore` 子进程**。表现是告警 4 秒内就响、`up` 归零，
但**显存 90 秒都不回收**，孤儿进程还活着；下一次拉起实例直接撞
`ValueError: No available memory for the cache blocks`。清掉那个子进程后显存 **5 秒**回收、server **28 秒**恢复。

→ 判断服务是否干净退出，必须**同时看 `up` 和 GPU 显存是否回到基线**。这是我在做故障演练时才暴露出来的，
纸面上想不到。

### Q16. 4bit 量化能省多少显存？能不能同时提速？

**省显存要说清分母；提速在我这台机器上一个点都没有。**
M18 实测 W4A16（AWQ 与 GPTQ，两者 bits=4、group_size=128 完全相同，只差算法与 zero-point）：
整体权重 0.5B **−53.6%**、1.5B **−62.8%**，都达不到 4bit 的理论 −75%。
原因不是猜的 —— 不变量核对了 `embed`/`lm_head` 的字节数在三个配置里**一模一样**，它们根本没被量化；
剔除后可量化部分是 **−74.0%**，几乎就是理论值。
所以有一条能外推的规律：**embed 占比决定稀释程度**（0.5B 占 27.6%、1.5B 占 15.1%），**模型越小整体收益被摊薄越狠**。

速度上 **28 个对照点 0 个变快**，23 个是 CI 排除 0 的显著变慢。
我特意做了一个反驳自己的检查：decode 变慢可以推给 M7 的 CPU 提交地板，
但 M7 同时证过**长 prefill 是 GPU 计算受限**，而 W4A16 在那一段**仍然更慢**（+9.2% / +6.6%）
→ 慢的来源包含 **Marlin GEMM 本身**。至于是 dequant 开销还是 sm_120 未优化，
**我没做 kernel profiling，所以不下判断**。

另外我在设计冻结里写的假设「模型越大越容易显现收益」**被数据推翻了**（1.5B 的 decode 惩罚反而更大），
这条我原样留在了结论里。

---

## 三、方法学问题（这类题最能拉开差距）

### Q17. 你怎么保证一次对照实验的结论是可信的？

四条，都有本 lab 的实例：

1. **先冻结再跑**：判据、网格、样本量、「不做的事」全部写在设计文档里，跑完不改。
   M17 的 I5-b 就是先冻结趋势判定规则，再去跑。
2. **写死隐藏变量**：M19 写死了 `async_scheduling=False` 和 `disable_log_stats=False`；
   M18 发现量化 checkpoint 声明 `float16`、BF16 基线声明 `bfloat16`，**不显式冻结激活 dtype 就会把两个变量混成一个数**。
3. **不变量优先**：先定义可验证的守恒律（`hit == actual − scheduled`、
   服务端 counter 增量 == 客户端 bench 汇总），跑完逐条核对。
   而且**布尔判据必须有正交的第二证据源** —— M17 曾把 chunked 判据方向写反，60/60 样本被误标却不报错，
   就是因为当时只有一个证据源。
4. **统计口径规范**：median difference 的 **Bootstrap 95% CI（固定 seed）**。
   CI 含 0 只能说「**未观察到超出噪声的差异**」，不能说「没有差异」。
   早期我写过 "3.5σ" 这种非标准量，后来整版撤回了。

### Q18. 你犯过什么错，怎么发现的？

我把这些都留在文档里，因为它们比顺利的结论更能说明问题：

- **M17 首版**：单样本 + JIT 污染得出「≈3×7ms」，加 warmup 和大样本后**共撤回 12 项不可支持的表述**。
- **M21 首版 E3**：用 200 请求/档，与 M12 的 60 秒窗口口径不可比 → 重跑，
  而且这个错误本身变成了一条结论：**短到达时长会系统性低估持续排队的尾延迟**。
- **M21 的 E4-B3**：我在前一个实验没结束时启动了第二个进程，两者轮流抢端口，
  写出「同一测量值被标成两个不同参数」的交错记录，**整批数据作废**，重跑并加了互斥锁。
- **M18/M21 的审计**：把 `X×(R+Z)` 误称「并发用户」（其实是闭环用户 population，
  系统内并发请求数是 `X×R`）；把秒值写进 `_ms` 字段（19.57s 被记成 19.57ms，
  会得出「客户端有问题」的**相反**排障结论）。都在 evidence audit 里修正并加了回归守卫。
- **一次 invariant 覆盖缺口**：13 条不变量全 PASS，但有个脚本存在 SyntaxError ——
  因为没有任何一条判据要求「脚本至少得是可解析的 Python」。补了 compile 守卫，并用注入错误验证它不是空转。

### Q19. 你没做的事情有哪些？

主动说，不等对方问（详见 [`portfolio.md`](portfolio.md) 的能力边界总表）：
**TensorRT-LLM 与跨引擎对照**（项目 D 因此未完成）、**PP/DP/EP/CP/P-D 分离**（阻塞于多卡机时）、
**长上下文 Ring Attention**（仅理论）、**模型压缩**（推理岗只需理论）、
**Kubernetes / 多副本 / 跨机 SLA**、**kernel 级 profiling 解释 W4A16 为何慢**。
