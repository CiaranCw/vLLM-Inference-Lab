# M16 · 面试化表达（30 秒 / 3 分钟 / 10 分钟 + 重点题）

> 每条回答都锚定本 lab 的**真实证据**（模块号 + 数据），而非教科书复述。数据均限本机（RTX 5080/sm_120、WSL2、Qwen2.5-0.5B、eager），面试时先声明边界。

---

## 一、整个项目的三档陈述

### 30 秒版
> 我在一个固定的 vLLM commit 上，对 V1 单机推理主链路做了逐 step 结构化追踪，验证了 continuous batching、KV block 分配、slot mapping 和 paged attention 的**动态行为**；然后用 torch.profiler 建立基线并定位瓶颈，做了一次 profiler 驱动的真实代码修改并做了正确性与性能回归，还为上游补了一个缺失的单元测试。关键是——每个结论都有 trace/profiler/benchmark 证据，包括一份**诚实的无收益报告**。

### 3 分钟版
> 分五步：**(1) 能证**——用只读 trace 钩子把"调度输出 → InputBatch → BlockTable → Attention → KV 写/读 → 采样"串成一次运行的闭环证据，验证了 `query_start_loc` 差分 = 本轮 scheduled tokens、slot = 物理块×block_size+偏移 这些不变量。**(2) 能测量**——进程内闭环 benchmark：decode 单步近平、吞吐近线性，compiled 比 eager 快 2.5–2.85×；KV≈12KiB/token。**(3) 能定位**——profiler 证实 decode 的 ~7ms 地板是 CPU kernel 提交开销（GPU 仅忙 23%、每步 257 次提交），长 prefill 才是 GPU 计算受限。**(4) 能改**——按 profiler 证据合并 decode 的两处 H2D，机制成立、正确性完整，但 E2E 收益落在噪声内，我如实保留为无收益报告，因为 microbench 提前算出天花板只有 0.13%。**(5) 服务与贡献**——用真实 serve + 开环 Poisson 测出最大稳定 QPS≈40，并发现过载是尾延迟劣化而非吞吐崩溃；还为 vLLM 补了一个 slot_mapping 的单元测试。

### 10 分钟版（讲的时候配 README 的 4 张图 + evidence_index）
1. **动机与方法**：不横向堆 demo，锁定单机主链路，按"观察真实运行→验证源码理解→建基线→定位瓶颈→改一处→验证正确性与性能"闭环推进；固定 commit 避免追 main。
2. **主链路闭环（M1–M5）**：展示 trace 事件 schema、`SchedulerOutput→KV 写(slot_mapping)→读(block_table)` 闭环；说明 enforce_eager 其实是一票否决执行模式，默认 O2 在 sm_120 上真跑通 torch.compile+CUDA Graph（捕获 51 PIECEWISE+35 FULL），且 compiled 与 eager 从第 5 token 起分叉是编译数值差异、不是 bug。
3. **基线与瓶颈（M6→M7，配图 1/2）**：decode 近平/吞吐近线性、compiled 2.5–2.85×；profiler 拆 step 的**统计口径**（GPU 时间取 kernel 事件区间并集、提交计数含 runtime+driver+graph 三类）；结论 decode=CPU 提交受限、长 prefill=GPU 计算受限，每条都有单变量对照。
4. **能写 kernel（M8）**：自写 Triton slot_mapping 生成 kernel，与 PyTorch oracle + vLLM 官方 kernel 三路交叉 22/22，microbench 与官方同量级。
5. **能改 + 诚实（M9）**：profiler 驱动、门控默认关可回退、correctness 完整；E2E 无收益就如实报告，并用 microbench 解释为什么没收益。
6. **量化与在线服务（M10/M12，配图 3/4）**：FP8 显存收益明确但 decode 反常变慢（工作负载相关）；开环容量规划 + 过载是尾延迟劣化。
7. **上游贡献（M13）**：定位到 `compute_slot_mapping` 零测试覆盖 + 一个 stale 的前车 PR，补齐当前 API 的单元测试（26/26）。
8. **能力边界**：明确区分"已验证 / 只读源码未获硬件 / 仅理论"（如 TP 单卡只做到源码定位）。

---

## 二、8 道重点题（锚定本 lab 证据）

### Q1. Continuous Batching 为什么提高吞吐？
传统按请求静态成批，短请求要等长请求、GPU 空转。Continuous batching 以 **step 为粒度**动态组批：每步按 token 预算把 waiting 里的 prefill 和 running 里的 decode 混进同一批，完成的请求立即让位。**本 lab 证据（M2/M6）**：同一批里能同时看到 prefill 与 decode 请求；decode 吞吐随 batch 近线性上升（图 1 右）。本质是"把零散小 kernel 合并、摊薄固定开销"，与 Q7 的 CPU 提交瓶颈同源。

### Q2. PagedAttention 解决了什么内存问题？
解决 KV cache 的**碎片与预留浪费**：不为每个请求预留连续最大长度，而是按**固定大小 block 分页**、用 block_table 做逻辑→物理映射，按需分配、可共享（prefix）、可回收。**证据（M3/M6）**：KV≈12KiB/token（实测=理论），block 复用/prefix 命中/抢占重算全程可追踪；KV 池容量决定"最大并发等价值"（M6 实测 357×，但运行时并发另受 max_num_seqs=64 限）。

### Q3. block_table 和 slot_mapping 有什么区别？
**block_table**：每请求一行 `[逻辑块 → 物理块号]`，进入 **attention kernel** 供 paged attention 按块读 KV。**slot_mapping**：每个 token 一个**物理槽位** `= 物理块号×block_size + 块内偏移`，进入 **KV 写入**（reshape_and_cache）决定这个 token 的 K/V 写到哪。一个管"读"（按块），一个管"写"（按 token）。**证据（M4/M8）**：M4 trace 到二者分别进入读/写路径的闭环；M8 我亲手写了 slot_mapping 的生成 kernel 并三路交叉验证。

### Q4. Prefill 和 Decode 的瓶颈分别是什么？
**Prefill = GPU 计算受限**（一次算整个 prompt，大 GEMM/attention 主导）；**Decode = 访存 + CPU 提交受限**（一次一个 token，是 GEMV、算得少、每步大量小 kernel 提交）。**证据（M7）**：长 prefill gpu_frac=0.88、cutlass tensor-core GEMM 14.6ms 主导；decode eager gpu_frac 仅 0.23、每步 257 次 CPU 提交、GPU 头号是 GEMV。这也解释了为何 CUDA Graph 主要救 decode、量化对二者影响相反（Q6）。

### Q5. KV Cache 为什么影响最大并发数？
每个活跃请求的每个 token 都要占 KV 槽位；KV 池是固定显存换算出的**有限 token 容量**，并发请求越多、上下文越长，占用越大。KV 满 → 触发抢占/重算。**证据（M3/M6）**：KV≈12KiB/token，池容量 eager 73.2 万 / compiled 72.2 万 token（compiled 少因 CUDA Graph 占显存）；"最大并发 357×"是容量等价值。所以扩并发要么省 KV（量化/更小 block）要么加显存。

### Q6. 如何确定一次优化真的有效？
**先建基线（warmup 分离、重复≥15、报 median/P95/stdev、固定 model/commit/dtype/seed），再单变量对照，最后看收益是否超过噪声（stdev）**。**证据（M9）**：我合并了 decode 两处 H2D，机制成立（cudaMemcpyAsync/step 11→10）、正确性完整（开=输出与基线逐一致），但 E2E 差异 ±0.01–0.02ms **远小于** stdev 0.11–0.30ms，且 microbench 提前算出天花板仅 0.13%——所以我判定**无收益并如实报告**。能区分"真收益 vs 噪声"、能解释"为什么没收益"，比强行报一个数更重要。

### Q7. 如何用 profiler 区分 CPU 和 GPU 瓶颈？
看 **GPU 忙碌占比（gpu_frac）与 kernel launch gap**：GPU 时间取 chrome trace 里 `cat==kernel` 事件的**区间并集**（不对 key_averages 求和，避免高层算子/底层 kernel 双计），CPU 提交计数要覆盖 runtime(`cudaLaunchKernel`)+driver(`cuLaunchKernel`)+graph(`cudaGraphLaunch`) 三类。**证据（M7）**：decode eager gpu_frac 0.23 + launch gap 明显 → CPU 提交受限；compiled FULL 图把提交 257→15、wall 7.44→2.50ms 证实假设。长 prefill gpu_frac 0.88 → GPU 计算受限。**单变量对照**（增大 batch / 换 compiled）是坐实假设的关键。

### Q8. 为什么 TP=2 不一定比单卡快？
TP 把每层切到两张卡，但每层要插入 **AllReduce/AllGather** 通信；小 batch 时单卡本就没吃满、计算量小，通信与同步开销可能盖过切分收益，反而更慢。收益出现在**计算量大到通信可被摊薄 / 单卡显存放不下**时。**本 lab 边界（M11）**：本机单卡，此题目前只做到**理论 + 源码定位**（vLLM 并行组、QKV 切分、collective 位置），**未做多卡实测**——面试会如实说明这是 T+O 级、非 H 级经验（见 ROADMAP T/E/O/H 约定）。

---

## 三、诚实与边界（这本身是加分项）
- 所有绝对值限本配置（sm_120/eager/0.5B/WSL2），不外推 compiled/生产/跨机。
- 明确标记能力等级：**已真实验证** vs **只读源码未获硬件** vs **仅理论**（T/E/O/H）。
- 保留失败/无收益报告（M9），不伪造收益、不为补表格伪造不支持的路径（M10 未强做 sm_120 不支持项）。
- 这种边界意识面试中主动讲——它证明系统工程的严谨，比"什么都会一点"更可信。
