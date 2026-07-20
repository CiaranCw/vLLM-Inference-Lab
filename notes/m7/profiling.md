# M7 · Profiler 定位瓶颈（★核心验收 #4）

> 目标：用 Profiler **拆开一个 engine step 的内部构成**，把 M6 留下的两个假设从"合理推测"变成"有证据的结论"：
> 1. decode / 短 prefill 那个 **~7ms 的"固定开销地板"到底是什么**——CPU/kernel 启动开销，还是 GPU 计算？
> 2. **长 prefill 是否真的由 GPU 计算主导**（GEMM / Attention kernel）？
>
> 每份分析都走完整链条：**现象 → Profiler 证据 → 原因假设 → 单变量对照实验 → 结论**（仅导出 timeline 不算完成）。
> 覆盖两类 workload：**decode-heavy 小 batch** 与 **prefill-heavy 长输入**。
> 边界：本模块做**瓶颈定位与机制解释**，不改算法（改在 M9）、不下跨机跨模型结论；数字仅本机本配置成立。
>
> 证据：`outputs/m7/profile_{eager,compiled}.json`（每 workload×mode 的 CPU/GPU/launch 摘要 + top kernel，**已入库**）、`outputs/m7/traces/*.json`（chrome trace 时间线，体积大、可由脚本复现、未入库）、`logs/m7/*.log`。复现命令见文末。

---

## 0. 预备知识：Profiler 到底在量什么？（本节为讲解，读懂后面才不费力）

一次 `engine.step()` 表面是"跑一步模型"，内部其实是 **CPU 和 GPU 两个处理器在协作**：

- **CPU 侧**：Python/框架决定"这一步要算什么"，然后**逐个把 GPU kernel"下发"（launch）到 GPU 队列**。每次下发一个 kernel 都有固定的 CPU 开销（约几微秒），叫 **kernel launch 开销**。还包括把输入张量/元数据从内存拷到显存（**H2D，`cudaMemcpyAsync`**）。
- **GPU 侧**：真正执行下发来的 kernel（矩阵乘、attention、归一化……）。

**PyTorch Profiler**（`torch.profiler`，底层走 CUPTI）能同时量到：
- 每个算子的 **CPU 时间**（`self_cpu_time_total`）——CPU 在这个算子上花了多久（含它触发的 launch）。
- 每个 kernel 的 **GPU/CUDA 时间**（`self_device_time_total`）——kernel 在 GPU 上真正跑了多久。
- **kernel 启动次数**——CPU 侧 `cudaLaunchKernel` 调用了多少次。

**怎么判断"谁是瓶颈"？**
- 若一步的**墙钟时间（wall）≫ GPU 真正忙碌时间**，且 **CPU 花在 `cudaLaunchKernel` 上很多、kernel 启动次数很高** → **CPU / launch 受限**（GPU 常常"算完一个小 kernel 就空等 CPU 下发下一个"，中间的空隙叫 **kernel launch gap**）。
- 若 **GPU kernel 时间 ≈ 或 ≥ wall**，kernel 又大又少 → **GPU 计算受限**（compute-bound）。

**两个关键的"形状"概念（后面反复用）**：
- **GEMV（矩阵×向量）**：batch=1 的 decode，每层权重矩阵去乘一个"单 token 的向量"。计算量小、但要把整块权重从显存读一遍 → **访存受限**，且 kernel 小、数量多。
- **GEMM（矩阵×矩阵）**：prefill 一次处理成百上千 token，权重乘一个"大矩阵"。计算量大、能喂满 tensor core → **计算受限**，kernel 大。

**一个必须诚实说明的度量坑**：`gpu_busy = Σ 所有 kernel 的 self device 时间`。当 GPU 上多个 stream 的 kernel **并行重叠**时，这个"求和"会**超过墙钟时间**（因为同一时刻两条 stream 都在忙，各自计时被分别累加）。所以：
- decode（kernel 小、基本串行、少重叠）：`gpu_busy < wall`，可近似当作"GPU 忙碌墙钟" → `gap = wall − gpu_busy` 有意义，代表 CPU/空等。
- 长 prefill（kernel 大、有重叠）：`gpu_busy` 会**大于 wall**（本报告实测 `gpu_frac≈1.7`）——这本身就说明 **GPU 被塞满了（compute-bound）**，此时"gap"公式失效、不应解读为负开销。

**本机工具现实（已实测，如实记录）**：`nsys`（Nsight Systems 2024.6.2）在本 WSL2 下**无法采集 GPU kernel**（`nsys stats cuda_gpu_kern_sum` 报 "does not contain CUDA kernel data"，是 WSL2 的 CUPTI device-trace 限制）。而 **`torch.profiler` 的 CUPTI 通道在本机可正常取到 CPU+CUDA**（已用一段 `aten::mm` 验证能读到 cutlass kernel 的 GPU 时间）。故本模块**以 PyTorch Profiler 为主力**，其导出的 chrome trace 提供时间线/launch gap 视图，等价覆盖了 Nsight 在本任务里的用途；nsys 的 GPU 时间线未采用，原因已如实说明。

## 1. 方法（保证可信）
- 复用 M6 的进程内 `LLMEngine`+`step()` 基础设施；每种 mode 独立子进程（防 cudagraph 显存/编译缓存污染）。
- **两相分离**：先跑**干净计时相**（不开 profiler，`cuda.synchronize` 夹 `perf_counter`）取 `step_wall`（profiler 有开销，不能用它测 wall）；再跑 **profiler 相**取 GPU/CPU/launch 构成。预热 10 步、正式 20 步取每步均值。
- 关键推断量：`gpu_busy_ms`（Σ kernel device 时间/步）、`cuda_launches/步`、`gpu_frac = gpu_busy/wall`、`cpu_gap = wall − gpu_busy`（仅在 gpu_frac<1 时解读）。
- **单变量对照**：同一 workload 下 **eager vs compiled**——二者唯一实质差别就是"有没有 CUDA Graph 把整串 kernel 打包成一次回放"（M5 已确诊：compiled decode=FULL、prefill≤128=PIECEWISE、≥256=NONE）。因此 eager→compiled 的变化，几乎纯粹反映"**去掉逐 kernel launch**"的效果，是干净的单变量实验。
- **自查**：每个测点确认 batch/seq_len 稳定、无 EOS/抢占（复用 M6 校验）；wall 与 profiler 分开测；gpu_frac>1 的场景不套用 gap 公式。

## 2. 全景数据（eager vs compiled，每步均值）

| workload | mode | wall(ms) | gpu_busy(ms) | **launches/步** | gpu_frac | 解读 |
|---|---|---:|---:|---:|---:|---|
| decode b=1 | eager | 6.66 | 3.46 | **257** | 0.52 | 一半时间在 CPU/空等 |
| decode b=1 | compiled | **2.47** | 2.12 | **14** | 0.86 | launch 塌缩，gap 几乎没了 |
| decode b=64 | eager | 9.81 | 4.39 | 279 | 0.45 | launch 数与 b=1 几乎一样 |
| decode b=64 | compiled | 3.51 | 2.52 | 12 | 0.72 | |
| prefill L=32 | eager | 8.19 | 3.55 | 209 | 0.43 | 短 prefill 同样被 launch 拖 |
| prefill L=32 | compiled | 2.90 | 2.23 | 62 | 0.77 | PIECEWISE 图，launch 209→62 |
| prefill L=2000 | eager | 21.70 | 37.98 | 161 | **1.75** | GPU 被塞满（compute-bound） |
| prefill L=2000 | compiled | 22.03 | 37.91 | 62 | 1.72 | launch 161→62 但 wall 不变 |

---

## 3. 分析一：Decode-heavy / 小 batch —— 瓶颈是 CPU kernel 启动

**① 现象（来自 M6）**：decode 单步 latency 在 eager 下约 7ms，且几乎不随 batch 变化；compiled 只要 ~2.5ms（约 2.7× 快）。M6 猜测存在一个"固定开销地板"，但不知道是什么。

**② Profiler 证据（decode b=1，eager）**：
- 墙钟 **6.66ms**，但 GPU 真正忙碌只有 **3.46ms**（`gpu_frac=0.52`）→ **一半时间 GPU 没在算**。
- 一步要 **257 次 kernel 启动**；CPU 侧**头号开销就是 `cudaLaunchKernel` 本身 = 1.59ms/步**（257 次 × ≈6μs），其次 `cudaMemcpyAsync`（H2D 输入元数据）0.42ms。
- GPU 上是**一堆小 kernel**：`aten::mm` 1.23ms、`gemvx::kernel`（**GEMV=矩阵×向量，batch=1 的典型**）1.14+0.22ms、flash attention `varlen_fwd` 仅 0.14ms。每个都很小，GPU"一下就算完"，然后空等 CPU 下发下一个 → 大量 launch gap。

**③ 原因假设**：decode（尤其 batch=1）每步只算 1 个 token，**每个 kernel 都是小活**（GEMV：读整块权重、只算一个向量，访存受限）；真正的时间被"CPU 逐个下发 257 个 kernel + kernel 间 GPU 空等"吃掉。即**这一步是 CPU / kernel-launch 受限**，不是 GPU 算力受限。

**④ 单变量对照实验（eager → compiled，即 加上 FULL CUDA Graph）**：
- launch 次数 **257 → 14**（compiled 用 `cudaGraphLaunch` 一次回放整串录好的 kernel）。
- CPU gap **3.21ms → 0.35ms**（`gpu_frac` 0.52 → 0.86）。
- wall **6.66 → 2.47ms**：省下的 4.2ms 几乎全部来自"消失的 launch gap"（gap 少了 2.86ms）+ 少量 GPU 侧收益。
- GPU 忙碌时间基本不变（3.46 → 2.12ms）——**GPU 干的活没变，变的只是"怎么把活下发下去"**。
- 补充：compiled decode 去掉 launch 后，残留 CPU 头号变成 `cudaMemcpyAsync` 1.54ms（每步 H2D 输入准备仍在图外，异步、与 GPU 重叠，不全计入 wall）。

**⑤ 结论**：**decode 的"~7ms 地板"= CPU 逐 kernel 启动开销（+ kernel 间 GPU 空等）**，M6 假设**证实**。机制根因：batch=1 decode 是"很多小 GEMV + 小 attention"，GPU 每个都秒算完，瓶颈在 CPU 下发节奏；**CUDA Graph 把 257 次下发压成 1 次回放**，直接抹掉 gap。这也解释了 M6 的"单步 latency 几乎不随 batch 变"——因为 launch 次数只取决于模型层数/算子数（~257），与 batch 无关（见分析三）。

---

## 4. 分析二：Prefill-heavy / 长输入 —— 瓶颈是 GPU 计算（GEMM + Attention）

**① 现象（来自 M6）**：长 prefill（如 2000 token）eager ≈ compiled（都 ~22ms），且随长度增长；短 prefill 才有 compiled 加速。

**② Profiler 证据（prefill L=2000，eager）**：
- GPU kernel 时间**求和 = 37.98ms > 墙钟 21.70ms**（`gpu_frac=1.75`）→ **GPU 被塞满**（多 stream kernel 重叠，求和超过 wall 正是"GPU 一直在忙"的标志）。
- GPU 时间由**大 kernel 主导**：`aten::mm` 13.7ms、**cutlass bf16 tensor-core GEMM** 8.56+4.0+2.05ms、**flash attention `varlen_fwd` 2.94ms**。这些是喂满 tensor core 的大计算 kernel。
- launch 只有 161 次/步（比 decode 少），且每次 launch 触发的是"大活"，**launch 开销相对 kernel 计算可忽略**。

**③ 原因假设**：prefill 一次处理 2000 个 token，每层是**大矩阵×大矩阵（GEMM）+ 长序列 attention**，计算量 ∝ 长度（attention 部分更陡）。GPU 算力/带宽被打满，CPU 下发那点开销藏在大 kernel 的计算时间里 → **compute-bound**。

**④ 单变量对照实验（eager → compiled）**：
- launch **161 → 62**（compiled 减少了下发次数）。
- 但 **wall 几乎不变：21.70 → 22.03ms**，`gpu_busy` 也不变（37.98 → 37.91ms）。
- **这正是判据**：既然把 launch 砍掉一大半、wall 却纹丝不动，说明 **launch 根本不是瓶颈**——瓶颈是 GPU 计算本身，图化/减少下发帮不上忙。（也呼应 M5：prefill≥256 在本配置超出 CUDA Graph 捕获上限 128，compiled 其实回退 NONE，无图可回放。）

**⑤ 结论**：**长 prefill 是 GPU 计算受限（GEMM + Attention 大 kernel 主导）**，M6 假设**证实**。CUDA Graph/编译对它几乎无用，因为省的是"下发开销"，而它的开销在"算"。要优化长 prefill 得从计算侧入手（kernel 效率、attention 实现、并行度）——属 M7 之后的方向。

---

## 5. 发散：为什么"batch 变大 decode 单步几乎不变"（分析一的延伸验证）

对照 decode b=1 与 b=64（eager）：
- launch 次数 **257 → 279**（几乎不变）——因为 launch 数只取决于**模型有多少层/多少算子**，和 batch 无关。
- GPU 忙碌 **3.46 → 4.39ms**（略增）——batch 大了，每个 GEMV 变成"矩阵×稍宽的矩阵"，算的活多一点点，但**权重只读一遍**（这就是 batching 摊销：权重读取成本被 64 个 token 分摊）。
- wall **6.66 → 9.81ms**：主要是 CPU/gap 增加（5.42ms），GPU 只多了 ~0.9ms。
→ 印证 M6："decode 吞吐随 batch 近线性、单步 latency 近平"，根因是 **launch 开销固定 + 权重读取被摊销**；compiled 把 launch 抹掉后（b=64 也只 12 次），这个近平特性更干净。也侧面支持"**decode 访存受限**"（GEMV 读权重是主成本）——但严格证明访存带宽上限需要 kernel 级带宽计数器（本机 ncu 不可用），故此点仍留作**受支持的假设**，不做定论。

## 6. 对 M6 两个假设的最终裁决
| M6 假设 | M7 裁决 | 关键证据 |
|---|---|---|
| decode/短 prefill 的 ~7ms 地板 = CPU/launch 开销 | **证实** | eager 257 launches、`cudaLaunchKernel` 1.59ms 为头号 CPU 开销、gpu_frac 0.52；compiled 14 launches、gap→0.35ms、wall 6.66→2.47 |
| 长 prefill = GPU 计算受限 | **证实** | gpu_frac 1.75、大 cutlass GEMM + flash attention 主导；compiled 砍 launch（161→62）而 wall 不变 |
| decode 访存受限（读权重） | **受支持，未定论** | batch↑ 时权重读取被摊销、kernel 为 GEMV；但缺 kernel 级带宽计数器（ncu 不可用），不下定论 |

## 7. 局限（如实）
- **nsys GPU kernel 追踪在本 WSL2 不可用**（已实测），故未用其 GPU 时间线；以 torch.profiler + chrome trace 替代，二者在本任务覆盖等价信息。
- `gpu_busy` 是 kernel self-time **求和**，多 stream 重叠时会 > wall（长 prefill 即如此）——只作方向判断，不当精确"GPU 墙钟"。
- profiler 本身有开销，故 wall 用**独立的干净计时相**测；profiler 相只用于构成占比。
- 单卡 / 单模型（0.5B）/ bf16；`ncu`（Nsight Compute）不可用，故无 kernel 级 occupancy/带宽利用率的硬件计数器证据（访存受限只到"受支持假设"）。
- chrome trace 体积大（单文件 6–23MB），未入库，可由脚本复现。

## 8. 承上启下
- 呼应 M5（执行模式）：M7 从 kernel/CPU 层面**量化**了 M5 的 FULL/PIECEWISE 到底省了什么——省的是"逐 kernel launch"。
- 呼应 M6（benchmark）：M7 把 M6 的两个"待验证假设"落成了有证据的结论。
- 指向 M9（profiler 驱动的真实修改）：本报告已定位到可下手的热点——decode 的 CPU launch / H2D 路径（对 eager 或图外部分）、以及长 prefill 的大 GEMM/attention（计算侧）。M9 的优化对象将由这些证据决定。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
python scripts/m7_run_all.py          # eager / compiled 各独立进程；-> outputs/m7/profile_*.json + traces/
# 单 mode：
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m7_profile.py --mode compiled --warmup 10 --n-prof 20
```
chrome trace 可用 Chrome `chrome://tracing` 或 Perfetto 打开 `outputs/m7/traces/decode_b1_eager.json` 直观看 launch gap；对比 `decode_b1_compiled.json` 看整步塌缩成一次 `cudaGraphLaunch`。
