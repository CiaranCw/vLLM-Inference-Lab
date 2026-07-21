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
>
> 注：统计口径经**审查修订**（GPU 时间改用 kernel 事件区间并集、不再对 key_averages 求和；CPU 提交计数覆盖 runtime+driver+graph 三类 API）。修订点见文末「统计实现修订记录」。

---

## 0. 预备知识：Profiler 到底在量什么？（本节为讲解，读懂后面才不费力）

一次 `engine.step()` 表面是"跑一步模型"，内部其实是 **CPU 和 GPU 两个处理器在协作**：

- **CPU 侧**：Python/框架决定"这一步要算什么"，然后**逐个把 GPU kernel"提交"（submit/launch）到 GPU 队列**。每次提交一个 kernel 都有固定 CPU 开销（约几微秒），叫 **kernel launch 开销**。还包括把输入张量/元数据从内存拷到显存（**H2D，`cudaMemcpyAsync`**）。
- **GPU 侧**：真正执行提交来的 kernel（矩阵乘、attention、归一化……）。

**PyTorch Profiler**（`torch.profiler`，底层走 CUPTI）导出的 chrome trace 里，每条事件带 `cat`（类别）：
- `cat=="kernel"`：**真正在 GPU 上执行的底层 kernel**（这才是 GPU 时间的唯一可靠来源）。
- `cat=="cuda_runtime"` / `"cuda_driver"`：CPU 侧的 CUDA API 调用（提交 kernel、拷贝内存等）。
- `cat=="cpu_op"`：高层 PyTorch 算子（如 `aten::mm`）。

**⚠️ 两个统计坑（本报告已按审查修正）**：
1. **GPU 时间不能对 `key_averages()` 的 device 时间求和**——因为一个高层算子（`aten::mm`）会把它启动的 kernel 的 device 时间"归属"到自己名下，而那个 kernel 又作为独立行再计一次，**求和 = 双计**（实测 decode 双计到 ~2×）。正确做法：只取 `cat=="kernel"` 的事件，算它们的**时间区间并集**（多 stream 重叠也不重复计）——这才是"真正的 GPU 忙碌墙钟"。本报告同时给出 kernel 时间**求和**（sum）与**并集**（union）；本机各场景实测 **只有单条 CUDA stream（strm=1）、无重叠，故 union==sum**。
2. **提交计数要覆盖所有 CPU 提交 API**——不只 `cudaLaunchKernel`/`cudaLaunchKernelExC`（runtime），还要算 `cuLaunchKernel`（**driver API，同样是 CPU 直接提交 kernel**）和 `cudaGraphLaunch`（**CUDA Graph 回放**）。只数 runtime 会严重漏计（实测长 prefill 漏掉近 100 次/步的 driver 提交）。本报告分列 `direct_runtime_launches / direct_driver_launches / graph_replays / total_cpu_submission_calls`。

**怎么判断"谁是瓶颈"？**
- 若一步**墙钟（wall）≫ GPU 忙碌并集**，且 **CPU 提交次数很多、`cudaLaunchKernel` 占 CPU 大头** → **CPU / launch 受限**（GPU 常常"算完一个小 kernel 就空等 CPU 提交下一个"，中间空隙 = **kernel launch gap**）。
- 若 **GPU 忙碌并集 ≈ wall**（gpu_frac 接近 1），kernel 又大又少 → **GPU 计算受限**（compute-bound）。
- 提示：`gpu_frac = gpu_union/wall` 只在**明显 <1 时**是干净的"GPU 占比"；接近或略超 1 时表示"已 GPU 受限"，其精确值受"干净计时相 vs profiler 相"两次测量差影响，不宜细抠。

**两个"形状"概念（后面反复用）**：
- **GEMV（矩阵×向量）**：batch=1 的 decode，权重矩阵乘"单 token 向量"。计算少、要把整块权重从显存读一遍 → **访存受限**，kernel 小而多。
- **GEMM（矩阵×矩阵）**：prefill 一次处理成百上千 token，权重乘大矩阵，喂满 tensor core → **计算受限**，kernel 大。

**本机工具现实（已实测，如实记录）**：`nsys`（Nsight Systems 2024.6.2）在本 WSL2 下**无法采集 GPU kernel**（`cuda_gpu_kern_sum` 报 "does not contain CUDA kernel data"，WSL2 的 CUPTI device-trace 限制）；而 **`torch.profiler` 的 CUPTI 通道在本机可正常取 CPU+CUDA**。故以 **PyTorch Profiler 为主力**，其 chrome trace 同时提供 kernel 时间线与 launch gap，等价覆盖 Nsight 在本任务的用途。`ncu`（Nsight Compute）不可用，故无 kernel 级带宽/occupancy 硬件计数器。

## 1. 方法（保证可信）
- 复用 M6 进程内 `LLMEngine`+`step()`；每 mode 独立子进程（防 cudagraph 显存/编译缓存污染）。
- **两相分离**：干净计时相（不开 profiler，`cuda.synchronize` 夹 `perf_counter`）取 `step_wall`；profiler 相取构成。预热 10 步、正式 20 步取每步均值。
- **GPU 时间**：从导出的 chrome trace 取 `cat=="kernel"` 事件，算**区间并集**（`gpu_busy_union`）与求和（`sum`）、每 stream 忙碌、kernel 计数。
- **CPU 提交**：runtime(`cudaLaunchKernel`/`ExC`) + driver(`cuLaunchKernel`) + graph(`cudaGraphLaunch`)，分列并汇总。
- **单变量对照**：eager vs compiled——唯一实质差别是"有没有把整串 kernel 打包成图回放"（M5 已确诊：decode=FULL、prefill≤128=PIECEWISE、≥256=NONE）。
- **自查**：每测点确认 batch/seq_len 稳定、无 EOS/抢占；wall 与 profiler 分开测；gpu_frac 只在 <1 时细读。

## 2. 全景数据（eager vs compiled，每步均值）

| workload | mode | wall(ms) | GPU忙碌并集(ms) | gpu_frac | rt提交 | drv提交 | graph回放 | **总提交/步** |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| decode b=1 | eager | 7.44 | 1.70 | **0.23** | 257 | 0 | 0 | **257** |
| decode b=1 | compiled | **2.50** | ~2–3 | ≈1（GPU受限） | 14 | 0 | 1 | **15** |
| decode b=64 | eager | 9.51 | 2.20 | 0.23 | 279 | 97 | 0 | 376 |
| decode b=64 | compiled | 3.46 | 2.10 | 0.61 | 12 | 1 | 1 | 14 |
| prefill L=32 | eager | 7.59 | 1.77 | 0.23 | 209 | 96 | 0 | 305 |
| prefill L=32 | compiled | 3.02 | 1.71 | 0.57 | 62 | 0 | 25 | 87 |
| prefill L=2000 | eager | 21.60 | 18.92 | **0.88** | 161 | 96 | 0 | 257 |
| prefill L=2000 | compiled | 22.16 | 18.97 | 0.86 | 62 | 193 | 0 | 255 |

（本机各场景均 **单 CUDA stream、无 kernel 重叠**，故 GPU 并集 = 求和。decode b=1 compiled 的 GPU 并集略超 wall 属两相测量差，视为"已 GPU 受限"。）

---

## 3. 分析一：Decode-heavy / 小 batch —— 瓶颈是 CPU kernel 提交

**① 现象（来自 M6）**：decode 单步 latency eager 约 7ms、几乎不随 batch 变；compiled 只要 ~2.5ms。M6 猜有"固定开销地板"，但不知是什么。

**② Profiler 证据（decode b=1，eager）**：
- 墙钟 **7.44ms**，但 GPU 真正忙碌只有 **1.70ms（gpu_frac 仅 0.23）**→ **77% 的时间 GPU 没在算**。
- 一步 **257 次 CPU 提交**（全是 runtime `cudaLaunchKernel`）；CPU 头号开销就是 **`cudaLaunchKernel` 本身 1.77ms/步**（257 次 × ≈7μs），其次 `vllm::unified_attention_with_output` 0.84ms、`aten::mm` 0.63ms、`cudaMemcpyAsync` 0.38ms（H2D，35 次/步）。
- GPU 上是**一堆小 kernel**：头号是 **`gemvx`（GEMV=矩阵×向量，batch=1 典型）1.36ms**，flash attention `varlen_fwd` 仅 0.14ms，其余是零碎 elementwise。每个都很小 → GPU 秒算完就空等 CPU 提交下一个 → 大量 launch gap。

**③ 原因假设**：batch=1 decode 每步只算 1 个 token，**每个 kernel 都是小活（GEMV：读整块权重、只算一个向量）**；时间被"CPU 逐个提交 257 个 kernel + kernel 间 GPU 空等"吃掉 → **CPU / kernel-launch 受限**，非 GPU 算力受限。

**④ 单变量对照实验（eager → compiled，即 加上 FULL CUDA Graph）**：
- **总提交 257 → 15**（compiled 用 1 次 `cudaGraphLaunch` 回放整串录好的 kernel，+ 14 个图外零碎）。
- wall **7.44 → 2.50ms**：省下的 ~5ms 几乎全部来自"消失的 launch gap"。
- gpu_frac 从 0.23 升到 ≈1 → 去掉提交开销后，**这一步变成 GPU 受限**（GPU 干的活没变，变的只是"怎么把活提交下去"）。

**⑤ 结论**：**decode 的"~7ms 地板"= CPU 逐 kernel 提交开销（+ kernel 间 GPU 空等）**，M6 假设**证实且更强**（GPU 仅占 23%）。机制根因：batch=1 decode 是"很多小 GEMV + 小 attention"，瓶颈在 CPU 提交节奏；**CUDA Graph 把 257 次提交压成 1 次回放**，直接抹掉 gap。这也解释 M6 的"单步 latency 几乎不随 batch 变"——提交次数只取决于模型层数/算子数（见分析三）。

---

## 4. 分析二：Prefill-heavy / 长输入 —— 瓶颈是 GPU 计算（GEMM + Attention）

**① 现象（来自 M6）**：长 prefill（2000 token）eager ≈ compiled（都 ~22ms），随长度增长；短 prefill 才有 compiled 加速。

**② Profiler 证据（prefill L=2000，eager）**：
- GPU 忙碌并集 **18.92ms**，墙钟 **21.60ms** → **gpu_frac 0.88**，GPU 几乎全程在算（compute-bound）。
- GPU 由**大 kernel 主导**：**cutlass bf16 tensor-core GEMM 14.6ms** + **flash attention `varlen_fwd` 2.9ms**，其余零碎不到 1ms。
- 一步共 **257 次 CPU 提交**（161 runtime + **96 driver `cuLaunchKernel`**）——注意 driver API 提交占了近四成，**只数 runtime 会严重漏计**。

**③ 原因假设**：prefill 一次处理 2000 token，每层是大 GEMM + 长序列 attention，计算量 ∝ 长度；GPU 算力被打满，CPU 提交那点开销藏在大 kernel 计算时间里 → **compute-bound**。

**④ 单变量对照实验（eager → compiled）**：
- **总提交几乎不变：257 → 255**（compiled 只是把部分 runtime 提交换成 driver 提交：rt 161→62、drv 96→193——因 prefill≥256 在本配置超出 CUDA Graph 捕获上限 128，是 **NONE 模式无图可回放**，只是 inductor 编译过的 kernel 逐个提交）。
- **wall 几乎不变：21.60 → 22.16ms**，GPU 忙碌也不变（18.92 → 18.97ms）。
- **这正是判据**：提交方式变了、提交总数没降、wall 纹丝不动 → **瓶颈不在提交，而在 GPU 计算**；图化/编译帮不上（本来也没图）。

> 更正：初版报告曾写"compiled 把 launch 161→62"——那是**只数了 runtime API、漏掉 driver `cuLaunchKernel`**造成的错觉。补全后**总提交 257→255 基本不变**，结论（compute-bound）不变但证据更准确。

**⑤ 结论**：**长 prefill 是 GPU 计算受限（tensor-core GEMM + flash attention 主导）**，M6 假设**证实**。要优化长 prefill 得从计算侧入手（kernel 效率、attention、并行度），属后续方向。

---

## 5. 发散：CPU-bound 与 GPU-bound 的分界，以及 batch 摊销

把四个场景按 gpu_frac 排开，分界一目了然：
- **CPU/launch 受限区（gpu_frac≈0.23）**：decode（任意 batch）、短 prefill(L=32)。共同点：GPU 只忙 ~1.7–2.2ms，提交次数几百。→ CUDA Graph（decode 用 FULL、短 prefill 用 PIECEWISE，实测短 prefill compiled 有 **25 次/步 graph 回放**、总提交 305→87）能大幅提速（wall 减半以上）。
- **GPU 计算受限区（gpu_frac≈0.88）**：长 prefill(L=2000)。GPU 忙 ~19ms、提交不再是瓶颈。→ 图化/编译无用。

**batch 摊销**（decode b=1 vs b=64，eager）：GPU 忙碌 1.70→2.20ms（只多一点，因权重只读一遍、被 64 token 摊销），而 wall 7.44→9.51ms 主要是 CPU/提交增加（提交 257→376）。→ 印证 M6"decode 吞吐随 batch 近线性、单步 latency 近平"；也侧面支持"decode 访存受限"（GPU 头号是读权重的 GEMV），但严格证明带宽上限需 kernel 级计数器（**ncu 本机不可用**），故仅列**受支持假设**。

## 6. 对 M6 两个假设的最终裁决
| M6 假设 | M7 裁决 | 关键证据（修订后） |
|---|---|---|
| decode/短 prefill 的 ~7ms 地板 = CPU/launch 开销 | **证实（更强）** | eager gpu_frac 0.23、257 次提交、`cudaLaunchKernel` 1.77ms 为头号 CPU；compiled 总提交→15（含 1 次 graph 回放）、wall 7.44→2.50 |
| 长 prefill = GPU 计算受限 | **证实** | gpu_frac 0.88、cutlass GEMM 14.6ms + flash attn 2.9ms 主导；compiled 总提交 257→255 基本不变而 wall 不变 |
| decode 访存受限（读权重的 GEMV） | **受支持，未定论** | GPU 头号为 GEMV、batch↑ GPU 仅微增（权重被摊销）；缺 kernel 级带宽计数器（ncu 不可用） |

## 7. 局限（如实）
- **nsys GPU kernel 追踪在本 WSL2 不可用**（已实测）；以 torch.profiler + chrome trace 替代。
- **GPU 时间用 kernel 事件区间并集**（非 key_averages 求和，避免双计）；本机均单 stream，故并集=求和。
- gpu_frac 只在明显 <1 时是干净"GPU 占比"；≈1 或略 >1（如 compiled decode）表示"已 GPU 受限"，精确值受两相测量差影响。
- profiler 有开销，故 wall 用独立干净计时相测。
- 单卡 / 单模型(0.5B) / bf16；`ncu` 不可用，访存受限只到"受支持假设"。
- chrome trace 体积大（6–23MB/文件），未入库、可复现。

## 8. 承上启下
- 呼应 M5：M7 从 CPU/kernel 层面**量化**了 FULL/PIECEWISE 省的到底是什么——省的是"CPU 逐 kernel 提交"。
- 呼应 M6：把 M6 两个"待验证假设"落成了有证据的结论。
- 指向 M9：热点已定位——decode/短 prefill 的 CPU 提交与 H2D 路径（对 eager 或图外部分）、长 prefill 的大 GEMM/attention（计算侧）。M9 优化对象由这些证据决定。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
python scripts/m7_run_all.py          # eager / compiled 各独立进程；-> outputs/m7/profile_*.json + traces/
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m7_profile.py --mode compiled --warmup 10 --n-prof 20
```
chrome trace 用 Perfetto / `chrome://tracing` 打开 `outputs/m7/traces/decode_b1_eager.json` 看 launch gap；对比 `decode_b1_compiled.json` 看整步塌缩成一次 `cudaGraphLaunch`。

---

## 统计实现修订记录（针对审查 2 点，均据 chrome trace 实测核实并修复）
1. **GPU 时间双计**：原 `gpu_us = Σ key_averages().self_device_time` 会把"高层算子的 device 归属"与"底层 kernel"重复相加（实测 decode 双计到 ~2×：3.46ms→真实 1.70ms）。→ 改为从 chrome trace 取 `cat=="kernel"` 事件、算**时间区间并集**（另存求和与每-stream 供参考）；本机单 stream 故并集=求和。修订后 decode gpu_frac 0.52→**0.23**、长 prefill 0.88（旧 1.75 系双计产物，且实为单 stream 无重叠，旧"重叠致 >wall"的解释一并纠正）。
2. **提交计数漏计**：原只数 `cudaLaunchKernel`/`ExC`（runtime）。→ 补 `cuLaunchKernel`（driver）与 `cudaGraphLaunch`（图回放），分列 `direct_runtime/direct_driver/graph_replays/total_cpu_submission_calls`。修订后长 prefill 提交 eager=257(rt161+drv96)、compiled=255(rt62+drv193)——**基本不变**，纠正了初版"launch 161→62"的错觉；decode compiled 总提交=15（含 1 次 graph 回放）。
