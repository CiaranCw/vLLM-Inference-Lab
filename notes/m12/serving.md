# M12 · 在线/开环 Serving 与 SLO（★核心验收 #第一阶段收口项）

> 目标：从"离线闭环引擎基准（M6）"迈到**真实 HTTP 在线服务 + 有限 QPS 开环（Poisson 到达）**，测 TTFT/ITL/TPOT/E2EL 的 P50/95/99、成功率、Goodput、**队列稳定性与最大稳定 QPS**、过载/背压、容量随负载形状的差异。
> 与 M6 本质区别：**M6=离线闭环、进程内 proxy 指标；M12=真实 vLLM OpenAI 服务 + 开环、真正的在线 TTFT/ITL/E2EL（HTTP+流式）**。
> 边界：单卡 RTX 5080(sm_120) / Qwen2.5-0.5B / **enforce_eager**（结论绝对值限 eager，不外推 compiled）；进程内客户端指标非跨机在线 SLA。零 vLLM 源码改动。
>
> 证据：`outputs/m12/{config.json, knee.json, sweep.csv, summary.json, raw/*(bench summary+detailed), metrics/*.jsonl, metadata/*}`、`logs/m12/*`。工具=真实 `vllm serve` + 官方 `vllm bench serve`（开环负载器）。复现见文末。

## 0. 概念（先厘清，读懂后面才不费力）
- **闭环（M6）**：下一步/请求受完成节流，测饱和吞吐；**开环（M12）**：请求按外部时钟 **Poisson 到达**（`--request-rate R --burstiness 1.0`），到达与完成解耦、队列可累积，测"给定 QPS 下的延迟分布 + 能否稳住"。
- **在线指标（源码口径）**：`TTFT`=发出→首 token（含 HTTP/queue/prefill）；`ITL`=同一请求相邻输出 token 间隔（**跨请求合并**后取 percentile，反映流式卡顿尾）；`TPOT`=`(E2EL−TTFT)/(out−1)`（**每请求**平均 decode 速度）；`E2EL`=端到端。percentile=`np.percentile`(linear)。
- **Goodput**：`--goodput ttft:X tpot:Y e2el:Z`，**每请求需同时满足所有 SLO(AND)** 才算 good；`request_goodput=good/时长`。
- **饱和吞吐 vs 最大稳定 QPS**：前者=系统吐得出的最高 req/s；后者=**满足 SLO** 的最高 offered QPS（通常 < 饱和吞吐，因尾延迟先破线）。

## 1. 固定配置（跑前冻结，`outputs/m12/config.json`）
- **Server（唯一命令）**：`vllm serve Qwen2.5-0.5B-Instruct --enforce-eager --dtype auto --max-model-len 2048 --gpu-memory-utilization 0.6 --max-num-seqs 64 --max-num-batched-tokens 4096 --no-enable-prefix-caching --no-async-scheduling --enable-chunked-prefill --disable-uvicorn-access-log`（attention=FLASH_ATTN、kv=auto）+ env `VLLM_USE_FLASHINFER_SAMPLER=0`。
- **Workload**：`random, random_range_ratio=0, ignore_eos, burstiness=1.0, temperature=0, top_p=1`；**Shape A=in256/out128**（decode 偏重）、**Shape B=in1024/out32**（prefill 偏重）。**唯一自变量=request-rate(QPS)**。
- **Calibration(rate=1，实测无排队 waiting≡0)** → 低负载 P99：`ttft 24.2 / tpot 9.7 / e2el 1250 / itl 13.2 ms`。**冻结实验 SLO（跑前写死、跑后不改）**：`ttft_p99≤300ms（=max(300,24×3)）、tpot_p99≤20ms、e2el_p99≤2499ms、itl_p99≤39ms、success≥0.99、goodput≥0.95、drain≤20s、preempt=0、timeout=0`。
- **run_stable/qps_stable/队列稳定判据** 全部数值化冻结（斜率>0.5、tail/head 差≥5且比≥2、end_waiting>5、drain>20s；3 seed 全过=qps_stable，worst-seed P99 判 SLO）。详见 config.json。

## 2. 核心结果 —— Shape A 的 QPS→延迟曲线与最大稳定 QPS

**粗扫（单 seed，achieved 吞吐 & P99 TTFT）**：
| offered QPS | 1 | 8 | 16 | 24 | 32 | 48 |
|---|---|---|---|---|---|---|
| achieved req/s | 0.98 | 7.74 | 15.5 | 23.1 | 30.9 | 45.5 |
| P99 TTFT (ms) | 22 | 25 | 25 | 29 | **32** | **242** |
| peak waiting | 0 | 0 | 0 | 0 | 0 | 9 |
→ QPS ≤ 32 时队列恒空、P99 TTFT 平稳 ~25–32ms；到 48 出现拐点（P99 TTFT 跳到 242ms、队列开始堆积）。

**Knee 精扫（3 seed，worst-seed P99 TTFT 判 SLO=300ms）**：
| offered QPS | 3-seed 稳定 | worst P99 TTFT | 各 seed P99 TTFT |
|---|---|---|---|
| 36 | ✅ | 121ms | 34 / 28 / 121 |
| **40** | ✅ | **209ms** | 56 / 209 / 194 |
| 44 | ❌ | 583ms | 117 / **583** / **387** |
| 48 | ❌ | 663ms | 220 / **663** / 383 |

**→ 最大稳定 Offered QPS ≈ 40 req/s**（容量区间 **(40, 44]**，测试网格步长 4 req/s；"最大稳定 QPS"=当前网格中 3 seed 全部满足冻结 SLO 的最大离散档，非系统精确极限）。

## 3. 过载是"尾延迟劣化"而非"吞吐崩溃"（关键洞察）
- 即使 offered 44/48，**achieved 吞吐仍紧跟 offered（43/47 req/s）、队列峰值不大（5–36）、到达结束时 waiting=0、drain ~1s**——系统没有"吞吐崩溃/持续堆积"。
- 破线的是 **TTFT 尾部（P99 583/663ms > 300ms SLO）**：高 QPS 下 Poisson 到达的**瞬时突发**让部分请求短暂排队，抬高 P99 尾。**所以 SLO 定义的实用容量（40 req/s）低于原始饱和吞吐（~46 req/s）**。
- **seed 方差大**（如 qps40：56/209/194ms；qps44：117/583/387ms）——尾延迟对到达随机性敏感，故用 **3 seed + worst-seed** 判定（单 seed 会误判）。

## 4. Burst（All-at-once，request-rate=inf，300 请求同时到达）
`peak_waiting=236、peak_running=64(=max_num_seqs 上限)、P99 TTFT 5343ms、E2EL P99 6416ms、achieved 46.6 req/s、输出 5972 tok/s、drain ~1s`。
→ 一次性灌入时队列瞬间堆到 236、running 顶到调度上限 64；**饱和吞吐 ≈ 46.6 req/s**（与 §2 拐点一致）。**独立实验，不参与 QPS 曲线/最大稳定 QPS 判定**。

## 5. 容量随负载形状变化（Shape A vs B）
| | Shape A (in256/out128, decode 偏重) | Shape B (in1024/out32, prefill 偏重) |
|---|---|---|
| qps32 achieved | 30.9 req/s | 31.7 req/s |
| qps32 P99 TTFT | 32ms | **61ms**（prefill 更重、首 token 更慢） |
| qps32 total token 吞吐 | ~10k tok/s | **33.5k tok/s** |
- **req/s 容量两者都很高（≥32 仍稳）**，但 **token 吞吐差一个量级**（Shape B 每请求 1024 输入 token）；prefill-heavy 的 **TTFT 更高**。→ **"服务容量"不是模型的单一常数，取决于输入/输出 token 形状**。（Shape B 未扫到过载，容量 ≥32 req/s。）

## 6. 与 M6 的对比（能力跃迁）
- M6：离线、进程内、闭环、proxy 指标（`decode_step_latency` 等），测单步/饱和。
- M12：真实 HTTP 服务、开环 Poisson、**真在线** TTFT/ITL/E2EL + 队列/背压 + SLO 容量。**首次给出"这个服务在多大 QPS 下还能满足延迟 SLO"的可运营答案**（本机本配置：~40 req/s）。

## 7. 局限（如实）
- **enforce_eager**：所有绝对值（TTFT/TPOT/最大稳定 QPS/饱和吞吐）**限 eager**；开环排队/尾延迟的**定性规律**可一般推断，但 compiled/CUDA Graph 会显著降低单步延迟、抬高容量——**不声称 eager 的 40 req/s 代表生产**。
- 单卡 / 0.5B（很小，故容量高、KV 压力低，kv_usage 峰值仅 3%）；进程内客户端指标非跨机 SLA；随机语料、贪心。
- 网格步长 4 req/s（(40,44] 区间未再细分）；burst 用 300 请求（非长时 soak）。
- Shape B 未扫到过载（容量 ≥32）；max-concurrency admission、compiled 对照 为可选未做。

## 8. 实现踩坑（诚实记录，全部已修）
过程中修复 6 处并**据实修正实验网格**：① `Sampler._stop` 遮蔽 `Thread._stop()`；② burst `rate="inf"` 与 timeout 算术冲突；③ `--stage` argparse choices 缺项；④ summarize `ndarray.ptp()` NumPy 2.0 移除→`np.ptp`；⑤ `_explor_stable` 误用未定义 `load()`；⑥ 清理脚本 `pkill -f "vllm serve"` **自匹配 kill 了执行 shell**。**最关键的诚实修正**：初始 smoke（少样本+冷启动）误判"rate6 过载"，加 warmup + 稳态大样本后发现**容量远高于估计**，遂把粗扫网格从 {0.5..6} 扩到 {1..48} 并改为 **in-script 动态 knee 检测 + fine 三 seed 精扫**，最终在 {36,40,44,48} 定位到真实拐点。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
python scripts/m12_run_sweep.py                          # 全流程：server 生命周期+calibration+冻结SLO+粗扫+动态knee精扫(3seed)+burst+ShapeB
python scripts/m12_run_sweep.py --stage fineonly --fine-rates "36,40,44"   # 追加细扫 knee
python scripts/m12_summarize.py                          # -> sweep.csv / summary.json（最大稳定 QPS 区间）
```
逐 run：`outputs/m12/raw/<run_id>.json`（bench summary+detailed）、`metrics/<run_id>.jsonl`（/metrics 队列时序）、`metadata/<run_id>.json`；冻结配置/SLO：`config.json`；knee：`knee.json`。
