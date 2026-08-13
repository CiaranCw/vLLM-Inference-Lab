# M21 · 实测结论（监控与容量规划）

> **状态**：**M21 T+E+O COMPLETE**（2026-08-12）。E1–E7 全部执行；不变量 `check_invariants.py --module m21` **14 PASS / 0 FAIL**。
> **2026-08-12 evidence audit（两轮）**：
> 第一轮修正 8 处**口径/术语/单位**问题（E5 闭环 population 语义、E6 timeout 的秒/毫秒、E2 因果归因过强、
> 「5s 窗口」表述不准、OOM 与 GPU-compute 术语过强、容量记号、环境指纹增强）；
> 第二轮做**静态可复现性**修复（`m21_gate1.py` 的 SyntaxError、新增脚本编译守卫、Triton 术语消歧、E1 schema 说明）。
> **两轮都未修改任何原始实验数据，未重跑 E1–E7、未启动任何 GPU 实验**；
> 受影响的 derived artifact 由已落盘 raw 离线重算（旧版备份在 `outputs/m21/pre_audit_20260812/`）。逐条见 §10。
> **边界**：单卡 RTX 5080(sm_120) / WSL2 / Qwen2.5-0.5B-Instruct / `enforce_eager` / 单副本 / 客户端与服务端同机。
> 绝对值**不外推**到 compiled、其他模型、多卡或跨机 SLA。
> 设计冻结见 [`monitoring_and_capacity.md`](monitoring_and_capacity.md)；栈的用法与踩坑见 [`../../monitoring/README.md`](../../monitoring/README.md)。

---

## 0. 一页结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | 服务端指标**可信**：21 个压测窗口里 counter 增量与客户端 bench 汇总**逐条精确相等** | `outputs/m21/e2_validation.json` |
| 2 | 但**服务端直方图近似分位数 ≠ 客户端 exact percentile**：P99 中位偏差 TTFT **+9.7%** / ITL +9.0% / E2EL +12.7%，最大 **+68%**。两者测量边界与统计表示均不同，**不可互换** | 同上，`figures/fig11_m21_capacity.png` (b) |
| 3 | 容量拐点**对到达时长（arrival horizon）极敏感**：同一系统，固定 200-request 短时实验判定「QPS 44 仍稳」，60s offered-arrival horizon 判定「QPS 32 已破线」 | `e3_shapeA_short200.csv` vs `e3_shapeA_60s.csv`，fig11 (a) |
| 4 | 过载时**先堆队列，而粗粒度 GPU util 仍低**（窗口均值仅 25→32%）→ 与「GPU 算力饱和」不符；本环境无 SM-active 指标，故只能说**未见 GPU 算力压力**，不能断言 GPU 完全空闲 | fig11 (c)，E4 |
| 5 | 四类压力源在面板上有**可区分的指标签名**（见 §5 对照表），但「KV 受限」只压显存预算造不出来，必须从**需求侧**（长输出）构造 | `outputs/m21/e4_bottleneck.json` |
| 6 | **监控栈自身开销 ≲1%**（ON/OFF 单变量 A/B，差异小于 seed 噪声）→ 高 QPS 下的劣化不是监控造成的 | `outputs/m21/e7_observability_cost.json` |
| 7 | **`SIGKILL` 掉 vLLM API server 不会杀死持有显存的 `EngineCore` 子进程**：`up==0` 但显存 90s 不回收，新实例随后启动失败；清掉子进程后 **5s** 回收、28s 恢复 | `outputs/m21/e6_drills.json` |
| 8 | 7 条告警规则**全部真实 firing 过**（不是只写在 yaml 里） | 同上 |
| 9 | 本次最后一个达标档为 **24 req/s**、第一个不达标档为 32（真实容量 C ∈ [24, 32)），显著低于 M12 的 40，且**跨会话不可复现**（同日另一会话 32 仍达标）。**改结论不改数据**：本机在拐点附近不具备跨会话可比性 | §4、§3 |

---

## 1. 监控栈（E1 Gate 1 PASS）

`vllm serve`(8014) + `dcgm-exporter`(9400) + 自写 `m21_host_exporter`(9836) → Prometheus(9090，**1s 抓取**) → Grafana(3000，面板由 provisioning 加载)。

Gate 1 判据全绿：四个 target `up`、7 条规则加载、16 条关键 PromQL 全部取到数、datasource 与 dashboard 均由文件 provision（`outputs/m21/e1_targets.json`）。

> **`e1_targets.json` 的 schema 说明（2026-08-12 audit）**：该文件是 Gate 1 运行的**唯一原始记录**
> —— 它没有 `raw/` 对应物，也**无法离线重算**（重生成必须重跑 Gate 1，要起 vLLM server）。
> 因此按本仓「原始数据不可修改」的纪律，它被视为**历史 raw evidence，保持原样不动**。
> 后果是：文件里的 key 仍是 audit 前的 `vllm_ttft_p99` / `vllm_itl_p99`（**单位是秒**），
> 而 `scripts/m21_gate1.py` 现已改为显式的 `vllm_ttft_p99_seconds` / `vllm_itl_p99_seconds`。
> **两者数值口径一致（都是秒），只是命名不同**；下次跑 Gate 1 会自然产出新 key。
> 对照：`e2_validation.json` / `e5_user_capacity.json` / `e6_drills.json` 都有 raw 来源、可离线重算，
> 所以它们在 audit 中被重新生成；`e1_targets.json` 不属于这一类。

**Gate 0 的预判被实测推翻（记录在案）**：规划时我判断「WSL2 大概率不支持 DCGM，需退回自写 NVML exporter」。实测 `nv-hostengine` 正常启动、`dcgmi discovery` 找到 GPU、`dcgm-exporter` 暴露 15 条指标 → **改走标准方案**。
**但有一个硬限制**：`dcgmi dmon -e 1001,1002,1005`（`gr_engine_active` / `sm_active` / `dram_active`）在本环境**全为 N/A** → 本模块**拿不到 SM 占用率**，所有「GPU 忙不忙」的判断只能基于粗粒度 `DCGM_FI_DEV_GPU_UTIL`（NVML 口径的时间占比）。**不得声称测到了 SM 利用率。**（证据：`logs/m21/gate0_dcgm_probe.txt`）

### 冷启动会真的把告警打响
warmup 首请求含 **Triton GPU-kernel JIT**（此处 Triton 指 **Triton GPU 编程语言/编译器**的 kernel JIT，
**不是** NVIDIA Triton Inference Server —— 后者是 M20 的对象，**M21 主实验完全没有使用推理服务器**；
下文简称 *Triton kernel JIT*），30s rate 窗口内服务端 P99 TTFT ≈ **0.42s** > 300ms SLO →
`VLLMTTFTP99AboveSLO` firing。这**不是误报**，而是「warmup 与 measure 必须分开」在监控口径下的同一件事（AGENTS.md §5）。

---

## 2. E2 · 指标真值校验（**先过这关，后面才值得看**）

21 个 measure 窗口，全部满足：

| 不变量 | 结果 |
|---|---|
| I1 Δ`vllm:request_success_total` == bench `completed` | 21/21 精确相等 |
| I2 Δ`vllm:prompt_tokens_total` == bench `total_input_tokens` | 21/21 精确相等 |
| I3 Δ`vllm:generation_tokens_total` == bench `total_output_tokens` | 21/21 精确相等 |
| I4 `num_requests_running` ≤ `max_num_seqs`(64) | 全程满足 |
| I5 `kv_cache_usage_perc` ∈ [0,1] | 满足 |
| I6 counter 增量非负 | 满足 |
| I7 1s 抓取覆盖率 ≥ 0.8 | 满足 |

### 量化结论（非判据）：两种 P99 不是同一个数

| 指标 | 服务端直方图近似分位数相对客户端 exact percentile 的中位偏差 | 范围 |
|---|---|---|
| TTFT p99 | **+9.7%** | −10.4% ~ **+68.2%** |
| ITL p99 | +9.0% | −6.0% ~ +50.4% |
| E2EL p99 | +12.7% | +1.9% ~ +32.1% |

**归因必须克制**（2026-08-12 audit 修正：首版写成「bucket 上界向上取整所致」，是过强的单因归因）。
两者本就不是同一个量，观测到的偏差**至少**可能由三类原因共同产生，本实验**无法**把 +9.7% 归到其中任何一项：

1. **统计表示不同**：Prometheus classic histogram 只保存 bucket 计数，`histogram_quantile()` 是基于 bucket
   分布的**近似**估计，不是原始样本的精确分位数；
2. **观测边界不同**：server-side TTFT 与 client-observed TTFT 的计时起止点不同（客户端还含 HTTP/网络栈与客户端调度）；
3. **窗口聚合**：1s 抓取 + `increase()` over window 本身带来的近似。

→ 能下的结论只有一条：**两者不可互换**。
客户端 exact percentile 是 **client/SLO 口径的 ground truth**，**不是** server metric 的 ground truth；
面板 P99 更适合趋势与告警，**本模块的 SLO 验收一律以客户端 exact percentile 为准**。

### I8：两条独立采集路径的交叉校验救了一次
DCGM 与 NVML 的 GPU util 峰值差中位仅 **2.0**，但最大 **66.0** —— 那次极端差值来自 `dcgm-exporter` 的**默认 30s 采集间隔**：秒级压测窗口里 Prometheus 每秒抓到的都是最多 30 秒前的旧值（DCGM 报 2%，NVML 同期 33%）。改成 `-c 1000` 后两者收敛（同一窗口峰值均为 38%）。
→ **只有一个数据源时，你不会知道它是错的。**（证据：`outputs/m21/raw/smoke_runs.jsonl` vs `smoke2_runs.jsonl`）

---

## 3. E3 · 容量阶梯（Shape A in256/out128，7 档 × 3 seed × 60s 窗口）

判据沿用 **M12 冻结 SLO**：worst-seed 客户端 P99 TTFT ≤ 300ms 且 success ≥ 0.99。

| offered QPS | worst-seed P99 TTFT | achieved(中位) | waiting 峰值 | kv 峰值 | GPU util(窗口均值中位) | 进程 CPU 峰值 | 判定 |
|---|---|---|---|---|---|---|---|
| 8 | 42ms | 7.8 | 0 | 0.010 | 25.1% | 120% | ✅ |
| 16 | 43ms | 15.6 | 0 | 0.019 | 25.8% | 129% | ✅ |
| 24 | 46ms | 23.4 | 0 | 0.027 | 26.4% | 138% | ✅ |
| 32 | **415ms** | 31.1 | 11 | 0.029 | 29.4% | 149% | ❌ |
| 40 | 7024ms | 35.7 | 251 | 0.031 | 31.5% | 150% | ❌ |
| 44 | 14649ms | 34.9 | 515 | 0.031 | 30.9% | 152% | ❌ |
| 48 | 20977ms | 35.8 | 705 | 0.031 | 32.2% | 154% | ❌ |

**本次会话：最后一个达标档 = 24 req/s，第一个不达标档 = 32 req/s。**

两种记号必须分清（audit 修正）：

| 记号 | 含义 |
|---|---|
| 拐点定位到 **(24, 32]** | 在本测试网格上，「达标 → 不达标」的**翻转发生在**这个半开区间内（左开右闭，因为 32 是第一个观测到失败的点） |
| 真实容量 **C ∈ [24, 32)** | C = 未知的最大 SLO 达标 offered QPS。已知 24 达标 → C ≥ 24；已知 32 不达标 → C < 32 |

网格步长 8 req/s 决定分辨率；两者都**不是**系统精确极限。
（M12 文档里的 `(40, 44]` 是同一种「拐点定位」记号，其历史数据与表述本次不作改动。）
achieved 吞吐在 **~35 req/s** 饱和；过载后 achieved 不再跟随 offered，队列持续堆积（waiting 705）→ 与 M12「过载=尾延迟劣化而非吞吐崩溃」的结论**不完全一致**：本次在 40 以上已经出现**吞吐跟不上 + 队列持续堆积**。

### 到达时长（arrival horizon）把容量结论直接改写了（fig11-a）

**先纠正一处口径表述**（2026-08-12 audit）：首版把短实验称为「5s 窗口」是不准确的 ——
短实验的实际设计是**每档固定 200 个请求**，其 *nominal arrival horizon* = `num_prompts / offered_qps`
随 QPS 变化（25.0s @QPS8 → 4.2s @QPS48），只有 QPS40 那档恰好 ≈5s。
而 `duration_s`（bench 墙钟）还包含客户端建构/收尾等生命周期开销，**也不等于** arrival horizon。

| offered QPS | 固定 200-request：nominal horizon / 实际 duration | worst P99 TTFT | 60s horizon：nominal / 实际 duration | worst P99 TTFT |
|---|---|---|---|---|
| 8 | 25.00s / 35.0s | 32ms | 60.0s / 70.8s | 42ms |
| 16 | 12.50s / 22.4s | 32ms | 60.0s / 71.5s | 43ms |
| 24 | 8.33s / 18.2s | 42ms | 60.0s / 72.3s | 46ms |
| 32 | 6.25s / 16.1s | **35ms** | 60.0s / 72.9s | **415ms** |
| 40 | 5.00s / 14.8s | 157ms | 60.0s / 79.3s | 7024ms |
| 44 | 4.55s / 14.6s | 210ms | 60.0s / 87.3s | 14649ms |
| 48 | 4.17s / 14.2s | 374ms | 60.0s / 92.9s | 20977ms |

**数据支持的结论**：在拐点附近，固定 200-request 的较短 arrival horizon **显著低估了持续排队造成的尾延迟** ——
把 offered-arrival horizon 拉到约 60s 后，QPS32 的 worst-seed P99 TTFT 从 **35ms 量级** 变成 **约 415ms**，
QPS44 从 210ms 变成 14.6 秒。机制上排队积压需要时间才显形，短到达时长根本等不到尾部。
（因此 M20-C 的 200-request 对照只适合做 C0/C1 **配对比较**，不能当容量结论用。）

---

## 4. 与 M12 的差异：诚实处理（**改结论不改数据**）

M12（2026-07-24，同配置同 bench 参数、同 60s 窗口）测得最大稳定 QPS ≈ **40**；本次为 **24**。我逐条排查：

| 假设 | 检验方式 | 结论 |
|---|---|---|
| 监控栈开销 | **E7 单变量 A/B**（同一 server 进程内切换三容器 ON/OFF，3 seed × 2 档交错） | **否定**。QPS32：achieved 31.23 vs 31.25、ITL 11.44 vs 11.34ms；QPS40：achieved 38.98 vs 38.70、ITL 11.75 vs 11.86ms。差异 ≲1%，小于 seed 噪声 |
| GPU 降频/过热 | 全程 NVML 时钟/温度/功耗 | **否定**。SM 时钟窗口均值稳定 ~2500MHz，温度 ≤63°C，功耗 ≤156W（上限 360W） |
| 机器整体变慢 | E4-B1 单流（rate=1）ITL | **否定**。低负载 ITL 中位 **9.40ms**，与 M12 低负载基线同量级 |
| 高并发下单步变慢 | 同 QPS40 逐项对比 | **成立**。M12：ITL 中位 9.2ms / TPOT 9.3ms / max_concurrent 108；本次同点 11.9ms / 12.1ms / 127 |

**机制**（可算的账）：Little's law 下所需并发 ≈ QPS × (out_tokens × TPOT)。
TPOT 9.3ms → 40 × 128 × 9.3ms ≈ **48 并发**，低于 `max_num_seqs=64`，队列不堆；
TPOT 12.1ms → 40 × 128 × 12.1ms ≈ **62 并发**，顶到 64 的硬闸 → 开始排队 → 排队又拉长 ITL → **正反馈**。
→ **拐点附近，单步解码 +30% 就足以让容量结论翻盘**；这也解释了 M12 自己在 QPS40 上的巨大 seed 方差（56/209/194ms）。

**根因未定**：TPOT 为何在高并发下比 M12 慢 ~30%，**无法归因** —— M12 当年**没有落盘环境指纹**（驱动/torch/内核版本、CPU、后台负载），今天无从比对。
→ 本模块补了 `scripts/m21_env_fingerprint.py`，输出 `outputs/m21/env_fingerprint.json`（驱动 610.62 / torch 2.11.0+cu130 / triton 3.6.0 / 内核 6.18.33.2-microsoft-standard-WSL2 / 24 逻辑核），**给后续模块留基线**。

**同日跨会话也不可复现**：E7 会话（新起 server、只跑 32/40 两档）在 QPS32 上 worst P99 仅 **174ms（达标）**，而 E3 会话同档为 415ms（破线）。
→ **诚实表述**：本机在拐点附近的容量**跨会话不稳定**，区间为 **24–32 req/s**；**不宣称**「M12 的 40 是错的」，也**不宣称**已复现 40。

---

## 5. E4 · 瓶颈签名矩阵（回答「瓶颈是谁」）

每场景独立 server 生命周期、60s 窗口。**整套矩阵在同一个进程里跑完**（见 §7 的教训）。

> **口径说明（audit 修正）**：`run_id` 为兼容既有原始数据仍叫 `B2_gpu_compute`，但**表述一律降到
> 「GPU-compute pressure 更强 / prefill-heavy signature」**。M21 的 dashboard 层只能识别到
> 「压力更强」；**长 prefill 是 compute-bound 的更底层机制证据来自 M7 的 profiler**（gpu_frac 0.88），
> 不是本模块的粗粒度 `GPU_UTIL` 能证的。

| 场景 | 构造 | GPU util 峰值 | 进程 CPU 峰值 | KV 峰值 | waiting 峰值 | 抢占 | ITL 中位 |
|---|---|---|---|---|---|---|---|
| **B1** CPU 提交受限 | Shape A，rate=1（并发≈1） | 31% | 110% | 0.002 | 0 | 0 | 9.40ms |
| **B2** prefill 计算压力（GPU-compute-heavy signature） | Shape B in1024/out32，rate=24 | **66%** | 131% | 0.030 | 0 | 0 | 10.24ms |
| **B3** 只压显存预算 | Shape A，util 0.20→0.10 | 50–53% | 143–149% | 0.12→**0.59** | 7–9 | **0** | ~10.4ms |
| **B3b** 需求侧长输出 | Shape C in128/out1900，util 0.12 | 85% | 130% | **0.9997** | 89 | **290** | 10.55ms |
| **B4** 队列受限 | Shape A，rate=64 | 57% | **163%** | 0.030 | **1196** | 0 | 11.30ms |

### 症状 → 指标签名 → 根因（面板上怎么读）

| 现象 | 指标签名 | 判断 |
|---|---|---|
| 延迟不低但队列空 | GPU util **低**(~31%) + 进程 CPU >100% + waiting=0 | **CPU 提交受限**（承接 M7：eager decode 的固定提交开销） |
| TTFT 偏高、队列不长 | 粗粒度 GPU util **明显更高**(66% vs B1 的 31%) + prefill 重的 shape | **GPU compute pressure 明显更强**（prefill-heavy signature）。**注意**：本环境 `DCGM_FI_PROF_SM_ACTIVE/GR_ENGINE_ACTIVE/DRAM_ACTIVE` 全为 N/A，仅凭 `GPU_UTIL=66%` **不能**断言「GPU 算力已饱和 / 是唯一瓶颈」 |
| 完成延迟抖动、吞吐掉 | `kv_cache_usage_perc`→1 **且** `num_preemptions_total` 上升 | **KV 受限**（两条正交证据缺一不可） |
| 尾延迟爆炸但吞吐仍在 | waiting 持续堆积 + GPU/KV 都不满 | **队列/准入受限**（offered > 容量） |

### 一个必须如实记录的负结果
**只压 `--gpu-memory-utilization` 造不出 KV 压力**：Shape A 下从 0.40 一路压到 **0.10**，KV 使用率也只到 **0.59**，抢占恒为 0。
原因是需求侧被 `max_num_seqs=64 × 384 tokens` 封顶了。必须换成**需求侧构造**（Shape C 长输出，每条序列的 KV 随解码持续增长），才在 util=0.12 时把 KV 打满到 **0.9997** 并触发 **290 次抢占**。
→ 教训：**「调小某个 limit」不等于「造出了该资源的瓶颈」**，得先算清需求侧上限。

---

## 6. E5 · 一张卡承载多少用户（Q4）

用 **Interactive Response Time Law**（闭环交互系统的 population 估计，可由 Little's law 推出）：

```text
closed_loop_user_population  N = X × (R + Z)
average_in_system_requests   L = X × R      ← 真正「在服务器里」的请求数
estimated_thinking_users     T = X × Z      ← 正在思考、不在服务系统内
```

`X` = SLO 达标的最大 offered QPS，`R` = 服务响应时间(E2EL)，`Z` = 用户思考时间。

**语义红线（2026-08-12 audit 修正）**：首版把 `X × (R + Z)` 命名为 `estimated_concurrent_users`
并称「并发用户」，是**误导性表述** —— 它是**闭环用户总数**（含正在思考、根本不在系统里的人），
**绝不能**读作「同时有 161–761 个请求在服务器里执行」。后者是 `L`，量级差一个数量级。

输入（本会话）：`X = 24 req/s`，该档 E2EL 中位 **R = 1.7247 s**。

**平均在系统内的请求数 `L = 24 × 1.7247 ≈ 41.4`**（这才是「服务器里同时有多少请求」）。

| 思考时间 Z | 周期 R+Z | 思考中用户 T=X·Z | **闭环用户总数 N=X·(R+Z)** |
|---|---|---|---|
| 5s | 6.73s | 120 | **≈161** |
| 15s | 16.7s | 360 | ≈401 |
| 30s | 31.7s | 720 | ≈761 |

**必须连同假设一起引用**：① 每用户串行（收到完整回答后思考再发下一条）；② 请求形状固定 in256/out128；
③ SLO = P99 TTFT ≤300ms；④ 单卡/eager/0.5B/单副本。
且因 §4 的跨会话不稳定，**这些人数只应读作数量级（10² 量级），不是可承诺的容量**：
若取另一会话的 X=32（该档 E2EL 中位 1.73s），同样三档思考时间给出 **215 / 535 / 1015** 人闭环总数，
对应 `L ≈ 32 × 1.73 ≈ 55.4` 个在系统内请求。

---

## 7. E6 · 故障定位演练（Q5，4/4 PASS，7 条告警全部真实触发）

| 演练 | 注入方式 | 定位证据 | 结果 |
|---|---|---|---|
| **worker 退出** | `kill -9` 本脚本自己启动且 PID 已落盘的 server | `up{job="vllm"}` 1→0、`m21_target_up` 1→0，两条告警 **4s** 内 firing | ✅ |
| **启动期显存预算拒绝**（`startup_memory_rejection`，artifact key 仍为 `oom`） | `--gpu-memory-utilization 0.97` | 原始报错：`ValueError: Free memory on device cuda:0 (14.6/15.92 GiB) on startup is less than desired GPU memory utilization (0.97, 15.44 GiB)` | ✅ |
| **请求超时** | 过载 QPS=64 下用 0.5s 超时的探针客户端 | 探针 **60/60 全部超时**；同期服务端 `waiting` 峰值 **774**、服务端直方图 TTFT P99 ≈ **19.57 s**（≈19568 ms），背景负载客户端 P99 TTFT ≈ **17.37 s** → 两者同为「严重排队」量级，判定是**服务端排队**而非客户端网络问题 | ✅ |
| **抢占** | Shape C 长输出 + util=0.12 | `num_preemptions_total` 上升，`VLLMPreemptionHappening`(28s) 与 `VLLMKVCacheNearFull`(32s) 相继 firing | ✅ |

### 本模块最有操作价值的一条发现

**`SIGKILL` 打在 vLLM API server 父进程上，不会杀死持有显存的 `EngineCore` 子进程。**

实测（`outputs/m21/e6_drills.json` → `worker_exit`）：
1. kill 前落盘子进程 PID：`[15471(resource_tracker), 15472(VLLM::EngineCore)]`；
2. kill 父进程后：告警 4s 内 firing，但 **90 秒后显存仍占 13307 MiB 未回收**，`VLLM::EngineCore` 仍存活；
3. 清理**已记录的** `EngineCore` PID 后：显存 **5s** 回收；
4. 随后 server **28s** 正常拉起。

**运维含义**：`up==0` 只说明「端口没了」，**不代表资源已释放**。判断服务是否真的干净退出，必须同时看 **GPU 显存是否回到基线**；否则下一次拉起会直接撞
`ValueError: No available memory for the cache blocks`（本次首轮重启失败就是这么来的）。
面板上对应的组合是：`up{job="vllm"}==0` **且** `DCGM_FI_DEV_FB_USED` 居高不下。

> **口径说明（audit 强化）**：这里的 "OOM" 是 vLLM **启动期显存预算校验失败** —— EngineCore 主动拒绝初始化，
> 准确名称为 `startup_memory_rejection`（artifact 的场景 key 为兼容既有数据仍是 `oom`，另有 `display_name` 字段）。
> **本模块没有注入运行中的 CUDA allocator OOM**（`torch.cuda.OutOfMemoryError`）——那属于 scope boundary /
> optional extension，不为补齐表格去做危险实验。两者不得混称。

---

## 8. 局限与未做

- ⬜ **未做**：Kubernetes / 多副本 / 长时 soak(>1h) / 跨机 SLA / Triton 指标面板 / compiled 模式对照。
- ⬜ **拿不到**：`DCGM_FI_PROF_*`（SM/tensor/dram active）在本环境全 N/A；WSL2 下 `nvidia-smi` 看不到进程级显存 → **无法把显存归因到单个进程**。
- ⚠️ **容量数字跨会话不稳定**（§4），只能给区间与机制，不能给承诺值。
- ⚠️ **P99 双口径**必须分开引用（§2），面板数字不能当 SLA 验收。
- ⚠️ M12→M21 的高并发 TPOT 回归**根因未定**（M12 无环境指纹）。
- ⬜ **未注入 runtime CUDA allocator OOM**：E6 的 "OOM" 只覆盖启动期显存预算拒绝，运行中 OOM 属 optional extension。
- ⚠️ **B2 只能证明「GPU compute pressure 更强」**，不能证明「GPU 算力饱和」——本环境没有 SM-active 指标。

## 9. 过程中的错误（诚实记录）

1. **首轮 E3 用 200 样本/档**，与 M12 的 60s 窗口不可比 → 已按 M12 口径重跑；旧数据保留为 `e3_shapeA_short200*`，并因此产出了 §3 的「采样窗口敏感性」结论。
2. **E4 的 B3 阶梯曾被并发污染**：我在前一个实验未结束时启动了第二个进程，两者轮流抢 8014 端口，写出「同一 kv 测量值被标成两个不同 util」的交错记录。
   → 污染数据留档为 `outputs/m21/e4_bottleneck_contaminated_20260811.json`，整套矩阵已在**单进程**内重跑；并新增 `m21_common.acquire_lock()` 全模块互斥锁防止再犯。
3. **首版「显存已回收」判据只等 10s**，把 §7 的孤儿进程现象误判成「未回收」而非找根因 → 改为轮询并记录真实回收耗时。
4. **Gate 1 首版把「空载无告警」判在 warmup 样本仍在 rate 窗口内**，导致冷启动告警被当成 Gate 失败 → 改判据，并把冷启动现象作为结论留下。

---

## 10. 2026-08-12 evidence audit（只改口径与 derived，不动原始数据）

| # | 原问题 | 为什么错 | 改了什么 | 是否影响核心实验结论 |
|---|---|---|---|---|
| P0-1 | E5 把 `X×(R+Z)` 命名为 `estimated_concurrent_users`／「并发用户」 | 这是**闭环用户 population**（含正在思考、不在系统内的人），不是系统内并发请求数（那是 `X×R`），量级差一个数量级 | `scripts/m21_capacity_users.py` 字段改为 `average_in_system_requests` / `estimated_thinking_users` / `estimated_closed_loop_users`；headline 重写；离线重算 `e5_user_capacity.json`；§6 重写 | **不影响**。数值未变（161/401/761 仍是闭环总数），新增 `L≈41.4` |
| P0-2 | `e6_drills.json` 的 `server_hist_ttft_p99_ms_during_probe=19.57` 实为**秒** | `vllm:time_to_first_token_seconds` 单位是秒，`prom_scalar_at()` 不换算，脚本漏了 ×1000 | `m21_fault_drills.py` 补 `_s` 原值 + `_ms` 换算；`m21_common.prom_scalar_at` 加单位警告；`m21_gate1.py` key 改为显式 `_seconds`；新增 `scripts/m21_e6_consolidate.py` 从 raw 离线重建 derived | **不影响** E2/E3（`m21_capacity_sweep.py` 一直有 ×1000）。仅 E6 timeout 的一个字段：19.57ms → **19568ms** |
| P0-3 | E2 把偏差归因为「bucket 上界向上取整所致」 | 单因归因过强：还有观测边界差异与窗口聚合，本实验无法区分各自贡献 | `m21_validate.py` 的 note 改为三因并列 + 「不可互换」结论；重算 `e2_validation.json`（**实测数值未变**）；§2 重写；图注与 README 同步 | **不影响**。+9.7% / −10.4%~+68.2% 原样保留 |
| P0-4 | 把固定 200-request 的短实验称为「5s 窗口」 | 短实验固定的是**请求数**不是时长，nominal arrival horizon 随 QPS 从 25.0s 变到 4.2s；`duration_s` 还含客户端开销 | 全仓改为「固定 200-request 短时实验 / offered-arrival horizon」，§3 表格同时给出 nominal horizon 与实际 duration；`m21_figures.py` 图例标题重写并离线重出 fig11 | **不影响**。结论改为更准确的表述：短 arrival horizon 低估持续排队尾延迟 |
| P1-1 | `m21_fault_drills.py` 仍称「进程级 CUDA OOM」 | 实际是**启动期显存预算校验失败**（EngineCore 拒绝初始化），不是 `torch.cuda.OutOfMemoryError` | 场景 key 保留 `oom`，新增 `display_name=startup_memory_rejection` 与 `scope_note`；docstring/§7/§8 明确「未注入 runtime OOM」 | **不影响**。演练本身与原始报错原样保留 |
| P1-2 | B2 直接称「GPU 计算受限」 | 本环境 `DCGM_FI_PROF_*` 全 N/A，仅凭粗粒度 `GPU_UTIL=66%` 不足以证明算力饱和 | §5 表格与对照表改为「GPU-compute pressure 更强 / prefill-heavy signature」，并注明底层机制证据来自 M7 profiler；fig11(c) 标题去掉「never saturates」 | **不影响**。数值与场景构造未动 |
| P1-3 | 「容量区间 (24, 32]」一种写法承担两种含义 | 拐点定位区间 ≠ 真实容量 C 的取值范围 | §3 拆成「拐点定位到 (24,32]」与「C ∈ [24,32)」两行；`e5_user_capacity.json` 增 `capacity_notation`；M12 的 `(40,44]` 历史文档不改，仅在此说明 | **不影响** |
| P2 | 环境指纹字段不足以定位跨会话回归 | 缺 vLLM/torch/triton 版本、源码 SHA 与 dirty flag、`VLLM_*` 环境变量等 | `m21_env_fingerprint.py` 扩充字段并重采一次（**不占 GPU、不建引擎**） | **不影响** |

### 第二轮（同日，静态/离线修复，**未启动任何 GPU 实验**）

| # | 原问题 | 为什么错 | 改了什么 | 是否影响既有核心实验结论 |
|---|---|---|---|---|
| R-P0-1 | `m21_gate1.py` **SyntaxError**：第一轮改注释时把逗号写进了行尾注释（`... # DCGM 该字段单位为 MiB,`） | 该脚本**根本无法解析**，Gate 1 不可复现 | 逗号移出注释；未改任何实验逻辑 | **不影响**。E1 的既有结果由 audit 前可运行的版本产生 |
| R-P0-2 | 13 条 invariant 全 PASS，却没发现上面那个 SyntaxError | 说明判据**完全没有覆盖「脚本至少是可执行 artifact」** | 新增 `m21_scripts_compile`：对 12 个 M21 正式脚本**真实执行** `py_compile.compile(doraise=True)`（写临时目录，不污染 `__pycache__`），不是查文件存在。**非空转验证**：临时注入语法错误 → FAIL 并报出 `SyntaxError: '(' was never closed`，还原后回到 PASS | **不影响**。判据总数 13 → **14 PASS / 0 FAIL** |
| R-P1-1 | `outputs/m21/e1_targets.json` 仍是 audit 前的 key（`vllm_ttft_p99` / `vllm_itl_p99`） | 与已改名的脚本不一致 | **不改该文件**：它是 Gate 1 的唯一原始记录、无 `raw/` 对应物、无法离线重算，按「原始数据不可修改」纪律保留，并在 §1 写明 schema 差异与两者单位一致（都是秒） | **不影响**。数值一个 bit 未动 |
| R-P1-2 | 「Triton JIT」在 M21 语境下有歧义 | M21 主实验**没有使用 NVIDIA Triton Inference Server**，此处 Triton 指 **Triton GPU 编程语言/编译器**的 kernel JIT；混淆会让人以为 M21 跑了推理服务器 | 首次出现处改为「**Triton GPU-kernel JIT**（指 Triton GPU 编程语言/编译器，不是 NVIDIA Triton Inference Server）」，后续简称 *Triton kernel JIT*；涉及 `results.md` §1、`m21_common.py`、`m21_capacity_sweep.py`、`m21_gate1.py`、`monitoring/README.md`。**M20 的 Triton Inference Server 历史结论一字未改** | **不影响** |

**新增两条更严格的不变量**（**不是**放宽既有判据，既有 11 条一字未改）：
- `m21_e5_closed_loop_semantics`：E5 必须分列 `L=X·R` 与 `N=X·(R+Z)`，且不得再出现 `estimated_concurrent_users` 命名；
- `m21_e6_ttft_unit_consistency`：服务端与客户端 TTFT P99 必须同量级（0.1×~10×）。
  **非空转验证**：用 audit 前的旧 `e6_drills.json` 跑该判据，比值 **0.001126 → FAIL**（正确捕获单位错）；修正后为 1.13 → PASS。
加上第二轮的 `m21_scripts_compile`，`check_invariants.py --module m21` 由 11 PASS 变为 **14 PASS / 0 FAIL**。

**未重跑**：E3 / E4 / E7 / E6 的任何故障注入。
**重新生成的 derived artifact**：`e5_user_capacity.json`、`e6_drills.json`、`e2_validation.json`、`env_fingerprint.json`、
`figures/fig11_m21_capacity.png`、`figures/fig12_m21_bottleneck_signature.png`；旧版备份在 `outputs/m21/pre_audit_20260812/`。
**原始 raw（`outputs/m21/raw/*.jsonl`、bench 结果 JSON、`logs/m21/*`）一个字节未改。**

---

## 附：复现

```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0

~/bin/docker compose -f monitoring/docker-compose.yml up -d
nohup python scripts/m21_host_exporter.py --port 9836 \
  --pid-file outputs/m21/raw/server.pid > logs/m21/host_exporter.log 2>&1 &

python scripts/m21_env_fingerprint.py                     # 环境指纹
python scripts/m21_gate1.py                               # E1 栈自证
python scripts/m21_capacity_sweep.py --shape A --rates 8,16,24,32,40,44,48 \
       --seeds 0,1,2 --measure-seconds 60 --tag e3_shapeA_60s   # E3（+E2 快照）
python scripts/m21_validate.py                            # E2 校验
python scripts/m21_bottleneck.py                          # E4 瓶颈矩阵
python scripts/m21_capacity_users.py                      # E5（离线）
python scripts/m21_fault_drills.py                        # E6 故障演练
python scripts/m21_observability_cost.py                  # E7 监控开销 A/B
python scripts/m21_e6_consolidate.py                      # E6 derived 合并（离线，含单位修正）
python scripts/m21_figures.py                             # -> fig11 / fig12
python scripts/check_invariants.py --module m21 --verbose  # 14 PASS
```
