# M21 · 实测结论（监控与容量规划）

> **状态**：**M21 T+E+O COMPLETE**（2026-08-12）。E1–E7 全部执行；不变量 `check_invariants.py --module m21` **11 PASS / 0 FAIL**。
> **边界**：单卡 RTX 5080(sm_120) / WSL2 / Qwen2.5-0.5B-Instruct / `enforce_eager` / 单副本 / 客户端与服务端同机。
> 绝对值**不外推**到 compiled、其他模型、多卡或跨机 SLA。
> 设计冻结见 [`monitoring_and_capacity.md`](monitoring_and_capacity.md)；栈的用法与踩坑见 [`../../monitoring/README.md`](../../monitoring/README.md)。

---

## 0. 一页结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | 服务端指标**可信**：21 个压测窗口里 counter 增量与客户端 bench 汇总**逐条精确相等** | `outputs/m21/e2_validation.json` |
| 2 | 但**服务端直方图分位数 ≠ 客户端真值分位数**：P99 中位偏差 TTFT **+9.7%** / ITL +9.0% / E2EL +12.7%，最大 **+68%** | 同上，`figures/fig11_m21_capacity.png` (b) |
| 3 | 容量拐点**对采样窗口极敏感**：同一系统，5s 窗口判定「QPS 44 仍稳」，60s 窗口判定「QPS 32 已破线」 | `e3_shapeA_short200.csv` vs `e3_shapeA_60s.csv`，fig11 (a) |
| 4 | 过载时**先堆队列，GPU 从未打满**（窗口均值 GPU util 仅 25→32%）→ 本机瓶颈不是 GPU 算力 | fig11 (c)，E4 |
| 5 | 四类瓶颈有**可区分的指标签名**（见 §5 对照表），但「KV 受限」只压显存预算造不出来，必须从**需求侧**（长输出）构造 | `outputs/m21/e4_bottleneck.json` |
| 6 | **监控栈自身开销 ≲1%**（ON/OFF 单变量 A/B，差异小于 seed 噪声）→ 高 QPS 下的劣化不是监控造成的 | `outputs/m21/e7_observability_cost.json` |
| 7 | **`SIGKILL` 掉 vLLM API server 不会杀死持有显存的 `EngineCore` 子进程**：`up==0` 但显存 90s 不回收，新实例随后启动失败；清掉子进程后 **5s** 回收、28s 恢复 | `outputs/m21/e6_drills.json` |
| 8 | 7 条告警规则**全部真实 firing 过**（不是只写在 yaml 里） | 同上 |
| 9 | 本次测得的最大稳定 QPS（**24**）显著低于 M12 的 40，且**跨会话不可复现**（同日另一会话为 32）。**改结论不改数据**：本机在拐点附近不具备跨会话可比性 | §4 |

---

## 1. 监控栈（E1 Gate 1 PASS）

`vllm serve`(8014) + `dcgm-exporter`(9400) + 自写 `m21_host_exporter`(9836) → Prometheus(9090，**1s 抓取**) → Grafana(3000，面板由 provisioning 加载)。

Gate 1 判据全绿：四个 target `up`、7 条规则加载、16 条关键 PromQL 全部取到数、datasource 与 dashboard 均由文件 provision（`outputs/m21/e1_targets.json`）。

**Gate 0 的预判被实测推翻（记录在案）**：规划时我判断「WSL2 大概率不支持 DCGM，需退回自写 NVML exporter」。实测 `nv-hostengine` 正常启动、`dcgmi discovery` 找到 GPU、`dcgm-exporter` 暴露 15 条指标 → **改走标准方案**。
**但有一个硬限制**：`dcgmi dmon -e 1001,1002,1005`（`gr_engine_active` / `sm_active` / `dram_active`）在本环境**全为 N/A** → 本模块**拿不到 SM 占用率**，所有「GPU 忙不忙」的判断只能基于粗粒度 `DCGM_FI_DEV_GPU_UTIL`（NVML 口径的时间占比）。**不得声称测到了 SM 利用率。**（证据：`logs/m21/gate0_dcgm_probe.txt`）

### 冷启动会真的把告警打响
warmup 首请求含 Triton JIT，30s rate 窗口内服务端 P99 TTFT ≈ **0.42s** > 300ms SLO → `VLLMTTFTP99AboveSLO` firing。这**不是误报**，而是「warmup 与 measure 必须分开」在监控口径下的同一件事（AGENTS.md §5）。

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

| 指标 | 服务端直方图相对客户端真值的中位偏差 | 范围 |
|---|---|---|
| TTFT p99 | **+9.7%** | −10.4% ~ **+68.2%** |
| ITL p99 | +9.0% | −6.0% ~ +50.4% |
| E2EL p99 | +12.7% | +1.9% ~ +32.1% |

原因是 `histogram_quantile` 只能在**桶边界**间插值。→ **面板上的 P99 适合看趋势与告警，不适合当 SLA 验收数字**；验收要用客户端真值。

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

**本次会话的最大稳定 QPS = 24 req/s，容量区间 (24, 32]**（网格步长 8，非系统精确极限）。
achieved 吞吐在 **~35 req/s** 饱和；过载后 achieved 不再跟随 offered，队列持续堆积（waiting 705）→ 与 M12「过载=尾延迟劣化而非吞吐崩溃」的结论**不完全一致**：本次在 40 以上已经出现**吞吐跟不上 + 队列持续堆积**。

### 采样窗口把容量结论直接改写了（fig11-a）
同一套脚本、同一台机器，只把每档样本量从 **200 条(≈5s)** 换成 **60s 到达量**（M12 口径）：

| offered QPS | 5s 窗口(200 样本) worst P99 | 60s 窗口(M12 口径) worst P99 |
|---|---|---|
| 8 | 32ms | 42ms |
| 16 | 32ms | 43ms |
| 24 | 42ms | 46ms |
| 32 | **35ms** | **415ms** |
| 40 | 157ms | 7024ms |
| 44 | 210ms | 14649ms |
| 48 | 374ms | 20977ms |

5s 窗口下 QPS 32 看起来毫无压力（35ms）、44 也「达标」（210ms）；60s 窗口下 32 已破线、44 烂到 14.6 秒。
→ **短窗口压测会系统性高估容量**：队列积压需要时间才能显形，5 秒根本等不到尾部。
（这也意味着 M20-C 的 200 样本对照只适合做 C0/C1 **配对比较**，不能当容量结论用。）

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

| 场景 | 构造 | GPU util 峰值 | 进程 CPU 峰值 | KV 峰值 | waiting 峰值 | 抢占 | ITL 中位 |
|---|---|---|---|---|---|---|---|
| **B1** CPU 提交受限 | Shape A，rate=1（并发≈1） | 31% | 110% | 0.002 | 0 | 0 | 9.40ms |
| **B2** GPU 计算受限 | Shape B in1024/out32，rate=24 | **66%** | 131% | 0.030 | 0 | 0 | 10.24ms |
| **B3** 只压显存预算 | Shape A，util 0.20→0.10 | 50–53% | 143–149% | 0.12→**0.59** | 7–9 | **0** | ~10.4ms |
| **B3b** 需求侧长输出 | Shape C in128/out1900，util 0.12 | 85% | 130% | **0.9997** | 89 | **290** | 10.55ms |
| **B4** 队列受限 | Shape A，rate=64 | 57% | **163%** | 0.030 | **1196** | 0 | 11.30ms |

### 症状 → 指标签名 → 根因（面板上怎么读）

| 现象 | 指标签名 | 判断 |
|---|---|---|
| 延迟不低但队列空 | GPU util **低**(~31%) + 进程 CPU >100% + waiting=0 | **CPU 提交受限**（承接 M7：eager decode 的固定提交开销） |
| TTFT 偏高、队列不长 | GPU util **明显更高**(66%) + prefill 重的 shape | **GPU 计算受限**（长 prefill） |
| 完成延迟抖动、吞吐掉 | `kv_cache_usage_perc`→1 **且** `num_preemptions_total` 上升 | **KV 受限**（两条正交证据缺一不可） |
| 尾延迟爆炸但吞吐仍在 | waiting 持续堆积 + GPU/KV 都不满 | **队列/准入受限**（offered > 容量） |

### 一个必须如实记录的负结果
**只压 `--gpu-memory-utilization` 造不出 KV 压力**：Shape A 下从 0.40 一路压到 **0.10**，KV 使用率也只到 **0.59**，抢占恒为 0。
原因是需求侧被 `max_num_seqs=64 × 384 tokens` 封顶了。必须换成**需求侧构造**（Shape C 长输出，每条序列的 KV 随解码持续增长），才在 util=0.12 时把 KV 打满到 **0.9997** 并触发 **290 次抢占**。
→ 教训：**「调小某个 limit」不等于「造出了该资源的瓶颈」**，得先算清需求侧上限。

---

## 6. E5 · 一张卡承载多少用户（Q4）

Little's law：`N_users = QPS_stable × (E2EL + think_time)`。

- 输入（本会话）：`QPS_stable = 24 req/s`，该档 E2EL 中位 **1.72s**。
- 敏感性：

| 思考时间 | 周期 | 估算并发用户 |
|---|---|---|
| 5s | 6.73s | **≈161** |
| 15s | 16.7s | ≈401 |
| 30s | 31.7s | ≈761 |

**必须连同假设一起引用**：① 每用户串行（收到完整回答后思考再发下一条）；② 请求形状固定 in256/out128；③ SLO = P99 TTFT ≤300ms；④ 单卡/eager/0.5B/单副本。
且因 §4 的跨会话不稳定，**这里的用户数应读作数量级（10² 量级），不是可承诺的容量**：
若取另一会话的 QPS_stable=32（该档 E2EL 中位 1.73s），同样三档思考时间给出 **215 / 535 / 1015** 人。

---

## 7. E6 · 故障定位演练（Q5，4/4 PASS，7 条告警全部真实触发）

| 演练 | 注入方式 | 定位证据 | 结果 |
|---|---|---|---|
| **worker 退出** | `kill -9` 本脚本自己启动且 PID 已落盘的 server | `up{job="vllm"}` 1→0、`m21_target_up` 1→0，两条告警 **4s** 内 firing | ✅ |
| **OOM（启动期）** | `--gpu-memory-utilization 0.97` | 原始报错：`ValueError: Free memory on device cuda:0 (14.6/15.92 GiB) on startup is less than desired GPU memory utilization (0.97, 15.44 GiB)` | ✅ |
| **请求超时** | 过载 QPS=64 下用 0.5s 超时的探针客户端 | 探针超时数 > 0，同期服务端 waiting 峰值与 TTFT 尾同步升高 → 判定是**服务端排队**而非客户端网络问题 | ✅ |
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

> 口径说明：这里的 "OOM" 是 vLLM **启动期显存预算校验失败**（拒绝启动），**不是**运行中的 CUDA OOM 现场。不得混称。

---

## 8. 局限与未做

- ⬜ **未做**：Kubernetes / 多副本 / 长时 soak(>1h) / 跨机 SLA / Triton 指标面板 / compiled 模式对照。
- ⬜ **拿不到**：`DCGM_FI_PROF_*`（SM/tensor/dram active）在本环境全 N/A；WSL2 下 `nvidia-smi` 看不到进程级显存 → **无法把显存归因到单个进程**。
- ⚠️ **容量数字跨会话不稳定**（§4），只能给区间与机制，不能给承诺值。
- ⚠️ **P99 双口径**必须分开引用（§2），面板数字不能当 SLA 验收。
- ⚠️ M12→M21 的高并发 TPOT 回归**根因未定**（M12 无环境指纹）。

## 9. 过程中的错误（诚实记录）

1. **首轮 E3 用 200 样本/档**，与 M12 的 60s 窗口不可比 → 已按 M12 口径重跑；旧数据保留为 `e3_shapeA_short200*`，并因此产出了 §3 的「采样窗口敏感性」结论。
2. **E4 的 B3 阶梯曾被并发污染**：我在前一个实验未结束时启动了第二个进程，两者轮流抢 8014 端口，写出「同一 kv 测量值被标成两个不同 util」的交错记录。
   → 污染数据留档为 `outputs/m21/e4_bottleneck_contaminated_20260811.json`，整套矩阵已在**单进程**内重跑；并新增 `m21_common.acquire_lock()` 全模块互斥锁防止再犯。
3. **首版「显存已回收」判据只等 10s**，把 §7 的孤儿进程现象误判成「未回收」而非找根因 → 改为轮询并记录真实回收耗时。
4. **Gate 1 首版把「空载无告警」判在 warmup 样本仍在 rate 窗口内**，导致冷启动告警被当成 Gate 失败 → 改判据，并把冷启动现象作为结论留下。

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
python scripts/m21_figures.py                             # -> fig11 / fig12
python scripts/check_invariants.py --module m21 --verbose  # 11 PASS
```
