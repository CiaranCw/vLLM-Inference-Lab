# M21 · 监控与容量规划 —— 设计冻结（规划稿）

> **状态：本文件是 M21 的设计冻结稿；实验已全部执行，结论见 [`results.md`](results.md)。**
> **2026-08-12 evidence audit** 已同步修正本文中的口径表述（E5 闭环 population 语义、E2 归因、
> OOM 与 GPU-compute 术语）；原始设计意图与「不做的事」保持不变。
> 下面这段是规划当时的原文状态说明：
> ~~**状态：⬜ 规划中，未执行任何实验。**~~ 本文件只冻结「要测什么 / 判据是什么 / 不做什么」，
> 所有表格中的数字位一律留空，**执行后才填**，且必须能回溯到 `outputs/m21/`。
> 边界：单卡 RTX 5080(sm_120) / WSL2 / Qwen2.5-0.5B-Instruct / `enforce_eager` / 单机单副本。

---

## 0. 本模块要回答的问题（学习主线原文，`study_path_with_tech_stack_roadmap.md` §15.3）

| # | 工程问题 | 本模块对应实验 |
|---|---|---|
| Q1 | 最大稳定 QPS 是多少？ | E3（复用 M12 冻结 SLO，**服务端指标视角**复核） |
| Q2 | 并发增加到什么位置后 P99 开始失控？ | E3（QPS 阶梯 + 队列/延迟指标同步时序） |
| Q3 | 瓶颈是 GPU 计算、显存、KV Cache、CPU 还是请求队列？ | **E4（瓶颈签名矩阵，本模块核心）** |
| Q4 | 怎样估算一张卡承载多少用户？ | E5（Little's law + 显式假设，给区间不给单点） |
| Q5 | OOM / 请求超时 / worker 退出怎样定位？ | **E6（可控故障注入 + 告警真触发）** |

必须搭建（学习主线要求，逐项对应）：Prometheus ✓ / Grafana ✓ / GPU 指标 ✓ / vLLM 服务指标 ✓ /
Triton 服务指标（可选，M20-B 已验证 `/metrics` 可用）/ 自定义压测结果 ✓。

**学习主线的红线**：「理论只能定义指标，不能替代真实压测和监控面板」→ 本模块**不接受**
「写了 dashboard JSON 但没截图 / 没触发过告警」这类纸面交付。

---

## 1. 与既有模块的关系（避免重复劳动，也避免口径混淆）

| 模块 | 已有结论 | M21 的增量 |
|---|---|---|
| M12 | 客户端口径开环压测：最大稳定 QPS≈40、饱和≈46.6、过载=尾延迟劣化 | M21 换成**服务端指标口径**看同一现象；并把「拐点」从 CSV 变成**可观测面板 + 告警** |
| M7 | decode ~7ms 地板 = CPU 提交开销（eager GPU 仅忙 23%） | M21 要证明：**这个瓶颈在监控面板上长什么样**（高延迟 + 低 GPU util 的组合签名） |
| M17 | APC / chunked prefill | 提供 `prefix_cache_hits/queries` 指标的可解释来源 |
| M20 | 同镜像 Triton vs vllm serve；入口排队税 | Triton 指标为**可选补充**，主线仍用 host `vllm serve` |

**口径红线（写进脚本注释，防止事后混淆）**：
- **服务端 histogram 分位数**（`histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[...]))`）
  与 **客户端 exact percentile**（`vllm bench serve` 的 `np.percentile`）**不是同一个数**。
  差异至少可能同时来自：① histogram 只存 bucket 计数、`histogram_quantile()` 是近似估计；
  ② server 与 client 的观测边界不同；③ scrape/窗口聚合的近似。**本实验无法把偏差归因到单一项**。
  两者分别命名 `server_hist_p99_*` / `client_p99_*`，**禁止混用**；客户端 exact percentile 是
  **client/SLO 口径**的 ground truth，**不是** server metric 的 ground truth。
- host `vllm serve`（lab commit `ee0da84`）与 M20 容器内 vLLM `0.24.0+092c4842` **不是同一版本**，
  M21 主线只用 host，**不得跨版本相减**。

---

## 2. 环境事实（已实测，2026-08-11）

| 项 | 实测结果 | 影响 |
|---|---|---|
| GPU | RTX 5080 16GB，空闲时约 3.9GiB 被桌面占用 | 每次跑前必须 `nvidia-smi`；预算按 ~12GiB 可用算 |
| Docker | 29.7.2，需经 `~/bin/docker` wrapper（session 缺 `docker` 补充组） | 所有 compose/run 命令统一走 wrapper |
| NVML（WSL2） | `utilization / memory / power / temperature / clocks` **均可读** | 自写 exporter 可行 |
| NVML 进程级 | `nvidia-smi` 显示 `No running processes found` | **无法做 per-process GPU 显存归因**，如实记录为限制 |
| DCGM | **Gate 0 实测：可用**（见下方「Gate 0 结论」） | 走标准方案 G0-a：`dcgm-exporter` |
| DCGM PROF 字段 | `gr_engine_active/sm_active/dram_active` 实测 **N/A** | **拿不到 SM 占用率**，GPU 忙碌只能用粗粒度 `GPU_UTIL` |
| Python | venv 内已有 `prometheus_client` / `psutil` / `pynvml` | 自写 exporter 无需新装包 |

---

## 3. 监控栈方案（含 fallback，先测后定）

```
┌ host ───────────────────────────────────────────────┐
│  vllm serve (:8000/metrics)   ← vLLM 官方指标        │
│  m21_gpu_exporter.py (:9835)  ← NVML: util/mem/power │
│  m21_proc_exporter.py (:9836) ← psutil: server 进程   │
│         CPU% / RSS / 线程数（回答 Q3 的 CPU 分支）    │
└──────────────┬──────────────────────────────────────┘
               │ scrape (1s)
┌ docker (--network host) ────────────────────────────┐
│  prom/prometheus (:9090)  + alert rules             │
│  grafana/grafana (:3000)  + provisioned dashboard    │
└─────────────────────────────────────────────────────┘
```

**Gate 0 结论（已执行，2026-08-11，证据 `logs/m21/gate0_dcgm_probe.txt`）**

规划时我的预判是「WSL2 大概率不支持 DCGM，需退回自写 NVML exporter」。**实测推翻了这个预判**，
按项目规则**改结论不改数据**，记录如下：

| 探测项 | 实测结果 |
|---|---|
| `/dev` 设备节点 | 只有 `/dev/nvidiactl` + `/dev/dxg`，**无** `/dev/nvidia0`（WSL2 特征） |
| host `dcgmi` | 未安装 |
| 容器内 DCGM 4.5.3（Triton 镜像）| `nv-hostengine` 启动成功；`dcgmi discovery -l` → **1 GPU found (Active)**；`dcgmi dmon` 能读 GPUTL/FBUSD/POWER |
| **DCGM PROF 字段** | `dcgmi profile -l` 列出字段，但 `dmon -e 1001,1002,1005` 实测全部 **N/A** |
| `dcgm-exporter:latest`（digest `sha256:613ab03c…`） | 启动成功，`:9400/metrics` 暴露 **15 条** `DCGM_*` |

→ **采用 G0-a（标准 dcgm-exporter）**；**但必须写明限制**：本环境拿不到 `DCGM_FI_PROF_*`
（SM active / tensor active / dram active 全 N/A），所以「GPU 到底忙不忙」只能用**粗粒度**
`DCGM_FI_DEV_GPU_UTIL`（NVML 口径的时间占比，**不是** SM 占用率）。
E4 的 B1/B2 判据必须建立在这个粗粒度指标上，**不得声称测到了 SM 利用率**。

**自写 exporter 仍然要写**（`scripts/m21_host_exporter.py`），但职责改为：
① 进程级 CPU%/RSS/线程数（回答 Q3 的 CPU 分支，dcgm-exporter 不提供）；
② NVML 显存/util 读数作为 **DCGM 的正交第二证据源**（交叉校验，见 I8）。

**镜像拉取**：`prom/prometheus`、`grafana/grafana`（合计约 0.5–1GB）。
→ **需所有者确认**是否允许拉取（模型下载需授权；镜像 M20 已有先例，但仍先问）。

**配置入库**：`monitoring/{prometheus.yml, alerts.yml, grafana/dashboards/*.json, docker-compose.yml}`
—— dashboard 用 provisioning 落盘为文本，**保证可复现，不靠 UI 手点**。

---

## 4. 实验设计

### E1 · 栈自证（Gate 1）
- 判据：Prometheus `up{job=~"vllm|gpu|proc"} == 1` 三个 target 全绿；Grafana 面板出图；
  落盘 `outputs/m21/e1_targets.json` + 面板截图 `figures/`（截图仅作佐证，结论仍以 CSV 为准）。

### E2 · 指标真值校验（**必做，防止"面板好看但数字是假的"**）
同一次压测中同时采集服务端指标与客户端 bench 结果，核对：

| 不变量 | 判据 |
|---|---|
| I1 | Δ`vllm:request_success_total` == bench `completed` |
| I2 | Δ`vllm:prompt_tokens_total` == bench `total_input_tokens` |
| I3 | Δ`vllm:generation_tokens_total` == bench `total_output_tokens` |
| I4 | 全程 `vllm:num_requests_running` ≤ `max_num_seqs`(64) |
| I5 | `vllm:kv_cache_usage_perc` ∈ [0, 1] |
| I6 | 所有 counter 在 scrape 序列上单调不减 |
| I7 | scrape 间隔 ≈ 配置的 `scrape_interval`（丢点率记录） |

**量化结论（非判据）**：`server_hist_p99_ttft` vs `client_p99_ttft` 的相对偏差 →
说明「Prometheus 分位数是 bucket 近似」这一常被面试问到的点，**用本机数字讲，不引用二手说法**。

### E3 · 容量与失控点（Q1/Q2）
- 复用 **M12 冻结配置与冻结 SLO**（`outputs/m12/config.json`），QPS 阶梯 `{8, 16, 24, 32, 40, 44, 48}`。
- 每档 30 warmup + 200 measure（沿用 M20-C 口径），**服务端 1s 采样全程落盘**。
- 产出：QPS → {client P99 TTFT, server hist P99 TTFT, peak waiting, peak running, kv_usage, GPU util} 主表。
- **交叉验证**：能否复现 M12 的「≈40 稳定 / ≥44 破线」。
  **若不一致（版本/环境已变），改结论不改数据**，并在 `results.md` 说明差异来源。

### E4 · 瓶颈签名矩阵（**本模块核心增量，Q3**）
用四个可控 knob 分别制造四类瓶颈，看**哪个指标先饱和**，产出「症状 → 指标签名 → 根因」对照表：

| 场景 | 构造方式（单变量） | 预期签名（**执行前假设，可能被推翻**） |
|---|---|---|
| B1 CPU 提交受限 | Shape A 低 QPS、小 batch（承接 M7） | 延迟不低但 **GPU util 低**、proc CPU% 高 |
| B2 prefill 计算压力 | Shape B in1024/out32（prefill 重） | **粗粒度 GPU util 更高**、TTFT 高、waiting 低（只能说 compute pressure 更强，**不能**断言算力饱和——本环境无 SM-active 指标） |
| B3 KV Cache 受限 | 降 `--gpu-memory-utilization` / 拉长输出 | **kv_cache_usage_perc 逼近 1**、`num_preemptions` > 0 |
| B4 队列受限 | QPS 超容量（48） | **waiting 堆积**、TTFT 尾爆、吞吐仍跟随（M12 已见） |

- 每个场景都要有**第二证据源**（AGENTS.md §6：布尔判据必须正交），例如 B3 除 `kv_cache_usage_perc`
  外还要看 `num_preemptions_total` 与 bench 侧完成延迟。
- **诚实预案**：B1 若在本机 0.5B 上做不出干净对照（GPU util 本来就低），**如实写"未能构造干净单变量对照"**，
  不得把 M7 的旧结论搬过来充数。

### E5 · 一张卡承载多少用户（Q4）
- 用 **Interactive Response Time Law**（可由 Little's law 推出）：
  `闭环用户总数 N = X × (R + Z)`；同时必须分开报告 `平均在系统内请求数 L = X × R` 与 `思考中用户 T = X × Z`。
  **N 是用户 population，不是「系统内并发请求数」**（后者是 L），两者量级差一个数量级。
- **必须显式写出假设**：会话长度、思考时间、输入/输出 token 形状、SLO 口径。
- 输出**区间 + 敏感性**（思考时间 5s/15s/30s 三档），**不给单点数字**，
  且必须标注边界：0.5B / eager / 单卡 / 本 shape。

### E6 · 故障定位演练（Q5，**只对自己启动的进程操作**）
| 故障 | 注入方式（非破坏、可回滚） | 要抓的定位证据 |
|---|---|---|
| 启动期 OOM | `--gpu-memory-utilization` 调高 / `--max-model-len` 调大，直到失败 | 原始报错全文 + 失败瞬间 NVML 显存曲线 |
| 请求超时 | 客户端 timeout 收紧 + 过载 QPS | 客户端 timeout 计数 vs 服务端 `waiting` / queue_time |
| worker 退出 | `kill` **本脚本自己拉起的** server PID | Prometheus `up` 由 1→0、告警触发、面板断点 |
| 抢占（preemption） | 由 B3 场景自然触发 | `vllm:num_preemptions_total` 上升时刻 |

**E6 安全约束（所有者 2026-08-11 授权时明确规定，逐条写死进脚本）**：
1. 只能操作 **M21 脚本自身创建、并已把 PID 显式落盘记录**的服务进程。
2. **禁止**模糊 `pkill` / `killall`（M12 曾因 `pkill -f "vllm serve"` 自匹配 kill 掉执行 shell）；
   只允许 `kill <已记录PID>`，且 kill 前需校验 `/proc/<pid>/cmdline` 匹配预期。
3. **禁止** `nvidia-smi --reset`、**禁止**重启系统 / WSL / Docker daemon、**禁止**操作任何其他进程。
4. OOM 注入必须**可恢复**（实测落到的是**启动期显存预算拒绝**，非 runtime CUDA allocator OOM）；失败后必须逐条验证：
   ① 进程已退出　② NVML 显存已回收（回到注入前基线 ±阈值）　③ 正常 server 能重新启动。
5. 一旦出现 GPU / driver 异常（NVML 报错、显存不回收、`nvidia-smi` 挂起）→ **立即停止该类实验并上报**。

- **规则遵守**：绝不 kill 他人进程、绝不 `nvidia-smi --reset`；注入前后各记一次 `nvidia-smi`。
- **告警必须真触发**：`alerts.yml` 里每条规则都要有一次实际 firing 的证据（Prometheus API 查询落盘），
  **没触发过的规则标 `[ ] 未验证`**。

---

## 5. 交付物清单

| 类型 | 路径 |
|---|---|
| 配置 | `monitoring/{docker-compose.yml, prometheus.yml, alerts.yml, grafana/**}` |
| 脚本 | `scripts/m21_{gpu_exporter,proc_exporter,capacity_sweep,fault_drills,figures}.py` |
| 数据 | `outputs/m21/{e2_validation.json, e3_capacity.csv, e4_bottleneck.csv, e5_user_capacity.json, e6_drills.json, raw/*.jsonl}` |
| 日志 | `logs/m21/*` |
| 图 | `figures/fig11_m21_capacity.png`、`figures/fig12_m21_bottleneck_signature.png`（**离线由 CSV 生成**） |
| 笔记 | `notes/m21/{README.md, monitoring_and_capacity.md(本文), results.md}` |
| 不变量 | `scripts/check_invariants.py --module m21` |
| 同步 | `ROADMAP.md` / `HANDOFF.md` / `notes/evidence_index.md` / `README.md` |

---

## 6. 明确**不做**的事（防止范围膨胀）

- ⬜ 不做 Kubernetes / 多副本 / 服务发现 / 长时 soak（>1h）——单机单副本已达学习主线要求。
- ⬜ 不做跨机 SLA（客户端与服务端同机，指标非真实网络 SLA）。
- ⬜ 不重跑 M12 的完整 3-seed knee 精扫（复用其冻结 SLO 与结论，只做服务端视角复核）。
- ⬜ Triton 指标面板为**可选**；若做，只作为 M20 的补充，不与 host vLLM 混算。
- ⬜ 不做 compiled 模式对照（全模块 `enforce_eager`，与 M12 对齐）。

---

## 7. 执行顺序（每步产出可核验的落盘证据）

1. Gate 0：DCGM 探测 → 定 exporter 方案（G0-a / G0-b）
2. 写 exporter + monitoring 配置 → Gate 1（E1 三 target 全绿）
3. E2 真值校验（**不过不许往下走**——指标不可信则后续全是沙上塔）
4. E3 容量阶梯 → E4 瓶颈矩阵（共用采样框架）
5. E5 推导（纯离线计算，不占 GPU）
6. E6 故障演练 + 告警触发验证
7. 出图 → `results.md` → 同步 ROADMAP / evidence_index / README / HANDOFF
