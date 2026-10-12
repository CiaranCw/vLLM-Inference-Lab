# M28 · KV Offloading 与 Serving 可靠性 —— 实验计划（DESIGN FROZEN v1）

> **状态：DESIGN FROZEN v1（2026-10-12）· CPU_VALIDATED · GPU NOT EXECUTED**。本文没有任何实测数字。
> 冻结配置的唯一来源是 `scripts/m28_common.py` 的 `FROZEN`；§4 的 JSON 块由代码生成，
> `python scripts/m28_validate.py` 的 `x_plan_frozen_matches_code` 规则逐键核对。
> 改设计 = 升 `plan_version`，并在 [`execution_record.md`](execution_record.md) 记录原因。不在机时中临场改问题。
> 源码依据见 [`source_review.md`](source_review.md)。

## 0. 范围与模块关系

- **A**：vLLM 原生 `OffloadingConnector`（GPU→CPU，`CPUOffloadingSpec`）。证明三条路径真实发生——GPU 热命中、
  GPU miss + CPU hit、完整重算——并量化它们的 online TTFT 随前缀长度的变化。文件 / SSD 层不在核心范围。
- **B**：Serving 可靠性。B1 = 在途请求下 **SIGKILL EngineCore 子进程**；B3 = 在途请求下 SIGTERM API，
  `shutdown_timeout` 取 0 / 不足 / 充足三档；B2（可选、高风险、需单独授权）= 短暂 SIGSTOP/SIGCONT EngineCore。
- **不重复**：M17 测的是 in-process APC（GPU 前缀缓存）；M21 E6 杀的是 **API 父进程且无在途请求**；M27 是 PD 分离。
  M28 不改它们的 raw 与结论。**不改 vLLM / SGLang 上游源码，不重做 Mini Lab，不实现跨 worker 续跑或 fencing。**

## 1. 目标栈

| 层 | 冻结值 | 核对方式 |
|---|---|---|
| GPU | 单卡 RTX 5080 16 GB（sm_120），WSL2 | `nvidia-smi`（整卡口径；WSL2 无进程级显存） |
| vLLM | `../vllm-src` editable，tag v0.24.0 / HEAD `ee0da84`，脏文件 22 个（M28 相关路径默认惰性，见 source_review §1） | lifecycle 记录 `env_fingerprint` |
| 模型 | Qwen2.5-0.5B-Instruct 本地快照 commit `7ae5576…`，以**目录路径**启动 + `--served-model-name`（不走 HF hub，杜绝自动下载） | `model_identity()`：blob sha 必须等于冻结值，否则门禁拒绝 |
| 运行形态 | `vllm serve` 子进程（API 进程 + `VLLM::EngineCore` 子进程），HTTP 流式客户端，**请求严格串行**（A） | lifecycle 记录 `tree_ready` |

## 2. 与仓库默认规则的偏离（逐条说明）

1. **`VLLM_ENABLE_V1_MULTIPROCESSING=1`**（仓库默认要求 0）：0 是给 in-process `LLM` 用的。serve 路径的 `AsyncLLM`
   总是 MP client（source_review §3），B 组的研究对象本身就是 EngineCore 子进程，因此显式设为 1，避免歧义。
   `VLLM_USE_FLASHINFER_SAMPLER=0` 照旧。
2. **指标命名**：A 组是 HTTP 流式客户端测到的首 token 时间，字段名 `online_ttft_ms`，`metric_scope=online_http_stream_serial`
   （串行、batch=1），与 M12 的开环负载 TTFT 不可直接比较。
3. **B 组重复次数**：B1 每档 3 次、B3 每档 3 次、B2 1 次。这是**行为刻画**（错误形态 / 时间线 / 清理），不做性能统计，
   不套用"≥15 measure"的性能实验规则；各次结果逐条报告，不合并成均值。
4. **以快照目录启动模型**：原因见 source_review §2 最后一行（`HF_HUB_OFFLINE=1` 下 `IncompleteSnapshotError`）。

## 3. 研究问题

- **RQ-A1**：在本机上，原生 OffloadingConnector 能否产生**被指标证实**的 GPU miss + CPU hit？（门禁：external hit 已确认）
- **RQ-A2**：CPU 命中的输出是否与 GPU 热命中逐位一致？与独立重算（关 prefix caching 的进程）是否一致？不一致时能否用证据解释？
- **RQ-A3**：P ∈ {1024, 4096, 8192, 16384} 上，CPU 命中相对完整重算、相对 GPU 热命中的 online TTFT 差多少？
  connector 本身是否给冷 / 热路径带来可观测开销？
- **RQ-A4（可选）**：CPU 层容量不足时，LRU 淘汰是否让"CPU 命中"退化为重算，并且在指标上可见？
- **RQ-B1**：EngineCore 子进程被 SIGKILL 时，在途 streaming / 非 streaming 请求、新请求、`/health`、API 进程、端口、显存和重启各自怎样表现？
- **RQ-B3**：SIGTERM 下 `shutdown_timeout` 为 0 / 不足 / 充足时，在途请求完成率、新请求的拒绝形态、退出耗时和资源清理有何不同？
- **RQ-B2（可选）**：EngineCore hang（SIGSTOP）期间 `/health` 能否发现异常？SIGCONT 后能否恢复？

## 4. 冻结配置

<!-- M28_FROZEN_CONFIG:BEGIN -->
```json
{
  "A": {
    "arm_order": [
      "offload",
      "base",
      "ref",
      "evict"
    ],
    "arms": {
      "base": {
        "connector": false,
        "paths": [
          "cold",
          "gpu_hot",
          "recompute_evicted"
        ],
        "prefix_caching": true
      },
      "evict": {
        "connector": true,
        "optional": true,
        "paths": [
          "cold",
          "gpu_hot",
          "cpu_evicted"
        ],
        "prefix_caching": true
      },
      "offload": {
        "connector": true,
        "paths": [
          "cold",
          "gpu_hot",
          "cpu_hit"
        ],
        "prefix_caching": true
      },
      "ref": {
        "connector": false,
        "paths": [
          "ref_a",
          "ref_b"
        ],
        "prefix_caching": false
      }
    },
    "cpu_bytes_to_use": 4294967296,
    "evict_cpu_bytes_to_use": 268435456,
    "evict_prefix_lens": [
      4096
    ],
    "filler_len": 12288,
    "max_model_len": 16640,
    "max_tokens": 16,
    "measure_trials": 15,
    "metrics_settle_s": 0.3,
    "n_fillers": 2,
    "num_gpu_blocks_override": 1280,
    "prefix_lens": [
      1024,
      4096,
      8192,
      16384
    ],
    "request_timeout_s": 120,
    "server_warmup_requests": 2,
    "store_quiesce_s": 0.5,
    "store_quiesce_timeout_s": 20,
    "suffix_len": 20,
    "token_id_range": [
      1000,
      150000
    ],
    "warmup_trials": 5
  },
  "B": {
    "B1": {
      "fault_after_first_token_s": 3.0,
      "max_tokens": 2048,
      "observe_s": 60,
      "probe_new_request_at_s": [
        1.0,
        8.0
      ],
      "repeats": 3
    },
    "B2": {
      "fault_after_first_token_s": 3.0,
      "max_tokens": 2048,
      "observe_s": 40,
      "repeats": 1,
      "requires_separate_authorization": true,
      "stop_s": 10
    },
    "B3": {
      "fault_after_first_token_s": 1.0,
      "max_tokens": 1024,
      "observe_s": 90,
      "probe_interval_s": 0.25,
      "repeats": 3,
      "shutdown_timeouts": [
        0,
        2,
        60
      ]
    },
    "gpu_poll_s": 1.0,
    "health_poll_s": 0.1,
    "max_model_len": 4096,
    "n_nonstream": 4,
    "n_stream": 4,
    "num_gpu_blocks_override": 2048,
    "prompt_len": 256,
    "server_ready_timeout_s": 300
  },
  "block_size": 16,
  "gpu_gate": {
    "max_baseline_used_mib": 6000,
    "reclaim_timeout_s": 240,
    "reclaim_tol_mib": 800
  },
  "host": "127.0.0.1",
  "kv_bytes_per_token_theory": 12288,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "model_safetensors_blob_sha256": "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
  "model_snapshot_commit": "7ae557604adf67be50417f59c2c2f167def9a775",
  "plan_version": "v1",
  "port": 8028,
  "server_common_args": [
    "--seed",
    "0",
    "--enforce-eager",
    "--dtype",
    "auto",
    "--gpu-memory-utilization",
    "0.45",
    "--max-num-seqs",
    "16",
    "--max-num-batched-tokens",
    "8192",
    "--no-async-scheduling",
    "--enable-chunked-prefill",
    "--disable-uvicorn-access-log",
    "--enable-prompt-tokens-details"
  ],
  "server_env_must_unset": [
    "VLLM_LAB_TRACE",
    "VLLM_LAB_BLOCK",
    "VLLM_LAB_M9_COALESCE",
    "VLLM_KEEP_ALIVE_ON_ENGINE_DEATH"
  ],
  "server_env_set": {
    "HF_HUB_OFFLINE": "1",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "1",
    "VLLM_USE_FLASHINFER_SAMPLER": "0"
  },
  "stats": {
    "bootstrap_iters": 10000,
    "bootstrap_seed": 0,
    "ci": 0.95
  },
  "vllm_source_ref": "ee0da84",
  "vllm_src_dirty_files_expected": 22
}
```
<!-- M28_FROZEN_CONFIG:END -->

## 5. A 组设计

### 5.1 Arm（每个 arm 一个独立 `vllm serve` 进程，顺序执行，不并行）

| arm | connector | prefix caching | 每个 trial 的被测路径 | 作用 |
|---|---|---|---|---|
| `offload` | OffloadingConnector，CPU 4 GiB | 开 | cold → gpu_hot → (2 个 filler) → **cpu_hit** | 主 arm |
| `base` | 无 | 开 | cold → gpu_hot → (2 个 filler) → recompute_evicted | 无 connector 的同协议对照：被挤出后只能重算 |
| `ref` | 无 | **关** | ref_a、ref_b（同一 prompt 两次） | 独立重算参照 + 进程内确定性基线 |
| `evict`（可选） | OffloadingConnector，CPU 256 MiB | 开 | cold → gpu_hot → (2 个 filler) → cpu_evicted | RQ-A4：CPU 层被 filler 挤满后的退化 |

- GPU KV 池固定为 `--num-gpu-blocks-override 1280`（20480 tokens，240 MiB）；两个 filler 共 24576 个互不相同的 token，
  一定能把被测前缀挤出 GPU。filler 也会被 offload 进 CPU 层；4 GiB 足以同时容纳一个 trial 的前缀与 filler。
- prompt = 随机 token id（区间 `[1000, 150000)`，由 `(P, idx)` 经 sha256 播种，**所有 arm 同一序列**），后缀 20 个 token，
  `N = P + 20`，`H = ⌊(N−1)/16⌋·16 = P + 16`。同一 trial 内三次请求发送**完全相同的 prompt**，因此可以直接比较输出。
- 采样：`temperature=0`、`ignore_eos=true`、`max_tokens=16`、`logprobs=1`、`return_tokens_as_token_ids=true`、流式 + `include_usage`。
- 每个被测请求之前先等 GPU→CPU store 计数器连续两次不变（`store_quiesce`；无 connector 的 arm 等相同时长），
  避免在途的异步 store 干扰下一次测量。
- 预热：server 启动后，每个 P 先发 2 个非计量请求，再发 1 个 filler（触发 Triton JIT）；
  然后每个 P 5 个 warmup trial + 15 个 measure trial，trial 内的 P 顺序由 `idx` 播种打乱（避免漂移和顺序混在一起）。
- **冒烟（smoke）**：正式 arm 之前，offload arm 先跑 1 个 P=1024 的 trial（phase=smoke，单独文件，不计入统计），
  用来确认 connector 能启动、cpu_hit 能被指标证实、日志里 `cudaHostRegister` 是否失败。

### 5.2 路径证明（逐样本，runner 判一次，validator 独立重算一次）

| 路径 | Δlocal_hits | Δext_hits | Δload_bytes | cached_tokens |
|---|---|---|---|---|
| cold / recompute_evicted / ref_* | 0 | 0 或不存在 | 0 或不存在 | 0 |
| gpu_hot | H | 0 | 0 | H |
| cpu_hit | 0 | H | > 0 | H |
| cpu_evicted | 0 | 观测量 | 观测量 | = Δext_hits |

所有路径还必须满足 `Δrequest_success == 1 ∧ Δprompt_tokens == N`（证明这次抓取覆盖到本请求）。
不满足的样本 `success=False`，**照样落盘、计入 n_fail，不静默剔除**。

### 5.3 正确性门禁

- **G-A-corr-1（硬）**：offload arm 中每个 measure trial 的 `cpu_hit` 与 `gpu_hot` 输出 token **和 logprob** 逐位相等。
- **G-A-corr-2（硬）**：`ref_a == ref_b`（确定性基线）；`offload.cold == ref_a`、`base.cold == ref_a`（都是完整 prefill）。
- **G-A-corr-3（解释性）**：`cpu_hit` 与 `ref_a` 可能因计算形状不同（缓存前缀 + 部分 prefill vs 完整 prefill）出现数值分歧。
  判据：cpu_hit 相对 ref 的分歧样本集合必须**等于** gpu_hot 相对 ref 的分歧集合——即 offload 没有在 GPU 前缀缓存之外引入新分歧。

### 5.4 统计

- 每个 (arm, P, path)：n、median、stdev、p95（只用 measure ∧ success）。
- **配对**（同一 arm 同一 trial）：`cpu_hit − cold`、`cpu_hit − gpu_hot`、`base.recompute_evicted − base.cold`；
  报 median of paired differences 的 bootstrap 95% CI（10000 次，seed 0）。
- **非配对**（跨进程）：`offload.cpu_hit vs base.recompute_evicted`、`offload.cold vs base.cold`、`offload.gpu_hot vs base.gpu_hot`；
  报 median difference 的 bootstrap 95% CI。CI 含 0 时只能写"未观察到超出噪声的差异"。
- load 带宽 = `Δload_bytes / Δload_time`（connector 自报），仅作参考，不作为结论依据。

### 5.5 A 组退出条件

- 冒烟中 connector 起不来、cpu_hit 无法被指标证实、或 KV 往返破坏输出（G-A-corr-1 失败）→ **停止 A 组**，
  原始日志 + 命令 + 报错落盘，状态记为 `BLOCKED`；**不 patch vLLM**，不换后端绕开。
- `cudaHostRegister` 失败但功能正常 → 继续执行，结论中把"unpinned DMA"作为限定条件写明。

## 6. B 组设计

### 6.1 公共

- server：无 connector、关 prefix caching、`--max-model-len 4096`、`--num-gpu-blocks-override 2048`（384 MiB KV）。
- 在途负载：4 条流式 + 4 条非流式 `/v1/completions`，prompt 256 tokens，`ignore_eos`。
  **在途证据（两个独立来源）**：4 条流式请求都收到首 token，且 `vllm:num_requests_running == 8`；不成立则不注入，记 BLOCKED。
- 每次演练前后：`nvidia-smi`、进程树（pid / ppid / sid / pgid / cmdline / create_time）、端口、`/dev/shm` 列表。
  演练后要求：无存活的已知 PID、端口释放、`/dev/shm` 无新增、整卡显存回到演练前基线 + 800 MiB 以内（≤240 s）。
  不满足时**中止后续演练**并报告。
- 观测：故障后每 0.1 s 采样一次 `/health` 状态码、API 是否存活、EngineCore 是否存活、端口是否在监听；每 1 s 记一次 `nvidia-smi`。
  服务端日志按 `[shutdown] …` 等关键行计数（第二证据源）。

### 6.2 B1：SIGKILL EngineCore 子进程（每档 3 次）

在途 `max_tokens=2048`；首 token 后 3 s 发送 SIGKILL 给 EngineCore；在故障后 1 s 和 8 s 各发一个新请求（探针）；观察 60 s。
如果 API 进程没有自行退出，由 runner 兜底清理，并把 `cleanup_needed=true` 作为**发现**记录下来。
之后用同一配置重启，`/health` 返回 200 且能完成 1 个请求才算 `restart_ok`。

### 6.3 B3：SIGTERM API × `shutdown_timeout ∈ {0, 2, 60}`（每档 3 次）

在途 `max_tokens=1024`。同一进程先跑一批同形状的校准请求，测出完成时长 `D_cal`；然后在途批次首 token 后 1 s 发 SIGTERM 给 API 进程；
每 0.25 s 发一个新请求探针，直到连续 3 次连接被拒；观察 90 s。
**前提核对**（validator `b_b3_timeout_precondition`）：不足档的剩余工作 `D_cal − (t_fault − t_send) > 2 + 1 s`；充足档 `D_cal < 60 − 5 s`。

### 6.4 B2（可选，需单独授权）：SIGSTOP EngineCore 10 s → SIGCONT

在途 `max_tokens=2048`，观察 40 s；在 1 s 和 8 s 各发一个探针。SIGCONT 有三重兜底：正常路径、`finally`、`atexit` 加 SIGINT/SIGTERM handler；
`--cleanup-orphans` 也会先对候选进程发 SIGCONT 再发 SIGKILL。

### 6.5 进程归属与安全（代码见 `m28_common.Server`）

- server 一律 `start_new_session=True` 启动：API 进程 pid == sid == pgid。
  "本轮脚本启动"的判据是：sid == 我们 Popen 的 pid，且 create_time 不早于启动时刻。
- 发信号前再次核对：B1/B2 的目标必须是**唯一**的 `VLLM::EngineCore`，其 ppid == API pid；B3 的目标必须是 Popen 对象本身且仍存活。任何一项不符就不发信号。
- 禁止 pkill / killall / 按名字匹配；禁止 `nvidia-smi --reset`；锁文件保证 A/B 不同时运行 GPU 引擎。
- 人工停止路径：`kill -INT <runner pid>`（pid 在锁文件 `outputs/m28/raw/.m28_experiment.lock` 中）→ runner 自己收尾；
  runner 已死时，用 `python scripts/m28_faults.py --cleanup-orphans` 先 dry-run 查看候选，再加 `--really`。它只按 `m28_active_server.json` 的记录行事。

## 7. 不变量（validator 规则 → 含义）

| 规则 | 含义 |
|---|---|
| `x_plan_frozen_matches_code` | §4 JSON == `FROZEN` |
| `a_counts` / `a_measure_all_success` | 每 (arm,P,path) 5 warmup + 15 measure，idx 连续；measure 失败必须列出 |
| `a_path_recheck` | validator 独立重算路径判据，与 runner 的 success 一致 |
| `a_cached_eq_metrics` | `cached_tokens == Δlocal_hits + Δext_hits`（请求级 vs Prometheus 两个来源） |
| `a_gate_external_hit` | 每个 P 的 cpu_hit 15/15 被证实（A 组门禁） |
| `a_load_bytes_consistent` | `Δload_bytes/Δext_hits` 恒定；非 CPU 命中路径 load 为 0 |
| `a_prompt_pairing` | 所有样本 prompt sha == 冻结生成器重算值 |
| `a_corr_*` | §5.3 的正确性门禁 |
| `a_csv_matches_raw` | 派生 CSV 的 n / median == raw 重算 |
| `a_lifecycle_clean` / `a_env_fingerprint` / `a_connector_log_evidence` | 进程清理、显存回收、环境变量、vllm-src 状态、模型身份、connector 日志证据 |
| `b_target_identity` / `b_inflight_before_fault` / `b_all_requests_terminal` | 注入目标身份、在途证据、每条请求都有终态 |
| `b_cleanup_clean` / `b_b1_restart_ok` / `b_b3_timeout_precondition` / `b_b2_resumed` | 清理、重启、B3 前提、B2 恢复 |
| `x_no_engine_overlap` | 所有 server 生命周期互不重叠 |

validator 的检出能力由 `--selftest` 证明：把基准数据集复制到临时目录，逐条注入反例，每条都必须让目标规则从 PASS 变成 FAIL。
反例只存在于临时目录。

## 8. 资源预算（GPU 授权申请的依据）

| 项 | 预估 | 依据 |
|---|---|---|
| 单个 server 的 GPU 占用 | 实际约 2–3 GiB（权重约 0.99 GB + KV 240/384 MiB + CUDA 上下文 / 激活）；**上限**是 vLLM 预算 `0.45 × 16 GiB ≈ 7.2 GiB` | 冻结参数；启动时 vLLM 要求预算 ≤ 当前空闲显存 |
| 整卡峰值 | 现有基线 4708 MiB（Gate 0 读数；授权申请前复核为 4991 MiB；`nvidia-smi` 不列出计算进程，来源需所有者确认）+ 单个 server ≤ 7.2 GiB ≈ **≤ 12.2 GiB / 16 GiB** | 同一时刻只运行一个 server；门禁要求基线 ≤ 6000 MiB |
| 主机内存 | offload arm：`/dev/shm` 4 GiB + server RSS 约 3–4 GiB ≈ 8 GiB（当前可用 57 GiB；`/dev/shm` 有 32 GiB 空闲） | `free -g`、`df -h /dev/shm` |
| 时长 | A 组 4 个 arm 约 30–40 min；B1 约 12 min；B3 约 25 min；B2 约 2 min；合计约 1–1.5 h | 请求数见 `m28_offload.py --dry-run`（约 987 个 A 请求） |

## 9. 风险

- WSL2 pinned memory：`cudaHostRegister` 可能失败，或 load 带宽低于原生 Linux（P-A4）。只报实测，不外推到原生 Linux 或服务器 GPU。
- 0.5B 模型、eager 模式下 prefill 本身很快，CPU 命中的收益可能只在长前缀上才能浮出噪声。小前缀上"未观察到收益"是有效结论。
- B3 `timeout=0` 的结果可能存在竞争（P-B3-1），3 次重复内可能出现不同形态，按次逐条报告。
- WSL2 下 SIGKILL 后显存归还有延迟（M21 E6 实测），因此显存回收采用轮询（≤240 s）。

## 10. 状态标签

`DESIGN_ONLY`（只有设计）/ `CPU_VALIDATED`（脚本、mock 演练、validator 自检通过，没碰 GPU）/ `GPU_VALIDATED`（已在 GPU 上跑通但未全部完成）/
`COMPLETE`（冻结矩阵全部完成且不变量全部 PASS）/ `BLOCKED`（有证据的阻塞）/ `SKIPPED`（按授权或范围不做）。

| 子项 | 当前状态 |
|---|---|
| A 冒烟 / A 正式（offload/base/ref） | CPU_VALIDATED（GPU 未执行） |
| A evict（可选） | CPU_VALIDATED（GPU 未执行） |
| B1 / B3 | CPU_VALIDATED（GPU 未执行；需故障注入授权） |
| B2（可选） | CPU_VALIDATED（需单独授权） |

## 11. 执行顺序与命令

```bash
source .venv/bin/activate
python scripts/m28_offload.py --smoke --i-have-gpu-authorization          # A 冒烟（门禁）
python scripts/m28_offload.py --i-have-gpu-authorization                  # A：offload → base → ref
python scripts/m28_offload.py --arm evict --i-have-gpu-authorization      # A 可选
python scripts/m28_faults.py --drill b1 --drill b3 --i-have-gpu-authorization --i-have-fault-authorization
python scripts/m28_faults.py --drill b2 --append --authorize-b2 --i-have-gpu-authorization --i-have-fault-authorization  # 可选
python scripts/m28_summarize.py && python scripts/m28_validate.py && python scripts/m28_figures.py
```
