# M27 · H20 正式实验计划（DESIGN FROZEN v1）

> **状态：DESIGN FROZEN v1（2026-10-03）· E/H NOT EXECUTED**。本文没有任何实测数字。
> 冻结配置的唯一来源是 `scripts/m27_common.py` 的 `FROZEN`；下方 JSON 块必须与它逐键相等，
> 由 `check_invariants.py --module m27` 的 `m27_config_freeze` 自动核对。
> 改设计 = 升 `plan_version`，并在 [`execution_record.md`](execution_record.md) 记录原因。**不得在机时中临场改问题。**
> 源码依据见 [`pd_disagg_nixl_theory.md`](pd_disagg_nixl_theory.md)；执行步骤见 [`M27_H20_RUNBOOK.md`](M27_H20_RUNBOOK.md)。

## 0. 范围与模块关系

- 只研究 **phase disaggregation**：每个 P/D 实例都是 TP=1，不开量化，不开 spec decode，不引入 SGLang。
- 承接 M24 中没有做的 P-D 部分（M24 计划 P3 “P-D 1P1D smoke” 未执行）。**M24 的冻结计划、矩阵和结论都不改**；
  对 M24 笔记的事实更正只记录在 M27 的 theory 笔记 §2。
- 要回答的问题是：**PD 付出了多少 KV transfer 和编排成本，换来了多少 prefill–decode 干扰隔离，
  以及这种交换在什么 workload / 资源配比下值得。** “KV 传输明显增加，但尾部 ITL 隔离收益有限”是有效结论。

## 1. 目标栈（上机后必须实测回填，不能只写期望值）

| 层 | 冻结值 | 上机核对方式 |
|---|---|---|
| GPU | NVIDIA H20（M22 实测 97871 MiB/卡） | preflight：`nvidia-smi` / `-L` / `topo -m` |
| 拓扑 | **不假设** GPU0–7 在同一 NVLink 域；M22 时容器只看到 GPU0–3（两两 NV18） | Gate 0 由 `topo -m` 求最大两两 NV# 集合 → `configs/gpu_map_<tag>.json` |
| Driver / CUDA | M22：driver 535.247.01 + cuda-compat 13.0（`LD_LIBRARY_PATH`） | 只记录，**不改** driver/CUDA |
| Python / torch | 3.11.x / 2.11.0+cu130 | preflight 版本比对 |
| vLLM | 0.24.0 wheel（源码参照 `ee0da84`） | preflight + `/server_info` |
| NIXL | 1.2.0（`nixl` meta 包 + `nixl-cu13` 后端；内置 UCX） | preflight 导入、创建 UCX agent、检查 vLLM 的 `NixlWrapper` |
| 模型 | Qwen2.5-7B-Instruct，**M22 的离线副本**，不重新下载；文件内容与 HF `Qwen/Qwen2.5-7B-Instruct@a09a3545` 一致（+ ModelScope 的 `configuration.json`） | preflight 用仓库内 `notes/m27/model_artifact/qwen2.5-7b-instruct.sha256` 做 `sha256sum -c`；config 架构字段比对（见 [`model_artifact_manifest.md`](model_artifact_manifest.md)） |

## 2. 冻结配置

<!-- M27_FROZEN_CONFIG:BEGIN -->
```json
{
  "plan_version": "v1",
  "software": {
    "python_minor": "3.11",
    "torch": "2.11.0+cu130",
    "cuda": "13.0",
    "vllm_wheel": "0.24.0",
    "vllm_source_ref": "ee0da84",
    "nixl": "1.2.0",
    "nixl_backend_wheel": "nixl-cu13==1.2.0"
  },
  "model": {
    "name": "Qwen2.5-7B-Instruct",
    "artifact": "M22 offline copy (no re-download)",
    "hf_repo": "Qwen/Qwen2.5-7B-Instruct",
    "hf_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
    "identity_manifest": "notes/m27/model_artifact/qwen2.5-7b-instruct.sha256",
    "identity_manifest_sha256": "157f08c69340e16b5d0358a31a4d20716f73a3b90ce6538ed557cff5fbe49638",
    "file_count": 15,
    "total_size_bytes": 15242807272,
    "torch_dtype": "bfloat16",
    "max_position_embeddings": 32768,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "hidden_size": 3584,
    "head_dim": 128,
    "kv_dtype_bytes": 2
  },
  "engine": {
    "tensor_parallel_size": 1,
    "dtype": "bfloat16",
    "seed": 0,
    "gpu_memory_utilization": 0.85,
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "enforce_eager": true,
    "enable_prefix_caching": false,
    "async_scheduling": false,
    "enable_chunked_prefill": true,
    "block_size": 16,
    "max_model_len": 32768,
    "attention_backend": "FLASH_ATTN",
    "stream_interval": 1,
    "kv_cache_dtype": "auto",
    "quantization": null,
    "speculative_config": null
  },
  "max_prompt_len": 16384,
  "legal_fallback": {
    "max_model_len": 16384,
    "max_prompt_len": 8192,
    "rule": "only if config.json max_position_embeddings < 32768; drop context points >= 16384"
  },
  "kv_cache_layout": "HND",
  "kv_transfer": {
    "kv_connector": "NixlConnector",
    "producer_role": "kv_producer",
    "consumer_role": "kv_consumer",
    "kv_load_failure_policy": "fail",
    "kv_buffer_device": "cuda",
    "backends": [
      "UCX"
    ]
  },
  "kv_bytes_per_token": 57344,
  "warmup": 5,
  "measure": 15,
  "e1": {
    "contexts": [
      512,
      2048,
      4096,
      8192,
      16384
    ],
    "output_tokens": 8
  },
  "e2": {
    "d_prompt": 128,
    "d_output": 512,
    "inject_after_token": 64,
    "p_lens": [
      0,
      2048,
      8192,
      16384
    ],
    "p_output": 1,
    "arms": [
      "colo1_mbt2048",
      "colo1_mbt4096",
      "colo1_mbt8192",
      "pd_1p1d"
    ]
  },
  "e3": {
    "shapes": {
      "d_heavy": [
        256,
        256
      ],
      "balanced": [
        2048,
        256
      ],
      "p_heavy": [
        8192,
        64
      ]
    },
    "mixture": {
      "d_heavy": 0.5,
      "balanced": 0.3,
      "p_heavy": 0.2
    },
    "seeds": [
      0,
      1,
      2
    ],
    "burstiness": 1.0,
    "duration_s": 60,
    "load_fractions": [
      0.5,
      0.75,
      0.9
    ],
    "arms": [
      "colo1_mbt8192",
      "colo2_mbt2048",
      "colo2_mbt4096",
      "colo2_mbt8192",
      "pd_1p1d"
    ],
    "calibration_arm": "colo2_mbt8192",
    "slo_multipliers": [
      2,
      5,
      10
    ]
  },
  "e4": {
    "shapes": [
      "d_heavy",
      "p_heavy",
      "mixture"
    ],
    "arms": [
      "colo4_mbt8192",
      "pd_1p3d",
      "pd_2p2d",
      "pd_3p1d"
    ],
    "calibration_arm": "colo4_mbt8192"
  },
  "e5_optional": {
    "arms": [
      "colo8_mbt8192",
      "pd_2p6d",
      "pd_4p4d",
      "pd_6p2d"
    ],
    "gate": "all 8 GPUs pairwise NV# in nvidia-smi topo -m"
  },
  "bootstrap": {
    "iters": 10000,
    "seed": 0
  }
}
```
<!-- M27_FROZEN_CONFIG:END -->

### 2.1 引擎参数与回读

每个实例都是一个独立的 `vllm serve` 进程（一个进程一个引擎），命令由 `m27_common.serve_cmd()` 生成，
参数已用 pinned 树的 arg parser 离线解析过（theory §7）。环境变量对所有 arm 相同：
`VLLM_KV_CACHE_LAYOUT=HND`、`VLLM_SERVER_DEV_MODE=1`、`HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
`VLLM_USE_FLASHINFER_SAMPLER=0`、`PYTHONHASHSEED=0`，加上每实例唯一的 `CUDA_VISIBLE_DEVICES` 和
`VLLM_NIXL_SIDE_CHANNEL_PORT=5600+i`。

启动后 launcher 做**双来源回读**，任一项不符就停掉全部进程并 STOP，不进入测量：

1. `GET /server_info?config_format=json`：max_model_len、dtype、seed、enforce_eager、quantization、TP、gmu、
   block_size、prefix caching、cache dtype、max_num_seqs、mbt、chunked、async、stream_interval、attention backend、
   spec config、kv_connector / kv_role / failure policy、`vllm_env.VLLM_KV_CACHE_LAYOUT`；
2. 服务日志：引擎配置行（max_seq_len / eager / APC / chunked / TP）、KV layout 行 == HND、
   attention backend 行 == FLASH_ATTN；P/D 还要求 “NIXL is available”。

### 2.2 唯一合法回退

只有当模型 `config.json` 的 `max_position_embeddings < 32768` 时，才允许把 `max_model_len` 降到 16384，
同时最长 prompt 降到 8192，并删除 ≥16384 的 context 点（E1 的 16384、E2 的 p=16384）。
这个判定由 Gate 0 自动写入 `gpu_map_<tag>.json` 的 `max_model_len_fallback`，并在回读中核对。
**不允许其他任何回退**（例如改 gmu、改 mbt、关分块、开 APC）。遇到 OOM 等问题就 STOP 并询问。

## 3. 对照结构

| 层 | 比较 | 性质 |
|---|---|---|
| **A（因果）** | `colo1_mbt8192` vs `pd_1p1d`：mbt=8192、chunked ON，其余 EngineArgs 全同 | EngineArgs 层面最接近单变量：只差“是否解耦”（以及随之而来的 kv_role 与 proxy 路径，见 §6）。**但 GPU 数不同（1 卡 vs 2 卡）**，所以只回答机制问题，不回答同成本效率 |
| **B（chunk 调优）** | `colo{1,2}_mbt{2048,4096,8192}` vs `pd_1p1d` | **不是单变量**：colocated 允许用合法的 chunk 预算调优，检验文档中“合适的 chunk size 也能做到同样效果”。在 E2 中用 colo1（机制层面），在 E3 中用 colo2（部署层面） |
| **GPU 预算匹配** | 在线实验中 PD(2 卡) vs `colo2_*`(2 个副本) | E3 主比较；colo1 只作为 A 的参考（只有 1 张卡，在同样到达率下必然更早饱和） |

**证据边界（E2 与 E3 分工）**：

- **E2 = mechanism / interference-isolation 实验**。colo1 用 1 张卡，1P1D 用 2 张卡，GPU 预算不同。
  E2 回答：“phase separation 是否把长 prefill 从 decode 的 timeline 上移走了？”（以及 chunk 调优能否做到类似效果）。
  E2 **不能单独回答**“同样 GPU 成本下 PD 是否更高效”。
- **E3 = equal-GPU-budget deployment 对照**（`pd_1p1d` vs `colo2_*`，都是 2 卡），承担部署层面的证据；E4 同理（4 卡）。

**关分块从不作为正式 arm**：对 generate 模型手动关分块会触发官方 warning，而且关分块要求 mbt ≥ max_model_len，
无法构造“关分块 + 小预算”（M17 §E3 已实证）。

## 4. 研究问题 → 实验

| RQ | 问题 | 实验 | 主指标 | 允许的结论 |
|---|---|---|---|---|
| RQ1 | PD 是否真的隔离了长 prefill 对正在 decode 的请求的尾部 ITL 干扰？ | E2（A + B） | D 的 max stall、注入窗口 ITL p99、整体 ITL p99 | 隔离 / 部分隔离 / 未观察到超出噪声的差异 / PD 更差 |
| RQ2 | NIXL 的 bytes、post 时间、transfer 时间、等效带宽如何随 context 增长？ | E1 | 实测 bytes vs 理论 bytes；xfer/post ms；GB/s；TTFT 分解 | 任何单调或非单调关系，按实测报告 |
| RQ3 | 调优后的 colocated 与 PD 各自适合什么 workload？ | E3（GPU 预算匹配） | TTFT/ITL 分位数、goodput（多档 SLO 倍数）、achieved rate | **允许“PD 无明显收益 / 有额外开销”** |
| RQ4 | 4 / 8 卡时 P:D 配比如何随 workload 变化？ | E4（+ 可选 E5） | 同 E3 | 不预设最优配比 |

## 5. 实验矩阵

所有请求都经过 `m27_proxy.py`（colocated 用 passthrough 模式，PD 用 pd 模式），客户端使用 token-id prompt、
`temperature=0`、`ignore_eos`、`stream=true`、`return_token_ids=true`。prompt 由 request key 的 sha256 确定性生成，
所以同一个 key 在所有 arm 上逐 token 相同（`m27_prompt_identity`）。

### E0 · Gate 1 正确性（`--stage e0`）

| arm | 样本 | prompt / output |
|---|---|---|
| `pd_1p1d`（允许故障注入） | 3 warmup + 5 measure + 1 fault probe | 2048 / 32 |
| `colo1_mbt8192` | 3 warmup + 5 measure | 同 key |

PASS 判据（每个 measure 样本，等 metrics 稳定后读取）：

- D 的 `prompt_tokens_by_source{external_kv_transfer}` 增量 == prompt_len，D 的 `local_compute` 增量 == 0，
  P 的 `local_compute` 增量 == prompt_len；`nixl_xfer_count` 增量 ≥ 1，`nixl_num_failed_transfers` 增量 == 0；
- 服务端 `usage.completion_tokens` == 客户端计数 == 32；服务端回显的 prompt sha256 == 发送值；
- 日志：D 有 “NIXL compatibility check passed”，没有 KV load failure，没有 Traceback；
- fault probe（proxy 丢弃 `kv_transfer_params`）→ HTTP 502，D 的 prompt_tokens 增量 == 0（证明不会静默回退）；
- 与 colocated 同 key 的首 token、整段输出 sha256 一致性**只记录，不作为 PASS 条件**
  （bf16 + 不同 batch 形状下逐 token 一致不是不变量；不一致时要解释）。

### E1 · KV transfer scaling（`--stage e1`，只跑 `pd_1p1d`）

context ∈ {512, 2048, 4096, 8192, 16384}（回退时去掉 16384），每点 5 warmup + 15 measure，**串行**（同一时刻只有 1 个请求），
output = 8 token（D 在 KV 到位后再正常 decode 7 步，用来确认 D 侧确实接管了请求；TTFT 不受影响）。

每个样本记录：D 侧 NIXL 增量（bytes、xfer 秒、post 秒、handle 数、descriptor 数、失败数）、
理论 bytes（token 口径与整 block 口径）、客户端 TTFT、proxy 时间戳（P 阶段耗时、D 首字节耗时）。
等效带宽 = 实测 bytes / Σxfer 时间（多 handle 并发时这是逐 handle 时间之和，会**低估**墙钟带宽，报告中要注明）。

### E2 · 受控 prefill 干扰（`--stage e2`）—— 最重要的机制实验

> 证据边界：colo1（1 卡）vs 1P1D（2 卡），GPU 预算不同。E2 只回答干扰是否被隔离，不作为同成本部署效率的证据（那由 E3 承担，见 §3）。

- D 请求：prompt 128 / output 512 / ignore_eos；
- 客户端收到 D 的**第 64 个 token** 时（事件驱动，不靠 sleep）立即注入 P 请求：prompt ∈ {0(对照), 2048, 8192, 16384}，output 1；
- 在 PD 中，注入请求也走完整的 PD 流程（P prefill → D 拉 KV → D 重算最后 1 token）；
- **有效性判据**：D 在注入请求的首 token 之后还至少收到 16 个 token（P 的整个 prefill 时间段落在 D 的 decode 时间段内）；
  不满足的样本记为 `success=False`，并计入 `m27_sample_count` 失败，不静默丢弃；
- arm：`colo1_mbt2048`、`colo1_mbt4096`、`colo1_mbt8192`（B / A）与 `pd_1p1d`；每 (arm, p_len) 5 warmup + 15 measure。

每样本指标：注入后 D 的最大 ITL（max stall）、注入窗口（注入至注入首 token 后 500 ms）的 ITL p95/p99、
D 全程 ITL p50/p95/p99/max、注入请求的 TTFT，以及 PD 中注入阶段的 NIXL 增量。p_len=0 的对照取第 64 个 token 之后的最大 ITL。

### E3 · 2 卡在线（`--stage e3cal` → `--stage e3`）

| 维度 | 冻结值 |
|---|---|
| shape | d_heavy 256/256、balanced 2048/256、p_heavy 8192/64、mixture（逐请求按 0.5 / 0.3 / 0.2 抽样） |
| 到达 | Poisson（burstiness=1，指数间隔），开环，每次 60 s 到达窗口，等全部完成后再统计 |
| seed | 0 / 1 / 2（同 (shape, seed, rate) 在所有 arm 上 schedule sha256 相同） |
| 负载 | C_cal × {0.5, 0.75, 0.9} |
| arm | `pd_1p1d`、`colo2_mbt2048`、`colo2_mbt4096`、`colo2_mbt8192`、`colo1_mbt8192` |
| warmup | 每 (arm, shape) 16 个请求，phase=warmup，不进统计 |

**速率校准**（在 GPU 预算匹配的 colocated 基线 `colo2_mbt8192` 上做一次，冻结为 `configs/<tag>_e3cal_rates.json`）：

1. 逐 base shape 测无负载中位 TTFT 与 ITL（5 warmup + 15 measure，串行）；
2. 从 0.5 req/s 起按 ×√2 爬坡，每级 30 s；通过条件：achieved ≥ 0.95 × offered、无失败、TTFT p99 ≤ 10 × 无负载中位 TTFT
   （mixture 用各组成 shape 中最大的无负载中位）；
3. C_cal = 最后一个通过的速率；如果第一级就不通过或 16 级都通过（梯子用尽），**STOP**（rc=3），不进入 E3。

SLO 不照搬 M12 的 300 ms：goodput 用**多档倍数** m ∈ {2, 5, 10}，好请求定义为
TTFT ≤ m × 无负载中位 TTFT(shape) 且 该请求 ITL p95 ≤ m × 无负载中位 ITL(shape)。报告全部三档，不挑选。

**基线是共享的，不是逐 arm 的**：“无负载中位”只取自校准 arm（E3 是 `colo2_mbt8192`，E4 是 `colo4_mbt8192`）
在 `--stage e3cal` / `e4cal` 中的测量，冻结在 `configs/<tag>_e{3,4}cal_rates.json`。同一实验的所有 arm 使用**同一组绝对 ms 门槛**
（`m27_summarize.py::_online_stats` 从这个文件读取）。每个 run summary 都记录 `rates_file_sha256`，
`m27_paired_schedules_{exp}` 要求所有 arm 的值与当前文件一致，所以事后修改基线会让不变量 FAIL。
因此 goodput 可以在**同一实验内**做同门槛的跨 arm 比较。边界：

- 门槛来自被比较的 arm 之一（colocated 校准 arm），不是独立的“生产 SLO”，只表示“相对校准基线的 m 倍”；
- E3 与 E4 的基线不同，goodput 不跨实验比较；
- 主报告必须同时给出绝对 TTFT / ITL / E2E 分位数；goodput 是补充，不能替代绝对值。

`colo1_mbt8192` 在 2 卡速率下会更早饱和，它的 E3 数据只用于 A 层参考，不参与 GPU 预算匹配的结论。

### E4 · 4 卡 P:D 配比（`--stage e4cal` → `--stage e4`；需 Gate 0 判定 4 卡两两 NV 互联）

arm：`colo4_mbt8192`（校准基线）、`pd_1p3d`、`pd_2p2d`、`pd_3p1d`；shape：d_heavy、p_heavy、mixture；
seed、负载档、时长、warmup 与 E3 相同。proxy 在多 P / 多 D 时按 round-robin 分配。

### Gate 8 → E5（可选，8 卡）

只有同时满足以下两条才执行：(1) Gate 0 实测 8 张卡**两两** NV# 互联（`e5_allowed=true`）；
(2) 在 [`execution_record.md`](execution_record.md) 中，根据 E4 证据写下选择 2–3 个代表配置的理由
（候选 `pd_2p6d` / `pd_4p4d` / `pd_6p2d`，基线 `colo8_mbt8192`），然后用 `--arms` 显式点名。否则 E5 记为“未执行（Gate 8 不满足）”。

### P1 · profiler（可选，**未实现**）

“NIXL READ 是异步的”**不等于** “传输与计算重叠”。要声称重叠，必须有 timeline / profiler 证据；本轮没有实现 P1，
results 中关于重叠的任何表述都只能写“未验证”。

## 6. 隐藏变量

| 变量 | 风险 | 处理 |
|---|---|---|
| KV cache layout | 不处理时 PD 为 HND、colocated 为 NHD | 全 arm `VLLM_KV_CACHE_LAYOUT=HND`，server_info + 日志双来源回读。**M27 正式结论只适用于 HND 布局**（见下方跨模块边界） |
| attention backend | 自动选择可能因配置不同而不同 | `--attention-backend FLASH_ATTN`，双来源回读 |
| proxy 路径 | 只有 PD 经过 proxy 会多一跳 | colocated 也经过 proxy（passthrough），客户端口径一致 |
| GPU 预算 | PD 用 2 卡，colo1 只用 1 卡 | 在线实验主比较为 PD vs `colo2_*`；E4 用 `colo4` |
| 物理 GPU | P 与 colo1 都在 gpu_ids[0]，D 在 gpu_ids[1] | 记录 GPU 映射；同型号卡，差异视为噪声来源之一 |
| 时间漂移 | arm 顺序执行 | 记录每个 arm 的起止时间；局限性中如实说明，不做跨天拼接 |
| eager 模式 | ITL 绝对值被 CPU launch 抬高（M7） | 全 arm 相同；结论只谈相对差异，并注明边界 |
| prefix caching / async scheduling | 改变命中与重叠行为 | 全 arm 关闭并回读 |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `vllm serve` 总是走 MP client，设 0 不生效 | 仍按 AGENTS.md 设置，只做记录 |

**跨模块边界（绝对数不相减）**：

- **M27 formal conclusions are scoped to HND layout.** M22 / M24 没有设置 `VLLM_KV_CACHE_LAYOUT`，也没有 KV connector，
  按源码默认（`kv_connector/utils.py:34-47`）应为 NHD；这是推断，它们的日志里没有布局行可以回读确认。
  所以 M27 的绝对延迟与 M22 / M24 的绝对延迟只作背景，不作单变量对照。
- 两者的口径也不同：M22 / M24 是 in-process 单步延迟，M27 是经 proxy 的在线客户端 TTFT / ITL。
- **M22 absolute latency numbers and M27 absolute latency numbers must not be directly subtracted and attributed
  solely to parallel strategy.** M27 的正式结论只来自模块内部同条件的 A/B 对照。

## 7. 统计口径

- 汇总只用 `phase=measure` 且 `success=True` 的样本；warmup、calibration、fault probe 单独标记 phase。
- E2：逐样本指标的 **median difference（colo − pd）Bootstrap 95% CI**，`random.Random(0)`，10000 次（与 M17 同算法）。
  CI 不含 0 才能说“观察到差异”；含 0 只能说“未观察到超出噪声的差异”。
- E3/E4：尾部分位数用**以请求为簇的 Poisson bootstrap**（`numpy default_rng(0)`，10000 次）给出分位数差的 95% CI。
- 所有分位数都是客户端逐请求精确值（线性插值），不使用服务端直方图近似。
- 不使用 “3.5σ” 这类非标准量。

## 8. 时间预算与降级阶梯（估算，不是测量）

| 段 | 估算 | 说明 |
|---|---|---|
| Gate 0 + 安装 NIXL + transfer test | 0.5 h | 包含 checksum |
| E0 | 0.3 h | 2 次拓扑启动 |
| E1 | 0.3 h | 100 个串行请求 |
| E2 | 1.2 h | 4 arm × 4 p_len × 20 样本，每样本约 10 s |
| E3cal + E3 | 4.5–5 h | 5 arm × 4 shape × 9 run × 约 80 s，加上启动与校准 |
| E4cal + E4 | 3 h | 4 arm × 3 shape × 9 run |
| Package + 回传 | 0.3 h | |

机时不足时，按优先级执行，**不能改问题**：

1. **最低成功线（必须）**：Gate 0 → Gate 1(E0) PASS → E1 → E2 的 A 层（`--arms pd_1p1d,colo1_mbt8192`）。
2. **主线**：E2 的 B 层（`colo1_mbt2048,colo1_mbt4096`）→ E3（先 `pd_1p1d,colo2_mbt8192`，再其余 colo2，最后 colo1）。
3. **扩展**：E4 → Gate 8 → E5。

没跑到的部分在 results 中标“未执行”，不得用理论或本地数据补位。

## 9. 相对任务说明的设计调整（全部在本轮冻结）

| # | 原设计 | 问题 | 冻结设计 |
|---|---|---|---|
| 1 | E2 D output ≈256 | 注入发生在第 64 token 时，D 只剩约 192 token；16384 prompt 的 prefill + KV 传输可能比这更长，P 段可能落在 D 结束之后 | D output = 512，并设有效性判据（注入首 token 之后 D 仍有 ≥16 token） |
| 2 | “D 进入稳定 decode 后注入” | 靠 sleep 无法保证 | 收到 D 第 64 个 token 的事件触发注入；记录相对时间戳 |
| 3 | E2 只比 colocated vs 1P1D | 无法检验“合适的 chunk size 也能做到” | 加 `colo1_mbt2048/4096`（B 层）与 p_len=0 对照 |
| 4 | E1 output 1 | 只有 1 个 token 时无法确认 D 之后能正常 decode | output = 8（TTFT 口径不变） |
| 5 | E3 colocated vs PD | colo1 只有 PD 一半的 GPU，比较的是资源而不是架构 | 加入 GPU 预算匹配的 `colo2_*`，作为 E3 主比较；colo1 保留为 A 参考 |
| 6 | E3 速率未定 | 固定 req/s 在不同 shape 上负载差异极大 | 在 `colo2_mbt8192` 上逐 shape 校准 C_cal，按 0.5/0.75/0.9 取点，速率文件冻结并以 sha256 绑定 |
| 7 | 不照搬 300 ms SLO | — | 多档 SLO 倍数 {2, 5, 10} × 无负载中位值 |
| 8 | 官方 toy proxy | 拿不到 `kv_transfer_params` 时会静默回退成 D 本地 prefill | 自写 fail-closed proxy；E0 用故障注入验证 502 且 D 未收到请求 |
| 9 | 未提及 KV layout | PD 强制 HND、colocated 默认 NHD，是隐藏变量 | 全 arm 设 HND 并回读 |
| 10 | 8 卡候选 2P6D / 4P4D / 6P2D | 拓扑未知（M22 只看到 4 卡） | Gate 0 拓扑判定 + Gate 8 依据 E4 证据选 2–3 个 |
| 11 | NIXL 1.2.0 | PyPI `nixl` 是 meta 包，同时依赖 cu12 与 cu13 后端；直接 install 可能动到 torch | `pip install --no-deps --target` 到独立目录，只装 `nixl` + `nixl-cu13`，不改 M22 的 venv（见 offline_package.md） |
| 12 | 混合比例由设计决定 | — | 0.5 d_heavy / 0.3 balanced / 0.2 p_heavy：以 decode 为主，同时保证约 1/5 的请求带 8K 长 prefill，形成真实干扰 |

### 9.1 Pre-H20 closure correction（2026-10-03，不改实验设计，plan_version 仍为 v1）

| 项 | 内容 |
|---|---|
| 模型身份 | 冻结块 `model` 增加 HF repo / revision / 身份 manifest 及其 sha256 / 文件数 / 总字节 / dtype / max_position_embeddings；manifest 入库，preflight 默认用它校验 |
| 模型 Gate | 恢复副本的 config 实读：28 层 / 28 Q heads / 4 KV heads / hidden 3584（head_dim 128）/ bf16 / max_position_embeddings 32768 → 与冻结值一致，**不触发回退**；KV 理论值仍为 57344 B/token |
| E2 证据边界 | §3 / §5 明确：E2 = 机制实验（1 卡 vs 2 卡），E3 = 同 GPU 预算的部署对照 |
| SLO 基线 | §5 明确：共享的校准 arm 基线、冻结并由 `rates_file_sha256` 不变量约束；绝对值为主 |
| HND 边界 | §6 明确：结论只适用于 HND；不与 M22 / M24 的绝对数相减 |
