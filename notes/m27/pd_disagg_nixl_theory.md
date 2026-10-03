# M27 · Prefill–Decode 解耦 + NIXL KV Transfer：源码与理论（T）

> 状态：**T（源码核实）+ O（工程准备）完成；E/H NOT EXECUTED**。本文没有任何性能数字。
> 源码基准：`../vllm-src @ ee0da84`（只读）。该工作树有 22 个带学习注释的脏文件
> （含 `scheduler.py` / `gpu_model_runner.py`），**所以下面的行号一律取自
> `git show ee0da84:<path>`（pristine）**，而不是工作树里的文件。
> H20 运行时用 vLLM **0.24.0 wheel**，不是本源码树；行号用于定位机制，wheel 中的行号可能不同。

---

## 1. 两个进程、两侧 connector

| 角色 | 配置 | 证据 |
|---|---|---|
| P（prefill） | `kv_role=kv_producer` | `vllm/config/kv_transfer.py:40-42` |
| D（decode） | `kv_role=kv_consumer` | 同上 |
| `kv_both` | 对 NixlConnector **已弃用**（只发 warning，不是全局弃用） | `nixl/connector.py:122-128` |
| 默认连接器类 | `NixlConnector = NixlPullConnector`（D 主动 READ P 的 KV） | `nixl/connector.py:386` |
| 失败策略 | `kv_load_failure_policy: "recompute" \| "fail" = "fail"` | `config/kv_transfer.py:69`；调度器读取于 `scheduler.py:143-144` |

**旧注释「Currently only 1P1D is supported.」**（`config/kv_transfer.py:45-48`）写在 `kv_rank` 上，
这是老式 P2P 连接器的 rank 语义；NixlConnector 不读 `kv_rank`。官方 NIXL 文档与测试用 proxy
的 `--prefiller-hosts/--decoder-hosts` 都是 `nargs="+"`
（`tests/v1/kv_connector/nixl_integration/toy_proxy_server.py:98-113`），即 xPyD 通过 proxy 轮转实现。

## 2. KV 生命周期（Pull 模式，按调用方向）

```text
client ──► proxy ──► P: max_tokens=1, kv_transfer_params={do_remote_decode: True}
                     P 本地 prefill（prompt_tokens_by_source{local_compute} += L）
                     P.request_finished → 返回 kv_transfer_params
                        {do_remote_prefill, remote_block_ids, remote_engine_id,
                         remote_request_id, remote_host, remote_port, tp_size, remote_num_tokens}
                        （pull_scheduler.py:181-275；P 的 block 保留到 D 通知或租约到期）
       proxy ──► D: 原请求 + 上述 kv_transfer_params
                     scheduler.get_num_new_matched_tokens (scheduler.py:725)
                        → pull_scheduler.py:34-111：返回 (整 prompt 的 token 数, async=True)
                     prefill_stats.set(local_cached, external_cached)   (scheduler.py:766)
                     update_state_after_alloc                            (scheduler.py:902)
                     status = WAITING_FOR_REMOTE_KVS                     (scheduler.py:921)
                     build_connector_meta                                (scheduler.py:1105)
                     worker: bind_connector_metadata → start_load_kv（forward 前，发起异步 READ）
                        (kv_connector_model_runner_mixin.py:89 / :95)
                     _pop_done_transfers → xfer_stats.record_transfer  (base_worker.py:2003-2020)
                     _update_from_kv_xfer_finished                      (scheduler.py:2422)
                     _update_waiting_for_remote_kv：全命中时
                        num_computed_tokens = num_tokens - 1           (scheduler.py:2355 / :2385)
                        → D 只重算最后 1 个 token 以产生 logits，然后正常 decode
```

**M24 笔记的更正**（M24 冻结，不改原文，只在这里记录）：

1. `context_parallel_ring_disagg.md` §3.2 写「逐层：attention 包装器在算完后 `save_kv_layer`」。
   对 NIXL 不成立：`save_kv_layer` / `wait_for_layer_load` 都是空实现
   （`nixl/connector.py:276-288`：“NixlConnector does not do layerwise saving / does not save explicitly”）。
   NIXL 的传输发生在 D 侧 `start_load_kv`（`nixl/connector.py:342`）发起的异步 READ。
2. 同节 mixin 行号 `86-100` 在 pristine 中对应 `kv_connector_model_runner_mixin.py:78-100`
   （`_get_kv_connector_output`；`bind` :89、`start_load_kv` :95、`wait_for_save` :100）。
3. 「`kv_rank` 注释限制当前是 1P1D」不适用于 NixlConnector（见 §1）。
4. 「≈12 KiB/token」是 **0.5B** 的值（2×24×2×64×2 = 12288 B）。M27 的 7B 是 **57344 B/token**（§4），
   P=2048 时理论 KV 是 112 MiB，而不是 24 MiB。

## 3. 正确性与 fail-closed 的源码依据

| 风险 | 源码事实 | M27 对策 |
|---|---|---|
| proxy 拿不到 `kv_transfer_params` 会**静默**回退成 D 本地 prefill | 官方 toy proxy：`if kv_transfer_params: req_data[...] = ...`（`toy_proxy_server.py:235-237`） | 自写 `m27_proxy.py`：参数缺失、`do_remote_prefill≠True`、缺 remote 键或 block 为空时直接返回 502 `fail_closed:<reason>`，**绝不转发 D**；E0 用故障注入验证 |
| KV load 失败时的行为 | `fail`：`_handle_invalid_blocks`（`scheduler.py:2554`）日志 “Failing %d request(s) due to KV load failure”（:2605）；`recompute` 时为 “Recovered from KV load failure”（:2614） | 显式 `kv_load_failure_policy=fail`；日志两种模式都解析，任一出现即判 FAIL |
| P/D 模型或配置不一致 | `compute_nixl_compatibility_hash`（`nixl/metadata.py:79+`）哈希 vllm 版本、connector 版本、**`model_config.model`（路径字符串）**、dtype、kv heads、head size、层数、attn backend、cache dtype、cross-layer、HMA；不一致时报 “NIXL compatibility hash mismatch”（`base_worker.py:597`），一致时 INFO “NIXL compatibility check passed (hash: %s)”（:610） | P/D 用**同一个模型路径字符串**；日志要求 D 有 compat pass、无 mismatch |
| D 是否真的没有重算 prompt | `vllm:prompt_tokens_by_source{source=local_compute\|local_cache_hit\|external_kv_transfer}`（`metrics/loggers.py:643-656`；语义 `metrics/stats.py:282-299`：computed + local_cache_hit + external = total） | 每个 PD 样本：D external == prompt_len、D local_compute == 0、P local_compute == prompt_len（末 token 的 1 次重算不计入 local_compute，见 §2 :2385） |
| NIXL 不可用时 vLLM 只发 warning | `vllm/distributed/nixl_utils.py:46-72`：导入失败 → `NixlWrapper=None` + “NIXL is not available” | preflight 断言 `NixlWrapper is not None`；launcher 要求 P/D 日志出现 “NIXL is available” |

## 4. KV 字节理论值

```text
kv_bytes_per_token = 2 (K,V) × L × H_kv × d_head × bytes(bf16)
                   = 2 × 28 × 4 × 128 × 2 = 57344 B/token（56 KiB）
```

NIXL 以 block（16 token）为单位搬运，因此另记整 block 口径：`ceil(L/16) × 16 × 57344`。

| context | 理论 bytes（token 口径） | 整 block 口径 |
|---:|---:|---:|
| 512 | 29,360,128（28 MiB） | 同 |
| 2048 | 117,440,512（112 MiB） | 同 |
| 4096 | 234,881,024（224 MiB） | 同 |
| 8192 | 469,762,048（448 MiB） | 同 |
| 16384 | 939,524,096（896 MiB） | 同 |

第二来源（已有数据，非 M27 实测）：M22 H20 TP1 formal 中，引擎报告
`sum_available_kv_bytes=70034921575` 对应 `sched_kv_cache_size_tokens=1221312`，
而 `70034921575 // (16×57344) × 16 = 1221312`（`outputs/m22/raw/h20_formal_tp1_run.jsonl`；
`check_invariants m27_kv_theory_cross_m22` 自动核对）。

**理论 bytes ≠ NIXL 实测 bytes**：实测来自 D 侧 telemetry `totalBytes`（`nixl/stats.py:47-50`：
`xferDuration/1e6` 秒、`postDuration/1e6` 秒、`totalBytes`、`descCount`），可能因 HND 布局的描述符切分、
跨层 block、只传未命中 block 等与理论值不同。M27 两者都记录，**不把相等作为判据**。

## 5. 隐藏变量：KV cache layout

- 有 NIXL 时：`NixlConnector.get_required_kvcache_layout` 返回 **HND**（`nixl/connector.py:141-157`）。
- 无 connector（colocated）时：默认 **NHD**（`kv_connector/utils.py:34-47`）。
- 二者都在 `get_kv_cache_layout()` 中被环境变量 `VLLM_KV_CACHE_LAYOUT` 覆盖，且环境变量的优先级
  **高于** connector（`attention/backends/utils.py:83-109`，日志 “`VLLM_KV_CACHE_LAYOUT` environment
  variable detected. Setting KV cache layout to HND.”）；FlashAttention 本身不强制布局
  （`v1/attention/backend.py:378` 返回 None）。

如果不处理，colocated 与 PD 会同时在「是否解耦」和「KV 内存布局」两个变量上不同。
**对策：所有 arm 都设 `VLLM_KV_CACHE_LAYOUT=HND`**，启动后从 `/server_info` 的 `vllm_env` 与日志两个来源回读。

## 6. Side channel / 租约 / 其他默认值

| 项 | 默认 | 源码 | M27 |
|---|---|---|---|
| `VLLM_NIXL_SIDE_CHANNEL_HOST` | `localhost` | `envs.py:198` | 显式 `127.0.0.1` |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` | `5600 + dp_index` | `envs.py:199`；`base_scheduler.py:65-67` | 同机每实例唯一：`5600 + i` |
| `kv_lease_duration` | 30 s | `base_scheduler.py:70-76` | 默认；P 侧 block 最长保留时间 |
| `bidirectional_kv_xfer` | False | `base_scheduler.py:149-151` | 默认（不启用多轮回传） |
| `kv_recompute_threshold` | 64 | `base_scheduler.py:141-143` | 只作用于 bidirectional 路径，M27 不触发 |

## 7. 服务端口径

- `vllm serve`（AsyncLLM）总是使用 MP client（`v1/engine/core_client.py:109-132` `make_async_mp_client`），
  所以 `VLLM_ENABLE_V1_MULTIPROCESSING=0` 对在线服务不生效（仍按 AGENTS.md 设置，只做记录）。
- prompt 长度必须 **严格小于** `max_model_len`（`v1/engine/input_processor.py:400-432`），因此 mml=32768 时
  最长 prompt 16384 的余量充足；回退到 16384 时，最长 prompt 降到 8192。
- 关分块时要求 `max_num_batched_tokens >= max_model_len`（`config/scheduler.py:271-283` `verify_max_model_len`），
  并且对 generate 模型手动关分块会给出警告 “does not officially support disabling chunked prefill”
  （`arg_utils.py:2471-2480`）。唯一会强制关分块的分支是 RISC-V CPU（`arg_utils.py:2510-2520`），与 H20 无关；
  生效值仍由启动回读核对。M27 **从不**把关分块作为正式 arm。
- `--attention-backend`（`arg_utils.py:915`）钉为 `FLASH_ATTN`；`--stream-interval`（:1452）钉为 1。
- `/server_info?config_format=json` 需要 `VLLM_SERVER_DEV_MODE=1`
  （`entrypoints/serve/dev/server_info/api_router.py`），它会用 pydantic 导出完整 `VllmConfig` 和 `vllm_env`。
  其中 attention backend 导出的是**枚举值（类路径）**
  `vllm.v1.attention.backends.flash_attn.FlashAttentionBackend`；日志打印 `AttentionBackendEnum.FLASH_ATTN`
  （`platforms/cuda.py:420`）。launcher 会把这两种写法都归一为 `FLASH_ATTN`。
- 引擎配置日志：“Initializing a V1 LLM engine (v%s) with config: …”（`v1/engine/core.py:114-118`），
  作为 `/server_info` 之外的第二来源。

本地离线验证（不建引擎、不占 GPU）：用 pinned 树的 `make_arg_parser` + `EngineArgs.from_cli_args`
解析 `m27_common.serve_cmd()` 生成的全部参数，并用本地 0.5B 快照的 config `create_engine_config()`
导出 JSON，确认 launcher 的 `READBACK_KEYS` 全部可以取到，且值与请求一致。

## 8. 官方文档的定性（`docs/features/disagg_prefill.md:1-16`）

- **experimental**；
- 动机是「分别调 TTFT 与 ITL」和「控制 tail ITL」；
- 「Chunked prefill with a proper chunk size also can achieve the same goal」；
- 「Disaggregated prefill **DOES NOT improve throughput**」。

因此 M27 的研究问题不以吞吐为成功线，并且显式包含 B（调好 chunk 的 colocated）对照来检验
「合适的 chunk size 也能达到同样效果」这一说法。**「PD 无收益」是允许的正式结论。**

## 9. 机制上的预期（是假设，不是结论）

```text
colocated ITL 尾部 ≈ T_decode_step + 被插入的 prefill chunk（≤ mbt token）
PD        ITL 尾部 ≈ T_decode_step_isolated + D 侧调度开销（含 KV READ 期间的 step）
PD        TTFT     ≈ T_prefill(P) + T_proxy + T_KV_READ(L) + T_first_decode(D)
```

- mbt 越小，colocated 被插入的 chunk 越短，尾部 ITL 可能越低，但长 prompt 的 TTFT 会更长（B 的取舍）。
- PD 把 GPU 预算拆成 P 和 D：同样 2 张卡时，colocated 有 2 个完整副本，PD 只有 1 个 decode 实例。
  所以在线实验的主比较是 **PD vs colo2_***（相同 GPU 预算），colo1 只作为 A 的因果参考。
- 以上都是要被 E1–E4 检验的假设，不是可以提前写进结论的事实。
