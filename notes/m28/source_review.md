# M28 · 源码核查（只读 `../vllm-src`）

> 状态：**静态核查完成**（2026-10-12）。本文只有源码事实（带 `file:line`）和**待实验检验的预测**，
> 没有任何实测结论。预测是否成立以 `results.md` 的实测为准；与实测矛盾时改结论，不改数据。
> 外部 Source Mapping 报告（`../source-mapping-research/reports/05_m28_decision.md`）只作线索，所有条目已在本仓库源码上逐条复核。

## 0. 版本与工作树

| 项 | 值 | 证据 |
|---|---|---|
| vLLM 源码 | tag `v0.24.0`，HEAD `ee0da84ab9e04ac7610e28580af62c365e898389`；editable 安装自报 `0.24.1.dev0+gee0da84ab.d20260714` | `git -C ../vllm-src rev-parse HEAD` / `describe --tags` |
| 脏文件 | 22 个（21 个 M + `vllm/_lab_trace.py` 新增），**无一属于 offloading / kv_connector / entrypoints 服务端路径** | `outputs/m28/m28_vllm_src_ast_audit.json`（`offloading_files_dirty: []`） |
| 模型 | 本地快照 `models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae5576…`，`model.safetensors` blob sha256 `fdf756fa…fe` | `m28_common.model_identity()` |

## 1. 脏工作树对 M28 的影响（AST 语义比对）

`scripts/m28_source_audit.py` 对每个已修改的 `.py`：先去掉所有裸字符串语句（docstring / 中文注释块），再比较
`ast.unparse(HEAD 版本)` 与 `ast.unparse(工作树版本)`。结果落盘在 `outputs/m28/m28_vllm_src_ast_audit.json`。

- **语义完全相同（15 个）**：`entrypoints/llm.py`、`offline_utils.py`、`v1/engine/core.py`、`v1/engine/core_client.py`、
  `v1/engine/llm_engine.py`、`attention/backend.py`、`fa_utils.py`、`registry.py`、`selector.py`、`kv_cache_coordinator.py`、
  `kv_cache_utils.py`、`single_type_kv_cache_manager.py`、`block_table.py`、`gpu_input_batch.py`、`worker/utils.py`。
  → **B 组用到的 EngineCore / MPClient 生命周期代码与上游语义一致**；API server、launcher、`/health` 根本未修改。
- **有差异（6 个）**：`attention.py`、`flash_attn.py`、`block_pool.py`、`kv_cache_manager.py`、`scheduler.py`、`gpu_model_runner.py`。
  差异中所有不带门控标识的行都位于以下三类判断之下（逐条人工核对过 audit JSON 的 `ungated_changed_lines`）：
  1. `if _lt.enabled():`：`VLLM_LAB_TRACE` 未设置时 `enabled()` 为 False，`trace_event()` 第一行就 return（`vllm/_lab_trace.py`）；
  2. `if request.request_id in _lt.blocked_ids():`（`scheduler.py:547`、`:806`）：`VLLM_LAB_BLOCK` 未设置时 `blocked_ids()` 返回空集；
  3. `if self._m9_coalesce:`（`gpu_model_runner.py:908-912`）：`VLLM_LAB_M9_COALESCE` 默认 `"0"`，else 分支与原版逐句等价。
  其余是 `try/except Exception: pass` 包装与无条件的 `_lt.bump_step()`（`scheduler.py:494`，只是模块级整数自增）。
- **结论（静态）**：在 `VLLM_LAB_TRACE` / `VLLM_LAB_BLOCK` / `VLLM_LAB_M9_COALESCE` 均未设置时，这些钩子不改变调度、缓存与计算行为；
  残留的只是每 step / 每层的少量 Python 调用开销（各 arm 相同）。M28 runner 因此**强制从 server 环境中删除**这三个变量，
  并在每个 lifecycle 记录里落盘 `server_env_unset_ok`（validator 规则 `a_env_fingerprint`）。
  这不是"绝对延迟与上游 wheel 相同"的证明：钩子的固定开销没有被单独测量（见 `limitations.md`）。

## 2. A · OffloadingConnector 入口

| 事实 | 位置 |
|---|---|
| 配置：`--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"cpu_bytes_to_use":N}}'`；`cpu_bytes_to_use` 必填，是**所有 worker 合计**字节数 | `docs/features/kv_offloading_usage.md:27-37,70`；`vllm/v1/kv_offload/cpu/spec.py:63-66` |
| 简写 `--kv-offloading-size <GiB>` 会被展开成同一个 connector（M28 不用简写，直接写 JSON，避免两层转换） | `vllm/config/vllm.py:781-800` |
| 默认 `spec_name=CPUOffloadingSpec`、`eviction_policy=lru`、`store_threshold=0`、`offload_prompt_only=true`、offload `block_size` = GPU block size | `docs/features/kv_offloading_usage.md:65-79` |
| **（v2 更正）** `CPUOffloadingSpec.create_handlers` 调 `CpuGpuOffloadingHandlers` 时**不传** `mmap_region`，CPU 层是 EngineCore 进程内的 `torch.zeros((num_cpu_blocks, page), int8, pin_memory=PIN_MEMORY)` | `vllm/v1/kv_offload/cpu/spec.py:134-139`；`vllm/v1/kv_offload/cpu/gpu_worker.py:478-516` |
| `PIN_MEMORY = is_pin_memory_available()`；CUDA 平台在 WSL2 上返回 `VLLM_WSL2_ENABLE_PIN_MEMORY`（默认 0）→ 本机 CPU 层是 **pageable** 内存 | `vllm/utils/torch_utils.py:71`；`vllm/platforms/cuda.py:285-303`；`vllm/envs.py:264,1855` |
| `/dev/shm/vllm_offload_{id}.mmap` + `cudaHostRegister` 只在 **tiering** spec 里使用（`SharedOffloadRegion` 只被 `tiering/spec.py` 构造），与 M28 用的 CPU spec 无关 | `vllm/v1/kv_offload/tiering/spec.py:104,170-177`；`shared_offload_region.py:56-117`；`gpu_worker.py:126-153` |
| GPU→CPU 用 `ops.swap_blocks_batch`；CPU→GPU 只在每个 page < 28 KiB 且 8B 对齐时才用 Triton kernel。本配置是 cross-layer KV 布局（日志 "Allocating a cross layer KV cache of shape (1280, 2, 24, 2, 16, 64)"、"Allocating 1 CPU tensors"），page = 2×24×2×16×64×2 B = **196608 B ≥ 28 KiB** → 两个方向都走 `ops.swap_blocks_batch` | `gpu_worker.py:38-61`；`swap_blocks_triton.py:20`（`THRESHOLD_BYTES = 28*1024`） |
| 调度：先查本地 prefix cache，再对**剩余部分**问 connector；`connector_prefix_cache_queries = num_tokens − local_hits`，`hits = ext_tokens` | `vllm/v1/core/sched/scheduler.py:860-932` |
| 请求级 `num_cached_tokens` = local + external（`PrefillStats`），经 `--enable-prompt-tokens-details` 进入 `usage.prompt_tokens_details.cached_tokens` | `vllm/v1/metrics/stats.py:248-264`；`vllm/v1/engine/output_processor.py:628-632`；`entrypoints/openai/completion/serving.py:447-449` |
| Prometheus：`vllm:prefix_cache_{queries,hits}`、`vllm:external_prefix_cache_{queries,hits}` | `vllm/v1/metrics/loggers.py:558-592` |
| Prometheus：`vllm:kv_offload_{load,store}_{bytes,time}`（+ size histogram） | `vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py:25-77` |
| `get_model_path` 只在 `HF_HUB_OFFLINE=1` 时调用 `snapshot_download(local_files_only)`；本地快照缺 `.gitattributes/LICENSE/README.md` → `IncompleteSnapshotError`（2026-10-12 用 vLLM 自己的 parser 在 CPU 上复现）。**M28 改用快照目录路径 + `--served-model-name` 启动**，`os.path.exists(model)` 直接返回，不走 hub | `vllm/transformers_utils/repo_utils.py:225-242` |

**由此得到的 A 组证据设计**：每个被测请求前后各抓一次 `/metrics`，三个相互独立的来源必须同时成立：
(1) `Δprefix_cache_hits` / `Δexternal_prefix_cache_hits`；(2) `Δkv_offload_load_bytes`；(3) 请求级 `cached_tokens`。
另用 `Δrequest_success == 1 ∧ Δprompt_tokens == N` 证明这次抓取确实覆盖到本请求（防止指标滞后导致误判）。

**待检验预测（A）**
- P-A1：GPU 被挤出后再发同一 prompt，`Δlocal_hits = 0`、`Δext_hits = H = ⌊(N−1)/16⌋·16`、`Δload_bytes > 0`。
- P-A2：`Δload_bytes / Δext_hits` 恒等于 `24 层 × 2(K,V) × 2 KV 头 × 64 × 2 B = 12288 B/token`。
- P-A3：CPU 命中与 GPU 热命中走相同的"缓存前缀 + 只算剩余 token"计算形状，输出 token 与 logprob 应逐位相同（KV 往返无损的强证据）。
- P-A4（v2 改写）：本机 CPU 层是 pageable 内存（见上表），load 带宽可能明显低于 pinned 配置；load 仍应正确工作。
  v1 写的是"`cudaHostRegister` 可能失败"——那条路径不属于 CPU spec，是 v1 核查的错误，v1 冒烟实测（`/dev/shm` 无新文件、日志无 mmap 行）推翻了它。

> **v1 核查错误记录**：v1 把 tiering spec 的 `SharedOffloadRegion`（`/dev/shm` mmap + `cudaHostRegister`）误认为 CPU spec 的 CPU 层，
> 并按"每层 8 KiB page"推断 CPU→GPU 走 Triton kernel；实际是 cross-layer 布局的单个 196608 B page。两处均在 v1 冒烟后、正式 arm 之前更正。

## 3. B · 生命周期 / 健康检查 / 停机

| 事实 | 位置 |
|---|---|
| serve 路径的 `AsyncLLM` **总是** MP client（EngineCore 是子进程）；`VLLM_ENABLE_V1_MULTIPROCESSING` 只影响 `LLMEngine`（离线 `LLM`） | `vllm/v1/engine/async_llm.py:146`；`vllm/v1/engine/llm_engine.py:207,224` |
| 子进程名 `VLLM::EngineCore`（setproctitle）；单 API server 时 API 进程不改名 | `vllm/v1/engine/core.py:1347-1350`；`vllm/utils/system_utils.py:184-198`；`vllm/v1/utils.py:503` |
| `/health`：`check_health()` 只看 `errored`；EngineDeadError → 503，否则 200 | `vllm/entrypoints/serve/instrumentator/health.py:22-33`；`async_llm.py:900-903,1054-1055` |
| EngineCore 存活监控：`monitor_engine_liveness()` 用进程 sentinel 等待；任一退出 → `engine_dead=True` → `MPClient.shutdown()` | `vllm/v1/engine/utils.py:222-242`；`core_client.py:736-752` |
| API watchdog 每 5 s 检查 `engine.errored and not is_running`，成立且未设 `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` → `server.should_exit = True` | `vllm/entrypoints/launcher.py:156-178` |
| EngineDeadError 的 HTTP 映射：非流式由 `engine_error_handler` 返回 JSON（`create_error_response` 的 else 分支 → 500 `InternalServerError`）；流式已发 200，异常作为下一个 SSE chunk 发出，然后 `[DONE]` | `vllm/entrypoints/serve/utils/server_utils.py:327-367`；`serve/utils/error_response.py:58-61`；`completion/serving.py:469-475` |
| `--shutdown-timeout`：int，默认 0；"0 = abort, >0 = wait" | `vllm/engine/arg_utils.py:1562-1567`；`vllm/config/vllm.py:380` |
| API 收到 SIGTERM/SIGINT → `engine_client.shutdown(timeout)`（在 executor 里跑）→ 结束后才让 uvicorn 退出 | `launcher.py:95-130` |
| 进程管理器：先 `proc.terminate()`（SIGTERM 给 EngineCore），等到 `deadline = now + timeout`，仍存活 → `kill_process_tree` | `vllm/v1/utils.py:584-627` |
| EngineCore 收到 SIGTERM → `shutdown_state=REQUESTED`；`timeout==0` → `finish_requests(ABORTED)` + 发 abort 输出；`>0` → drain 到没有工作 | `vllm/v1/engine/core.py:1391-1401,1517-1563` |
| shutdown 期间的新请求由 `_reject_add_in_shutdown` 以 abort 输出回给客户端 | `core.py:1574,1600-1609` |

**待检验预测（B）**
- P-B1-1：SIGKILL EngineCore 后约 1 s 内（sentinel 等待粒度）`/health` 变 503。
- P-B1-2：在途 streaming 请求收到 200 流中的错误事件 + `[DONE]`；非流式请求收到 HTTP 500。
- P-B1-3：API 进程在 ≤ 1 个 watchdog 周期（5 s）+ 关闭时间内自行退出，端口释放；显存随 EngineCore 死亡释放（与 M21 E6"只杀父进程 → EngineCore 成孤儿占显存"形成对照）。
- P-B3-1：`timeout=0` 时进程管理器的 deadline 立即到期 → EngineCore 刚收到 SIGTERM 就被 `kill_process_tree`；EngineCore 自己的 abort 输出可能来不及送达，客户端看到的可能是连接中断而不是 `finish_reason=abort`。**这是源码推出的竞争，必须由实测判定**。
- P-B3-2：`timeout=2`（不足）→ drain 2 s 后被强杀，在途请求被截断；`timeout=60`（充足）→ 在途请求全部完成后才退出。
- P-B3-3：drain 期间 HTTP 仍接受连接，新请求被 EngineCore 以 abort 拒绝；engine client 停止后端口关闭 → 连接被拒。
- P-B2：SIGSTOP EngineCore 期间 `/health` 仍为 200（`check_health` 只看 `errored` 标志），即 **hang 对 `/health` 不可见**；SIGCONT 后在途请求恢复。

## 4. 不在 M28 范围内

跨 worker 的请求续跑 / fencing、文件或 SSD 层（`TieringOffloadingSpec`）、多实例共享 CPU 层、LMCache 后端。
