# M3 · Trace 改动清单

M3 全部为**被动只读**钩子（`# VLLM_INFERENCE_LAB_TRACE`，component `kv`/`scheduler`），`VLLM_LAB_TRACE` 关闭时零输出、零逻辑改变（已用 E1 回归验证：归一化后与之前 trace 逐行一致）。未新增文件、未改 `_lab_trace.py`（沿用 M1/2 的 JSONL/组件过滤开关）。

## 新增钩子（3 个文件，只读打印 shape/id/标量/计数）

| 文件 | 方法 | 记录 |
|---|---|---|
| `vllm/v1/core/kv_cache_manager.py` | `get_computed_blocks`（2 处：命中 / 关闭） | req_id、num_new_computed_tokens、hit_block_ids、prefix_caching |
| 同上 | `allocate_slots`（成功 / `allocate_slots_FAILED`） | req_id、num_new_tokens、new_block_ids、free_blocks_after / required vs available（失败） |
| 同上 | `free` | req_id、释放前的 block_ids、free_blocks_before |
| `vllm/v1/core/block_pool.py` | `get_new_blocks` | 分配的 block_ids + ref_cnts |
| 同上 | `free_blocks` | freed_cached_tail（带 hash 保留）/ freed_evict_first（无 hash 优先淘汰） |
| 同上 | `_maybe_evict_cached_block` | 被驱逐 block_id + num_evicted_hashes |
| `vllm/v1/core/sched/scheduler.py` | `_preempt_request` | victim_id、num_preemptions、num_computed_tokens_after_reset(=0) |

- 标记计数：kv_cache_manager 5、block_pool 3、scheduler +1（preempt；另有 M1 的 4 个 TRACE、E5 的 1 个 EXPERIMENT）。
- 备份：`backups/day3/{kv_cache_manager,block_pool,scheduler}.py.pre_day3`。
- 删除：移除各文件中标 `# VLLM_INFERENCE_LAB_TRACE`（M3 新增）的代码块即可；均为 try/except 包裹、`if _lt.enabled()` 或 `trace_event` 内部 enabled 门控。
- 关于 "resumed"：`scheduled_resumed_reqs` 在 schedule 钩子处已被清空，故未在 schedule 补 resumed；改用 `scheduler/preempt` 事件 + per_req 中 victim 以 `computed_before=0` 重现（恢复/重算）来观察，更可靠。

## git 状态
M3 前后 `vllm-src` 文件集合**完全一致**：21 个 `M`（含本轮三文件，本就因注释/前几日插桩为 M）+ 1 个 `?? vllm/_lab_trace.py`。未 `git add/commit/stash`、未改 baseline venv、未编译 C++/CUDA、未加依赖、未删用户注释。

## lab 侧新增
- `scripts/m3_kv_driver.py`：KV 实验驱动（`num_gpu_blocks_override`、`enable_prefix_caching` 开关、到达表、长生成）。

## K5 追加：waiting 循环实验性 skip + evict 覆盖增强

为验证"抢占后尾部优先淘汰"，新增：
| 位置 | 标记 | 门控 | 作用 |
|---|---|---|---|
| `scheduler.py` waiting 准入循环（`peek_request()` 之后） | `# VLLM_INFERENCE_LAB_EXPERIMENT` | `VLLM_LAB_BLOCK=1` + id 命中 | 跳过被 block 的 waiting 请求（`pop+step_skipped_waiting.prepend+continue`），使被抢占请求不被重新准入、保持 cached-free。**会改变调度**，与被动 trace 区分；复用 running 循环同款 `blocked_ids()`。 |
| `block_pool._maybe_evict_cached_block` 入口 | `# VLLM_INFERENCE_LAB_TRACE` | trace | 在 `reset_hash` 前捕获 `block.block_hash_num_tokens`，随驱逐事件记录（判断尾部/头部）。 |
| `scripts/m3_kv_driver.py` | — | — | 新增 `run_k5` 控制流：priority 调度 + 抢占检测（scheduler 自省）+ set/clear_blocked + 动态加压。 |

- scheduler 现有 2 处 `# VLLM_INFERENCE_LAB_EXPERIMENT`（E5 running 跳过 + K5 waiting 跳过），均 `VLLM_LAB_BLOCK` 门控、默认关、已回归验证零副作用（E1 归一化一致）。
- 备份：`backups/day3/{scheduler,block_pool}.py.pre_k5`。
- git 状态 K5 前后文件集合不变（21 `M` + 1 `?? _lab_trace.py`）；未 `git add/commit/stash`。
