# M17 · Prefix Caching & Chunked Prefill 深化（研究/设计 —— GPU-free 部分）

> **模块目标（ROADMAP 第二阶段 阶段 E，T+E+O，单卡可做）**：在 M3/M6 已触及 prefix 的基础上，**用真实受益场景（长共享前缀）系统化证明** APC 的作用与边界，并打通 chunked prefill 对调度/TTFT 的影响。
>
> **本文件状态（GPU 繁忙下推进）**：
> - ✅ **T 理论** + **O 源码定位（带 file:line）** + **实验设计冻结** + **可运行脚本骨架** —— 全部 GPU-free，已完成（见 §1–§5）。
> - ⏸ **E 实测**（跑脚本产出 `outputs/m17/*` 与 `reports/06_prefix_cache.md`）—— **需 GPU，未跑**。GPU 空闲后见 §6 一键复现。
> - 恢复入口：读本文件 §5「续传清单」即可立刻接上。
>
> **边界**：本机 RTX 5080/sm_120、WSL2、Qwen2.5-0.5B、`enforce_eager`；结论限本配置。源码行号基于 pinned `ee0da84`/v0.24.0，动手前可再核。

---

## 0. 与 M3 / M6 的关系（承接，不重复）

| 已有 | 结论 | M17 深化点 |
|---|---|---|
| **M3**（`notes/day3/*`） | block_table 逻辑→物理、prefix 命中(hash/ref_cnt/共享/对齐)、释放→cached-free→复用、抢占重算、K5 尾部优先淘汰/前缀优先保留 | 把这些**跨请求**串成一条 block 生命周期时间线，并对上源码 file:line |
| **M6**（`outputs/m6/baseline.csv` 的 `prefix` 行） | prefix on/off/cold/hot：hot 命中省 94% 计算 token（scheduled 16 vs 256、hit 240）**但首 token 延迟改善落噪声内**（off 6.99ms / hot 7.88ms，ON 短 prompt 反略慢）→「省计算≠省延迟」 | **M6 用的是短前缀（body≤256）**，正好落在 APC 收益不明显区；M17 **用长共享前缀（512/1024/2048）**——APC 文档明说的真实受益场景——看延迟收益能否浮出噪声 |

> **一句话**：M6 诚实地证明了「短前缀下 APC 收益在噪声内」；M17 要验证「**长前缀下 APC 是否真的省出可测的 TTFT**，以及它的边界（只省 prefill 不省 decode）」。

---

## 1. T · 理论（APC 与 Chunked Prefill）

### 1.1 Automatic Prefix Caching（APC）
- **是什么**（`docs/features/automatic_prefix_caching.md`）：缓存已有请求的 KV，新请求若**共享相同前缀**，直接复用其 KV、**跳过共享部分的 prefill 计算**。
- **两个真实受益场景**：① 长文档反复提问（长 doc 只 prefill 一次）；② 多轮对话（历史只算一次，后续轮复用）。
- **关键边界（Limits，务必诚实）**：APC **只减少 prefill 时间，不减少 decode 时间**。所以当**输出很长**（时间主要花在 decode）或**新请求不共享前缀**时，APC **没有收益**。← 这正是 M6 短前缀结论的理论根源。
- **对齐要求**：命中以 **完整 block** 为单位（block_size 对齐）；不足一个 block 的尾部 token 不参与 block hash（见 §2）。

### 1.2 Chunked Prefill
- **是什么**：一个长 prompt 的 prefill 不必在一个 step 内算完，可按**剩余 token 预算**（`max_num_batched_tokens`）切成多个 chunk 跨多个 step 完成。
- **为什么**：① 让**长 prefill 不独占**整个 step，短请求/decode 能插进同一批 → 降低 decode 卡顿、平滑 ITL；② 配合 `long_prefill_token_threshold` 限制单请求单步最多算多少 prefill token。
- **与 APC 的联动**：命中的前缀 token 直接算作 `num_computed_tokens`，**不占本轮 token 预算**；只有未命中的尾部才需要（可能被 chunk 的）prefill。

---

## 2. O · 源码定位（v0.24.0/ee0da84 真实 file:line）

### 2.1 Prefix Caching / KV block 生命周期（`vllm/v1/core/`）
| 环节 | 位置 | 作用 |
|---|---|---|
| **block hash 链式计算** | `kv_cache_utils.py:584` `hash_block_tokens()`；`:696` `get_request_block_hasher()`（`:701` 仅完整 block 才 hash、`:714` 从前一 block hash 链式累积） | hash 输入 = `(parent_hash, token_ids, extra_keys)`，形成前缀唯一指纹 |
| **命中查找** | `kv_cache_manager.py:273` `get_computed_blocks()`（`:318` 最多命中到倒数第二个 token、`:326` `find_longest_cache_hit`）；`block_pool.py:256` `get_cached_block()` | 返回 `(computed_blocks, num_new_computed_tokens)`＝命中了多少完整 block / 少算多少 token |
| **ref_cnt 增** | `block_pool.py:629`（分配 `get_new_blocks` +1）、`:707`（命中复用 `touch()` +1，并从 free 队列移出） | 引用计数 |
| **ref_cnt 减 + 放回队列** | `block_pool.py:724` `free_blocks()` −1；`:745-746` **无 hash block `prepend`（优先驱逐）/ 有 hash block `append`（前缀保留）** | ref_cnt=0 进 free 队列；体现「尾部/无缓存价值优先死、前缀优先活」 |
| **LRU 队列** | `kv_cache_utils.py:182` `FreeKVCacheBlockQueue`（双向链表） | 驱逐顺序 |
| **驱逐已缓存 block** | `block_pool.py:651` `_maybe_evict_cached_block()`（由 `get_new_blocks` `:627` 调用） | 复用带旧 hash 的 block 前先清哈希映射 |
| **登记进缓存** | `block_pool.py:284` `cache_full_blocks()`；`:581` `_insert_block_hash()` | prefill 算完的完整 block 注册 hash → 后续可命中 |

### 2.2 Chunked Prefill（`vllm/v1/core/sched/scheduler.py`）
| 环节 | 位置 | 作用 |
|---|---|---|
| **token 预算** | `schedule()` `:513` `token_budget = max_num_scheduled_tokens`；`:512` `num_scheduled_tokens{}`；`:724` `token_budget -= num_new_tokens` | 每轮调度 token 总量上限 |
| **running(decode/prefill chunk) 优先** | `:538` `while running and token_budget>0`；剩余给 waiting `:790` | prefill 与 decode 在同一 step **共享**预算，running 先扣 |
| **waiting(新 prefill) 受预算限** | `:1003` `num_new_tokens = min(num_new_tokens, token_budget)` | 长 prefill 被切成 chunk 的关键 |
| **long prefill 阈值** | `:593`（running）/`:987`（waiting）`if 0<long_prefill_token_threshold<num_new_tokens: num_new_tokens=threshold`；配置 `config/scheduler.py:80`（`max_num_partial_prefills>1` 时默认 `int(max_model_len*0.04)`） | 限制单请求单步 prefill token，短请求可插队 |
| **关闭 chunked 时** | `:994` `if not enable_chunked_prefill and num_new_tokens>token_budget: break` | 放不下就整体等下一轮（不分块） |

### 2.3 开关
- **APC**：`config/cache.py:92` `enable_prefix_caching: bool = True`（`--no-enable-prefix-caching` 关）；scheduler 初始化传入 `scheduler.py:352 enable_caching=...`。
- **Chunked Prefill**：`config/scheduler.py:84` `enable_chunked_prefill: bool = True`（encoder-decoder 强制关）。

### 2.4 现成测试（供参考）
`tests/v1/core/test_prefix_caching.py`、`tests/v1/core/prefix_cache/test_partial_prefix_cache_primitives.py`、`tests/v1/core/test_kv_cache_utils.py`、`tests/v1/core/test_scheduler.py`。

---

## 3. 实验设计（跑前冻结，GPU 空闲后执行）

> 复用 M6 已验证的**闭环内省法**：`LLMEngine` + `ScheduleSpy` 抓 `SchedulerOutput`；**命中证据 = `actual_prompt_tokens − total_num_scheduled_tokens`**（命中→scheduled<actual）；`chunked = scheduled>actual`；计时用 `torch.cuda.synchronize()` 夹 `perf_counter()`。固定 `Qwen2.5-0.5B / enforce_eager / seed=0 / temperature=0`，warmup 与正式分离、重复≥15、报 median/P95/stdev。

### E1 · 长共享前缀 on/off（核心）
- 构造：共享前缀长度 `P ∈ {256, 512, 1024, 2048}`（block 对齐），各带一段**独立短后缀**打破完全命中。
- 序列：prefix ON 引擎里 `Cold`(全新 ids, miss, 填缓存) → `Hot`(同前缀+新后缀, 命中前缀)；prefix OFF 引擎跑同 ids 作对照。
- 记录（每样本）：`P`、`actual_prompt_tokens`、`scheduled_tokens`、`hit_tokens`、`prefill latency(ms)`、`chunked?`。
- **看点**：随 `P` 增大，Hot 的 `hit_tokens` 应≈`P` 对齐值，prefill latency 相对 Off 的**降幅是否超过 stdev**（M6 短前缀时没超）。

### E2 · block 生命周期（跨请求）
- 场景：请求 A（长前缀）完成→释放；请求 B（同前缀）到达命中；再制造 KV 压力触发驱逐。
- 记录：`new_block_ids`（ScheduleSpy）、命中 block 数；**（可选/需验证内省）** 通过 `get_scheduler().kv_cache_manager` / `block_pool` 读 `ref_cnt`、free 队列、`cached_block_hash_to_block`，产出 block 状态表（Block ID | Request | ref_cnt | hash | 状态 | Step）。
- **看点**：验证 §2 的「无 hash prepend 优先驱逐 / 有 hash append 前缀保留」。

### E3 · chunked prefill 交互
- 构造：长 prompt `L`（如 2048）> `max_num_batched_tokens`（如设 512）。
- 变量：`enable_chunked_prefill ∈ {on, off}`、`max_num_batched_tokens ∈ {512, 1024, 2048}`、`long_prefill_token_threshold`。
- 记录：每 step 的 `total_num_scheduled_tokens`（看分块序列，chunk 数、每块大小、Σ=L）、TTFT 代理；再**并发一个 decode 请求**，看 chunked on/off 下该 decode 的 ITL 是否被长 prefill 阻塞。
- **看点**：chunked off → 长 prefill 整块占 step、decode 被挤；chunked on → prefill 分块、decode 可插队。

### 指标与产出（待 GPU）
- `outputs/m17/prefix_cache_sweep.csv`（E1）、`outputs/m17/block_lifecycle.csv`（E2）、`outputs/m17/chunked_prefill.csv`（E3）、`outputs/m17/raw/*.jsonl`。
- `reports/06_prefix_cache.md`（或 `notes/m17/results.md`）：结论 + 与 M6 短前缀对照。

---

## 4. 不变量（冻结，可验证；跑后逐条核对）
- **I1 命中对齐**：APC on、共享前缀 `P`、block_size `B` → Hot 的 `num_computed_tokens(cached) ≈ floor(P/B)*B`（完整 block 才命中）。
- **I2 省计算守恒**：`hit_tokens`（少算的 prefill token 数）＝ 命中的缓存 token 数（`actual − scheduled`）。
- **I3 off 基线**：APC off → `hit_tokens = 0`（scheduled == actual）。
- **I4 分块守恒**：chunked on、`L > max_num_batched_tokens` → 该 prefill 跨多 step，每 step 调度 token ≤ 预算，**Σ(chunks) = 未命中 prefill token 数**。
- **I5 诚实边界**：APC 省 prefill 不省 decode——固定前缀、**加长 `max_tokens`（输出）**时，APC 的 E2E 相对收益应**变小**（复现 APC 文档的 Limits）。

---

## 5. 续传清单（GPU 空闲后从这里接）
- [x] T 理论、O 源码定位（本文件 §1–§2）
- [x] 实验设计 + 不变量冻结（§3–§4）
- [x] 可运行脚本骨架 `scripts/m17_prefix_cache.py`（E1/E3 主体，复用 M6 内省法）
- [ ] **跑 E1**（长共享前缀 on/off sweep）→ `outputs/m17/prefix_cache_sweep.csv`
- [ ] **跑 E3**（chunked prefill）→ `outputs/m17/chunked_prefill.csv`
- [ ] （可选）E2 block 生命周期内省——**先验证 `block_pool` 的 ref_cnt/free 队列内省 API 在本版本可读**（脚本里标了 TODO）
- [ ] 写 `reports`/结论，回填 evidence_index 与 ROADMAP M17 状态
- [ ] 逐条核对 I1–I5

**脚本已标注**：需要 GPU 的入口在 `main()`，未运行；命中证据法（`actual − scheduled`）已被 M6 验证可靠；E2 的 block_pool 深度内省标为待验证 TODO（不阻塞 E1/E3）。

---

## 6. 复现（待 GPU 空闲，注意先确认 GPU 不忙以免 OOM）
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m17_prefix_cache.py            # E1 + E3 -> outputs/m17/*.csv + raw/*.jsonl
# 产出后再写结论、回填 evidence_index / ROADMAP
```
