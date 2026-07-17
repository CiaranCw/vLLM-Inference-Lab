# Day 3 · K5 —— 抢占后 block chain 尾部优先淘汰、前缀优先保留

> 结论以真实 trace 为准：`outputs/day3/K5.jsonl`（提前解除→部分命中）、`outputs/day3/K5_full.jsonl`（不解除→全驱逐，展示完整单调顺序）、`logs/day3/K5*.log`。对照 `vllm-src` @ `ee0da84` (v0.24.0)。id 随机后缀已去除。

## 1. 实验配置与请求时序
- 引擎：Qwen2.5-0.5B、enforce_eager、`scheduling_policy="priority"`、`enable_prefix_caching=True`、`num_gpu_blocks_override=40`、`max_model_len=512`、`max_num_seqs=4`、`async_scheduling=False`、`VLLM_USE_FLASHINFER_SAMPLER=0`、seed/temp=0。驱动：`scripts/day3_kv_driver.py --scenario K5`（`run_k5` 控制流）。
- 两个新增可控机制（实验性、`VLLM_LAB_BLOCK` 门控、默认关、已回归验证零副作用）：
  1. **priority 调度让 D 成为抢占 victim**：D `priority=10`（数字大=最低优先级），压力请求 `priority=0`；victim=`max(running,key=(priority,arrival))` → D。
  2. **waiting 循环实验性 skip**（`# VLLM_INFERENCE_LAB_EXPERIMENT`）：D 被抢占后放入 blocked 集合，`scheduler` 在 waiting 准入循环跳过它 → D 保持 cached-free，不被重新准入。
- 时序：step0 加 D（`make_prompt(8)`，196 prompt token）→ 独自 prefill+decode 建块；step10 加 P1,P2（priority 0）逼 KV 耗尽；driver 自省检测 `D.status==PREEMPTED` → `set_blocked({D})`；随后每 4 步加一个小压力波 W（priority 0）逼取新块 → 驱逐 D 的 cached 块；`block_step+14` 后 `clear_blocked()` 放行 D 恢复。

## 2. D 抢占前的 block chain（trace 实测）
step46 `allocate_slots_FAILED(req=D, available_blocks=0)` → 抢占 D，free 出其完整链：
```
kv/free req_id=D block_ids=[1,2,3,4,5,6,7,8,9,10,11,12,13, 34, 37]   # 15 个物理块
scheduler/preempt victim_id=D num_computed_tokens_after_reset=0
```
逻辑→物理：逻辑块 i → 物理 id（下标即逻辑序）。头部（逻辑 0..）是 prompt 前缀（物理 1..13），尾部（物理 34,37）是 decode 追加块。各块 hash 覆盖的累计 token（`block_hash_num_tokens`）随逻辑序递增：物理 1→16、2→32、…、13→208、34→224、37→240。

free 时入队顺序（`free_blocks`）：
```
freed_cached_tail=[37, 34, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]   # 尾块 37 最先进队头 → 最先被驱逐
```
即 vLLM 在 `free()` 中**按逆序释放**，使尾部块位于空闲队列前端、最先被复用/驱逐（`block_pool.free_blocks` + `free()` 的 reverse 语义）。

## 3. 实际 eviction 顺序（`K5_full`：不解除，观察全 15 块）
D 被 block 期间持续加压，其 cached 块被逐个 `_maybe_evict_cached_block` 驱逐，**block_hash_num_tokens 严格递减**：
```
blk=37 cov=240   ← 尾部（覆盖最多）最先
blk=34 cov=224
blk=13 cov=208
blk=12 cov=192
blk=11 cov=176
blk=10 cov=160
blk=9  cov=144
blk=8  cov=128
blk=7  cov=112
blk=6  cov=96
blk=5  cov=80
blk=4  cov=64
blk=3  cov=48
blk=2  cov=32
blk=1  cov=16   ← 头部（覆盖最少）最后
```
**15 个块的驱逐严格按覆盖 token 数 240→16 单调递减 = 逻辑块从尾到头**。证明"尾部优先淘汰、头部前缀最后保留"。

## 4. D 恢复时的 prefix hit（`K5`：抢占 step46 后 +14 步解除）
到解除阻塞（step60）时，尾部 8 块已被驱逐（37,34,13,12,11,10,9,8，覆盖 240→128），头部 7 块（物理 1..7，覆盖 16→112）存活。放行后 D 恢复：
```
step62 kv/get_computed_blocks req_id=D num_new_computed_tokens=112 hit_block_ids=[1,2,3,4,5,6,7]
```
D 命中**存活的前 7 个连续头部块**，`num_new_computed_tokens=112=7×16`；随后重算被驱逐的尾部，最终跑完（D 从 requests 移除）。

## 5. 是否证明"尾部优先淘汰、前缀优先保留"
**是，完全证明**：
- 驱逐顺序 = block_hash_num_tokens 严格递减（尾→头），15 块无一例外（K5_full）。
- 抢占 M=8 个尾部块后恢复，命中的是**前 N−M 个连续头部块**（[1..7]），命中 token = (N−M)×block_size = 7×16 = **112**（K5）。
- **未出现**"头部块先被驱逐、但后部块仍被当作有效 prefix hit"的情况：命中集合始终是从逻辑 0 起的**连续前缀**。

机制解释（与 trace 一致）：`free()` 逆序释放使尾块在空闲队列更靠前 → 取新块 `popleft` 时尾块先被 `_maybe_evict_cached_block` 驱逐；prefix 命中要求**从头连续的整块**，故只要头部块存活即命中最长连续前缀。这一设计最大化保留"最可能被复用的公共前缀"。

## 6. 计量与"不硬编码 M"的说明
- 未硬编码淘汰数：M 由"持续加压 + 解除时机"共同决定。`K5_full`（不解除）→ M=N=15（全驱逐）；`K5`（+14 解除）→ M=8、N−M=7 存活。
- free 队列确实可能先消耗其他块：本轮 `available_blocks=0`（KV 全满）时抢占，D 的 15 块进入队列后由压力波逐步取用；trace 中对 D 块的驱逐从尾块 37 开始，说明其他未缓存空闲块此时已耗尽、直接命中 D 的尾块。
- 结论均以 trace 为据，未据源码臆断。

## 7. 与预期对照
| 预期 | 实测 | 结论 |
|---|---|---|
| 尾部逻辑块比头部更早驱逐 | 覆盖 240→16 严格递减驱逐 | ✅ |
| 尾部 M 被驱逐 → 恢复命中前 N−M 连续块 | M=8 → 命中 [1..7] | ✅ |
| 命中 token ≈ (N−M)×block_size | 7×16=112 实测 112 | ✅ |
| 不出现头部先驱逐但后部仍有效命中 | 命中恒为从头连续前缀 | ✅ |

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
VLLM_LAB_BLOCK=1 VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_COMPONENTS=kv,scheduler \
VLLM_LAB_TRACE_MAX_STEPS=400 VLLM_LAB_TRACE_JSONL=outputs/day3/K5.jsonl \
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
python scripts/day3_kv_driver.py --scenario K5 --max-steps 400 2>&1 | tee logs/day3/K5.log
# run_k5 中 `block_step + 14` 控制解除时机：调大 → 淘汰更多（趋近全驱逐 K5_full），调小 → 保留更多头部块
```
