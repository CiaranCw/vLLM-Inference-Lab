# Day 3 · KV Cache 与 Block 管理

> 全部结论以真实 trace 为依据：`outputs/day3/K{1,2,3,4}.jsonl`、`K2_off.jsonl`、`K4_on.jsonl` 与 `logs/day3/*.log`（component `kv`/`scheduler` 被动钩子，`# VLLM_INFERENCE_LAB_TRACE`，默认关、已验证零副作用），对照 `vllm-src` @ `ee0da84` (v0.24.0)。下文 trace 均已把 request_id 的随机后缀去掉。

## 0. 方法与背景
- 手动驱动 V1 `LLMEngine`（`scripts/day3_kv_driver.py`），Qwen2.5-0.5B、enforce_eager、`async_scheduling=False`、`VLLM_USE_FLASHINFER_SAMPLER=0`、seed/temp=0。
- 钩子（只读）：`kv_cache_manager.get_computed_blocks / allocate_slots / free`、`block_pool.get_new_blocks / free_blocks / _maybe_evict_cached_block`、`scheduler._preempt_request`。
- 基本事实（K1–K3，gpu_util=0.5）：**block_size=16**，物理块 id 从 **1** 起（**块 0 是 null block 占位**）。`req_to_blocks[req]` 的列表下标=逻辑块号，元素=物理块 id。

## 1. K1 —— 块分配 + block_table + 跨块边界（单请求，prompt=24 token）
```
step1 prefill: get_computed_blocks=0 → get_new_blocks [1,2](ref=1) → allocate_slots new=[1,2]   # ⌈24/16⌉=2 块
step2..9 decode(computed 24→31): allocate_slots new=[]                                          # 仍在逻辑块1(16-31)内，不分配
step10 decode(computed_before=32): get_new_blocks [3] → allocate_slots new=[3]                   # 跨入逻辑块2(32-47) → 新物理块
```
**结论**：prefill 一次分配 ⌈prompt/16⌉ 块；decode **只在 position 跨过 16 的整数倍时才分配一个新物理块**（此处 pos 32 → 第 3 块，物理 id=3）。逻辑→物理映射 = `[1, 2, 3]`（逻辑块 0/1/2 → 物理 1/2/3）。这就是 block_table 的本体。

## 2. K2 —— prefix 命中（hash / 共享物理块 / 块对齐）vs 关闭（A、B 相同 prompt=48 token）
prefix caching **ON**：
```
step1 A prefill: get_new_blocks [1,2,3] → allocate_slots new=[1,2,3]        # A 的整块写入 prefix cache
step3 B 到达:    get_computed_blocks num_new_computed_tokens=32 hit_block_ids=[1,2]   # 命中 A 的物理块
               → allocate_slots B num_new_tokens=16 new=[5]                 # 只 prefill 剩余 16，B computed_before=32
```
prefix caching **OFF**（`K2_off`）：
```
step3 B: get_computed_blocks num_new_computed_tokens=0                       # 不命中，B 全量 prefill 48
```
**结论**：
- 命中的是 **A 的物理块 `[1,2]`**——A、B **共享同一份 KV 物理块**（B 通过 `touch` 使其 ref_cnt→2，两请求共用）。这就是 Day2「B 的 prefill 被缩短」的机制。
- **为什么命中 32 而不是 48**：块 hash 只对**整块**计算（`kv_cache_utils`），且 `get_computed_blocks` 用 `max_cache_hit_length = num_tokens-1 = 47`（末 token 必须重算以取 logits）→ `47 // 16 = 2` 个整块 = 32 token。这正解释了 Day2 的 `computed_before=16`（当时共享 1 块）。**非块对齐的尾部 token 不会被缓存/命中**。
- 关闭 prefix caching → 命中恒 0、请求各用各的块、无共享。

## 3. K3 —— 释放 → cached-free → 复用（A 跑完，B 用相同 prompt 后到）
```
step1  A prefill new=[1,2,3]; step2 decode new=[4]
step4  A 完成 → free A block_ids=[1,2,3,4]
       free_blocks: freed_cached_tail=[3,2,1]  freed_evict_first=[4]        # 带 hash 的整块进队尾(保留可复用)，无 hash 的块进队头(优先淘汰)
step6  B(相同 prompt): get_computed_blocks=32 hit_block_ids=[1,2]           # 命中 A 释放但仍 cached 的块 → 复用！
       → get_new_blocks [4]（复用刚释放的 evict-first 块）
```
**结论**：请求结束 `free` 时，`ref_cnt→0` 的块回到空闲队列，但**带 block_hash 的整块进入队尾并保留在 prefix cache**（cached-free 状态），可被后续相同前缀的请求**命中复用**（K3 中 B 复用 A 的 `[1,2]`）；无 hash 的部分块进入队头优先被复用/淘汰（block 4 被 B 重新取用）。

## 4. K4 —— KV 不足 → 抢占 + 重算（`num_gpu_blocks_override=48`, `max_model_len=512`, 4 个独立请求）
> 注：块数须 ≥ ⌈max_model_len/16⌉ 才能通过 init 校验（否则 `_check_enough_kv_cache_memory` 报错，单请求都放不下），故 K4 把 max_model_len 收到 512、块数设 48。

**prefix OFF（`K4`）**：
```
step1  A,B,C,D 各 prefill 152 token（各 10 块，共 40 块）
step2..26 四者一起 decode，块用量增长
step26 available_blocks=0 → allocate_slots_FAILED(req=D) 
       → free D [31..40,44] → preempt victim=D, num_computed_tokens_after_reset=0
       该步 schedule: running=3 waiting=1 preempted=1（A/B/C 继续，D 回到 waiting 头）
step27..  D get_computed_blocks=0（prefix OFF）→ 恢复后从 0 全量重算
step60 A/B/C 完成释放块 → D 重新获得块、跑完（step94）
```
**prefix ON（`K4_on`）**：
```
step26 preempt victim=D, num_computed_tokens_after_reset=0
step27 D get_computed_blocks num_new_computed_tokens=176 hit_block_ids=[31..40,44]   # 命中自己被抢占前 cached 的块！
```
**结论（回答"KV 不足怎么办"）**：
- 触发点：某请求需要新块但 `allocate_slots` 返回 None（`available_blocks=0`）。
- victim 选择：非优先级策略下 `self.running.pop()`（最后进入 running 的，本例 D）。
- 处理：`free` victim 的块、`status=PREEMPTED`、**`num_computed_tokens` 归零**、放回 waiting **队头**；A/B/C 不受影响继续 decode。
- 恢复：victim 从 waiting 重新准入（需有空闲块，本例等 A/B/C 结束）。
  - **prefix OFF**：`get_computed_blocks=0` → **从头全量重算**（丢掉的 176 token 全部重算）。
  - **prefix ON**：`get_computed_blocks=176` → **命中自己仍 cached 的块，几乎不重算**。这是 prefix caching 的一个大价值：**让抢占-恢复的重算代价接近零**。

**两条不变量（study_path 要求）验证**：
1. **活跃请求引用的块不会进入普通 free 状态**：step26 抢占只 `free` 了 victim(D) 的块；A/B/C（ref_cnt>0，仍在 decode）的块保持占用，直到它们 step60 结束才释放。`free_blocks` 只对 ref_cnt→0 的块生效。
2. **被恢复请求不盲目 append 失效的旧 block_table**：`num_computed_tokens_after_reset=0` → 恢复时重新走 `get_computed_blocks` 重建映射（OFF 从 0、ON 从缓存命中），而非续用抢占前的旧块列表。

## 5. 逐条回答 Day3 目标
- **块怎么分配 / block_table**：按 block_size=16，逻辑块 i → 物理块 id（`req_to_blocks`）；prefill 批量分配、decode 跨 16 边界才加块（K1）。
- **prefix 命中机制**：整块算 hash 存入 `cached_block_hash_to_block`；后到请求 `get_computed_blocks` 查最长整块前缀命中、`touch` 共享物理块（ref_cnt++）；命中量受"整块对齐 + 末 token 需重算"限制（K2，解释 Day2 的 16）。
- **释放 / cached-free / 复用 / 驱逐**：`free` 使 ref_cnt→0，带 hash 块留队尾可复用（K3 命中复用）、无 hash 块进队头；取新块时若命中的是 cached 块会 `_maybe_evict_cached_block` 移除旧 hash（驱逐，K4 高压时可见）。
- **KV 不足 / 抢占 / 重算**：allocate 失败→pop victim→free+PREEMPTED+computed 归零→回 waiting；恢复时 OFF 全量重算、ON 靠自身缓存几乎免算（K4/K4_on）。

## 6. 与 Day1/2 的衔接
- Day1 见到 `kv_cache=T(34801,2,16,2,64)` 的 block_size=16、block_table=T(1,128)——Day3 把"块"从静态 shape 变成了可观察的**分配/共享/释放/抢占**动态。
- Day2 的 `computed_before=16`（前缀命中）在 K2 得到机制级解释（整块 hash + 命中对齐 + `num_tokens-1`）。

## 7. 未解决 / 下一步（Day4 起）
- Attention backend 如何消费 block_table + slot_mapping（Day4）。
- 优先级调度策略下 victim 选择（本轮为默认 FIFO `pop()`）。
- 前缀缓存跨请求命中率对吞吐/TTFT 的量化影响（留待性能日 Day6）。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
COMMON="VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_COMPONENTS=kv,scheduler VLLM_LAB_TRACE_MAX_STEPS=300 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0"
env $COMMON VLLM_LAB_TRACE_JSONL=outputs/day3/K1.jsonl python scripts/day3_kv_driver.py --scenario K1 |& tee logs/day3/K1.log
env $COMMON VLLM_LAB_TRACE_JSONL=outputs/day3/K2.jsonl python scripts/day3_kv_driver.py --scenario K2 |& tee logs/day3/K2.log
env $COMMON VLLM_LAB_TRACE_JSONL=outputs/day3/K2_off.jsonl python scripts/day3_kv_driver.py --scenario K2 --no-enable-prefix-caching |& tee logs/day3/K2_off.log
env $COMMON VLLM_LAB_TRACE_JSONL=outputs/day3/K3.jsonl python scripts/day3_kv_driver.py --scenario K3 |& tee logs/day3/K3.log
env $COMMON VLLM_LAB_TRACE_JSONL=outputs/day3/K4.jsonl python scripts/day3_kv_driver.py --scenario K4 |& tee logs/day3/K4.log
env $COMMON VLLM_LAB_TRACE_JSONL=outputs/day3/K4_on.jsonl python scripts/day3_kv_driver.py --scenario K4 --enable-prefix-caching |& tee logs/day3/K4_on.log
```
