# M2 · Scheduler 与 Continuous Batching

> 全部结论以真实运行日志为依据：`outputs/m2/E{1..4}.jsonl`（scheduler 事件）、`logs/m2/E*.log`、`outputs/m2/E*_timeline.md`（由 `scripts/m2_build_timeline.py` 生成），对照 `vllm-src` @ `ee0da84` (v0.24.0) 源码。

## 1. 目标与方法

M1 只看了单请求的 prefill→decode。M2 观察**多请求在同一引擎里如何共存、错峰到达、排队、交错 decode、以及长 prompt 如何被切分**。

方法（关键决策，见 `scripts/m2_engine_driver.py`）：
- **手动驱动 V1 引擎**：`LLMEngine.from_engine_args(EngineArgs(...), usage_context=ENGINE_CONTEXT, enable_multiprocessing=False)`（`vllm/v1/engine/llm_engine.py:211`），按「到达表」在指定 step 调 `add_request(rid, prompt, SamplingParams)`（:268）注入请求，`step()`（:368）逐步驱动，`has_unfinished_requests()`（:238）控制结束。offline `LLM.generate` 会一次性提交全部 prompt，无法表达到达时序，故不用。
- **`async_scheduling=False`**：每次 `step()` = 一次干净的 schedule→execute→update（无批队列流水线），便于逐 step 观察。（这也回答 M1 遗留问题：M1 默认 async 调度+batch queue=2 是流水线优化，本轮关掉后 step 序列更直观，调度语义一致。）
- **调度上限**：`max_num_seqs=2`（`config/scheduler.py:63`，默认 128）逼出排队；E4 用 `max_num_batched_tokens=64`（默认因 chunked prefill 被设为 8192）逼出 chunked prefill。
- **复用 M1 Trace**：`scheduler/schedule` 钩子已输出 `total_scheduled/n_new/n_cached/running/waiting/preempted/finished/per_req{scheduled,computed_before,num_prompt,phase}`；本轮仅给 `_lab_trace.py` 加了 `VLLM_LAB_TRACE_COMPONENTS`（只发 scheduler）与 `VLLM_LAB_TRACE_JSONL`（结构化输出），**未再改动 4 个 vLLM 源文件**。
- 采样器：见下 nvcc 结论，本轮仍用 `VLLM_USE_FLASHINFER_SAMPLER=0`（原生采样器）。固定 model=Qwen2.5-0.5B / seed=0 / temperature=0 / enforce_eager。
- **step 编号**：trace 的 `step` 从 1 开始（在 `Scheduler.schedule()` 递增），比 driver 的 step_idx 偏移 +1；下文表格用 trace step。

## 2. nvcc / FlashInfer 采样器 实测结论（据实，未伪造）

nvcc 12.8 已装（`/usr/local/cuda`），但：
- 仅装 nvcc **不足以**让 FlashInfer 采样器可用。设 `CUDA_HOME=/usr/local/cuda`（+ `PATH`，甚至 `TORCH_CUDA_ARCH_LIST=12.0`）后，`get_cuda_path` 不再报错，但 JIT 立即在 `flashinfer/jit/core.py:96 check_cuda_arch` 抛 `RuntimeError: FlashInfer requires GPUs with sm75 or higher`。
- 根因：`check_cuda_arch` 遍历 `current_compilation_context.TARGET_CUDA_ARCHS`，只要有 `major>=8` 即合格；RTX 5080 是 (12,0) 本应合格，报错说明**该集合为空**——即 flashinfer 0.6.12 未能为 sm_120 探测出目标架构（torch cu130 与 flashinfer/nvcc 12.8 的 arch 识别不一致）。
- 结论：**本机 sm_120 上 FlashInfer 采样器仍不可用**，M2 继续 `VLLM_USE_FLASHINFER_SAMPLER=0`。此为 flashinfer 版本/arch 识别问题，深挖超出 M2 范围（留作后续）。日志：`logs/m2/nvcc_flashinfer_check.log`。

## 3. 四场景时间线与解读

单元格：`P{n}`=prefill 本步调度 n token；`D{n}`=decode 本步调度 n token；`·`=本步未调度。

### E1 单请求（基线，验证手动驱动与 M1 一致）
R0（num_prompt=24）step1 prefill(P24)，step2–8 decode(D1)，running 恒为 1，waiting 恒为 0，与 M1 的 prefill→decode 结构一致 → 手动驱动路径可信。

### E2 错峰 2 请求、长度不同（**continuous batching 核心**）
A_long（num_prompt=96）step0 到达；B_short（num_prompt=24）晚到（driver step3 → 首次调度在 trace step4）。

| step | A_long | B_short | total | running | waiting | 事件 |
|---|---|---|---|---|---|---|
| 1 | P96 | · | 96 | 1 | 0 | ＋A_long |
| 2–3 | D1 | · | 1 | 1 | 0 | |
| **4** | **D1** | **P8** | **9** | **2** | 0 | ＋B_short |
| 5–15 | D1 | D1 | 2 | 2 | 0 | ✓B_short(15) |
| 16–24 | D1 | · | 1 | 1 | 0 | ✓A_long(24) |

- **step 4 一个 batch 内同时有 A 的 decode(D1) 和 B 的 prefill(P8)** —— 这就是 continuous batching 的本质：新请求的 prefill 直接插进正在 decode 的 batch，不必等 A 跑完。`total_scheduled=9` = A 的 1 + B 的 8。
- steps 5–15 稳定态：两请求同批各 decode 1 token（running=2）。B 先结束（12 tokens），A 独自 decode 到 24。
- **附带发现（prefix caching）**：B_short 首次调度是 `P8` 而非 `P24`。核对 JSONL：`scheduled=8, computed_before=16, num_prompt=24`。因 B 的 prompt 是 A 的前缀（都以同一句开头），命中 **1 个 block=16 token（block_size=16，M1 实测）** 的前缀缓存，只需 prefill 剩余 8 个。这不是「B 短所以 prefill 短」，而是前缀缓存把它进一步缩短了（M3 深入）。

### E3 四请求 + max_num_seqs=2（**排队 / 准入**）
A、B（各 num_prompt=24）step0 到达；C、D（各 24）driver step2 到达（trace step3）。

| step | A | B | C | D | total | running | waiting | 事件 |
|---|---|---|---|---|---|---|---|---|
| 1 | P24 | P8 | · | · | 32 | 2 | 0 | ＋A ＋B |
| 2 | D1 | D1 | · | · | 2 | 2 | 0 | |
| **3–12** | D1 | D1 | · | · | 2 | **2** | **2** | (C、D 到达即排队) |
| 12 | D1 | D1 | · | · | 2 | 2 | 2 | ✓A ✓B |
| **13** | · | · | **P8** | **P8** | 16 | 2 | 0 | ＋C ＋D |
| 14–24 | · | · | D1 | D1 | 2 | 2 | 0 | ✓C ✓D(24) |

- **running 恒为上限 2**；C、D 在 trace step3 进入 waiting，`waiting=2` 持续到 step12。
- **step12 A、B 结束 → step13 C、D 才被准入**（waiting 2→0，running 仍 2）。这直接对应 `scheduler.py` 等待队列准入门槛：`if len(self.running) == self.max_num_running_reqs: break`（running 满则不接纳新请求）。
- B、C、D 的首次 prefill 都是 P8（computed_before=16）—— 与 A 同 prompt，命中前缀缓存（同 E2）。

### E4 chunked prefill（长 prompt 被切分）
LONG（num_prompt=288）+ `max_num_batched_tokens=64`。

| step | LONG | total | running | waiting |
|---|---|---|---|---|
| 1–4 | **P64** | 64 | 1 | 0 |
| 5 | **P32** | 32 | 1 | 0 |
| 6–8 | D1 | 1 | 1 | 0 | ✓(8) |

- prefill 被切成 **64+64+64+64+32 = 288**，跨 5 个 step，每步受 `token_budget=max_num_batched_tokens=64` 限制（`scheduler.py` 运行循环 `num_new_tokens = min(剩余, token_budget)`）。核对：这 5 步同一请求 `phase=prefill`、`computed_before` 递增 0→64→128→192→256，直到追平 num_prompt=288，随后转 decode。
- 说明单个长 prompt 不会一次性吃满一步、阻塞其他请求——这正是 chunked prefill 的意义（M1 因 prompt 仅 11 token 未触发）。

## 4. 关键机制总结（回答 M2 观察点）

- **请求进入/退出**：`add_request` 进 waiting；被 schedule 准入后转 running（首次出现在 per_req）；产出末 token 后退出（末次出现）。见各表「事件」列。
- **每步 token 预算被谁消耗**：`total_scheduled` = 各请求 `scheduled` 之和；prefill 吃大块（P96/P64），decode 每请求各 1。
- **prefill/decode 交错**：E2 step4 同批 D1+P8 —— 同一 step 内既有 decode 又有 prefill。
- **running/waiting 动态**：E3 running 封顶=2，超出的请求 waiting 排队直到有空位。
- **首 token 时刻（TTFT 直觉）**：请求首次被 schedule（prefill 那步）产出首 token；E3 中 C/D 因排队，首 token 被推迟到 step13。
- **是否重排**：running 优先、waiting 其次；未见抢占（preempted 恒 0，KV 充足）。
- **chunked prefill**：E4 长 prompt 按 token_budget 分 5 步。
- **prefix caching（附带）**：相同前缀的后到请求，prefill 被缓存命中缩短（computed_before=16=1 block）。

## 5. 验收：只看 Trace/时间线能否解释调度？
可以。例如 E3 step13 为何突然调度 C、D：因为 step12 A、B 结束使 running 从 2 降到 0，等待队列（waiting=2）在 running 出现空位后被准入——完全由 `running/waiting/finished/n_new` 与 max_num_seqs=2 解释，无需读源码即可复述。

## 6. 与 M1 / 静态理解的对应与修正
- **验证**：`max_num_seqs` 封顶 running、waiting 排队准入、token_budget 切 prefill，均与源码逻辑一致。
- **修正/补充**：
  1. continuous batching 不是「一个请求跑完再跑下一个」，而是**新请求 prefill 与旧请求 decode 同批混跑**（E2 step4 铁证）。
  2. `async_scheduling` 默认开（M1）是**流水线优化**（batch queue=2），关掉后每 step 一次 schedule，观察更清晰，调度决策不变。
  3. 「prompt 短 → prefill 短」不严谨：**前缀缓存**也会缩短 prefill（E2/E3 的 P8 来自 16-token 缓存命中，而非 prompt 本身只有 8）。
  4. chunked prefill 下，单请求可连续多步处于 prefill，`scheduled < 剩余 prompt`，`computed_before` 递增。

## 7. 未解决问题（留 M3：KV Cache 与 Block 管理）
1. 前缀缓存的 block hash / ref_cnt / 命中判定细节（E2/E3 的 16-token 命中如何产生、跨并发请求如何共享）。
2. block_table 的逻辑块→物理块映射、block 的分配/释放/复用（请求结束后 block 去向）。
3. KV 不足时的 preemption（本轮 KV 充足未触发，preempted 恒 0）。
4. `max_num_batched_tokens` 与 `long_prefill_token_threshold` 对多请求混合 prefill/decode 的联合影响。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
for S in E1 E2 E3 E4; do
  VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_COMPONENTS=scheduler VLLM_LAB_TRACE_MAX_STEPS=200 \
  VLLM_LAB_TRACE_JSONL=outputs/m2/$S.jsonl VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m2_engine_driver.py --scenario $S 2>&1 | tee logs/m2/$S.log
  python scripts/m2_build_timeline.py outputs/m2/$S.jsonl --title "$S" > outputs/m2/${S}_timeline.md
done
```
