# Day 2 · E5 —— blocked decode + chunked prefill：调度优先级实证

> 直接回答用户问题：**当一个 decode 请求（D）被临时阻塞、让 chunked prefill 请求（P）先跑，解除阻塞后，调度器是继续让 P 的 prefill 进行、还是转去调度恢复的 D 而把 P 暂时搁置？** 用两个场景分别对应两种资源约束，结论均由真实 trace 给出。
> 数据：`outputs/day2/E5.jsonl`、`E5b.jsonl`、`E5_timeline.md`、`E5b_timeline.md`；日志 `logs/day2/E5_blocked_decode.log`、`E5b_one_slot.log`。

## 方法（实验性干预，非被动 trace）

vLLM 无「按请求 block」的公开 API，故新增了一个**默认关闭、env 门控（`VLLM_LAB_BLOCK=1`）、可回退**的实验性 scheduler skip 钩子（`scheduler.py` running 循环内，标 `# VLLM_INFERENCE_LAB_EXPERIMENT`）：当某 request_id 被 driver 标记为 blocked 时，该请求本步被跳过（`req_index+=1; continue`），获得 0 token、仍留在 `self.running`。driver 进程内在 `step()` 之间 `set_blocked/clear_blocked`。**回归已验证**：`VLLM_LAB_BLOCK` 未设时 E1 trace 与之前逐字节一致（除 request_id 随机后缀），即钩子默认零影响。

引擎：Qwen2.5-0.5B，enforce_eager，`max_num_batched_tokens=64`，`async_scheduling=False`，`enable_prefix_caching=False`，`VLLM_USE_FLASHINFER_SAMPLER=0`。请求：`D`=24-token prompt（max_tokens=20）；`P`=288-token prompt（触发 chunked prefill）。时序：step0 加 D → D prefill+decode → **step3 加 P 且仅该步 block D** → step4 起解除。

## 两种资源约束必须分开看
- **max_num_batched_tokens（每步 token 预算）**：决定「一步内」多少 token 被处理，在**已 running 的请求间**分配。
- **max_num_seqs（running 槽位数）**：决定「同时能有几个请求处于 running」，是 waiting→running 的**准入**上限。
E5 让预算成为瓶颈（2 个槽位够用），E5b 让槽位成为瓶颈。

## E5（max_num_seqs=2 → token 预算争用）

| step | D | P | total | running | waiting | 说明 |
|---|---|---|---|---|---|---|
| 1 | P24 | · | 24 | 1 | 0 | D prefill |
| 2 | D1 | · | 1 | 1 | 0 | D decode |
| **3** | **·(blocked)** | **P64** | 64 | 2 | 0 | D 被阻塞跳过 → P 独占全部 64 预算 |
| **4** | **D1** | **P63** | 64 | 2 | 0 | **解除后：D decode 1 + P prefill 63 同一步** |
| 5–6 | D1 | P63 | 64 | 2 | 0 | 持续共享（P computed 127→190→253） |
| 7 | D1 | P35 | 36 | 2 | 0 | P prefill 收尾（64+63+63+63+35=288） |
| 8–10 | D1 | D1 | 2 | 2 | 0 | P 转 decode，两者各 decode 1，P 于 step10 完成 |
| 11–21 | D1 | · | 1 | 1 | 0 | D 独自 decode 到 max_tokens |

**结论（E5）**：解除阻塞后，调度器在**同一步**里**既调度恢复的 decode（D 先拿 1 token），又继续 P 的 prefill（拿剩余 63）**。
- Prefill **不被搁置**；decode **不被推迟**——两者共享该步 64 的 token 预算。
- 顺序：`self.running` 按准入序为 `[D, P]`，running 循环先服务 D（decode 只需 1），再让 P 用剩余预算（63）。这解释了为什么是 **D1 + P63** 而不是 P64+D0 或 D1+P0。
- 对比阻塞步（step3 P64）与解除步（step4 P63）：D 的回归只让 P 少拿 1 个 token，代价极小——这正是 chunked prefill 与 decode 能高效混批的原因。

## E5b（max_num_seqs=1 → running 槽位争用）

| step | D | P | total | running | waiting | 说明 |
|---|---|---|---|---|---|---|
| 1 | P24 | · | 24 | 1 | 0 | D 占用唯一槽位 |
| 2 | D1 | · | 1 | 1 | 0 | D decode |
| **3** | **·(blocked)** | **·** | **0** | **1** | **1** | D 被跳过但**仍占槽位** → P 无法准入，只能 waiting；空步 |
| 4–21 | D1 | · | 1 | 1 | **1** | 解除后 D 继续 decode，**P 始终 waiting**（槽位一直被 D 占） |
| 22 | · | P64 | 64 | 1 | 0 | **D 结束（step21）释放槽位后，P 才被准入** → P64 |
| 23–26 | · | P64/P32 | — | 1 | 0 | P 独自 chunked prefill（64×4+32=288） |
| 27–29 | · | D1 | 1 | 1 | 0 | P decode 到完成 |

**结论（E5b）**：只有一个 running 槽位时，**阻塞 D（跳过调度）并不会让出槽位**给 P。恢复的 decode D 继续占用唯一槽位并跑到结束，P 只能一直 waiting，**直到 D 完成、槽位释放，P 才被准入**。即：
- **阻塞（skip 调度）≠ 抢占（preempt，释放槽位）**。跳过一个 running 请求只是让它本步 0 token，它仍是 running、仍占槽位。
- 槽位受限时，调度器不会为了让 prefill 进来而中止/挪走正在 decode 的请求。

## 一句话总结（回答用户）
- **预算受限、槽位够用（E5）**：恢复 decode 与在跑 prefill **同步进行、共享每步预算**（decode 先拿 1，prefill 拿剩余）；prefill 不被搁置。
- **槽位受限（E5b，只剩一个 running 槽位）**：恢复的 decode **独占槽位继续**，prefill **被搁置在 waiting**，直到 decode 结束释放槽位。

两者结合说明：decode 与 chunked prefill 的"共存"发生在**同一 running 集合内的 token 预算分配**层面；能否共存的前提是**槽位（max_num_seqs）允许它们同时 running**。

## 校验
- 两场景 D、P 均正常完成（E5 driver_steps=21，E5b=29）。
- 钩子默认关闭零影响（E1 回归一致）；scheduler 改动 env 门控、已备份（`backups/day2/scheduler.py.pre_e5`）、可回退。

## 复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
for S in E5 E5b; do
  VLLM_LAB_BLOCK=1 VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_COMPONENTS=scheduler \
  VLLM_LAB_TRACE_JSONL=outputs/day2/$S.jsonl VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/day2_engine_driver.py --scenario $S > logs/day2/${S}.log 2>&1
  python scripts/day2_build_timeline.py outputs/day2/$S.jsonl --title "$S" > outputs/day2/${S}_timeline.md
done
```
