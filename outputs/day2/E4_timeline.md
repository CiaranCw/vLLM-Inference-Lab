### 时间线 · E4 chunked prefill (max_num_batched_tokens=64)

数据：`outputs/day2/E4.jsonl`；单元格 `P{n}`=prefill 调度 n token，`D{n}`=decode 调度 n token，`·`=该请求本步未被调度。

| step | LONG | total_sched | running | waiting | 事件 |
|---|---|---|---|---|---|
| 1 | P64 | 64 | 1 | 0 | ＋LONG |
| 2 | P64 | 64 | 1 | 0 |  |
| 3 | P64 | 64 | 1 | 0 |  |
| 4 | P64 | 64 | 1 | 0 |  |
| 5 | P32 | 32 | 1 | 0 |  |
| 6 | D1 | 1 | 1 | 0 |  |
| 7 | D1 | 1 | 1 | 0 |  |
| 8 | D1 | 1 | 1 | 0 | ✓LONG |

#### 请求生命周期摘要

| request | num_prompt | 首次调度 step | 末次调度 step | 跨 step 数 |
|---|---|---|---|---|
| LONG | 288 | 1 | 8 | 8 |
