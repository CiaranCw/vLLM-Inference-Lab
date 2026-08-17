### 时间线 · E1 单请求基线

数据：`outputs/m2/E1.jsonl`；单元格 `P{n}`=prefill 调度 n token，`D{n}`=decode 调度 n token，`·`=该请求本步未被调度。

| step | R0 | total_sched | running | waiting | 事件 |
|---|---|---|---|---|---|
| 1 | P24 | 24 | 1 | 0 | ＋R0 |
| 2 | D1 | 1 | 1 | 0 |  |
| 3 | D1 | 1 | 1 | 0 |  |
| 4 | D1 | 1 | 1 | 0 |  |
| 5 | D1 | 1 | 1 | 0 |  |
| 6 | D1 | 1 | 1 | 0 |  |
| 7 | D1 | 1 | 1 | 0 |  |
| 8 | D1 | 1 | 1 | 0 | ✓R0 |

#### 请求生命周期摘要

| request | num_prompt | 首次调度 step | 末次调度 step | 跨 step 数 |
|---|---|---|---|---|
| R0 | 24 | 1 | 8 | 8 |
