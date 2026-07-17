# Day 2 · Trace 改动清单

Day 2 **未改动任何 vLLM 源文件**（Day1 的 4 个插桩文件保持原状），仅对自建的 `vllm/_lab_trace.py` 做了两处**向后兼容、默认关闭**的增量，并新增了 lab 侧脚本。

## `vllm-src/vllm/_lab_trace.py` 增量（Day1→Day2）

| 新增 | 环境变量 | 默认 | 作用 |
|---|---|---|---|
| 组件过滤 | `VLLM_LAB_TRACE_COMPONENTS` | 空=不过滤 | 逗号分隔，如 `scheduler`：只发这些 component 的事件（Day2 日志聚焦调度，屏蔽 attention 噪声） |
| JSONL 输出 | `VLLM_LAB_TRACE_JSONL` | 空=不写 | 事件同时以 JSON 行追加到该文件，得到机器可读的 `scheduler_steps` 供时间线分析 |

- 均在 `trace_event` 内生效；`VLLM_LAB_TRACE` 关闭时二者都不触发（自检：trace off 不建 JSONL 文件、无输出）。
- 备份：`backups/day2/_lab_trace.py.day1`（Day1 版本）。
- 关闭：不设上述变量即为 Day1 行为。删除：还原 `_components/_jsonl_path/_jsonable` 三个函数与 `trace_event` 内两处调用即可。

## lab 侧新增脚本（非 vLLM 源码）
- `scripts/day2_engine_driver.py`：手动驱动 V1 `LLMEngine`，按到达表注入请求观察 continuous batching。
- `scripts/day2_build_timeline.py`：读 `E*.jsonl` 生成 Markdown 时间线表。

## git 状态
Day2 前后 `vllm-src` 状态**完全一致**：21 个 `M`（Day1 注释+插桩）+ 1 个 `?? vllm/_lab_trace.py`。Day2 仅改 `_lab_trace.py` 内容（仍为 untracked，文件集合不变）。未执行 `git add/commit/stash`，未删用户注释，未编译 C++/CUDA，未改 baseline venv，未新增 Python 依赖。

## E5 追加：实验性「按请求 block」钩子（**会改变调度逻辑**，与被动 trace 区分）

为实现 E5（blocked decode）需要 vLLM 没有的「按请求跳过」能力，新增一个**默认关闭、env 门控、可回退**的实验性钩子：

| 位置 | 标记 | 门控 | 作用 |
|---|---|---|---|
| `vllm/v1/core/sched/scheduler.py` running 循环（`request = self.running[req_index]` 之后） | `# VLLM_INFERENCE_LAB_EXPERIMENT` | `VLLM_LAB_BLOCK=1` + id 命中 | 跳过被标记的 running 请求（`req_index+=1; continue`），本步 0 token、仍留 running |
| `vllm/_lab_trace.py` | — | 同上 | `_BLOCKED` 集合 + `set_blocked/clear_blocked/blocked_ids/block_enabled`；`blocked_ids()` 仅在 `VLLM_LAB_BLOCK=1` 时非空 |

- **与被动 trace 的区别**：Day1/2 的 `# VLLM_INFERENCE_LAB_TRACE` 只读不改逻辑；此 `# VLLM_INFERENCE_LAB_EXPERIMENT` **会改变调度**（跳过一个请求），故单独标注。
- **默认零影响**：未设 `VLLM_LAB_BLOCK` 时 `blocked_ids()` 返回空集，`request.request_id in set()` 恒 False → 从不跳过。**已回归验证**：`VLLM_LAB_BLOCK` 未设时 E1 trace 与之前一致（仅 request_id 随机后缀不同）。
- **备份/回退**：`backups/day2/{scheduler.py.pre_e5, _lab_trace.py.pre_e5}`。删除：移除 scheduler 中标 `# VLLM_INFERENCE_LAB_EXPERIMENT` 的代码块 + `_lab_trace.py` 的 blocked API。
- **git 状态**：E5 前后文件集合不变（scheduler.py 本就 `M`，_lab_trace.py 本就 `??`）；未 `git add/commit/stash`。
- driver 侧：`scripts/day2_engine_driver.py` 增加 E5/E5b 场景与 block 计划（进程内 `set_blocked/clear_blocked`，用 `add_request` 返回的内部后缀 id 精确匹配）。

