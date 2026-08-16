# M24 · 模块入口　✅ **T COMPLETE**（2026-08-14）　✅ **DCP E+O+H COMPLETE**（2026-08-16 · 4×H20）　⬜ Ring / P-D E

| 文件 | 内容 | 状态 |
|---|---|---|
| [`results.md`](results.md) | DCP 实测结论（唯一变量；含未做项） | ✅ E |
| [`context_parallel_ring_disagg.md`](context_parallel_ring_disagg.md) | 理论 + v0.24.0 源码路径 + 性能模型 | ✅ T |
| [`M24_H20_PLAN.md`](M24_H20_PLAN.md) | 4×H20 冻结矩阵（唯一变量 DCP；模型必须 1.5B） | 已执行 |
| [`H20_RUNBOOK.md`](H20_RUNBOOK.md) | 上机命令顺序 | 冻结 |
| [`REMOTE_AGENT_PROMPT.md`](REMOTE_AGENT_PROMPT.md) | 给内网新机器 Agent 的整份提示 | 冻结 |
| [`M24_EXECUTION_FREEZE.md`](M24_EXECUTION_FREEZE.md) | 上机前拷贝包快照 | 冻结 |

图：[`figures/fig15_m24_dcp.png`](../../figures/fig15_m24_dcp.png)（由 CSV 离线生成）。  
不变量：`python scripts/check_invariants.py --module m24` → 13 PASS。

**面试纪律**：**在 vLLM 里看到 CP 配置 ≠ 掌握 Ring Attention。**
