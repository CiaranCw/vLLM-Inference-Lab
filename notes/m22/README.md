# M22 · 模块入口

| 文件 | 内容 |
|---|---|
| [`tensor_parallel.md`](tensor_parallel.md) | **T+O**：TP 理论 + v0.24.0 源码定位（R1–R7，带 `file:line`） |
| [`results.md`](results.md) | **E+H**：4×H20 实测结论（D1/D2/D3 + P1），只放结果与边界 |
| [`M22_EXECUTION_FREEZE_MANIFEST.md`](M22_EXECUTION_FREEZE_MANIFEST.md) | 上机前冻结的矩阵 / 成功线 / 禁止项（审计留痕） |
| [`execution_record.md`](execution_record.md) | 实际执行环境、产物清单、复现命令（非操作剧本） |
| [`frozen_runtime_snapshot/`](frozen_runtime_snapshot/) | 与 Freeze 对齐的 invariants / repro 快照 |

**范围（诚实）**：本模块完成的是 **Tensor Parallel + collective communication**。  
**PP / DP / EP 未实现、未实测，仍为 ⬜。** CP / Ring / P-D 的 **T** 见 [`../m24/`](../m24/)（E 仍 ⬜）。

**方案原文**：[`../../MULTI_GPU_EXPERIMENT_PLAN.md`](../../MULTI_GPU_EXPERIMENT_PLAN.md)
