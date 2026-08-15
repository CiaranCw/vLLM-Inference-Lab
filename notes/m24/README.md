# M24 · 模块入口　✅ **T COMPLETE**（2026-08-14）　⬜ **E/H 待 4×H20**

| 文件 | 内容 | 状态 |
|---|---|---|
| [`context_parallel_ring_disagg.md`](context_parallel_ring_disagg.md) | 理论 + v0.24.0 源码路径 + 性能模型 | ✅ T |
| [`M24_H20_PLAN.md`](M24_H20_PLAN.md) | 4×H20 冻结矩阵（唯一变量 DCP；模型必须 1.5B） | 冻结 |
| [`H20_RUNBOOK.md`](H20_RUNBOOK.md) | 上机命令顺序 | 冻结 |
| [`REMOTE_AGENT_PROMPT.md`](REMOTE_AGENT_PROMPT.md) | 给内网新机器 Agent 的整份提示 | 冻结 |

**拷贝包位置（本机）**：`/home/ciaran/m24_h20_transfer/`  
把该目录整份拷到 H20 后，把 `REMOTE_AGENT_PROMPT.md` 交给远端 Agent。

**面试纪律**：**在 vLLM 里看到 CP 配置 ≠ 掌握 Ring Attention。**
