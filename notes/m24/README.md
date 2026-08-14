# M24 · 模块入口　✅ **T COMPLETE**（2026-08-14）　⬜ **E/H 多卡门控**

| 文件 | 内容 | 状态 |
|---|---|---|
| [`context_parallel_ring_disagg.md`](context_parallel_ring_disagg.md) | Context Parallel / Ring Attention / Prefill-Decode Disaggregation：理论、v0.24.0 源码路径、性能模型 | ✅ T |

**本轮只做 T。E/H 等下一次多卡机**，不用单卡模拟后写成落地。

**面试纪律（ROADMAP 原文）**：**在 vLLM 里看到 CP 配置 ≠ 掌握 Ring Attention。**
本树的 Decode CP 是 Q-allgather + 本地 KV shard + LSE 合并，**不是**环形传 KV。
