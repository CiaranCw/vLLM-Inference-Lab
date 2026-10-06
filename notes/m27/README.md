# M27 · Prefill–Decode Disaggregation & NIXL KV Transfer（Stage K · Distributed Serving Deep Dive）

**状态：DESIGN FROZEN v1；真实 H20 Gate 0/E0/E1 已通过，E2 A/B 已按统一代理策略完成，E3–E5 尚未完成；M27_E/H = NOT COMPLETE。**
当前机时受限，先生成中转包给下一台机器续跑；原始交接包另行提供。当前数字见 [`results.md`](results.md)，操作及旧策略归档见 [`execution_record.md`](execution_record.md)。M24 冻结计划与结论不改。

| 文件 | 内容 |
|---|---|
| [`pd_disagg_nixl_theory.md`](pd_disagg_nixl_theory.md) | `ee0da84` 源码事实（file:line）、KV 生命周期、fail-closed 依据、KV 理论值、隐藏变量、对 M24 笔记的更正 |
| [`M27_H20_PLAN.md`](M27_H20_PLAN.md) | 冻结设计 v1：目标栈、冻结配置 JSON、A/B 对照、RQ1–RQ4、E0–E5 矩阵、统计口径、降级阶梯、设计调整 |
| [`M27_H20_RUNBOOK.md`](M27_H20_RUNBOOK.md) | H20 执行顺序：Gate 0 → Gate 1(E0) → E1 → E2 → E3 → E4 → Gate 8 → E5 → Package，每步 PASS / STOP |
| [`offline_package.md`](offline_package.md) | NIXL wheel 清单与 sha256、安装方式、三部分交接包（code / wheelhouse / model）、未闭环风险 |
| [`model_artifact_manifest.md`](model_artifact_manifest.md) | 7B 模型身份：逐文件 sha256（[`model_artifact/`](model_artifact/)）、来源证据链、HF revision 比对、模型 Gate |
| [`execution_record.md`](execution_record.md) | 本机 H20 现场执行、失败与偏离、当前有效证据边界 |
| [`results.md`](results.md) | 当前有效 E0/E1 与 E2 待收口的部分实测结果；E3–E5 不做结论 |

## 脚本

| 脚本 | 职责 | 需要 GPU |
|---|---|---|
| `scripts/m27_common.py` | 冻结配置唯一来源、arm/GPU/端口构造、serve 命令、Prometheus 解析、formal 门禁 | 否（仅标准库） |
| `scripts/m27_preflight.sh` | Gate 0：版本、NIXL/UCX、模型 checksum、端口、GPU 进程、拓扑 → gpu_map | 读 `nvidia-smi` |
| `scripts/m27_launch.py` | 每实例一个 `vllm serve`；server_info + 日志双来源回读；生命周期记录 | 是 |
| `scripts/m27_proxy.py` | fail-closed proxy（colocated passthrough / PD），逐请求落盘时间点 | 否 |
| `scripts/m27_workload.py` | E0 正确性、E1 传输探针、E2 事件驱动注入、E3–E5 Poisson 在线、速率校准、1 Hz 指标采样 | 经 proxy |
| `scripts/m27_run_all.py` | 编排、断点续跑（只移动备份，不删除）、gpu_map、transfer test、打包 | 是 |
| `scripts/m27_summarize.py` | raw → CSV/JSON（Bootstrap CI）；缺 raw 时 PENDING | 否 |
| `scripts/m27_trace_analyze.py` | 服务日志证据汇总 | 否 |
| `scripts/m27_figures.py` | fig16–18，离线；缺 CSV 时 PENDING，不画占位图 | 否 |
| `scripts/check_invariants.py --module m27` | 静态项 PASS；raw 相关项 PENDING | 否 |

## 本轮本地验证（不是 H20 证据）

只做了 CPU / 静态检查：py_compile、`bash -n`、manifest 解析、用 pinned 树的 arg parser 解析全部 serve 参数、
用本地 0.5B 快照的 config 离线构造 `VllmConfig` 来核对回读 key、
proxy 在假 P/D 服务上的 fail-closed 自测，以及在 `/tmp` 隔离副本中用**合成数据**测试不变量检查器（能检测出注入的违规）。
这些合成数据从未进入仓库，也不构成任何实验结论。

2026-10-03 pre-H20 closure：找回 M22 的 7B artifact 并冻结身份（不是新下载）；在 `/tmp` 仓库副本中以
`local_preflight` 跑了一次 preflight，只用来验证 §5 的模型校验逻辑（本机 GPU / 版本 / NIXL 项按预期 FAIL），仓库内没有它的产物。
