# M27 · Prefill–Decode Disaggregation & NIXL KV Transfer（Stage K · Distributed Serving Deep Dive）

**状态：M27_H20_FORMAL_COMPLETE；M27_E/H = COMPLETE（单节点 8×H20 环境、Gate 0→E4 fresh 同实例，E0–E4 COMPLETE 为最终范围；正式实验覆盖 1/2/4 GPU，最大正式并发 4 GPU，P:D ratio 为 4-GPU E4；E5（可选 8-GPU）未执行，没有 8-GPU 实验；P1 profiler 未做）。2026-10-06 独立 post-hoc 审计 `M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`，修正已闭环（[`FINAL_CLOSURE_ADDENDUM.md`](FINAL_CLOSURE_ADDENDUM.md)）。**
正式结果与负结果见 [`results.md`](results.md) 顶部；上一台独立中转数字只作为其附录参考，不和本机 raw 池化。实例 ID `ins-a8qac2ve`，与上一实例意外同名 `TENCENT64.site`，以任务起点和新 raw/校准产生时间隔离证据。M24 冻结计划与结论不改。

| 文件 | 内容 |
|---|---|
| [`pd_disagg_nixl_theory.md`](pd_disagg_nixl_theory.md) | `ee0da84` 源码事实（file:line）、KV 生命周期、fail-closed 依据、KV 理论值、隐藏变量、对 M24 笔记的更正 |
| [`M27_H20_PLAN.md`](M27_H20_PLAN.md) | 冻结设计 v1：目标栈、冻结配置 JSON、A/B 对照、RQ1–RQ4、E0–E5 矩阵、统计口径、降级阶梯、设计调整 |
| [`M27_H20_RUNBOOK.md`](M27_H20_RUNBOOK.md) | H20 执行顺序：Gate 0 → Gate 1(E0) → E1 → E2 → E3 → E4 → Gate 8 → E5 → Package，每步 PASS / STOP（实际执行到 E4，E5 未执行） |
| [`offline_package.md`](offline_package.md) | NIXL wheel 清单与 sha256、安装方式、三部分交接包（code / wheelhouse / model）、未闭环风险 |
| [`model_artifact_manifest.md`](model_artifact_manifest.md) | 7B 模型身份：逐文件 sha256（[`model_artifact/`](model_artifact/)）、来源证据链、HF revision 比对、模型 Gate |
| [`execution_record.md`](execution_record.md) | H20 现场执行（§1–§9 前一实例历史、§10 当前实例）、失败与偏离、§11 post-hoc 修正 |
| [`results.md`](results.md) | 当前新实例 E0–E4 完整正式结果、RQ1–RQ4、负结果、口径/计费/证据存放；旧机独立附录与 Gate 8/E5 判定 |
| [`FINAL_CLOSURE_ADDENDUM.md`](FINAL_CLOSURE_ADDENDUM.md) | H20 原始最终报告与 post-hoc 修正两层；F-M01…F-M07 逐条闭环 |
| [`H20_COMMIT_PROVENANCE.md`](H20_COMMIT_PROVENANCE.md) | H20 原提交 `884503dc` / `c4db4561` 与规范化提交的对应、内容关系、超大 raw 排除策略 |
| [`H20_RAW_ARCHIVE_MANIFEST.tsv`](H20_RAW_ARCHIVE_MANIFEST.tsv) | 11 个 >100MB raw 的外部证据身份（size / sha256 / H20 原 blob） |
| [`audit/`](audit/) | 独立 post-hoc 审计报告与 requirement / headline / rebuild / findings 表 |

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
| `scripts/m27_final_evidence_audit.py` | 离线重算 final evidence audit 73 项并附证据；`--check` 与落盘结果比对；外部 raw 缺失时 INCOMPLETE | 否 |
| `scripts/check_invariants.py --module m27` | 静态 + raw/派生全链检查；外部 raw 不在本地时相关项 PENDING（不算通过） | 否 |

## H20 之前的本地验证（历史，不是 H20 证据）

只做了 CPU / 静态检查：py_compile、`bash -n`、manifest 解析、用 pinned 树的 arg parser 解析全部 serve 参数、
用本地 0.5B 快照的 config 离线构造 `VllmConfig` 来核对回读 key、
proxy 在假 P/D 服务上的 fail-closed 自测，以及在 `/tmp` 隔离副本中用**合成数据**测试不变量检查器（能检测出注入的违规）。
这些合成数据从未进入仓库，也不构成任何实验结论。

2026-10-03 pre-H20 closure：找回 M22 的 7B artifact 并冻结身份（不是新下载）；在 `/tmp` 仓库副本中以
`local_preflight` 跑了一次 preflight，只用来验证 §5 的模型校验逻辑（本机 GPU / 版本 / NIXL 项按预期 FAIL），仓库内没有它的产物。
