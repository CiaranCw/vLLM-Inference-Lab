# M27 · 结果（模板）

> **status = NOT EXECUTED**。没有任何 H20 formal raw（`outputs/m27/raw/h20_formal_*.jsonl` 不存在），
> 因此本文件**不含任何实测数字**。本地（RTX 5080 单卡、没有 NIXL）没有执行 P-D smoke，也没有 `local_preflight` 数据。
> 填写规则：每个数字都必须来自 `outputs/m27/` 的 CSV/JSON，并注明文件名；结论都要带适用边界
> （H20 / TP1 每实例 / Qwen2.5-7B bf16 / `enforce_eager` / 在线客户端口径 / 本次实测拓扑）。

## 1. Environment

NOT EXECUTED。来源将是 `outputs/m27/configs/preflight_h20_formal.json`、`gpu_map_h20_formal.json`，以及
[`execution_record.md`](execution_record.md) §2。

## 2. Formal configuration

冻结配置见 [`M27_H20_PLAN.md`](M27_H20_PLAN.md) §2（v1）。生效配置将取自 `outputs/m27/configs/h20_formal_*_effective.json`
（server_info + 日志双来源回读）。NOT EXECUTED。

## 3. E0 correctness

NOT EXECUTED。来源：`outputs/m27/h20_formal_e0_gate.json`、`h20_formal_log_evidence.json`。

| 判据 | 结果 |
|---|---|
| D external_kv_transfer == prompt tokens sent | NOT EXECUTED |
| D local_compute == 0 / P local_compute == sent | NOT EXECUTED |
| NIXL transfers ≥ 5 / failed == 0 | NOT EXECUTED |
| fault probe 502 且 D 未收到请求 | NOT EXECUTED |
| 日志：compat pass / 无 KV load failure / 无 Traceback / KV layout = HND | NOT EXECUTED |
| UCX 实际传输层 | NOT EXECUTED |

## 4. E1 KV transfer scaling（RQ2）

NOT EXECUTED。来源：`outputs/m27/h20_formal_e1_transfer.csv`、`figures/fig16_m27_e1_transfer_h20_formal.png`。

| context | 理论 bytes（block） | NIXL 实测 bytes | 实测/理论 | xfer ms | post ms | GB/s | 客户端 TTFT | proxy P 阶段 | D 首字节 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 29,360,128 | NOT EXECUTED | | | | | | | |
| 2048 | 117,440,512 | NOT EXECUTED | | | | | | | |
| 4096 | 234,881,024 | NOT EXECUTED | | | | | | | |
| 8192 | 469,762,048 | NOT EXECUTED | | | | | | | |
| 16384 | 939,524,096 | NOT EXECUTED | | | | | | | |

（理论列是公式计算值，不是测量值。）

## 5. E2 interference isolation（RQ1）

> 证据性质：**mechanism / interference-isolation**。colo1（1 卡）vs 1P1D（2 卡），不是同 GPU 预算的部署比较。

NOT EXECUTED。来源：`outputs/m27/h20_formal_e2_interference.csv`、`h20_formal_e2_effect.csv`、
`figures/fig17_m27_e2_interference_h20_formal.png`。

| p_len | arm | max stall 中位 | D ITL p99 中位 | 注入 TTFT 中位 | colo − pd 差值 [95% CI] |
|---:|---|---:|---:|---:|---|
| 0 / 2048 / 8192 / 16384 | colo1_mbt{2048,4096,8192} / pd_1p1d | NOT EXECUTED | | | |

## 6. E3 online workload（RQ3）

NOT EXECUTED。来源：`outputs/m27/h20_formal_e3_online.csv`、`h20_formal_e3_effect.csv`、`configs/h20_formal_e3cal_rates.json`、
`figures/fig18_m27_e3_online_h20_formal.png`。主比较是 `pd_1p1d` vs `colo2_*`（GPU 预算相同，**equal-GPU-budget deployment comparison**）。
goodput 门槛 = m × `colo2_mbt8192` 校准时的无负载中位（共享、冻结）；必须同时报告绝对 TTFT / ITL / E2E。

## 7. E4 P:D resource ratio（RQ4）

NOT EXECUTED。来源：`outputs/m27/h20_formal_e4_online.csv`、`h20_formal_e4_effect.csv`。

## 8. 8-GPU confirmation

NOT EXECUTED（Gate 8 尚未评估；M22 时容器只看到 4 张 GPU）。

## 9. Profiler evidence

NOT EXECUTED（P1 未实现）。在没有 timeline 证据之前，“NIXL 异步 READ 与 decode 计算重叠”只能写“未验证”。

## 10. Failed hypotheses

NOT EXECUTED。（要求：如实列出被数据否定的假设，例如“PD 降低 E3 的 ITL p99”若 CI 含 0 或方向相反。）

## 11. Limitations

- 单机 H20、每实例 TP1、Qwen2.5-7B bf16、`enforce_eager`（ITL 绝对值偏高）、客户端在线口径；不外推到跨节点、TP>1 或 compiled 模式。
- arm 顺序执行，时间漂移是伴随变量。
- B 层（调优后的 colocated）比较不是单变量对照。
- E2 是机制实验（colo1 = 1 卡，1P1D = 2 卡），只回答干扰是否被隔离；同 GPU 预算的部署结论只来自 E3（colo2 vs 1P1D）和 E4。
- goodput 门槛来自校准 arm 的共享基线（同一实验内所有 arm 门槛相同），只在同一实验内比较；主结论以绝对 TTFT / ITL / E2E 为准。
- M27 formal conclusions are scoped to HND layout。M22 / M24 未设布局（按源码默认推断为 NHD）且是 in-process 口径，
  绝对延迟只作背景，不与 M27 相减，也不归因于并行策略。
- 模型身份由 `notes/m27/model_artifact/qwen2.5-7b-instruct.sha256` 冻结（与 M22 下载时生成的 manifest 逐行相同；
  恢复副本被认为就是 M22 artifact，但没有密码学证据证明它与 M22 在 H20 上使用的那份逐位相同），
  见 [`model_artifact_manifest.md`](model_artifact_manifest.md)。H20 上的副本须在 Gate 0 重新校验。
- 理论 KV bytes ≠ NIXL 实测 bytes。
- 吞吐和尾部延迟隔离分开报告；官方文档明确说 P-D 不提升吞吐。

## 12. Interview answer

NOT EXECUTED。（待数据后用三句话回答：PD 付出了多少传输/编排成本、换来了多少干扰隔离、在什么 workload / 配比下值得；
允许答案是“不值得”。）

## 13. Evidence map

| 结论 | 证据文件 | 生成脚本 |
|---|---|---|
| （尚无结论） | `outputs/m27/raw/h20_formal_*.jsonl`（不存在） | `scripts/m27_run_all.py` |
| | `outputs/m27/h20_formal_*.csv` | `scripts/m27_summarize.py` |
| | `outputs/m27/h20_formal_log_evidence.json` | `scripts/m27_trace_analyze.py` |
| | `figures/fig1{6,7,8}_m27_*_h20_formal.png` | `scripts/m27_figures.py` |
| | 不变量 | `scripts/check_invariants.py --module m27` |
