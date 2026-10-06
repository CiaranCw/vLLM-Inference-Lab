# M27 · 当前实例 H20 正式结果

> **M27_H20_FORMAL_COMPLETE；M27_E/H = COMPLETE。** 2026-10-05 新实例 `ins-a8qac2ve`（hostname 仍为 `TENCENT64.site`）从原始交付 + 上台独立中转证据恢复后，重新采集了同实例 Gate 0→E0→E1→E2→E3cal→E3→E4cal→E4；Gate 8 选择跳过可选 E5。之前机器同名，不用 hostname 单独判定来源：当前 raw/校准/Gate 0 均晚于 `TASK_START=2026-10-05T12:14:20Z`，旧 raw 保留于 `raw/backup_*/`。本节为唯一正式 headline；下方附录保留原机原文，只作独立 replication reference，绝不 pool 样本。

## 环境与正确性

8×H20，各 97871 MiB、GPU0–7 两两 NV18、双 NUMA、driver 535.247.01；Python 3.11.13、torch 2.11.0+cu130、CUDA compat 13.0、vLLM 0.24.0、NIXL 1.2.0；Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`、15/15 文件校验。各实例 BF16 / TP1 / eager / HND、APC/async/spec/quant OFF；未触发 context fallback。fresh E0：D external KV=P local=sent=10240 tokens、D local=0，NIXL READ 5/失败 0；故障探针 HTTP 502 且 D prompt 增量 0；GPU KV 用 `cuda_ipc/cuda`，控制面 `tcp/bond1`。见 `outputs/m27/h20_formal_e0_gate.json`、`final_evidence_audit.json`。

## RQ1 · long prefill 隔离（E2）

四 arm × 四注入长度 × (5 warmup+15 measure)：320/320 overlap_valid，其中 240 个 measure；1 GPU colo1 对 2 GPU PD，只能回答机制，不能冒充同成本收益。8K 注入的 D max stall 中位：PD **105.05 ms**、colo1_mbt8192 **886.03 ms**；colo−PD **+780.98 ms**，Bootstrap 95% CI **[780.51, 781.36]**。无注入 `p=0` 时 PD **13.43 ms** 反而高于 colo **11.84 ms**（colo−PD −1.59 ms，CI [−1.95, −0.93]）。其余两种 mbt 在派生 CSV 中完整保留。

## RQ2 · KV transport 随 context 增长（E1）

每档 15 measure；理论 57,344 B/token。512→16384 token 时 NIXL 实测 bytes **29,360,128→939,524,096**，累计 xfer 中位 **6.914→208.242 ms**，有效 GB/s **4.25→4.51**；五档全表见 `outputs/m27/h20_formal_e1_transfer.csv`、`figures/fig16_m27_e1_transfer_h20_formal.png`。本例理论与实测恰相等，不是普适 invariant；xfer 是累计 handle 时间而非独立墙钟。

## RQ3 · 同成本部署（E3）

fresh `E3cal` 的 rates SHA256 `9d5e69466371ca72d9d45f830cf4049c24a1bef61bb039172d146c017f386e0a`；5 arm × 4 workload × 3 load × 3 seed = 180 runs，**78,685/78,685 请求成功**；Poisson open-loop、统一 `force_close=True`。同样 2 GPU 的 PD 1P1D 与 colo2_mbt8192 在 90% 校准负载：balanced ITL P99 的三 seed 中位 **102.5 vs 424.6 ms**（PD 低），但 TTFT P99 **28,728 vs 1,197 ms** 且 achieved **4.7 vs 6.9 req/s**（PD 差）；prefill-heavy ITL P99 **12.5 vs 893.3 ms**，TTFT P99 **38,764 vs 3,708 ms**；decode-heavy ITL P99 **32.7 vs 38.0 ms**，TTFT P99 **252 vs 82 ms**。`colo2_mbt2048/4096` 也完整采集；同 workload 的预先冻结共享 SLO 门槛与请求簇 Poisson bootstrap 见 `h20_formal_e3_{online,effect}.csv`。**不存在 PD 全面胜出**，不能把 ITL 改善当成 TTFT/E2E/吞吐改善；90% 档严重排队时本表的 achieved 和 offered 不同。

## RQ4 · 四 GPU ratio（E4）

fresh `E4cal` 的 rates SHA256 `5cd7a9637e870e1dd47a29ba6fd8a5ba3ed77aa235ed3b9d5206a03f40aa3cdd`；4 arm × 3 workload × 3 load × 3 seed = 108 runs，**130,428/130,428 请求成功**，每 arm 总占 4 GPU、各实例 TP1。90% 负载、三 seed 的组内 P99 中位（TTFT / ITL，ms）：

| workload | colo4 | 1P:3D | 2P:2D | 3P:1D |
|---|---:|---:|---:|---:|
| decode-heavy | 151 / 78 | 26,636 / 53 | 15,133 / 78 | 93,690 / 47 |
| prefill-heavy | 3,254 / 897 | 134,208 / 12 | 38,313 / 13 | 7,191 / 110 |
| mixed | 5,148 / 906 | 151,333 / 15 | 49,985 / 35 | 17,620 / 159 |

比率取舍随工作负载而变：偏 P 的 3P:1D 在 prefill-heavy/mixed 的 PD 选项中缓和 TTFT 排队，decode-heavy 的 1P:3D 减小 ITL tail；然而本轮各 90% 档 **colo4 的 TTFT P99 均优于 PD**，不能宣称某 PD ratio 全局最佳。对照 CI 和全部请求数见 `h20_formal_e4_{online,effect}.csv`。

## Gate 8、负结果与限制

Gate 8：E4 完成，M27 invariants 35 PASS/0 FAIL/0 PENDING，8 卡 NV18；但 E5 只属 scale confirmation，可选且需保留充分 closure 时间。本次 **E5=SKIPPED_BY_TIME_BUDGET_OR_GATE8**；P1 profiler 未做，不能称 compute/comm overlap proven。正式结果仅适用于本机 H20 / 7B / vLLM 0.24.0 / NIXL 1.2.0 / HND / eager / 这组 workload、负载与网络拓扑；不和 M22/M24 不同布局与口径绝对值相减。前一台独立花费 4.844513 GPU·h，本次 refreshed 168 GPU·h（整机 8×墙钟小时）。本轮强负结果为高载 PD TTFT 排队及无注入时略差的 stall。全部头条数字来源 `outputs/m27/h20_formal_e{0,1,2,3,4}_*`；审核 `outputs/m27/final_evidence_audit.json` 73/73 PASS；全仓 invariants 187 PASS。前机数据不参与本轮任一 CSV/CI。

---

## 附录：前一实例中途交接原文（独立参考，不参与当前正式结论）

以下保留原历史交接陈述；其中“本机”“当前”均指上一实例，不是本次 `ins-a8qac2ve`。旧机摘要：8K E2 stall PD 105.03 ms、colo 885.10 ms，E3/E4 未完成；只是定性 replication，严禁样本合并。

# M27 · H20 中途交接结果（非正式收口）

> **M27_H20_PARTIAL_HANDOFF**。当前统一上游连接策略 `force_close=True` 下 Gate 0、E0、E1 完成；E2 A/B 已重跑且通过 overlap/paired-key/transfer 不变量；E3cal/E3/E4/E5 未形成可用正式结果。M27_E/M27_H **NOT COMPLETE**。本文件不是正式结论。H20 / vLLM 0.24.0 / NIXL 1.2.0 / Qwen2.5-7B bf16 / TP1 / eager / HND / 本机 NV18 / 在线客户端口径。

## Gate 0 · 环境

8×NVIDIA H20 97871 MiB，driver 535.247.01 + 官方 cuda-compat 13.0；Python 3.11.13、torch 2.11.0+cu130、vLLM 0.24.0、NIXL 1.2.0。GPU0–7 两两 NV18，但 GPU0–3 属于 NUMA 0，GPU4–7 属于 NUMA 1。交接包 1129/1129 SHA256 OK；模型 15/15 OK，HF revision `a09a35458c702b33eeacc393d103063234e8bc28`，无 context fallback。详见 `outputs/m27/configs/preflight_h20_formal.json`、`gpu_map_h20_formal.json` 和 `logs/m27/gate0_environment_inventory.log`。

## E0 · 正确性（PASS）

当前有效 raw：`outputs/m27/raw/h20_formal_e0_pd_1p1d.jsonl` 与 `h20_formal_e0_colo1_mbt8192.jsonl`；派生 `outputs/m27/h20_formal_e0_gate.json`。PD 5 measure 中 D external KV tokens = P local compute = 发出 prompt tokens = 10240；D local compute = 0；NIXL transfers = 5、failed = 0、指标稳定；fault probe 返回 HTTP 502、D prompt 增量 0。D 日志 `UCX_PROTO_INFO=y` 记录 GPU `cuda_ipc/cuda` zero-copy、控制面 `tcp/bond1`；不以 `tcp` 字样误判 KV 数据通道。P/D `kv_load_failure_policy=fail`、HND 与配置双来源回读；服务日志无 KV failure、compat mismatch、Traceback。与 colocated 首 token/完整输出各 4/5 一致；不是通过条件。旧失败尝试在 `raw/backup_*/` 和 `logs/m27/failed_attempts/`，不得合并。

## E1 · KV 传输缩放（COMPLETE，RQ2 局部回答）

当前有效 raw：`outputs/m27/raw/h20_formal_e1_pd_1p1d.jsonl`；派生 `outputs/m27/h20_formal_e1_transfer.csv`。每点 5 warmup 与 15 measure 分离，下面均为成功 measure 的中位数；GB/s = 实测 NIXL bytes / 累计 xfer time，而非墙钟带宽。

| Context tokens | 理论整块 KV bytes | NIXL 实测 bytes | xfer ms | post ms | 等效 GB/s | 客户端 TTFT ms |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 29,360,128 | 29,360,128 | 7.354 | 5.761 | 3.992 | 85.726 |
| 2048 | 117,440,512 | 117,440,512 | 24.570 | 16.871 | 4.780 | 260.465 |
| 4096 | 234,881,024 | 234,881,024 | 48.820 | 41.481 | 4.811 | 502.584 |
| 8192 | 469,762,048 | 469,762,048 | 104.631 | 92.563 | 4.490 | 1033.345 |
| 16384 | 939,524,096 | 939,524,096 | 211.424 | 186.936 | 4.444 | 2235.930 |

本轮恰好数值相等是实测观察，**不是理论 bytes 必须等于 NIXL bytes 的一般不变量**；xfer 为 handle 时间之和，不能等同于独立通信墙钟时间；没有 profiler 时间线，不声称 compute/communication overlap。

## E2 · 受控 Prefill 干扰（RQ1）

**当前代理策略 E2 完成**：4 arm × 4 context × (5 warmup+15 measure) = 320/320 overlap_valid。
派生 `outputs/m27/h20_formal_e2_{interference,effect}.csv` 只取本轮当前 raw；旧策略 raw 已归档。
下表的 max stall 为客户端逐请求最大 ITL 的组内中位数；差值是 `colo1_mbt8192 − PD` 的 median difference 和冻结 Bootstrap 95% CI。

| 注入 tokens | PD max stall 中位 ms | colo1_mbt8192 max stall 中位 ms | 差值 [95% CI] ms |
|---:|---:|---:|---:|
| 0 | 13.09 | 11.77 | -1.32 [-1.86, -1.09] |
| 2048 | 29.86 | 210.70 | 180.84 [179.58, 181.49] |
| 8192 | 105.03 | 885.10 | 780.06 [778.73, 781.27] |
| 16384 | 200.60 | 1075.87 | 875.27 [874.61, 875.83] |

E2 是不同 GPU 数（1 vs 2）的干扰隔离机制比较，**不能**据此回答同成本部署效率；见 `h20_formal_e2_effect.csv` 的另外两个合法 colo chunk 对照。

## E3 / E4 / E5 · 尚未完成（RQ3/RQ4 暂无答案）

E3cal 曾按旧连接策略完成，**其速率文件、raw 现已移入 `outputs/m27/raw/backup_20261004T231827/`，不属于当前有效正式证据，不能沿用其 C_cal 或 SHA**。E3 PD 旧策略高负载发生过一次 `ServerDisconnectedError` HTTP 502，已完整保留原失败证据，但不能纳入汇总。新服务器在当前代理策略下必须从 `--stage e3cal` 重新生成共享基线，然后运行冻结 E3/E4；E5 只在 Gate 8 满足时运行。不得把未跑的 E3/E4/E5 写成负结果或 PASS。

## 局限和完整证据链

M27 目前可用 E1 回答 RQ2 的本机 KV 传输缩放、用完整 E2 回答 RQ1 的机制级干扰隔离；RQ3、RQ4 缺同预算 E3/E4，仍未回答。原启动、停机、校准窗口和代理修复均在 `notes/m27/execution_record.md` §8–9 有根因及旧 raw 归档；`outputs/m27/gpu_hour_ledger.csv` 含失败尝试用量。历史 M22/M24 口径/布局不同，不拿绝对延迟做直接相减归因。此中转包不是最终结果，不进行 final commit / push / tag / release。
