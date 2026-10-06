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
