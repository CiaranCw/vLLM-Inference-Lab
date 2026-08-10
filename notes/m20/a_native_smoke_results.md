# M20-A · Native serving semantics（轻量 smoke 结果）

> **时间**：2026-08-08 19:38–19:40 Asia/Shanghai  
> **脚本**：`scripts/m20_native_smoke.py`  
> **日志**：`logs/m20/a_native_smoke_run.log`  
> **边界**：host native `vllm.entrypoints.openai.api_server` / port 8020 / 0.5B / eager；**非**大规模 SLO，**无** Bootstrap。

## 结论（仅 smoke）

| ID | 结果 | 证据 |
|---|---|---|
| A1 stream vs non-stream | 协议语义可分：non-stream `Content-Type: application/json`，first-byte≈E2E；stream 为 SSE，first-byte 远早于 E2E。本 smoke **不做** streaming ITL 统计 | `outputs/m20/raw/a1_stream_smoke.json` |
| A2 prefix serving | `--enable-prefix-caching` 可起服并完成共享前缀请求；**无性能主张**（APC 见 M17） | `outputs/m20/raw/a2_prefix_smoke.json` |
| A3 health restart | `/health` 200 → stop → health 失败 → restart → 200 + generate OK；`/metrics` 在线可取 | `outputs/m20/raw/a3_health_restart_smoke.json` |

## A1 数字（n=3/模式，median）

| mode | median t_first_byte_ms | 备注 |
|---|---:|---|
| non_stream | ≈216.6 | ≈整包到达 |
| stream | ≈12.5 | 首包；不能当 non-stream TTFT 对照滥用 |

## 模块状态含义

- M20-A **轻量完成**。
- 后续 B/C 已于 2026-08-08/10 完成 → 见 [`results.md`](results.md)。
