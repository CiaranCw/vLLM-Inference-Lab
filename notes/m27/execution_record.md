# M27 · H20 执行记录

> **当前状态（2026-10-06 post-hoc 集成后）：M27_H20_FORMAL_COMPLETE；M27_E = COMPLETE、M27_H = COMPLETE。** 新实例 `ins-a8qac2ve` 同实例完成 Gate 0→E0→E1→E2→E3cal→E3→E4cal→E4（§10）；E5 可选、跳过；P1 profiler 未执行。独立 post-hoc 审计 `M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`，文档修正已闭环（§11）。只记实际观察和原始证据。
>
> **阅读顺序**：§1–§9 是**前一实例（2026-10-04，hostname 同为 `TENCENT64.site`）的历史会话记录**，其中“本机”“当前”“尚未完成”“下一台必须…”都指当时的状态，保留原文、不改写；§4 E3 之后的行和 §5、§6 已由本次实例的结果回填，并逐处标注。§10 是当前实例的正式执行，§11 是 2026-10-06 的 post-hoc 修正。
> 前一实例原文的状态行（历史）：“Gate 0/E0/E1/E2 已在本机实测；M27 模块不变量 31 PASS、全仓 183 PASS、0 FAIL；E3cal/E3/E4/E5 在当前代理策略下尚未完成。机时限制触发中途交接；M27_E/H 不得标 COMPLETE。”
> 偏离 [`M27_H20_PLAN.md`](M27_H20_PLAN.md) 的操作在 §9 登记。

## 1. 会话（前一实例，历史）

| 项 | 值 |
|---|---|
| 日期 / 时区 | 2026-10-04 / +08:00 |
| 操作者 | 本轮执行助手（工作副本 `/root/m27_work/vLLM-Inference-Lab`） |
| H20 hostname（preflight） | `TENCENT64.site` |
| lab commit（`git rev-parse HEAD`） | `423575cd73559825ea0595b94f608742714adb86` |
| 计划版本 | v1 |
| tmux 会话 | 未使用；独立 nohup 阶段命令 + 生命周期日志 |
| 交接包完整性 | `/root/m27_h20_handoff_423575cd/manifests/SHA256SUMS`：1129/1129 OK；原包代码 HEAD 正确，恢复上传时丢失的两个执行位后 git clean |
| GPU 预算 | 硬上限 168 GPU·h；`outputs/m27/gpu_hour_ledger.csv`；每阶段启动前预留预算，结束后依据生命周期核算 |

## 2. Gate 0 · preflight（前一实例，历史；当前实例见 §10）

| 项 | 实测值 | 期望 | 一致？ |
|---|---|---|---|
| 可见 GPU 数 / 型号 / 显存 | 8 × NVIDIA H20，各 97871 MiB | H20 / 97871 MiB | 是 |
| driver | 535.247.01；官方 cuda-compat 13.0 user-space (580.95.05) | （记录） | 是 |
| Python / torch / torch CUDA | 3.11.13 / 2.11.0+cu130 / 13.0 | 3.11.x / 2.11.0+cu130 / 13.0 | 是 |
| vLLM 版本 + 文件路径 | 0.24.0 / `/root/m27-venv/lib/python3.11/site-packages/vllm/` | 0.24.0 | 是 |
| nixl / nixl-cu13 | 1.2.0 / 1.2.0 | 1.2.0 / 1.2.0 | 是 |
| NIXL wheel sha256 | `eb666a3691e3d98be35164128175d7763eea3217125b5ead9fc54cd9754e0f16` / `dae1330fd5eeecd23b252b556fb1a528c22ea02ee2019cae92a5a3fad7e8f401`；wheelhouse 复验 OK | 见 offline_package.md §1 | 是 |
| NIXL 安装方式与目录 | `--no-deps --no-index --target /root/m27_offline/site`，`PYTHONPATH` 注入 | 同 | 是 |
| `NixlWrapper` 非 None / UCX agent 创建 | 两者皆是；agent 短进程退出触发 libnl-route `_dl_fini` 崩溃，见 §8 | 是 / 是 | 导入/创建为是；退出问题单独处理 |
| 模型路径 / checksum manifest（sha256） / 结果 | handoff 中 7B；冻结清单 `157f08c6…9638` / 15/15 OK（另见 `/root/model_sha256_check.log`） | 15/15 OK | 是 |
| config 架构 / 是否回退 | 28 层、28 Q heads、4 KV heads、hidden 3584、head_dim 128、bf16、MPE 32768、无量化；不回退 | 同冻结值 | 是 |
| `topo -m` 最大 NV 互联集合 | GPU0–7 **两两 NV18**；GPU0–3 NUMA 0，GPU4–7 NUMA 1；不把 NUMA 亲和性伪装为同一 NUMA 域 | 实测 | 是 |
| `gpu_ids` / e4_allowed / e5_allowed | `[0,1,2,3,4,5,6,7]` / true / true | 根据实际拓扑 | 是 |
| NIC / 端口 / GPU 进程 | 42 NIC（见 `nvidia-smi topo -m`），`UCX_NET_DEVICES=bond1`（平台原值）；预检端口空闲、GPU 无计算进程 | 空闲 / 无 | 是 |
| transfer test sha256（远端 / 本机） | `outputs/m27/transfer_test_h20_formal.txt.sha256` 在本机 sha256sum -c OK（未进行跨主机 pull） | 相同 | 本机验证是；跨主机未验证 |

preflight 日志：`logs/m27/preflight_h20_formal.log`；结果文件：`outputs/m27/configs/preflight_h20_formal.json`。

## 3. Gate 1 · E0（前一实例，历史；当前实例见 §10）

| 项 | 值 |
|---|---|
| 开始 / 结束时间 | 最终有效 E0 为 `e0_rerun_proxy_policy`，2026-10-04 22:53:02–22:56:44 +08:00，准确时间见 `outputs/m27/gpu_hour_ledger.csv` |
| 命令 | `UCX_PROTO_INFO=y python scripts/m27_budget_run.py --stage e0_rerun_proxy_policy --gpu-count 2 --reserve-hours 1 -- python scripts/m27_run_all.py --tag h20_formal --stage e0 --force --model "$M27_MODEL"`（此前不同代理策略的 raw 已归档；正式 raw 只有最新重跑） |
| UCX 实际传输层（D 日志，`UCX_PROTO_INFO=y`） | GPU KV：`cuda_ipc/cuda` zero-copy；host/control：`tcp/bond1`，不得混作 KV 数据通道；详见 `logs/m27/h20_formal/e0_pd_1p1d_D1.log` |
| `e0_gate.json`：d_external / sent / d_local / p_local / transfers / failed / settled | `10240 / 10240 / 0 / 10240 / 5 / 0 / true`；5/5 PD 与 5/5 colocated 完成，首 token 和全输出均 4/5 相同（此项非 PASS 条件） |
| fault probe：HTTP / D prompt Δ | 502 / 0；P 端返回后代理拒绝没有 KV 参数的 D 请求 |
| `check_invariants --module m27` | `logs/m27/interim_invariants_m27.log`：当前代理策略统一复验，无 FAIL / PENDING；trace：0 KV fail、0 compat mismatch、0 Traceback、全部 HND |
| Gate 1 判定（PASS / STOP + 原因） | **M27_E0 = PASS**；E1–E4 严格按冻结设计继续 |

## 4. E1–E5 逐阶段记录（E1/E2/首行 E3cal 为前一实例历史；其后各行为当前实例回填）

| 阶段 | arm | 开始 | 结束 | rc | 关键日志 / 异常 | invariants |
|---|---|---|---|---|---|---|
| E1 | pd_1p1d | 2026-10-04 22:57:34 +08:00 | 23:02:46 +08:00 | 0 | **当前代理策略有效 E1**：`logs/m27/run_e1_proxy_policy.log`，512/2048/4096/8192/16384 每点 5 warmup + 15 measure 均成功且 metrics settled，`outputs/m27/h20_formal_e1_transfer.csv` 从当前 raw 重建；此前 E1 已归档 | 中转前重新检查 |
| E2 | PD 1P1D、colo1_mbt8192、colo1_mbt2048/4096 | 2026-10-04 23:03 +08:00 | 详见 `gpu_hour_ledger.csv` 的 e2a/e2b | A=0 / B=0 | 当前代理策略 `logs/m27/run_e2{a,b}_proxy_policy.log`；较早 20:48–21:34 的完整结果属于旧策略，已/将归档，**不能混入本次有效 E2** | 见 `logs/m27/interim_invariants_m27.log`；机制 1 vs 2 GPU，非同成本比较 |
| E3cal | colo2_mbt8192 | 2026-10-04 21:49:33 +08:00 | 22:17:21 +08:00 | 旧策略先失败后 rc=0，但已失效 | 旧成功 rates SHA `2fce21ca…` 及 raw 均已移入 `outputs/m27/raw/backup_20261004T231827/`；与新代理策略不一致，**当前 E3cal = NOT EXECUTED，下一台必须重新校准并生成新的 rates SHA** | 旧 `invariants_e3cal.log` 仅为历史尝试，不是当前 PASS |
| E3cal（当前实例） | colo2_mbt8192 | ledger `fresh_e2` 13:33:03Z 之后 | 2026-10-05 14:01:46Z（`wall_clock_ledger.csv` `fresh_e3cal`） | 0 | 新 rates SHA `9d5e6946…f386e0a` | M27 PASS（§10） |
| E3（当前实例） | pd_1p1d、colo2_mbt{2048,4096,8192}、colo1_mbt8192 | 14:01:46Z 之后 | 2026-10-05 18:27:22Z（`fresh_e3`） | 0 | 5 arm × 36 runs、78,685 请求、0 fail | M27 PASS（§10） |
| E4cal（当前实例） | colo4_mbt8192 | 18:27:22Z 之后 | 2026-10-05 18:55:54Z（`fresh_e4cal`） | 0 | 新 rates SHA `5cd7a963…3aa3cdd` | M27 PASS（§10） |
| E4（当前实例） | colo4_mbt8192、pd_{1p3d,2p2d,3p1d} | 18:55:54Z 之后 | 2026-10-05 22:05:57Z（`fresh_e4`） | 0 | 4 arm × 27 runs、130,428 请求、0 fail | M27 35 PASS（Gate 8） |
| E5cal / E5 | — | — | — | — | **SKIPPED**（可选；Gate 8 决定见 §6） | 不适用 |

各阶段精确起止以 `outputs/m27/raw/h20_formal_lifecycle.jsonl` 为准；上表的结束时间是 `wall_clock_ledger.csv` 的阶段记账时刻。

## 5. 校准结果（当前实例回填；来自 `configs/h20_formal_e{3,4}cal_rates.json`，原样抄录）

| exp | shape | 无负载 TTFT 中位 ms | 无负载 ITL 中位 ms | C_cal req/s | 梯子是否用尽 | 校准级数 |
|---|---|---:|---:|---:|---|---:|
| e3cal（colo2_mbt8192） | balanced | 218.00 | 10.09 | 8.000 | 否 | 10 |
| e3cal | d_heavy | 34.87 | 10.40 | 22.627 | 否 | 13 |
| e3cal | p_heavy | 904.72 | 10.24 | 2.000 | 否 | 6 |
| e3cal | mixture | —（TTFT 门槛基准取 p_heavy 904.72） | — | 8.000 | 否 | 10 |
| e4cal（colo4_mbt8192） | d_heavy | 34.89 | 10.36 | 64.000 | 否 | 16 |
| e4cal | p_heavy | 905.49 | 10.20 | 4.000 | 否 | 8 |
| e4cal | mixture | —（TTFT 门槛基准取 p_heavy 905.49） | — | 16.000 | 否 | 12 |

每个 shape 都恰有一级未通过（即梯子在其上一级停下），且 75 个校准级的通过与否都只由 TTFT 判据决定，见 §11 的判据局限。

## 6. Gate 8 决策

**当前实例（2026-10-05 23:29:57Z，`wall_clock_ledger.csv` `gate8_e5_skipped`）**：

| 条件 | 实测 | 满足？ |
|---|---|---|
| 8 卡两两 NV 互联（`e5_allowed`） | GPU0–7 两两 NV18 | 是 |
| E4 完成且 invariants 0 FAIL | E4 完成，M27 35 PASS | 是 |
| 剩余机时 ≥ 3.5 h | 已用 11.26 h；ledger 原文：E5 + 重分析 + 收尾预计可能接近 19 h、工作负载过载时长不确定，需保留 ≥2 h 收尾 | 按额度数值满足，但执行方按预计总时长选择不跑 |
| 决定 | **E5 SKIPPED_BY_TIME_BUDGET_OR_GATE8**（E5 是可选的规模确认，优先完成全量 E3/E4 bootstrap、审计与打包） | — |

**前一实例（历史）**：

| 条件 | 实测 | 满足？ |
|---|---|---|
| 8 卡两两 NV 互联（`e5_allowed`） | GPU0–7 两两 NV18，true | 是 |
| E4 完成且 invariants 0 FAIL | E4 未执行 | 否 |
| 剩余机时 ≥ 3.5 h | 本机剩余机时仅 43 分钟，留给 E2 和中转 | 否 |
| 依据 E4 证据选择的配置及理由 | E4 尚无本机数据，不得事后挑 E5 配置 | 不适用；E5 未执行（Gate 8 未正式评估） |

## 7. 打包与回传（前一实例中转包，历史；当前实例最终 archive 见 §11）

| 项 | 值 |
|---|---|
| 中转包文件名 / sha256 / 大小 | 详见单独下载的 `/root/m27_download/M27_INTERIM_MANIFEST.tsv`、`INTERIM_REPORT.md`；本仓库文件写在 archive 内，故自身不能嵌入 archive digest |
| 本地 tar / checksum / raw→derived 校验 | `logs/m27/interim_{summarize,figures,trace,invariants_m27,invariants_full}.log`、下载目录中 `.sha256`；只有 E1/E2 图，E3 图因未执行不伪造 |
| 回传时间 | 本机 13333 已开放 localhost HTTP 200；用户下载未由本机证明，云安全组可能仍需允许 TCP/13333 |

## 8. STOP / 异常（前一实例，历史；原始命令 + 原始报错 + 根因判断）

1. Gate 0 原样预检（`logs/m27/preflight_h20_formal_initial.log`）返回 rc=1：NIXL UCX agent 创建成功、Python 打印 `RESULT PASS` 后，在进程退出期间 SIGSEGV。`logs/m27/nixl_libnl_exit_backtrace.log` 的 gdb 栈在 wheel 自带 `libnl-route-3-…so.200.26.0:rtnl_tc_unregister()` → `_dl_fini`，发生在共享库清理而非 agent 创建 / 数据传输。`openssl3-libs` 原本缺失，先由官方仓库安装，令 `libssl.so.3` / `libcrypto.so.3` 可用。隔离式 preflight 子进程在判断后 `os._exit`，既保存原报错，又不把 agent 创建当成真实 KV 传输的 PASS。修复后 `logs/m27/preflight_h20_formal.log` / `outputs/m27/configs/preflight_h20_formal.json` 为 Gate 0 PASS。
2. E0 首次运行（`logs/m27/failed_attempts/e0_missing_llguidance/run_e0.log`）：P0 rc=1，`ModuleNotFoundError: No module named 'llguidance'`；原始 P/D 日志在同目录。根因是 glibc 2.28 上 PyPI 预制 `llguidance>=1.7,<1.8` wheel 需要 glibc ≥2.31；依赖安装时刻意未采用不兼容 wheel。本轮从官方 PyPI 源码编译相同版本 1.7.6；Cargo 在平台代理下载大型索引受阻，依 `Cargo.lock` 从官方 `index.crates.io` / `static.crates.io` 预取并以锁文件 SHA256 核验 351/351 crate，离线编译成功（`/root/llguidance_offline_build.log`）；不修改模型、采样或研究口径。首次 E0 没产生测量 raw，账本计入启动资源占用。
3. E0 两次测量功能上成功（`h20_formal_e0_gate.json`：D external=sent=10240、D local=0、P local=10240、NIXL transfer=5 / fail=0、故障 502 且 D prompt Δ=0），但服务关闭时 vLLM `shutdown_timeout=0` 的 abort 触发 EngineCore 强杀和 `AsyncLLM output_handler failed` / `EngineDeadError`，使严格日志不变量 FAIL。原始日志/旧 raw 分别在 `logs/m27/failed_attempts/e0_shutdown_trace*/` 与 `outputs/m27/raw/backup_*/`，全部保留且不得纳入正式结果。源码核实 `/root/m27-venv/.../vllm/entrypoints/openai/api_server.py` 的 shutdown 使用 `vllm_config.shutdown_timeout`，因此为所有 arm 一致设置仅退出时生效的 `--shutdown-timeout 15`；并使 launcher 先通知 API 而非同时向 API+EngineCore 发 SIGTERM。从 E0 完整重跑，`logs/m27/invariants_e0_gate_pass.log` 实证 24 PASS / 0 FAIL；旧失败日志未清除。

## 9. 偏离计划的操作（前一实例登记；其修复在当前实例沿用）

- 原 runbook 设想复用 M22 `/root/.venv`、M22 模型与 cuda-compat；这台服务器没有现成环境，所以从已校验的 handoff wheel 和官方 PyPI / NVIDIA CUDA 仓库在 `/root/m27-venv` 安装目标原版本；模型直接只读使用 handoff artifact，未重下。系统 Python 3.11.13 和 `openssl3-libs` 来自 TencentOS/EPEL 官方仓库。结论不受环境路径变化影响；仍须 E0 生效配置双回读。
- Gate 0 的 NIXL agent 创建验证保留，但为避开已留证的 bundled libnl-route `_dl_fini` 崩溃，仅修改短命 preflight 子进程退出方式；失败依旧非零退出。该修改不能代替 E0 的真实 READ 与计数/日志验证。
- 增加 `scripts/m27_budget_run.py` 在 GPU 阶段启动前执行 168 GPU·h 硬预算检查，结束按生命周期起止统计；不改变 frozen matrix。
- `UCX_NET_DEVICES=bond1` 是进入本机时已有的平台变量；`UCX_PROTO_INFO=y` 的 D 侧实测显示 GPU KV 传输为 `cuda_ipc/cuda` zero-copy，控制/host memory transport 同时有 `tcp/bond1`，因此不能仅以 tcp 字样断言 GPU KV 走网络。
- 原实验脚本默认 `--shutdown-timeout 0` 并同时向 API/EngineCore 发 SIGTERM，测量结束后随机产生异常日志；改为所有正式 arm 同一 `--shutdown-timeout 15`（仅终止时生效），launcher 优先通知 API 主进程。`server_info` 增加实际值回读，`m27_arm_construction` 增加全部 arm 参数一致性的回归；不变更 frozen `FROZEN`、采样与测量阶段。`m27_no_failures_log` 仍对当前 raw 的实际服务错误/Traceback FAIL，但排除有独立归档的旧失败尝试，另增合成重试边界不变量，未抹除旧生命周期证据。
- E3cal 首次 rc=3（`logs/m27/failed_attempts/e3cal_mixed_rate_denominator/`、`outputs/m27/raw/backup_*`）：校准首级 d_heavy/balanced 全部成功且 TTFT p99 远低于 10× 基线，却分别因 `11/31.808 < 0.95×11/30` 和 `15/32.033 < 0.95×15/30` 被判失败。最后一个请求分别在 t=29.12/29.21 秒抵达，单请求 2.69/2.82 秒正常解码使 30 秒窗口的完成墙钟超过 31.58 秒；混用 **30s 到达 offered** 与 **包含 drain 的 wall achieved** 在该冻结负载下机械误判轻载，属于实现中的时间分母 bug。修复只针对校准：delivered/offered 都除以冻结 30s 到达窗口，实际 `n_ok/wall_s` 保持原 raw 和正式 online `achieved_rate` 不变，校准 rates 文件另记 `achieved_wall`；不更改 0.5 初速/√2 梯子、30s 到达、3 seeds、calibration arm、baseline、SLO 门槛、工作负载或 sampling。加 `m27_calibration_rate_window` 回归并从最早受影响的 E3cal 归档重跑，**旧 rates 不得用于 E3**。
- E3 PD 首次启动停留在 warmup、无任何正式请求：`warmup_online()` 调用 `build_schedule(shape,2.0,1e9,9999)[:16]`，切片发生在生成约 20 亿个 Poisson 请求之后；内存曾增长到 143+ GiB，P/D metrics 始终 0 请求。仅向本轮明确识别 PID 66328/69045/66340/66341 发送终止信号，全部 GPU 释放；本次 0.228 GPU·h 入账。原日志 `logs/m27/failed_attempts/e3_warmup_unbounded/`，生命周期中明确记录 `manual_abort_unbounded_warmup_schedule`；无测量 raw。修复为生成器可选 `max_requests=16`、保持前 16 请求同 seed/key/shape 不变和正式 60s schedule 无 cap；新增 `m27_warmup_schedule_bounded` 合成回归，重新从 E3 最早受影响的 PD arm 启动。
- E3 PD 第二次尝试在 `d_heavy`、seed 2、rate 20.364675、序列第 28 个请求发生 HTTP 502：`proxy_exception: ServerDisconnectedError: Server disconnected`，P 返回 200/合法 KV 参数，但在 D 请求拿到 HTTP 响应前断开；D 侧服务持续运行，前后共 10341 个成功响应，未看到 NIXL failed/EngineCore traceback。原始失败请求与 11,138 行不完整 raw 保留于 `outputs/m27/raw/backup_*/`，日志保留 `logs/m27/failed_attempts/e3_pd_upstream_disconnect/`，生命周期记录 `manual_abort_single_D_upstream_ServerDisconnectedError`；此数据不得进入正式结论。时序与旧上游 keep-alive socket 被服务端关闭后复用**一致但不能绝对证实**，因此在所有 PD 与 colocated arm 的同一个代理会话中统一设置 `aiohttp.TCPConnector(force_close=True)`，不重试已失败的 KV 请求、不隐藏失败；代理启动在 raw 写明实际连接策略并新增 `m27_upstream_connection_policy` 不变量。由于代理连接策略的观测口径已改变，旧 E0/E1/E2/E3cal 虽成功也不与新 E3 拼接：从 E0 整条证据链归档重跑，重新校准 shared baseline/rates_file_sha256 后再测 E3–E4。
RECOVERY_COMMIT_SHA=884503dccf4d8108326e34cabac0ff36c1fb8823
source_host=TENCENT64.site; archive SHA256=c154e65fcba25a9ea946285cc57f5ba7a67394c44deb47dc9084a1917c2e81b9; current instance=ins-a8qac2ve; task start=2026-10-05T12:14:20Z; previous-host E0-E2 retained as independent reference and not pooled.
Official distro Python 3.11.13 and openssl3-libs installed; isolated /root/m27-venv; install_binary unexpectedly yielded Python 3.14.3 and was not used.

## 10. 新实例完整正式执行（2026-10-05，UTC）

- 计时起点 `2026-10-05T12:14:20Z` / `1791202460`；云 metadata instance-id=`ins-a8qac2ve`，hostname 与前机**碰巧仍为** `TENCENT64.site`；metadata `payment/create-time=2025-03-27` 表示 PREPAID 机器历史创建，**不是本轮刷新额度的计费起点**，不能从这个不适用日期算本轮用量。按可验证 TASK_START 估算整机 8 GPU × 墙钟；详细 `outputs/m27/wall_clock_ledger.csv`。旧机历史用量 4.844513 GPU·h 不从新分配 168 GPU·h 扣除。
- 原 package 1129/1129、模型 15/15、原 anchor `423575cd73559825ea0595b94f608742714adb86`；原件 git 仅因上传丢失两个执行位非 clean，**只在工作 clone** 恢复执行位。interim tar SHA256 `c154e65fcba25a9ea946285cc57f5ba7a67394c44deb47dc9084a1917c2e81b9`、`zstd -t`、overlay 297/297 SHA/size PASS；中转补丁为审计物未重复 apply。recovery commit `884503dccf4d8108326e34cabac0ff36c1fb8823`。
- 由官方系统包获取 Python 3.11.13 / OpenSSL3；从 NVIDIA 官方 RHEL8 compat RPM `580.95.05` **仅解压** CUDA 13.0 用户空间到 `/root/cuda-compat/`，未修改驱动；venv `/root/m27-venv` 使用官方 PyTorch cu130、原 wheelhouse 中已校验 vLLM/NIXL wheel、overlay 中已校验 llguidance wheel；后期为三张图补装官方 PyPI matplotlib。`pip check` 的 `nixl-cu12` 元包要求未满足是为了避免 cu12/torch 污染，实际 NIXL/UCX + CUDA 13 GPU READ 由 E0 证实。`VLLM_USE_FLASHINFER_SAMPLER=0`、`VLLM_ENABLE_V1_MULTIPROCESSING=0` 每一正式阶段均设置。
- 新 Gate 0 PASS：`logs/m27/preflight_current_host.log`，新 preflight/gpu_map 均在 TASK_START 后生成；`transfer_test_h20_formal.txt.sha256` 在本机 PASS。模型 28 layers/28 Q heads/4 KV heads/3584 hidden/128 head_dim/bfloat16/32768，GPU0–7 双 NUMA、全部 NV18、无 context fallback；E0 真 READ `cuda_ipc/cuda`，控制面 `tcp/bond1`；5 次 transfer/0 fail，D 本地 prompt 0，fault 502/0 token。
- E0/E1 `--force` 使用脚本自动将先前独立机器正式 raw 备份；E2 四 arm 也都重跑（含 `colo1_mbt2048/4096/8192`、`pd_1p1d`）。首次整批 E2 因调用端空闲超时中断 `colo1_mbt8192` 的尾部，本轮仅向识别为 M27 的 API PID 24158/代理 25524 发 TERM，保留 partial raw 于 `backup_20261005T210958`；单 arm 无 `--force` 恢复时 launcher 自动归档并重新测满；PD 另用 `--force` 覆盖上一机正式 raw。E2 320/320 overlap_valid，正式 240 measure；`raw/backup_*/` 始终未参与 summarize。
- `E3cal` 当场重新校准，冻结 `rates_file_sha256=9d5e69466371ca72d9d45f830cf4049c24a1bef61bb039172d146c017f386e0a`，同一 `force_close=True` 连接策略；E3 5 arm × 36 measure runs、78,685 请求、0 fail；`E4cal` 重新校准共享 `colo4_mbt8192`，rates SHA `5cd7a9637e870e1dd47a29ba6fd8a5ba3ed77aa235ed3b9d5206a03f40aa3cdd`；E4 4 arm × 27 runs、130,428 请求、0 fail。每个正式在线 run 的 rates SHA 与对应文件逐字一致、相同 shape/fraction/seed 的跨 arm schedule SHA 一致；没有逐请求补发或修改原始 raw。
- E4 后立即进行 10,000 次请求簇 Poisson bootstrap，E4 派生 108 online/81 effect 行、E3 派生 180 online/108 effect 行。首次 E3 汇总因交互空闲超时只写到 online CSV，重运行 CPU 分析后 effect 完整生成；无 GPU benchmark 重跑。`m27_figures.py` 首次因 matplotlib 未装失败，补装 venv 依赖后重建图 16–18。第一次全仓不变量调用未继承两个冻结环境变量导致仅 env 2 FAIL；按冻结环境重新执行，结果 **M27 35 PASS、全仓 187 PASS，均 0 FAIL/0 PENDING**；初次异常没有触碰 raw，留在命令记录/交接材料。
- Gate 8：E4 后 M27 35/35 PASS、8 卡 NV18；E5 在 E4 发现的高负载 tail/TTFT 取舍上仅能增加规模确认，本次优先冻结全量 E3/E4 CI、审计与下载包，**E5 SKIPPED_BY_TIME_BUDGET_OR_GATE8**；P1 profiler NOT RUN，不能声称计算与通信重叠已证。最终显式审计 `outputs/m27/final_evidence_audit.json`：73/73 PASS、活跃 raw 修改时间均晚于本次 TASK_START、前一台数据只在 backup 下。不能仅以相同 hostname 作为来源判据。
- 新机意外/负结果：无 prefill 注入的 PD baseline stall 略高；E3 90% 负载 PD 用更低 decode ITL P99 换来显著更高 TTFT P99/更低 achieved rate；E4 4 卡 PD ratio 负载偏斜和严重 queueing 改变 Pareto 最优。完整数值见 `results.md` 顶部本机正式结果，不从历史附录摘 headline。最终 commit 和下载包标识由 final 报告给出；不 push、不 tag、不 release。

## 11. Post-hoc 审计与仓库集成（2026-10-06，本地，无 GPU）

本节不改写 §1–§10 的原始记录，只补充审计发现的遗漏，并说明证据在仓库中的存放方式。审计结论 `M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`（CRITICAL 0 / MAJOR 0 / MINOR 7），材料见 [`audit/`](audit/)，逐条闭环见 [`FINAL_CLOSURE_ADDENDUM.md`](FINAL_CLOSURE_ADDENDUM.md)。

- **最终 archive 身份**：`m27_h20_final_20261006T003354Z.tar.zst`，2,465,709,981 B，SHA256 `fb1664ca9506efb75bb784ab19a8b65186c6eabedf69127805e8e72b29633ad6`（以名称 + 大小 + SHA256 为准，与存放路径无关）。archive 内仓库的 H20 提交为 `884503dc…`（运行时修复 + 前机中途证据）与 `c4db4561…`（当前实例正式证据）；规范化后的对应提交见 [`H20_COMMIT_PROVENANCE.md`](H20_COMMIT_PROVENANCE.md)。
- **计费（F-M01）**：正式执行与收尾约 12.44 h ≈ 99.5 GPU·h（8 卡整机）。12.44 h 取自 H20 执行方最终回复（12.442 h）与所有者观察的 746 min，ledger 最后一行（打包前）为 12.308 h、原 FINAL_REPORT 快照为 12.3619 h。实验结束后实例仍保持分配，refreshed 168 GPU·h 额度（168/8 = 21 h）最终耗尽；其后约 8.56 h ≈ 68.5 GPU·h 是完成后的分配/空闲时间，不是正式实验负载。`wall_clock_ledger.csv` 与 `current_host_gpu_technical_ledger.csv` 原样保留，未改动。
- **校准判据（F-M04）**：§9 的 arrival-horizon 修正让 `delivered ≥ 0.95×offered` 等价于 `n_ok/n_offered ≥ 0.95`。在 0 失败时它不起区分作用：e3cal 39 级、e4cal 36 级中，achieved 每一级都等于 offered，且每一级是否通过都与 `TTFT P99 ≤ 10×空载中位` 完全一致（由 `configs/h20_formal_e{3,4}cal_rates.json` 逐级核对）。真正起门槛作用的是 TTFT 判据。这是判据设计的局限，不改变 raw、schedule 与正式结果。
- **首次全仓 invariants 失败的记录（F-M05）**：§10 记载第一次全仓 invariants 因调用端未继承 `VLLM_USE_FLASHINFER_SAMPLER=0` / `VLLM_ENABLE_V1_MULTIPROCESSING=0` 出现 env 类 2 FAIL，补上冻结环境变量后重跑，M27 35 PASS、全仓 187 PASS。**那次失败的原始完整输出没有作为独立日志保留**，只剩 ledger `final_invariants` 行的文字描述；本仓库不补造该日志。审计方在无这两个变量的条件下复现出相同的 2 个 env FAIL（审计目录 `invariants_full_noenv_control.log`，未入库），与该描述一致。H20 上的 `logs/m27/invariants_final_*.log` 只有结果摘要、没有命令/环境/时间；本轮集成后的复验日志 `logs/m27/invariants_final_integration_{m27,full}.log` 在开头记录了命令、环境变量和 UTC 时间。
- **前一实例 server log（F-M05）**：当前实例 E0–E2 重跑时覆盖了 `logs/m27/h20_formal/` 下 10 个同名 server log（`e0_colo1_mbt8192_colo0`、`e0_pd_1p1d_{P0,D1}`、`e1_pd_1p1d_{P0,D1}`、`e2_colo1_mbt{2048,4096,8192}_colo0`、`e2_pd_1p1d_{P0,D1}`），当前树中的是当前实例的版本。前一实例的原文没有备份到 `failed_attempts/`，只存在于提交历史中：H20 原提交 `884503dc` 与其规范化提交（见 provenance 文档 N1）逐字节相同，可用 `git show <N1>:logs/m27/h20_formal/<name>.log` 取回。前一实例数据本来就只作独立参考、不 pool，因此不影响任何正式结论。
- **超大 raw（F-M06）**：11 个单文件 >100MB 的 raw（合计 2,712,916,405 B）不进入普通 Git 历史，逐文件 size/SHA256/H20 原 blob 记录在 [`H20_RAW_ARCHIVE_MANIFEST.tsv`](H20_RAW_ARCHIVE_MANIFEST.tsv)，`.gitignore` 逐个路径列出。SHA256 是从 archive 直接解出的文件计算的。不变量 `m27_external_raw_identity` 在 raw 存在时逐字节核对，缺失时报 PENDING（不会静默跳过）。
- **final evidence audit 生成器（F-M07）**：新增 `scripts/m27_final_evidence_audit.py`，离线从 raw/derived/configs/logs 重算原 73 项，名称与顺序和 `final_evidence_audit.json` 一致，73/73 PASS，并附逐项证据，写入 `outputs/m27/final_evidence_audit_regenerated.json`。原 JSON 保持不变。反向测试（缺 raw、翻转 overlap、篡改 1 字节、伪造 n_fail、preflight FAIL）均被检出。
