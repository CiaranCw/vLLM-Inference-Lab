# M27 · H20 执行记录（模板）

> **状态：NOT EXECUTED**。本文件在 H20 上逐项填写；只能写观察到的事实（原始命令、原始输出、文件路径），
> 不写推断出的结论。偏离 [`M27_H20_PLAN.md`](M27_H20_PLAN.md) 的任何操作都必须在 §9 登记。

## 1. 会话

| 项 | 值 |
|---|---|
| 日期 / 时区 | |
| 操作者 | |
| H20 hostname（preflight） | |
| lab commit（`git rev-parse HEAD`） | |
| 计划版本 | v1 |
| tmux 会话 | |

## 2. Gate 0 · preflight

| 项 | 实测值 | 期望 | 一致？ |
|---|---|---|---|
| 可见 GPU 数 / 型号 / 显存 | | H20 / 97871 MiB | |
| driver | | （记录） | |
| Python / torch / torch CUDA | | 3.11.x / 2.11.0+cu130 / 13.0 | |
| vLLM 版本 + 文件路径 | | 0.24.0 | |
| nixl / nixl-cu13 | | 1.2.0 / 1.2.0 | |
| NIXL wheel sha256 | | 见 offline_package.md §1 | |
| NIXL 安装方式与目录 | | `--no-deps --target /root/m27_offline/site` | |
| `NixlWrapper` 非 None / UCX agent 创建 | | 是 / 是 | |
| 模型路径 / checksum manifest（sha256） / 结果 | | M22 副本或本地恢复副本 / `notes/m27/model_artifact/qwen2.5-7b-instruct.sha256`（`157f08c6…9638`）/ 15/15 OK | |
| `max_position_embeddings` / 是否回退 | | ≥32768 / 否 | |
| `topo -m` 最大 NV 互联集合 | | （记录，不假设） | |
| `gpu_ids` / e4_allowed / e5_allowed | | | |
| 端口 8300 / 8400–8407 / 5600–5607 | | 空闲 | |
| GPU 上已有进程 | | 无 | |
| transfer test sha256（远端 / 本地） | | 相同 | |

preflight 日志：`logs/m27/preflight_h20_formal.log`；结果文件：`outputs/m27/configs/preflight_h20_formal.json`。

## 3. Gate 1 · E0

| 项 | 值 |
|---|---|
| 开始 / 结束时间 | |
| 命令 | |
| UCX 实际传输层（D 日志，`UCX_PROTO_INFO=y`） | |
| `e0_gate.json`：d_external / sent / d_local / p_local / transfers / failed / settled | |
| fault probe：HTTP / D prompt Δ | |
| `check_invariants --module m27` | |
| Gate 1 判定（PASS / STOP + 原因） | |

## 4. E1–E5 逐阶段记录

| 阶段 | arm | 开始 | 结束 | rc | 关键日志 / 异常 | invariants |
|---|---|---|---|---|---|---|
| E1 | pd_1p1d | | | | | |
| E2 | | | | | | |
| E3cal | colo2_mbt8192 | | | | | |
| E3 | | | | | | |
| E4cal | colo4_mbt8192 | | | | | |
| E4 | | | | | | |
| E5cal / E5 | | | | | | |

## 5. 校准结果（来自 `configs/h20_formal_e{3,4}cal_rates.json`，原样抄录）

| exp | shape | 无负载 TTFT 中位 | 无负载 ITL 中位 | C_cal | 梯子是否用尽 |
|---|---|---|---|---|---|
| | | | | | |

## 6. Gate 8 决策

| 条件 | 实测 | 满足？ |
|---|---|---|
| 8 卡两两 NV 互联（`e5_allowed`） | | |
| E4 完成且 invariants 0 FAIL | | |
| 剩余机时 ≥ 3.5 h | | |
| 依据 E4 证据选择的配置及理由 | | |

## 7. 打包与回传

| 项 | 值 |
|---|---|
| package 文件名 / sha256 / 大小 | |
| 本地 `sha256sum -c`（tar 与逐文件） | |
| 回传时间 | |

## 8. STOP / 异常（原始命令 + 原始报错 + 根因判断）

（无则写“无”）

## 9. 偏离计划的操作

（无则写“无”。任何偏离都必须写：原计划 → 实际操作 → 原因 → 对结论的影响）
