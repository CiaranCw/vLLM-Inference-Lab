# M28 执行记录

> **状态：CPU_VALIDATED · GPU NOT EXECUTED**。本文件按时间记录做了什么、看到了什么、改了什么。
> 截至本次更新**没有启动过任何 vLLM 引擎，没有使用 GPU，没有做故障注入**；下文所有"通过"都是 CPU / mock 层面的，不是实验结论。

## 0. 环境快照（2026-10-12）

| 项 | 值 | 来源 |
|---|---|---|
| lab HEAD | `4c516a2`（M28 产物尚未提交时） | `git log -1` |
| vLLM 源码 | `../vllm-src` @ `ee0da84`（v0.24.0），editable `0.24.1.dev0+gee0da84ab.d20260714`；脏文件 22（基线） | `git -C ../vllm-src status --short \| wc -l` |
| vllm-src 脏文件语义 | 15 个脏 `.py` 去注释后 AST 与 HEAD 相同；6 个含 lab hook（`_lt.enabled()` / `blocked_ids()` / `VLLM_LAB_M9_COALESCE` 门控，另有一个无条件 `bump_step` 计数）；**offloading 相关文件均不脏** | `outputs/m28/m28_vllm_src_ast_audit.json`（`scripts/m28_source_audit.py`） |
| 模型 | `Qwen/Qwen2.5-0.5B-Instruct` 本地快照 `7ae5576…`，权重 blob sha256 `fdf756fa…` | `m28_common.model_identity()` |
| GPU | RTX 5080 16303 MiB；Gate 0 时已用 4708 MiB，授权申请前复核 4991 MiB；`nvidia-smi --query-compute-apps` 为空（WSL2 不做进程级归因） | `nvidia-smi` |
| 主机 | 内存 62 GiB（可用 57 GiB）；`/dev/shm` 32 GiB 空闲 | `free -g`、`df -h /dev/shm` |

## 1. 源码核查与计划冻结

- 只读核查 OffloadingConnector、`/health`、EngineCore 死亡路径、`--shutdown-timeout` 的实现，结论与预测见 [`source_review.md`](source_review.md)。
- 冻结计划 [`M28_PLAN.md`](M28_PLAN.md) v1：§4 的 JSON 块由 `m28_common.FROZEN` 生成，validator 规则 `x_plan_frozen_matches_code` 检查两者一致。
- 所有冻结的 server 参数都用 vLLM 自己的 CLI parser 在 CPU 上解析过一次（不建引擎），解析结果符合预期。

## 2. 冻结前发现并修正的问题（均在冻结 v1 之前，没有实验数据受影响）

1. **HF 离线加载**：`HF_HUB_OFFLINE=1` 下用 repo id 启动会抛 `IncompleteSnapshotError`，原因是本地快照缺少 `.gitattributes`、`LICENSE`、`README.md`（权重和配置文件齐全）。
   改为用快照目录路径启动并加 `--served-model-name Qwen/Qwen2.5-0.5B-Instruct`；GPU 门禁增加模型身份核对（快照 commit + 权重 sha256），不一致就拒绝启动。没有下载任何文件。
2. **B3 不足档不成立**：`max_tokens=512` 时在途批次的剩余工作可能少于 2 s，`shutdown_timeout=2` 就不再是"不够用"的档位。
   改成 `max_tokens=1024`，并加了 validator 前提规则 `b_b3_timeout_precondition`：用同进程校准批次的完成时长核对，不成立的重复记 FAIL，不能当作有效的 B3 样本。
3. **线程属性遮蔽**（`m28_faults.py`）：Observer 把 `threading.Event` 存在 `self._stop`，遮蔽了 `threading.Thread._stop`，`join()` 时报 `TypeError: 'Event' object is not callable`，mock 演练 5 次全部出错。
   改名为 `_halt` 后通过。这些出错的演练里清理逻辑仍然正确执行（端口释放、无残留进程）。
4. **在途证据计入失败请求**：信号量在请求失败时也会释放，可能把"失败"误判成"已收到首 token"。增加 `first_flags`，只在真的收到首 token 时置位。
5. **探针截止**：Prober 停止方式改为先 join 再 stop，避免最后一个探针被截断。

## 3. CPU / mock 验证（不是实验数据）

`scripts/m28_mock_server.py` **不是 vLLM**，只模拟 HTTP 接口、指标名、LRU 缓存、`VLLM::EngineCore` 子进程和关停语义，用来走通 runner、清理、summarize、validator 的全链路。
mock 产物只写在 `/tmp/m28_mock_rehearsal`，runner 拒绝把 mock 输出写进 `outputs/m28`。

| 检查 | 结果 | 日志 |
|---|---|---|
| 8 个 `scripts/m28_*.py --help` | 全部 exit 0 | 本节命令 |
| A dry-run | 打印 4 个 arm 的冻结矩阵与命令，约 987 个请求；理论 load_bytes（P=1024…16384：12779520 / 50528256 / 100859904 / 201523200 B） | `logs/m28/cpu_dryrun_a.log` |
| B dry-run | 打印 B1/B3/B2 矩阵、server 命令、目标身份规则 | `logs/m28/cpu_dryrun_b.log` |
| 未授权启动 | `m28_offload.py --smoke`、`m28_faults.py --drill b1`（缺故障授权 / 缺 GPU 授权）都拒绝启动，exit 2 | 本节命令 |
| mock A 演练 | 4 个 arm，每 (arm, P, path) 5 warmup + 15 measure，全部成功；evict arm 的 ext_hits=0；清理干净 | `/tmp/m28_mock_rehearsal`（临时） |
| mock B 演练 | 13 次演练（B1×3、B3×3 档×3、B2×1）全部 OK，清理干净 | 同上 |
| validator（mock 数据，`--allow-mock`） | 33 PASS / 0 FAIL / 0 PENDING | `logs/m28/mock_rehearsal_validate.log` |
| validator 自检 | 12 个注入反例全部检出（目标规则 PASS→FAIL），反例只在临时目录 | `logs/m28/validator_selftest_cpu.json` |
| `check_invariants.py --module m28` | 2 PASS / 0 FAIL / 3 PENDING（A raw、B raw、lifecycle 未执行） | `logs/m28/invariants_m28_cpu_validated.log` |
| `check_invariants.py`（全仓） | 192 PASS / 0 FAIL / 3 PENDING（3 个都是 M28 未执行项） | `logs/m28/invariants_full_cpu_validated.log` |
| `reproduce.py --run --module m28 --only a-smoke` | 拒绝启动（`formal_only`） | — |

## 4. 尚未执行

- A 冒烟、A 正式（offload / base / ref）、A evict（可选）：需要 GPU 授权。
- B1、B3：需要 GPU 授权 + 一次故障注入授权。B2：需要单独授权。
- `results.md`、`limitations.md`、面试笔记：等实测数据落盘后再写，现在不写任何性能或行为结论。
