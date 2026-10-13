# M28 执行记录

> 本文件按时间记录做了什么、看到了什么、改了什么。§0–§4 是 GPU 授权之前的 CPU / mock 阶段（那些"通过"都不是实验结论）；
> §5 起是授权之后的 GPU 执行。

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

§3 的日志已移到 `logs/m28/v1_cpu_validation/`（v1 配置下的结果）；v2 的同类日志在 `logs/m28/` 下重新生成，见 §6。

## 4. 授权（2026-10-12，所有者通过结构化问答确认）

- 整卡已有约 4.7–5.0 GiB 显存占用（`nvidia-smi` 不列进程）：所有者确认是自己的 / 系统占用，可以在旁边跑。
- GPU：授权 A 冒烟 + A 正式 + A evict（可选）。
- 故障注入：授权 B1 + B3（一次授权覆盖冻结矩阵）。
- B2：所有者选择不做 → **SKIPPED**。

## 5. v1 冒烟（GPU，第一次启动引擎）

命令 `python scripts/m28_offload.py --smoke --i-have-gpu-authorization`，exit 0，server 就绪 19.06 s，清理后显存回到基线。
产物原样保存在 `outputs/m28/raw_old_v1_smoke/`、`logs/m28/v1_smoke/`，**不进入任何统计或结论**。

- 按设计成功的部分：cold `cached_tokens=0`；gpu_hot `Δlocal_hits=1040`；两个 filler 之后 cpu_hit `Δext_hits=1040`、
  `Δload_bytes=12779520`（= 65 block × 196608 B，与理论值一致）；cpu_hit 与 gpu_hot 的 token 与 logprob 逐位相等。
- 暴露的问题一：server 日志 "Default vLLM sampling parameters have been overridden by the model's `generation_config.json`:
  `{'repetition_penalty': 1.1, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}`"。请求显式设了 temperature / top_p，
  但 `repetition_penalty=1.1` 仍生效，与计划"贪心"不符。
- 暴露的问题二：`/dev/shm` 没有出现 `vllm_offload_*.mmap`，日志没有 "Created mmap file"，而是 "Allocating 1 CPU tensors..."。
  回到源码确认：CPU spec 的 CPU 层是 `torch.zeros(pin_memory=PIN_MEMORY)`，WSL2 上默认非 pinned；mmap + `cudaHostRegister`
  只属于 tiering spec。v1 源码核查写错了（详见 `source_review.md` 的更正记录）。CPU→GPU 也不走 Triton（cross-layer 布局，page 196608 B ≥ 28 KiB）。
- 暴露的问题三：runner 的日志标记把自己写的分隔行（含完整命令行）也计入 `OffloadingConnector` 计数；而 mock 数据下
  `a_connector_log_evidence` 规则被跳过，所以 CPU 验证没发现它与真实日志不匹配。

## 6. v1 → v2 修订（正式 arm 之前）

- `server_common_args` 加 `--generation-config vllm`；`server_env_must_unset` 加 `VLLM_WSL2_ENABLE_PIN_MEMORY`（固定 pageable）。
  用 vLLM 自己的 parser 在 CPU 上确认 4 个 A arm 与 B 的参数都解析为 `generation_config='vllm'`。
- `scan_log` 合并进 `m28_common`，跳过 runner 分隔行；A 组标记改为真实日志行（connector / CPUOffloadingSpec / CPU tensor 分配 /
  采样覆盖 / 推理期 JIT / traceback / EngineDeadError，mmap 与 cudaHostRegister 两项保留但预期为 0）。
- validator `a_connector_log_evidence` 改为：connector arm 必须有三行初始化日志、非 connector arm 不得有，所有 arm 不得出现采样覆盖 /
  traceback / EngineDeadError；对 mock 数据同样生效（mock server 输出带 `mock:` 前缀的同名行）。新增 2 个反例。
- `M28_PLAN.md` 升 v2（§12 修订记录），冻结块由代码重新生成。
- v2 CPU 复验（mock 产物在 `/tmp/m28_mock_rehearsal_v2`）：A 4 个 arm + B 13 次演练全链路 OK；
  validator（`--allow-mock`）33 PASS / 0 FAIL；自检 **14/14** 反例检出。日志：`logs/m28/{cpu_dryrun_a,cpu_dryrun_b,mock_rehearsal_validate,validator_selftest_cpu}.log`、
  `logs/m28/validator_selftest_cpu.json`。

## 7. v2 冒烟（GPU）

exit 0，server 就绪 16.05 s。日志标记：connector 2 / CPUOffloadingSpec 2 / CPU tensor 分配 1 / **采样覆盖 0** / traceback 0 /
推理期 JIT 1（`_compute_slot_mapping_kernel`，发生在 server_warmup 阶段）/ mmap 0。

| 路径 | success | cached_tokens | Δlocal_hits | Δext_hits | Δload_bytes |
|---|---|---|---|---|---|
| cold | True | 0 | 0 | 0 | 无 |
| gpu_hot | True | 1040 | 1040 | 0 | 无 |
| cpu_hit | True | 1040 | 0 | 1040 | 12779520 |

cpu_hit 与 gpu_hot 输出 token 与 logprob 逐位相等（G-A-corr-1 成立）；cold 与 gpu_hot 从第 2 个 token 起不同——这属于计划的
G-A-corr-3（计算形状不同导致的数值分歧），是否被 offload 放大要看正式数据里 ref 对照。冒烟单样本延迟不作结论。
A 组门禁（connector 能起、cpu_hit 被指标证实、KV 往返不破坏输出）通过 → 进入正式 arm。产物：`outputs/m28/raw/m28_a_smoke.jsonl`、`logs/m28/a_smoke_runner.log`。

## 8. A 正式 + evict（GPU）

- `python scripts/m28_offload.py --i-have-gpu-authorization`：offload → base → ref 顺序执行，exit 0（`logs/m28/a_formal_runner.log`）。
  就绪 16.05 / 17.05 / 16.04 s；server 生命周期 449 / 444 / 202 s。
- `python scripts/m28_offload.py --arm evict --i-have-gpu-authorization`：exit 0，就绪 17.05 s，生命周期 125 s（`logs/m28/a_evict_runner.log`）。
- 每个 arm 日志标记：connector arm 三行初始化日志齐全；全部 arm 采样覆盖 0、traceback 0、EngineDeadError 0；推理期 JIT 各 1 次
  （`_compute_slot_mapping_kernel`），时间戳早于该 arm 第一个 measure 样本 29 s 以上（发生在 server_warmup 期间）。
- 显存：就绪后整卡 +1924～+1982 MiB；EngineCore RSS offload 6050.8 MiB / base 1953.7 MiB / evict 2210.8 MiB（`m28_lifecycle.jsonl`）。
- summarize + validate：A 组 21 PASS / 0 FAIL（当时 B 组 PENDING）。结果见 `results.md` §1。

## 9. B 第一轮（GPU，B1 ×3 + B3 ×9）

命令 `python scripts/m28_faults.py --drill b1 --drill b3 --i-have-gpu-authorization --i-have-fault-authorization`，exit 0，错误演练数 0（`logs/m28/b_runner.log`）。

- B1 ×3 按设计完成（`results.md` §2.1）。这 3 条记录**保留在正式 raw** 里，第二轮没有重跑 B1。
- B3 有 5 次演练（t0 ×3、t2 r0/r1）在 90 s 观测窗结束时 API 仍存活、在途请求未结束，runner 清理时 SIGKILL（rc −9）。
- 暴露的 runner 缺陷：
  1. 挂住的请求被记成 `{"missing": true}`，summarize 在空列表上取 median 抛 `StatisticsError`，validate 在空序列上 `min()` 抛 `ValueError`。
  2. 强制清理时 API 与 resource_tracker 同时被 SIGKILL，`/dev/shm` 每次留下 1 个 `sem.mp-*`，共 5 个（`oce92guz`、`mjeod_5h`、`uvel949z`、`mt1elx_i`、`3g9tarul`）。
     当时未删除；2026-10-13 所有者同意后删除（见 §12）。
- 处理：第一轮 B3 原样保存——`outputs/m28/raw_old_b3_run1/m28_b_drills.jsonl`（含 B1+B3 共 12 行，sha256 `4e01dbd6…`）及其派生 CSV、
  `logs/m28/b3_run1/`（9 个 B3 server 日志 + 当时的 runner / summarize / validate 日志，后两者含上述崩溃的 traceback）。正式 raw 只保留 B1 的 3 行（与第一轮逐字节相同）。`m28_lifecycle.jsonl` 里第一轮 B3 的 9 条生命周期记录保留未删，
  `x_no_engine_overlap` 把它们一并核对。

## 10. 执行期修订与 CPU 复验

修订内容见 `M28_PLAN.md` §12「执行期修订」（只改记录 / 清理逻辑，不改冻结块与矩阵）。修订中又发现并修正一个问题：清理后补记的判断
最初用"线程是否仍在运行"，而清理时线程已随连接断开结束，导致 `requests_after_cleanup` 从未写入。改为按观测窗记录里的 `terminal is False` 判断。
mock 复验（`/tmp/m28_mock_rehearsal_v3`，A 复制自 v2、B 重新跑 b1/b3/b2，mock t=0 档模拟挂住）：挂住的演练 8 条请求都有清理后终态；
validator（`--allow-mock`）33 PASS / 0 FAIL；自检 **15/15**（`logs/m28/validator_selftest_cpu.{log,json}`）。

## 11. B3 第二轮（GPU，同一冻结矩阵）

开跑前：整卡 3403 MiB、端口 8028 空闲、无 vLLM 进程、无实验锁；`/dev/shm` 只有第一轮遗留的 5 个 `sem.mp-*`（计入演练前快照，不算新增）。
命令 `python scripts/m28_faults.py --drill b3 --append --i-have-gpu-authorization --i-have-fault-authorization`，exit 0，错误演练数 0，用时约 17 min（`logs/m28/b3_runner.log`）。
结束后：无 vLLM 进程、端口空闲、整卡 3434 MiB、`/dev/shm` 无新增。

- 9 次结局见 `results.md` §2.2：t0 r0 8/8 abort（API 30.37 s 退出，等的是故障瞬间探针的 30 s 客户端超时），r1/r2 挂住；
  t2 r1 500 并退出，r0/r2 挂住；t60 3/3 正常完成。挂住的 4 次清理后补记：非流式 `RemoteDisconnected` ×4、流式 `eof_without_done` ×4。
- 与第一轮不同（t0 3/3 → 2/3 挂住），两轮不合并。对 12 次 t0/t2 server 日志逐个统计 EngineCore 的 abort / drain / "request processing complete" 与 EngineDeadError，
  用来解释结局差异（`results.md` §2.3）。
- summarize → validate：**33 PASS / 0 FAIL / 0 PENDING**（`logs/m28/{summarize,validate}.log`）；figures 离线生成 `fig19_m28_offload_ttft.png`、`fig20_m28_faults.png`；
  `check_invariants.py --module m28` 33/0/0，全模块 223/0/0（`logs/m28/check_invariants_{m28,all}.log`）。`../vllm-src` 脏文件数仍为 22。

## 12. 收尾：/dev/shm 遗留信号量（2026-10-13）

所有者同意删除 §9 记录的 5 个 `sem.mp-*`。删除前核对：5 个文件均存在（32 B，属主 ciaran，时间 20:57–21:11，与第一轮 5 次强制清理对应），
`/proc/*/fd` 与 `/proc/*/maps` 中没有任何进程打开或映射它们。按确切文件名 `rm` 这 5 个，没有使用通配符。删除后 `/dev/shm` 为空。
