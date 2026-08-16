# M24 · 远端 H20 运行记录（观察，不含结论数字）

> 结论以 `outputs/m24/` 的 raw jsonl / CSV 为准。本文件只记环境与降级。

## 环境（2026-08-16 实测）

- 机器：4× NVIDIA H20（97871 MiB/卡，NV18 全连接），driver 535.247.01，CUDA 12.2
- GPU 占用：开跑前 `nvidia-smi` 无计算进程，4 卡独占
- vLLM：**0.24.0 wheel**，`/root/m24-venv/lib/python3.11/site-packages/vllm/__init__.py`
- torch：2.11.0+cu130；Python：3.11.15（uv venv `/root/m24-venv`）
- OS/glibc：glibc 2.28（CentOS 系）
- 无 tmux/screen（系统未装，禁 sudo 不可装）→ 全部 GPU 任务用 `setsid` 后台脱离 SSH，日志落 `logs/m24/`。preflight 的 tmux WARN 由此而来。

## 与本机既有笔记的偏差（全部如实记录）

1. **llguidance = 1.7.6+m24stub（非官方 wheel）**
   - 原因：glibc 2.28 < 官方 wheel 的 manylinux_2_31；本机无 Rust 工具链，禁编 sdist
   - 影响：guided decoding 调用会 RuntimeError；M24 矩阵未使用 structured/guided decoding
   - 与 M22 同一处理口径；**不是** vLLM 产品代码改动；未升级 glibc / 未换系统
   - 门闩验证：`from vllm.v1.structured_output.backend_guidance import GuidanceBackend, GuidanceGrammar` → `import OK`
2. **cuda-compat 前向兼容**：driver 535.247.01 仅支持 CUDA 12.2，torch cu130 初始报 "driver too old"。
   按 M22 同法（M22 记录即 cuda-compat）：从 NVIDIA 官方仓库下载 `cuda-compat-13-0_580.95.05-0ubuntu1_amd64.deb`，
   **仅解包**到 `/root/cuda-compat/`（用户态，无 sudo、无系统安装），`LD_LIBRARY_PATH` 指向之。
   验证：torch.cuda.is_available()=True，ndev=4，sm_90，cuda:0 matmul 通过。
3. 环境注入脚本：`/root/m24_env.sh`（venv + compat + 必需环境变量）。每个建引擎进程均 source。
4. 依赖闭包用 `uv pip sync` 按 `vllm==0.24.0 + torch==2.11.0` 解析清单安装（仅 llguidance 换 stub）。
   transformers=5.15.0（闭包解析结果，与 M22 可能存在小版本差异）。

## 运行记录

- [x] preflight rc=0（`logs/m24/preflight.log`）；GQA 门 TP=4 DCP=2 合法（H_q=12 H_kv=2）
- [x] transfer-test：`outputs/m24/transfer_test_h20_formal.txt`（sha256 见 `.sha256`）
- [x] smoke：`h20_smoke_dcp2_run.jsonl`，dcp=2 引擎建成（回读 dcp=2、backend=mp、dcp_group=[2,2,2,2]），D1 成功。
  quick 网格 measure 记录 `success=False`，invalid_reason 全为「稳态 decode step 仅 4 个（<5）」——
  quick 输出 8 token 的数学必然（8-1-3=4），非引擎故障；formal 32 token 不受影响。
- [x] **P0 formal**：`h20_formal_dcp{1,2}_run.jsonl` 均非空完整。
  - dcp1 rc=0（102.5s）、dcp2 rc=0（152.9s）；各 90/90 measure success，steady_batch_stable=90/90 True，scheduler 内省 90/90 ok；D1 两档均 4 ranks success。
  - summarize 三 CSV 已生成（memory_kv / decode / decode_effect）；`logs/m24/environment.txt` 5 行（两档 effective_cfg 回读）。
  - 引擎回读确认：dcp1→dcp=1、dcp2→dcp=2，TP=4 两档一致。
- [x] **P1 profile（dcp=2）**：首跑 rc=1 且无 trace——0.24.0 的 torch profiler 需显式 `start_profile()/stop_profile()`，仅传 ProfilerConfig 不落盘。
  对 `scripts/m24_dcp.py` 做最小现场修正（**仅 profile 分支**包 start/stop；run shard 与矩阵未动，P0 数据不受影响），
  直调底层脚本重跑（未删首跑 jsonl，两条 profile 记录并存）。产物：4 rank chrome trace（`outputs/m24/traces/h20_formal_dcp2/rank{0..3}.*.pt.trace.json.gz`，各约 1.9MB）+ profiler_out。
  trace 实际可见（不预设、只记录）：`vllm::all_gather`、`vllm::reduce_scatter`、`_correct_attn_cp_out_kernel`、`vllm::merge_attn_states_kernel`、`FlashAttnFwdCombine`、NCCL allgather/RS kernel。
  注意：NCCL kernel 名中的 "RING_LL" 是 NCCL 集合通信内部算法，**与 Ring Attention 无关**。
  profile 样本本身 `success=False` 同 smoke 原因（8 token → 稳态 4<5），jsonl 中该记录标 `profile_only_not_formal`，不进汇总。
- [x] **P2 PCP 负例**：`outputs/m24/raw/h20_formal_pcp_negative.json`。引擎未建成（`built_engine=false`），
  但失败点比预期更早：PCP=2 乘进 world_size=TP4×PCP2=8 > 4 GPU，`ParallelConfig` pydantic 校验即拒（ValidationError）。
  未走到 `check_attention_cp_compatibility` 的 `supports_pcp` 断言。`ok_as_negative=false`（失败模式与脚本预期不同），
  完整 traceback 已存。结论留给本地：本 4 卡机 PCP>1 物理不可行。
- [ ] **P3 P-D smoke：未跑**（时间预算内优先保 P0+P1+P2 与回传）。
- 未做：`--force` 未使用；断点续跑机制未触发异常；无任何网格/模型降级（gpu_memory_utilization 保持 0.85，8192 档完整跑出）。

## 给本地的回传说明

- 回传包：`m24_h20_formal_*.tar.gz`（+ `.sha256`，manifest `outputs/m24/package_h20_formal.json`）；
  或直接 rsync 整目录 `outputs/m24/` + `logs/m24/`。
- trace 文件较大（4×1.9MB gz），已在包内。
- `/root/m24_env.sh` 为本机环境注入脚本（venv 路径 + cuda-compat + 必需变量），供复跑参照。
