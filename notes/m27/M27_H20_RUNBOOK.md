# M27 · H20 Runbook（按顺序执行；每个 Gate 都有 PASS / STOP 条件）

> 冻结设计：[`M27_H20_PLAN.md`](M27_H20_PLAN.md)（v1）。本 runbook 只描述**怎么跑**，不改问题。
> 任何 STOP：**保留 logs 和 raw，不删除、不 `--force` 覆盖**，在 [`execution_record.md`](execution_record.md) 记录原始命令、
> 原始报错和根因判断，然后停下来询问所有者。禁止临场改实验来“跑通”。
> 不改 driver/CUDA，不用 sudo，不下载模型，不 kill 别人的 GPU 进程。

## 0. 上机前（本地，所有者操作）

1. 使用 pre-H20 anchor commit（见 `HANDOFF.md` §5；未推送，由所有者决定何时 push）。在 execution record 中记下要在 H20 上使用的 lab commit。
2. 把**整个仓库**放到 H20 的 `/root/vllm-inference-lab`（M27 脚本按 `scripts/` 的上级目录定位 `outputs/`、`notes/`、`config/`，
   不能像 M22 那样只拷 `/root/scripts/`）：H20 能访问 GitHub 就 `git clone`；否则在本地
   `git archive --format=tar.gz -o m27_repo_<commit>.tar.gz <commit>` 后用 scp/rsync **直接**传过去，并核对 sha256。
3. 准备 NIXL wheel（二选一，**禁止第三方中转 / 网盘**）：
   - A（推荐，M22 时 H20 可以访问 PyPI）：在 H20 上 `pip download`，然后用 [`offline_package.md`](offline_package.md) 中的 sha256 校验；
   - B：从本地 `~/m27_offline/wheelhouse/` 用 scp/rsync **直接**传到 H20。
4. 模型（身份见 [`model_artifact_manifest.md`](model_artifact_manifest.md)）：首选 H20 上 M22 遗留的
   `/root/m22_offline/models/Qwen2.5-7B-Instruct`；不存在时，从本地 `/mnt/f/wsl_backup/m22_offline/models/Qwen2.5-7B-Instruct`
   用 scp/rsync **直接**传过去（≈15.2 GB）。不论哪种，都必须在 Gate 0 通过身份 manifest 的 `sha256sum -c`。

## 1. H20 环境（每个新 shell 都要执行）

```bash
tmux new -s m27            # 或 tmux attach -t m27
cd /root/vllm-inference-lab && git rev-parse HEAD   # 必须 == <LAB_COMMIT>；archive 方式则核对包名中的 commit
source /root/.venv/bin/activate
export LD_LIBRARY_PATH=/root/cuda-compat/usr/local/cuda-13.0/compat:${LD_LIBRARY_PATH:-}   # 与 M22 相同；以机器实际路径为准
export M27_MODEL=/root/m22_offline/models/Qwen2.5-7B-Instruct
# M27_MODEL_SHA256 不需要设置：preflight 默认用仓库内的 notes/m27/model_artifact/qwen2.5-7b-instruct.sha256
export PYTHONPATH=/root/m27_offline/site          # NIXL 独立安装目录（§2），不改 /root/.venv
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
mkdir -p logs/m27
```

M22 时 H20 上的环境是：`/root/.venv`（vllm 0.24.0 / torch 2.11.0+cu130 / Python 3.11.13）与
`/root/m22_offline/`（`models/`、`manifests/`、`wheelhouse/`）。**如果容器已经被回收、这些目录不存在**：不要在 H20 上重装或重新下载，
STOP 并询问。本地备份 `/mnt/f/wsl_backup/m22_offline/` 里有模型（§0 第 4 步）和 vllm 0.24.0 wheel（sha256 与 PyPI 相同，见 offline_package.md §2）。

launcher 会给每个实例再注入 `VLLM_KV_CACHE_LAYOUT=HND`、`VLLM_SERVER_DEV_MODE=1`、`HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1`、`PYTHONHASHSEED=0`、`CUDA_VISIBLE_DEVICES`、`VLLM_NIXL_SIDE_CHANNEL_{HOST,PORT}`。

## 2. 安装 NIXL（不改 M22 venv、不碰 torch）

```bash
mkdir -p /root/m27_offline/wheelhouse /root/m27_offline/site && cd /root/m27_offline/wheelhouse
# A：pip download --no-deps nixl==1.2.0 nixl-cu13==1.2.0 --python-version 3.11 --only-binary=:all: -d .
# B：已从本地直接拷入
sha256sum nixl-1.2.0-py3-none-any.whl nixl_cu13-1.2.0-cp311-cp311-manylinux_2_28_x86_64.whl
#   必须等于 offline_package.md §1 的值，不等 → STOP
pip install --no-deps --no-index --target /root/m27_offline/site \
    nixl-1.2.0-py3-none-any.whl nixl_cu13-1.2.0-cp311-cp311-manylinux_2_28_x86_64.whl
cd /root/vllm-inference-lab
python -c "import torch; print(torch.__version__)"    # 必须仍是 2.11.0+cu130
```

`--no-deps` 是必需的：`nixl` meta 包同时声明了 `nixl-cu12` 与 `nixl-cu13`，`nixl-cu13` 还声明了 torch / numpy。
不加 `--no-deps` 可能改动 torch。

## Gate 0 · 环境 / 拓扑 / 依赖 / 模型

```bash
bash scripts/m27_preflight.sh h20_formal 2>&1 | tee logs/m27/preflight_h20_formal.log
python scripts/m27_run_all.py --tag h20_formal --transfer-test       # T+0:10 内完成
# 本地：rsync -avP <USER>@<H20>:/root/vllm-inference-lab/outputs/m27/transfer_test_h20_formal.txt{,.sha256} ./m27_transfer_test/
#       cd m27_transfer_test && sha256sum -c transfer_test_h20_formal.txt.sha256
```

- **PASS**：preflight 退出码 0 且最后一行是 `Gate 0 PASS`；`outputs/m27/configs/{preflight,gpu_map}_h20_formal.json` 存在，
  其中 `e0_e3_allowed=true`；本地 pull 的 transfer test 通过 sha256 校验。
- **STOP**：任何 `[FAIL]` 行，例如版本不符、NIXL 导入或 UCX agent 失败、`NixlWrapper is None`、模型 checksum 不符、
  config 架构不符、端口占用、GPU 上已有进程、NV 互联 GPU 不足 2 张；transfer test 拉不回来。
- 记录：`gpu_map` 中的 `gpu_ids`、`e4_allowed`、`e5_allowed`、`max_model_len_fallback`（为 true 时自动使用唯一合法回退）。

## Gate 1 · E0：1P1D 正确性 / 真传输 / 无重算 / fail-closed

```bash
export UCX_PROTO_INFO=y        # 只在 E0 打开：让 D 日志打印 UCX 实际选择的传输协议
python scripts/m27_run_all.py --tag h20_formal --stage e0 --model $M27_MODEL 2>&1 | tee logs/m27/run_e0.log
unset UCX_PROTO_INFO
python scripts/m27_summarize.py --tag h20_formal --only e0
python scripts/m27_trace_analyze.py --tag h20_formal
python scripts/check_invariants.py --module m27
grep -iE "cuda_ipc|rc_mlx5|dc_mlx5|ud_mlx5|tcp|sysv|posix" logs/m27/h20_formal/e0_pd_1p1d_D1.log | head -40
```

- **PASS**（全部满足）：
  - `outputs/m27/h20_formal_e0_gate.json`：`d_external_kv_tokens == prompt_tokens_sent`、`d_local_compute_tokens == 0`、
    `p_local_compute_tokens == prompt_tokens_sent`、`d_nixl_transfers ≥ 5`、`d_nixl_failed == 0`、`metrics_settled_all == true`、
    `fault_probe_http_status == 502` 且 `fault_probe_d_prompt_tokens_delta == 0`；
  - `check_invariants --module m27`：0 FAIL；
  - `h20_formal_log_evidence.json`：kv_load_fail = 0、compat mismatch = 0、tracebacks = 0、所有实例的 KV layout 都是 HND；
  - 从 D 日志确认 UCX 的实际传输层，并写入 execution record。
- **STOP**：任何一项不满足；launcher 回读不符（`*_effective.json` 中有 `mismatches`）；
  UCX 选择的不是同机 GPU 直连（例如走 `tcp`，或者在没有 cuda_ipc 的情况下走 mlx5 网卡）→ 记录后询问，**不要自行改 `UCX_TLS`**。
- `first_token_match` / `full_output_match` 只记录，不作为 PASS 条件（见计划 §5 E0）。

## E1 · KV transfer scaling

```bash
python scripts/m27_run_all.py --tag h20_formal --stage e1 --model $M27_MODEL 2>&1 | tee logs/m27/run_e1.log
python scripts/m27_summarize.py --tag h20_formal --only e1 && python scripts/check_invariants.py --module m27
```

- **PASS**：raw 有 `arm_done`；每个 context 15 个 measure 全部成功；`m27_transfer_success` 与 `m27_theoretical_kv_metadata` PASS。
- **STOP**：任何 NIXL 失败、`metrics_settled=false`、D 侧出现 local_compute。实测 bytes 与理论 bytes 不相等**不是** STOP 条件，只记录和分析。

## E2 · 受控 prefill 干扰（最重要）

```bash
# 先跑 A 层（最低成功线），再跑 B 层
python scripts/m27_run_all.py --tag h20_formal --stage e2 --model $M27_MODEL --arms pd_1p1d,colo1_mbt8192 2>&1 | tee logs/m27/run_e2a.log
python scripts/m27_run_all.py --tag h20_formal --stage e2 --model $M27_MODEL --arms colo1_mbt2048,colo1_mbt4096 2>&1 | tee logs/m27/run_e2b.log
python scripts/m27_summarize.py --tag h20_formal --only e2 && python scripts/check_invariants.py --module m27
```

- **PASS**：每个 (arm, p_len) 15 个 measure 全部 `overlap_valid`；`m27_paired_keys_e2` PASS；PD 的注入阶段 D external == p_len。
- **STOP**：出现 `overlap_valid=false`（说明注入没有落在 D 的 decode 区间内）→ 不改 d_output 或注入点，记录并询问。

## E3 · 2 卡在线

```bash
python scripts/m27_run_all.py --tag h20_formal --stage e3cal --model $M27_MODEL 2>&1 | tee logs/m27/run_e3cal.log
python scripts/m27_run_all.py --tag h20_formal --stage e3 --model $M27_MODEL --arms pd_1p1d,colo2_mbt8192 2>&1 | tee logs/m27/run_e3a.log
python scripts/m27_run_all.py --tag h20_formal --stage e3 --model $M27_MODEL --arms colo2_mbt2048,colo2_mbt4096 2>&1 | tee logs/m27/run_e3b.log
python scripts/m27_run_all.py --tag h20_formal --stage e3 --model $M27_MODEL --arms colo1_mbt8192 2>&1 | tee logs/m27/run_e3c.log
python scripts/m27_summarize.py --tag h20_formal --only e3 && python scripts/check_invariants.py --module m27
```

- **PASS**：校准 rc=0（每个 shape 都有 C_cal，且梯子没有用尽）；`m27_paired_schedules_e3`、`m27_online_no_failed_requests_e3` PASS。
- **STOP**：校准 rc=3 → 不手调速率，记录 `configs/h20_formal_e3cal_rates.json` 后询问；任何 run 出现 `n_fail > 0` → 保留并询问。

## E4 · 4 卡 P:D 配比（需要 `gpu_map.e4_allowed=true`）

```bash
python scripts/m27_run_all.py --tag h20_formal --stage e4cal --model $M27_MODEL 2>&1 | tee logs/m27/run_e4cal.log
python scripts/m27_run_all.py --tag h20_formal --stage e4 --model $M27_MODEL 2>&1 | tee logs/m27/run_e4.log
python scripts/m27_summarize.py --tag h20_formal --only e4 && python scripts/check_invariants.py --module m27
```

- **PASS / STOP**：与 E3 相同。`e4_allowed=false` 时 run_all 会直接拒绝，results 记为“未执行（拓扑不满足）”。

## Gate 8 · 是否值得跑 8 卡

同时满足以下条件才进入 E5：

1. `gpu_map.e5_allowed=true`（8 张卡**两两** NV# 互联，由 `topo -m` 实测，不假设）；
2. E4 已完成且 invariants 0 FAIL；
3. 剩余机时 ≥ 3.5 h；
4. 在 execution record 中写下依据 E4 证据选出的 2–3 个配置及理由（候选 `pd_2p6d` / `pd_4p4d` / `pd_6p2d`）。

否则在 results §8 写“未执行（Gate 8：原因）”。

## E5 · 8 卡（可选）

```bash
python scripts/m27_run_all.py --tag h20_formal --stage e5cal --model $M27_MODEL 2>&1 | tee logs/m27/run_e5cal.log
python scripts/m27_run_all.py --tag h20_formal --stage e5 --model $M27_MODEL --arms colo8_mbt8192,<Gate 8 选出的配置> 2>&1 | tee logs/m27/run_e5.log
```

## P1 · profiler（可选，未实现）

本轮没有实现 P1 脚本。机时充足时也不要临场编写；“NIXL 异步 READ 与 decode 计算重叠”保持“未验证”。

## Package · 打包与回传

```bash
python scripts/m27_summarize.py --tag h20_formal
python scripts/m27_trace_analyze.py --tag h20_formal
python scripts/check_invariants.py --module m27 2>&1 | tee logs/m27/invariants_h20_formal.log
python scripts/m27_run_all.py --tag h20_formal --package          # → m27_h20_formal_<stamp>.tar.gz + .sha256 + 逐文件 SHA256SUMS
```

本地主动拉取（不使用第三方中转）：

```bash
rsync -avP <USER>@<H20>:/root/vllm-inference-lab/m27_h20_formal_<stamp>.tar.gz{,.sha256} ./
sha256sum -c m27_h20_formal_<stamp>.tar.gz.sha256
tar xzf m27_h20_formal_<stamp>.tar.gz          # 解到仓库根；先确认不会覆盖已有文件
sha256sum -c outputs/m27/SHA256SUMS_h20_formal_<stamp>.txt
python scripts/m27_figures.py --tag h20_formal  # 图在本地离线生成
python scripts/check_invariants.py
```

回传清单：`outputs/m27/{raw,configs,metrics}/`、`outputs/m27/*.csv|json`、`logs/m27/`、
`logs/m27/preflight_h20_formal.log`、execution record 中的手写记录。**数据没带回来 = 实验失败。**

## 通用失败处理

- 任意阶段出现异常退出：launcher 已停掉本 arm 的全部进程，并在 `raw/h20_formal_lifecycle.jsonl` 写入停机原因和日志证据；
  先跑 `m27_trace_analyze.py`，再看 `logs/m27/h20_formal/<exp>_<arm>_*.log`。
- 续跑：同一命令再执行一次即可，已有 `arm_done` 的 arm 会跳过；未完成的 arm 的旧文件会被**移动**到
  `outputs/m27/raw/backup_<stamp>/`（不删除）。`--force` 只能在记录原因之后使用。
- 端口被占用或 GPU 上有残留进程：先 `nvidia-smi` 和 `ss -ltnp` 确认是不是本实验残留；不是本实验的 → 不 kill，询问所有者。
