# M24 · H20 上机 Runbook

给**远端 Agent / 操作者**。理论见 [`context_parallel_ring_disagg.md`](context_parallel_ring_disagg.md)；矩阵见 [`M24_H20_PLAN.md`](M24_H20_PLAN.md)。完整指令见 [`REMOTE_AGENT_PROMPT.md`](REMOTE_AGENT_PROMPT.md)。

## 0. 拷到机器上之后

假设拷贝根目录为 `$XFER`（本机准备位置：`/home/ciaran/m24_h20_transfer`）。

```text
$XFER/
  REMOTE_AGENT_PROMPT.md    ← 先把这个整份交给远端 Agent
  README_COPY.md
  models/Qwen2.5-1.5B-Instruct/   # 扁平权重，约 2.9G
  vllm-inference-lab/             # 本仓库切片（无 .venv）
```

工作目录必须是 **`$XFER/vllm-inference-lab`**。模型用绝对路径：

```bash
export M24_MODEL="$XFER/models/Qwen2.5-1.5B-Instruct"
```

进入 **tmux**（SSH 断线保护）：`tmux new -s m24`

## 1. 环境变量（缺了实验作废）

```bash
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_LAB_TRACE
unset VLLM_LAB_M9_COALESCE
# 禁止下载
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

不要 `source` 本机 5080 的 `.venv`。用 H20 上已有的 Python/vLLM（M22 当时是 0.24.0 wheel + torch 2.11+cu130）。

## 2. 命令顺序（按这个走，不要发明）

```bash
cd "$XFER/vllm-inference-lab"
mkdir -p logs/m24 outputs/m24/raw

# T+0  自检（不下载、不建引擎）
bash scripts/m24_preflight.sh "$M24_MODEL" 2>&1 | tee logs/m24/preflight.log
# 有 [FAIL] → 停下报告，不要往下跑

# T+10 回传演练（本地必须马上 pull 一次）
python3 scripts/m24_run_all.py --tag h20_formal --transfer-test

# smoke（--quick，验证 DCP=2 能建起来）
python3 scripts/m24_run_all.py --tag h20_smoke --model "$M24_MODEL" --dcp 2 --quick
# smoke 失败 → 保存日志，不要直接开 formal

# P0 正式（断点续跑：已有 jsonl 会 SKIP）
python3 scripts/m24_run_all.py --tag h20_formal --model "$M24_MODEL" --dcp 1,2

# P1（可选）
python3 scripts/m24_run_all.py --tag h20_formal --model "$M24_MODEL" --dcp 2 --profile

# P2 PCP 负例
python3 scripts/m24_run_all.py --tag h20_formal --model "$M24_MODEL" --pcp-negative

# 打包
python3 scripts/m24_run_all.py --tag h20_formal --package
```

## 3. 失败怎么降级

| 现象 | 动作 |
|---|---|
| 只有 2 张 GPU | **停**。本矩阵需要 TP=4 |
| DCP=2 建引擎失败 | 保存完整 traceback；仍把 DCP=1 跑完；**不要改成 7B / 不要改 tp** |
| OOM | 先降 `gpu_memory_utilization=0.7` 记一笔；再不行去掉 `prompt_len=8192` 一档并记入日志 |
| vLLM 不是 0.24.0 | 记录版本，仍尝试；API 对不上就停，不要 pip 编译源码 |
| 机器上有别人的 GPU 进程 | **停下来问**，不得 kill |

## 4. 明确不要做

- 不要跑 PCP>1 当正式性能（P2 只允许负例）
- 0.24.0 的 torch profiler **必须** `engine.start_profile()` / `stop_profile()`；只传 ProfilerConfig 不落盘（H20 已踩）
- 不要找 Ring Attention kernel
- 不要 `pip install` 新模型 / 不要访问 huggingface.co 下权重
- 不要改 `vllm` 产品代码
- 不要在 ExampleConnector 上写「P-D 更快」
- 不要把 H20 数字和 5080 数字相减

## 5. 回传

本地执行（H20 填真实 host）：

```bash
rsync -avP <USER>@<H20>:$XFER/vllm-inference-lab/outputs/m24/ ./outputs/m24/
rsync -avP <USER>@<H20>:$XFER/vllm-inference-lab/logs/m24/ ./logs/m24/
```

图和不变量在**有 GPU 的本机不必跑**；回传后在 5080 侧：

```bash
python scripts/check_invariants.py --module m24
python scripts/m24_dcp.py --shard summarize --tag h20_formal
python scripts/m24_figures.py --tag h20_formal
```
