# M22 · 执行记录与复现（E/H 已完成后的留痕）

> 本文件**不是**上机操作剧本。远端窗口已结束；只记录「在什么环境跑了什么、产物在哪、如何离线复核」。
> 结论见 [`results.md`](results.md)；源码定位见 [`tensor_parallel.md`](tensor_parallel.md)。

## 1. 实际执行环境（`logs/m22/environment.txt`）

| 项 | 值 |
|---|---|
| 时间 | 2026-08-09（Asia/Shanghai），4×H20 窗口 |
| GPU | 4× NVIDIA H20，97871 MiB / 卡，NV18 全连接，driver **535.247.01** |
| vLLM | **0.24.0**（H20 上为 site-packages wheel，非本机 editable `ee0da84` 树） |
| torch / CUDA / Python | 2.11.0+cu130 / 13.0（cuda-compat）/ 3.11.13 |
| 模型 | `Qwen2.5-7B-Instruct`（离线包；未临时下载） |
| 执行模式 | `enforce_eager=True`，`async_scheduling=False`，`VLLM_ENABLE_V1_MULTIPROCESSING=0` |
| 正式 tag | `h20_formal` |
| 可选补强 tag | `h20_opt_profile`（补全 P1 chrome case 标签 + TP=1 profile） |

> **版本边界**：本机 lab 仍 pin `ee0da84`；H20 实测为发布版 **v0.24.0** wheel。结论不得与 5080 / 0.5B / `local_preflight` 混写。

## 2. 产物清单（正式线）

| 类型 | 路径 |
|---|---|
| 逐样本 | `outputs/m22/raw/h20_formal_tp{1,2,4}_run.jsonl` |
| 汇总 CSV | `outputs/m22/h20_formal_{memory_kv,decode,decode_effect,prefill,output_consistency}.csv` |
| Profile JSONL | `outputs/m22/raw/h20_formal_tp{2,4}_profile.jsonl` |
| Chrome traces | `outputs/m22/traces/`（formal 文件名曾全部带 `decode_small_batch` 前缀，见 results §P1） |
| 可选 profile | `outputs/m22/raw/h20_opt_profile_tp{1,2,4}_profile.jsonl` + 已按时间序 rename 的 traces |
| 通信解析 | `outputs/m22/h20_opt_profile_comm_profile.csv`（本机离线） |
| 图 | `figures/fig9_m22_tp_scaling_h20_formal.png` |
| 环境 / 日志 | `logs/m22/environment.txt`、`logs/m22/h20_formal_*.log` |
| 不变量 | `python scripts/check_invariants.py --module m22` → **22 PASS**（tag=`h20_formal`） |

Smoke / `local_preflight` 产物保留作对照，**不进入正式结论**。

## 3. 离线复核（本机，无需 GPU / 无需 H20）

```bash
source .venv/bin/activate
python scripts/check_invariants.py --module m22
python scripts/m22_tensor_parallel.py --shard summarize --tag h20_formal
python scripts/m22_figures.py --tag h20_formal
# 仅解析 opt 命名后的 traces（--tag 只影响输出文件名，不过滤输入目录）
python scripts/m22_trace_analyze.py --trace-dir outputs/m22/traces --tag h20_opt_profile \
  --out outputs/m22/h20_opt_profile_comm_profile.csv
```

重跑正式矩阵需要等价多卡环境与模型；**不要**用 `local_preflight`（RTX 5080 TP=1）冒充 H20 多卡结果。
