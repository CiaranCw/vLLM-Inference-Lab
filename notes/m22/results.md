# M22 · 实测结论（TP=1/2/4 · D1 显存/KV · D2 decode · D3 prefill · P1 通信）

> **状态**：**TP 主线 T+E+O+H COMPLETE**（数据采集 2026-08-09；本机收口 2026-08-09）。
> 设计 / 源码定位见 [`tensor_parallel.md`](tensor_parallel.md)；执行留痕见 [`execution_record.md`](execution_record.md)。
> 本文件只放结果与结论。
>
> **指标口径（勿与 M12 混淆）**：所有延迟均为 **in-process engine** 路径；
> 字段名用 `steady_step_ms_*` / `first_token_ms_*`，**不是**在线 TTFT/ITL。
> **`speedup_vs_tp1` 按吞吐（`output_tps`）派生**——大 batch 时会被 **并发 prefill 提速**拉高，
> 解读「小 batch 时 TP 为何更慢」必须以 **`steady_step_ms` + `decode_effect`** 为准。
>
> **适用边界**：4× NVIDIA H20（NV18）/ driver 535.247.01 / vLLM **0.24.0** wheel /
> `Qwen2.5-7B-Instruct` / `enforce_eager=True` / `async_scheduling=False` /
> in-process / `temperature=0` / `VLLM_ENABLE_V1_MULTIPROCESSING=0`。
> **所有绝对值仅限本配置**；不外推 compiled、其他模型、跨节点、PP/DP/EP、在线 serving。
>
> **未做（仍为 ⬜）**：Pipeline / Data / Expert Parallel、CP、P-D 分离——无脚本、无数据。

## 环境（`logs/m22/environment.txt`）

| 项 | 值 |
|---|---|
| vLLM | **0.24.0**（H20 site-packages；本机 lab pin 仍为 `ee0da84`） |
| 模型 | `Qwen2.5-7B-Instruct` |
| GPU | 4× H20，97871 MiB/卡，NV18 |
| torch / CUDA / Python | 2.11.0+cu130 / 13.0 / 3.11.13 |
| 采样 | 每配置 **5 warmup + 15 measure**；汇总只用 measure 且 `success=True` |
| 统计 | median difference 的 **Bootstrap 95% CI**（与 M17 同口径，固定 seed），见 `h20_formal_decode_effect.csv` |
| 正式 tag | `h20_formal` |
| 不变量 | `check_invariants.py --module m22` → **22 PASS / 0 FAIL / 0 PENDING** |

---

## 0. 一句话结论

1. **小 batch 稳态 decode 单步：TP 显著更慢**（相对 TP=1 约 +19–24%，Bootstrap CI 全程排除 0）——假说成立。  
2. **长 / 并发 prefill：TP 明显更快**（P=4096 时 TP=4 相对 TP=1 约 **3.39×** first-token）——计算受限侧切分有效。  
3. **大 batch 的吞吐 `speedup_vs_tp1>1` 主要来自 batch prefill 变快，不是 decode 单步变快**（bs=64 时 step 仍慢 ~19%，但 `first_token` 804→230 ms）。  
4. **每卡权重 ≈1/TP，KV 未复制**（7B / 4 KV heads 整除）；KV token 容量随 TP 上升。  
5. **P1**：TP≥2 观察到 custom all-reduce / NCCL / symm_mem；TP=1 无跨卡通信 kernel（`comm_share=0`）。

---

## 1. D1 · 每卡权重与 KV（`h20_formal_memory_kv.csv`）

观测方法：`collective_rpc(callable)`（非 driver 侧 `mem_snapshot`）。下列为各 TP 的 **rank0**（同 TP 各 rank 一致）。

| TP | weight_MiB | vs TP1 | 理想 1/TP | available_kv_MiB | KV tokens | attn/KV heads per rank | kv_heads_replicated |
|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 14632.69 | 1.000 | 1.000 | 66790.5 | 1 221 312 | 28 / 4 | False |
| 2 | 7327.57 | **0.501** | 0.500 | 74021.6 | 2 707 072 | 14 / 2 | False |
| 4 | 3720.51 | **0.254** | 0.250 | 77727.9 | 5 685 232 | 7 / 1 | False |

### 可支持
- 权重接近 **1/TP**（残留约 0.1–0.4 pp，符合 layernorm / 小复制项；属 **expected observation**，非硬不变量）。
- **`kv_heads_replicated=False`**：4 KV heads 在 TP∈{1,2,4} 均可整除，无复制。
- KV **容量等价值**随 TP 上升（tokens：1.22M → 2.71M → 5.69M）；运行仍受 `max_num_seqs` 等限制，**不等于**实测最大并发。

---

## 2. D2 · 稳态 decode 单步（回答「小 batch 时 TP 为何更慢」）

数据：`outputs/m22/h20_formal_decode.csv`（每格 n=15）、效应：`h20_formal_decode_effect.csv`。

`median_diff_ms = tp1_median − tp_k_median`（负值 = TP-k **更慢**）。

### 2.1 主表：`steady_step_ms_median`（ms）

| batch | TP=1 | TP=2 | TP=4 | TP2 − TP1（CI） | CI 排除 0？ | TP4 − TP1（CI） | CI 排除 0？ |
|---:|---:|---:|---:|---|:--:|---|:--:|
| 1 | 10.132 | 12.367 | 12.525 | **−2.235** [−2.266, −2.219] | ✅ | **−2.394** [−2.424, −2.375] | ✅ |
| 4 | 10.263 | 12.514 | 12.607 | **−2.251** [−2.272, −2.224] | ✅ | **−2.343** [−2.372, −2.331] | ✅ |
| 16 | 10.513 | 12.994 | 12.928 | **−2.480** [−2.503, −2.455] | ✅ | **−2.415** [−2.439, −2.389] | ✅ |
| 64 | 11.759 | 14.024 | 14.017 | **−2.265** [−2.284, −2.232] | ✅ | **−2.258** [−2.281, −2.222] | ✅ |

相对变慢（`rel_change_pct`，来自 effect CSV）：TP=2 约 **−19～−24%**（点估计；负号=变慢），TP=4 同量级。  
四档 batch **全部 CI 排除 0** → 本配置下「小 batch（乃至本矩阵全 batch）稳态单步 TP 更慢」是**显著现象**，不是噪声。

### 2.2 吞吐派生指标（勿单独当「decode 加速」）

| batch | TP2 `speedup_vs_tp1` | TP2 eff | TP4 `speedup` | TP4 eff | 备注 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.827 | 0.413 | 0.816 | 0.204 | 与 step 变慢一致 |
| 4 | 0.855 | 0.428 | 0.862 | 0.216 | 同上 |
| 16 | 0.934 | 0.467 | 0.992 | 0.248 | step 仍慢；吞吐接近打平 |
| 64 | **1.169** | 0.584 | **1.365** | 0.341 | **step 仍慢 ~19%**，但 completion 更快 |

**机制拆解（bs=64）**——同一批 CSV 行：

| TP | steady_step_ms | first_token_ms | completion_ms | output_tps |
|---:|---:|---:|---:|---:|
| 1 | 11.76 | **804.2** | 1610.5 | 2543 |
| 2 | 14.02 | **426.8** | 1378.1 | 2972 |
| 4 | 14.02 | **230.2** | 1180.2 | 3470 |

→ 大 batch「吞吐加速」主要是 **64 路并发 prefill 被 TP 切开后 first-token 大降**，摊薄了整段 completion；  
**不是**「decode 单步变快」。面试表述必须把这两层拆开。

### 2.3 与 T 假说的对齐
`tensor_parallel.md` §1.2：decode GEMV 访存受限 + 每层 2 次 all-reduce 固定开销 → 小 batch 可能更慢。  
本轮在 **eager / 7B / H20 NV18** 下：D2 全 batch 的 step 延迟均支持该方向；  
**batch 增大并未扭转 step 延迟**（仍慢），只扭转了 **含 prefill 的吞吐口径**。

---

## 3. D3 · Prefill first-token（`h20_formal_prefill.csv`，batch=1）

| prompt_len | TP=1 med (ms) | TP=2 | speedup | TP=4 | speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 29.33 | 17.21 | 1.70 | 13.02 | 2.25 |
| 1024 | 108.15 | 58.68 | 1.84 | 33.47 | 3.23 |
| 4096 | 424.27 | 226.86 | 1.87 | 125.13 | **3.39** |

- 全部 `prefill_step_count=1`，Σ scheduled = prompt_len（未触发分块把长 prompt 切成多步的本矩阵设定）。  
- 长度越长，TP=4 相对加速越接近线性容量直觉（仍 **<4**，通信与固定开销仍在）。  
- **计算受限侧**：TP 扩的是可切分的 GEMM，本数据支持「长 prefill 上 TP 有壁钟收益」。

---

## 4. 跨 TP 输出一致性（诊断，非硬通过条件）

`h20_formal_output_consistency.csv`：**120** 组对照（measure 段）。

| batch | output_hash 完全一致 |
|---:|---|
| 1 | **30/30** |
| 4 | **30/30** |
| 16 | **0/30** |
| 64 | **0/30** |
| 合计 | **60/120** |

- 输出**长度**跨 TP 一致（硬不变量 PASS）；prompt hash 一致。  
- bs≥16 的 token hash 不一致：**计划不要求 bit-identical**（切分/归约顺序 → 有限精度 greedy 可翻转；与 M5/M19 同类）。  
- **不得**把 hash 不一致写成「TP 实现 bug」；也**不得**写成「完全数值等价」。

---

## 5. P1 · Collective communication 证据

### 5.1 正式 profile
- `h20_formal_tp{2,4}_profile.jsonl`：各 3 case（`decode_small_batch` / `decode_large_batch` / `prefill_long`）均 `success=True`。  
- **文件名缺口**：chrome trace 落盘时文件名曾**全部**带 `decode_small_batch` 前缀；每 rank 实际有 3 个文件（体积约 1.6M / 1.6M / 125K），对应三次 profile。  
  → 正式包不是「没跑 large/prefill」，而是 **命名未区分 case**。

### 5.2 可选补强（`h20_opt_profile`）
按执行序 rename（见 `logs/m22/opt_trace_rename_map.txt`），并补 **TP=1** profile：

| | TP1 | TP2 | TP4 |
|---|---|---|---|
| cases success | 3/3 | 3/3 | 3/3 |
| chrome traces | 3 | 6 | 12 |

离线 `h20_opt_profile_comm_profile.csv`（21 traces）要点：

- **TP=1**：`comm_share=0`，backends 空 —— 无跨卡 collective。  
- **TP≥2 decode**：出现 `custom_all_reduce` / `nccl`（部分 case 还有 `torch_symm_mem`）。  
- **prefill_long**：通信占比明显低于小 batch decode（与「计算占比上升」一致）。  
- `comm_share` = 通信 kernel 占 **GPU busy** 的比例，**不是**净 wall-time 损失（可与计算重叠）。

> case 标签对 opt traces 依赖 **时间序启发式 rename**，写论文级断言时应同时引用 JSONL 的 `profile_prefix` 与 rename map。

### 5.3 Profile 内稳态单步（opt，与 formal D2 同方向）

| case | TP1 | TP2 | TP4 |
|---|---:|---:|---:|
| decode_small (bs=1) | 15.97 | 19.75 | 20.25 |
| decode_large (bs=16) | 16.18 | 19.86 | 20.36 |

（绝对值高于 formal D2：profile 路径开销不同；**只比相对方向**，不与 D2 绝对值混用。）

---

## 6. 图

离线生成（无需 GPU）：

```bash
python scripts/m22_figures.py --tag h20_formal
```

产出：`figures/fig9_m22_tp_scaling_h20_formal.png`  
（左：decode step vs batch；中：吞吐 speedup/efficiency；右：prefill first-token vs prompt_len）

---

## 7. 明确不支持 / 未执行

| 项 | 状态 |
|---|---|
| PP / DP / EP / CP / P-D | **⬜ 未执行**（无实现） |
| compiled / CUDA Graph 下的 TP | **未测** |
| 跨节点 NCCL / 多机 | **未测**（本机 NV18 单节点） |
| 用 `speedup_vs_tp1` 单独证明「decode 加速」 | **不支持**（见 §2.2） |
| 跨 TP bit-identical 输出 | **不要求**；bs≥16 实测常不一致 |
| 本机 `local_preflight` / `h20_smoke` | **不是**正式结论数据源 |

---

## 8. 证据索引（本模块）

| 结论 | 证据 |
|---|---|
| D2 step 变慢且显著 | `h20_formal_decode.csv` + `h20_formal_decode_effect.csv` |
| 大 batch 吞吐升来自 prefill | 同 CSV 的 `first_token_ms` vs `steady_step_ms` |
| D3 prefill 加速 | `h20_formal_prefill.csv` |
| 权重≈1/TP、KV 不复制 | `h20_formal_memory_kv.csv` |
| 输出长度一致 / hash 诊断 | `h20_formal_output_consistency.csv` |
| 通信后端 | `h20_opt_profile_comm_profile.csv` + traces |
| 不变量 | `scripts/check_invariants.py --module m22` |
| 原始样本 | `outputs/m22/raw/h20_formal_tp*_run.jsonl` |
