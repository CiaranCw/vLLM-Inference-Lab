# M20 · 实测结论（Production Serving + Triton · A/B smoke + C same-runtime）

> **状态**：**M20 COMPLETE**（2026-08-10）。T + A + B + **C（QPS 20/32/40）** + O。  
> 设计见 [`production_serving_and_triton.md`](production_serving_and_triton.md)；  
> B 解阻见 [`b_triton_runtime_unblocked.md`](b_triton_runtime_unblocked.md)；A 见 [`a_native_smoke_results.md`](a_native_smoke_results.md)。
>
> **求职一句话**：能在控制 Runtime Confound 的前提下，把 Triton + vLLM Backend 与同镜像  
> `vllm serve` 做真实开环对照，并说明 overhead 主要落在哪一段延迟。
>
> **适用边界**：单卡 RTX 5080 / WSL2 / Docker  
> `nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3`（digest `sha256:31e20bf…`） /  
> `Qwen2.5-0.5B-Instruct` / `enforce_eager` / Shape A **in256/out128** /  
> 客户端 **online streaming** 口径（TTFT/ITL/E2E）。  
> **不得**用 host-M12 数字减 container 数字称 Triton overhead。  
> 绝对值不外推 compiled、更大模型、多卡、生产 K8s。

## 环境

| 项 | 值 |
|---|---|
| 镜像 | `nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3` |
| 容器 vLLM | `0.24.0+092c4842.nv26.7…`（≠ host lab `ee0da84`） |
| 容器 torch / CUDA | `2.13.0a0…nv26.07` / 13.3 |
| Host GPU | RTX 5080 16GB，driver 610.62 |
| EngineArgs（C0=C1） | `gpu_memory_utilization=0.45`，`max_model_len=2048`，`max_num_seqs=64`，`max_num_batched_tokens=4096`，eager，chunked prefill on，prefix off |
| 负载 | Poisson `burstiness≡Exp` 到达；`temperature=0`，`ignore_eos`，每配置 **30 warmup + 200 measure** |
| 统计 | median difference 的 Bootstrap 95% CI（seed=0，10000 次），字段 `c0 − c1`（负值 = C1 更慢） |

**C0** = 同镜像内 `vllm serve`（OpenAI `/v1/completions` stream）  
**C1** = 同镜像内 `tritonserver` + vLLM backend（`/v2/models/qwen25_05b/generate_stream`）  
两侧 **串行**（16GB 不能双引擎）。

脚本：`scripts/m20_same_runtime_bench.py`  
数据：`outputs/m20/c_same_runtime_{sweep,effect,config}.*` + `outputs/m20/raw/c_c{0,1}_qps*.jsonl`

---

## 0. 一句话结论

1. **Same-runtime 下 Triton+backend 相对 `vllm serve` 有可测延迟代价**（TTFT/E2E 的 CI 在 QPS 20/32/40 **全部排除 0**）。  
2. **代价随 offered QPS 急剧放大在 TTFT（排队）**：QPS 20 时 TTFT median 差约 **22ms**；QPS 32/40 到约 **0.69s / 0.93s**。  
3. **稳态 ITL（解码步）接近**：两侧 ITL p50 约 **9–12ms**，说明主差不在「每 token 算得更慢」，而在 **入口路径 / 调度排队**。  
4. A/B smoke 与 C 可行性此前已确认；本轮补齐正式 C → 模块可标 COMPLETE。

---

## 1. M20-C 主表（streaming 客户端）

`outputs/m20/c_same_runtime_sweep.csv`（每格 n=200，success_rate=1.0）

| QPS | Backend | TTFT p50 | TTFT p99 | E2E p50 | E2E p99 | ITL p50 |
|---:|---|---:|---:|---:|---:|---:|
| 20 | C0 | 19.4 | 33.1 | 1193 | 1298 | 9.25 |
| 20 | C1 | 41.3 | 225 | 1357 | 1586 | 10.22 |
| **32** | C0 | **24.2** | **36.8** | **1415** | **1462** | 10.95 |
| **32** | C1 | **719** | **888** | **2137** | **2266** | 11.64 |
| **40** | C0 | **26.1** | **212** | **1482** | **1665** | 11.38 |
| **40** | C1 | **952** | **1098** | **2352** | **2434** | 11.51 |

单位 ms。ITL = 单请求内 stream chunk 间隔均值的分布分位数。

### 1.1 Bootstrap 效应（`c_same_runtime_effect.csv`）

`median_diff = median(C0) − median(C1)`（负 = C1 更慢）

| QPS | 指标 | C0 med | C1 med | diff | 95% CI | 排除 0？ |
|---:|---|---:|---:|---:|---|:--:|
| 20 | TTFT | 19.4 | 41.3 | **−21.9** | [−24.9, −19.9] | ✅ |
| 20 | E2E | 1193 | 1357 | **−164** | [−172, −156] | ✅ |
| 32 | TTFT | 24.2 | 719 | **−694** | [−734, −569] | ✅ |
| 32 | E2E | 1415 | 2137 | **−722** | [−767, −644] | ✅ |
| 40 | TTFT | 26.1 | 952 | **−926** | [−995, −720] | ✅ |
| 40 | E2E | 1482 | 2352 | **−869** | [−886, −847] | ✅ |

### 1.2 机制解读（本数据支持的）

- **低负载（QPS 20）**：C1 已有稳定但较小的 TTFT/E2E 溢价 → 存在固定的 Triton HTTP/backend 路径成本。  
- **近 M12 容量点（32/40）**：C0 TTFT p50 仍 ~24–26ms；C1 TTFT p50 跳到数百 ms～近 1s → **队列在 Triton 入口侧堆积**，不是 ITL 翻倍。  
- **成功率为 1.0**：本轮未观察到错误主导；差异是延迟分布，不是失败率。  
- **与 M12 的关系**：M12 给出 host native 的容量背景（Shape A 最大稳定 QPS≈40）。本轮 C **不**声称「Triton 把容量从 40 降到 X」——未按 M12 SLO 三 seed 网格重做容量判定；只报告 same-runtime 延迟对照。

---

## 2. A / B 摘要（已完成，非本轮重跑）

| 项 | 结果 | 证据 |
|---|---|---|
| A1–A3 host native smoke | stream/prefix/health-restart PASS | `a_native_smoke_results.md` |
| B Triton smoke | health/load/generate/`generate_stream`/metrics PASS | `b_triton_runtime_unblocked.md` |
| Same-runtime 可行性 | 同镜像可跑 `vllm serve` | `logs/m20/same_runtime_vllm_serve.txt` |

---

## 3. 图

```bash
python scripts/m20_figures.py
```

→ `figures/fig10_m20_same_runtime.png`

---

## 4. 明确不支持 / 未做

| 项 | 状态 |
|---|---|
| 用 host-M12 − Triton 称 overhead | **禁止**（本轮未做） |
| Triton OpenAI frontend（:9000）对照 | **未测**（C1 用原生 `generate_stream`） |
| compiled / CUDA Graph | **未测** |
| 多卡 / K8s / gRPC 压测 | **未测** |
| 按 M12 SLO 重测「Triton 最大稳定 QPS」 | **未做**（可选扩展） |
| M21 Prom/Grafana 深监控 | 留给 M21 |

---

## 5. 复现

```bash
export PATH="$HOME/bin:$PATH"   # WSL 若 docker 组未进 session
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
python scripts/m20_same_runtime_bench.py --phase all --rates 20,32,40 --warmup-n 30 --measure-n 200
python scripts/m20_same_runtime_bench.py --phase summarize --rates 20,32,40
python scripts/m20_figures.py
```

需：Docker + NVIDIA Container Toolkit、已拉取上述镜像、本地 HF 缓存中的 0.5B。
