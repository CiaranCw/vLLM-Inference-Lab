# M20 · Production Serving + Triton（设计冻结 / 修订版）

> **求职目标一句话**：能够部署并解释 Triton + vLLM Backend，并在控制 Runtime Confound
> 的前提下，与 Native vLLM 做真实服务对照。
>
> **状态（2026-08-10）**：**M20 COMPLETE**（A/B smoke + C same-runtime QPS 20/32/40）。  
> 结论见 [`results.md`](results.md)。
>
> **边界**：单卡 RTX 5080 / WSL2 / Qwen2.5-0.5B-Instruct / eager（与 M12 Shape A 对齐）。

---

## 0. 与既有模块的边界

| 模块 | M20 做什么 / 不做什么 |
|---|---|
| **M12** | 引用容量背景（QPS 32 稳定高负载、40 near-knee）。**禁止**用 host-M12 数字减 container 数字称 Triton overhead |
| **M17** | APC **性能**证据源。M20 只做 serving 侧「如何启用/观测」smoke |
| **M19** | 无关 |
| **M21** | metrics 能访问即可；Prom/Grafana/admission/backpressure 深度留给 M21 |

---

## 1. T · 架构（求职够用）

```text
Client
  ↓  HTTP/gRPC
Triton Server          ← 协议、模型生命周期、health/metrics、请求路由
  ↓
vLLM Backend (Python)  ← 把 Triton 请求转成 vLLM AsyncEngine 调用
  ↓
vLLM AsyncEngine       ← continuous batching / paged attention / Scheduler 仍在这里
  ↓
Scheduler / ModelRunner / GPU
```

**关键事实（官方 vLLM backend README）**：

- 请求进入 backend 后尽快放到 **vLLM AsyncEngine**；inflight batching 与 paged attention **由 vLLM 处理**。
- Triton **不替代** vLLM Scheduler。
- Model repository 典型含 `config.pbtxt` + `1/model.json`；`model.json` 键值喂给 vLLM EngineArgs
  （见 vLLM `arg_utils.py`）。
- 预构建镜像：`nvcr.io/nvidia/tritonserver:<xx.yy>-vllm-python-py3`。
- Generate 示例：`POST /v2/models/<name>/generate`；metrics：`:8002/metrics`（`vllm:` 前缀字段需
  `REPORT_CUSTOM_METRICS` + `disable_log_stats` 未关）。

**不深挖**：Triton C++ core、shared-memory、ensemble、sequence batcher、custom backend、K8s。

### Native OpenAI vs Triton API（形态差异）

| | Native `vllm serve` | Triton + vLLM backend |
|---|---|---|
| 典型端点 | `/v1/chat/completions`、`/health`、`/metrics` | `/v2/models/.../generate`、HTTP:8000、gRPC:8001、metrics:8002 |
| 配置入口 | CLI EngineArgs | model repository + `config.pbtxt` + `model.json` |
| Streaming | OpenAI SSE `stream=true` | generate `parameters.stream` / decoupled 模式（以所选版本为准） |

---

## 2. Gate0 结果（只读）

详见 [`logs/m20/gate0_probe.txt`](../../logs/m20/gate0_probe.txt)。

| 项 | 结果 |
|---|---|
| G0.1 Docker/Podman | **ABSENT** |
| G0.2 用户可跑 container | N/A |
| G0.3 NVIDIA Container Toolkit | **ABSENT** |
| G0.4 `--gpus` 见 5080 | N/A |
| G0.5 Host driver | 610.62（候选 26.07 矩阵要 610.43.02；host 更新，预期兼容待实跑） |
| G0.6 NGC | 可达 |
| G0.7 0.5B cache | ~954M 存在 |
| G0.8 磁盘/显存 | 563G free；16GB 对 0.5B 合理 |

**候选镜像（未拉取、未冻结）**：`nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3`  
矩阵：vLLM `0.24.0+092c4842.nv26.7…`（**NVIDIA build ≠ lab `ee0da84`**）、CUDA 13.3、~10.3GB。

---

## 3. 冻结实验矩阵

### M20-A · Native production semantics（轻量；无 Bootstrap）

| ID | 内容 | 规模 |
|---|---|---|
| A1 | `stream=true` vs `stream=false` 协议/首包时机；non-stream 禁用 streaming ITL 口径 | 少量请求 |
| A2 | `--enable-prefix-caching` serving smoke + 共享 prefix；引用 M17 | smoke |
| A3 | `/health` → stop/kill → FAIL → restart → PASS | 1 次故障环 |

Server 默认对齐 M12 指纹（见 `notes/m12/serving.md` §1），A2 仅改 prefix 开关。

### M20-B · Triton deployment（核心；需 Docker）

- 拉候选镜像 → model repo（0.5B bind-mount）→ `tritonserver` → health + generate smoke。
- 记录：镜像 tag、容器内 vLLM version/build、启动步骤、配置文件清单。

### M20-C · Same-runtime comparison（核心）

```text
同一 container / GPU / model / vLLM package / PyTorch·CUDA / EngineArgs / load generator
C0: container 内 vllm serve
C1: 同 container 内 tritonserver + vLLM backend
```

执行前必须核实：容器内是否有可用的 `vllm serve`；backend 真实入口；EngineArgs 映射。  
若不存在，据实调整命令，**不强造**。

| | 值 |
|---|---|
| Model | Qwen2.5-0.5B-Instruct |
| Shape | in256 / out128（M12 Shape A） |
| Arrival | Poisson（与 M12 对齐，除非 Triton 客户端限制） |
| 主点 | **QPS 32、40** |
| Optional | QPS 20 sanity |
| 统计 | 仅 C：warmup + formal measure + Bootstrap 95% CI |

**比较**：TTFT/ITL/E2E P50–99、request/output 吞吐、error rate；startup、health、metrics、配置文件事实清单。

**禁止**：`M12_host − Triton_container` 称为 Triton overhead。

---

## 4. 成功标准

**M20 COMPLETE** 当且仅当：T + B（真启动+生成）+ C（same-runtime ≥1 正式点，最好 32/40）+ O（repo/backend/config→Engine）。

**Gate0 blocked**：T 可完成；A 可完成；B/C blocked → **overall NOT COMPLETE** / `BLOCKED_ON_TRITON_RUNTIME`。

---

## 5. 产出布局

```text
notes/m20/
serving/                 # model repo 模板、workload yaml（B 就绪后）
scripts/m20_*.py
outputs/m20/
logs/m20/
reports/07_online_serving.md   # 仅 COMPLETE 时
```

---

## 6. 授权需求（停在此处等待所有者）

为实现 B/C，需要所有者明确授权：

1. **安装 Docker**（系统包；本机有 passwordless sudo，但按 AGENTS **不得擅自使用**）  
2. **安装 NVIDIA Container Toolkit**（GPU passthrough）  
3. **拉取** `nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3`（~10GB；磁盘充足）  

替代：若所有者提供已装好 Docker 的环境/其他机器，可迁移执行 B/C。

无授权时：完成本文件 + M20-A host smoke 后**暂停 M20**，可转其他单卡模块；**不**用重复 M12 冒充完成。
