# M20-B · Triton runtime 解阻 + 最小 smoke（2026-08-08）

> **本轮目标（历史）**：消掉 `BLOCKED_ON_TRITON_RUNTIME`，确认 M20-C same-runtime **可执行**。  
> **本轮非目标**：正式 QPS（已于 2026-08-10 在 `results.md` 完成）。  

## 1. Runtime 安装与验证

| 项 | 结果 |
|---|---|
| Docker Engine | **29.7.2**（`docker-ce` resolute）；`hello-world` OK |
| NVIDIA Container Toolkit | **1.19.1**；`nvidia` runtime 写入 `/etc/docker/daemon.json` |
| `--gpus all` | 容器内见 **RTX 5080**（driver 610.62） |
| 镜像 | `nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3` |
| Digest | `sha256:31e20bfbc65055d6b85553a00d45342a62c506874aa996c11ae53650151f05b1` |
| Size | ~11.0 GB |

日志：`logs/m20/install_docker.log`、`install_nvidia_container_toolkit.log`、`pull_triton_26.07.log`。

### 容器精确版本（`logs/m20/container_versions.txt`）

| 组件 | 版本 |
|---|---|
| Triton Release / Server | **26.07** / **2.71.0**（build 59591852） |
| Python | **3.12.3** |
| PyTorch | **2.13.0a0+9186a08b2c.nv26.07** |
| CUDA (torch) | **13.3** |
| vLLM | **0.24.0+092c4842.nv26.7.59534043**（`vllm.__version__`=`0.24.0+092c4842.dev`） |

**与 lab host 不同（必须写进 C 设计）**：

| | Host lab | Triton container |
|---|---|---|
| vLLM | `0.24.1.dev0+gee0da84ab`（editable `ee0da84`） | NVIDIA build `0.24.0+092c4842…` |
| torch | `2.11.0+cu130` | `2.13.0a0+…nv26.07` / CUDA 13.3 |

→ 正式 M20-C **只能**在同镜像内对照 `vllm serve` vs `tritonserver+backend`，禁止用 host-M12 减 container。

### Host 环境影响检查

安装后复测：host `torch 2.11.0+cu130`、`vllm 0.24.1.dev0+gee0da84ab`、`cuda_available True`、driver **610.62** 不变。未改 CUDA 本地 repo、未动 `vllm-src` / M22 pack。

## 2. 模型 bind-mount

- Host：`~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct`
- Container：`/models/qwen05b`（**必须挂整个 hub 目录**，因 snapshot 内是指向 `../../blobs` 的 symlink）
- `model.json`：`/models/qwen05b/snapshots/7ae557604adf67be50417f59c2c2f167def9a775`
- 容器内 `AutoConfig... local_files_only=True` → `qwen2` / 24 layers；权重 ~0.92 GiB 可读。

## 3. Triton + vLLM backend 最小 smoke

Repo：`serving/triton_model_repository/qwen25_05b/`（`config.pbtxt` + `1/model.json`）。

| 检查 | 结果 |
|---|---|
| `/v2/health/live` + `/ready` | PASS |
| 模型加载 | PASS（`nv_model_load_duration_secs≈31.6`） |
| `POST .../generate` non-stream | **HTTP 200**，有 `text_output` |
| Streaming | **`/generate` + stream=true 不行**（报错提示用 stream endpoint）；**`POST .../generate_stream` HTTP 200**，SSE `data:` 多 chunk |
| `:8002/metrics` | PASS；含 `vllm:prompt_tokens_total` / `vllm:time_to_first_token_seconds` 等 |

证据：`outputs/m20/raw/b_triton_generate_smoke.txt`、`b_triton_stream_smoke.txt`、`logs/m20/tritonserver_smoke.log`。

EngineArgs（smoke，非 C 正式）：`enforce_eager`、`max_model_len=2048`、`gpu_memory_utilization=0.40`、`max_num_seqs=32`、`max_num_batched_tokens=4096`、`enable_chunked_prefill=true`；env `VLLM_USE_FLASHINFER_SAMPLER=0`、`HF_HUB_OFFLINE=1`。

## 4. M20-C same-runtime 可行性

| 问题 | 结论 |
|---|---|
| 同 Triton 镜像内能否 `vllm serve`？ | **能**。`/usr/local/bin/vllm` 存在；`--gpus` 下 `vllm serve --help` OK |
| 同 build / 同模型 / 同 EngineArgs？ | **能对齐到同一 package + 同一 bind-mount 路径 + 同一 CLI/json 参数集合** |
| 实测 | 停 Triton 后，同镜像 `vllm serve` @8021 → `/health` + `/v1/chat/completions` **PASS**；`system_fingerprint=vllm-0.24.0+092c4842.dev-…` |

证据：`logs/m20/same_runtime_vllm_serve.txt`。

**约束（正式 C 前已知）**：

1. 16GB 上 **不能同时** 跑 Triton backend 引擎 + `vllm serve` 引擎 → C0/C1 必须串行、同镜像、同参数。
2. API 面不同：OpenAI `/v1/chat/completions` vs Triton `/v2/models/.../generate[_stream]` → 负载发生器需适配，比较口径要统一到 TTFT/ITL/E2E/吞吐/错误率。
3. Streaming：Triton 用 **`generate_stream`**，不是 `/generate`+`stream:true`。

## 5. 模块状态

| | |
|---|---|
| Runtime blocker | **已消除** |
| M20-B 最小 smoke | **PASS** |
| M20-C 可行性 | **CONFIRMED** |
| M20-C 正式 QPS | **DONE**（20/32/40；见 `results.md`） |
| M20 overall COMPLETE | **是**（2026-08-10） |

容器跑完已清理。正式 C 数据：`outputs/m20/c_same_runtime_*.csv`。
