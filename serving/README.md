# Serving / Triton model repository（M20）

> **状态**：M20-B smoke 已在本机跑通（2026-08-08）。正式 M20-C 未跑。

## 结构

```text
serving/triton_model_repository/
  qwen25_05b/
    config.pbtxt
    1/
      model.json
```

## 关键事实

- 镜像：`nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3`（digest `sha256:31e20bf…`）
- 模型必须挂 **整个** HF hub 目录（含 `blobs/`），不要只挂 `snapshots/`（symlink 会断）
- Non-stream：`POST /v2/models/qwen25_05b/generate`
- Stream：`POST /v2/models/qwen25_05b/generate_stream`（`/generate`+`stream:true` 会报错）
- Metrics：`http://127.0.0.1:8002/metrics`（需 `REPORT_CUSTOM_METRICS=true`）
- Same-runtime Native：同镜像内 `vllm serve <same snapshot path> ...`（与 Triton **串行**，16GB 不能双引擎）

## 启动示例（smoke）

```bash
MODEL_ROOT=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct
REPO=$PWD/serving/triton_model_repository

sudo docker run --rm --gpus all --net=host --shm-size=1G \
  -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  -v "$REPO:/models_repo:ro" \
  -v "$MODEL_ROOT:/models/qwen05b:ro" \
  nvcr.io/nvidia/tritonserver:26.07-vllm-python-py3 \
  tritonserver --model-repository=/models_repo
```

细节与证据：`notes/m20/b_triton_runtime_unblocked.md`。
