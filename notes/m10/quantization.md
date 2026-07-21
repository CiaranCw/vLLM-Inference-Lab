# M10 · 量化对比（★核心验收 #9）：BF16 / 在线 FP8 W8A8 Linear / FP8 KV Cache

> 目标：在本机 sm_120 上，用**固定公共参数 + 单变量对照**，比较三种量化在**显存 / 性能 / 质量 / 源码机制**上的真实差异，所有结论由真实配置、真实 Tensor、真实运行与逐样本原始数据支撑，**如实报告（含反常项）**。
> 边界：进程内 `LLMEngine`，主实验 `enforce_eager=True`（隔离 cudagraph/compile）；指标为**进程内 proxy**（`engine_first_step_latency_ms`≈TTFT、`steady_decode_step_latency_ms`≈TPOT，**非在线**）。不改 vLLM 源码、不下载量化模型。
>
> 证据：`outputs/m10/{summary.json,latency.csv,summary.csv, configs/*, memory/*, quality/*, raw/*_perf.jsonl}`、`logs/m10/*`。复现见文末。

## 0. 环境与配置矩阵（公共参数固定）
RTX 5080 sm_120(12.0) / torch2.11+cu130 / triton3.6 / vLLM ee0da84 / Qwen2.5-0.5B-Instruct / TP=1。
**COMMON（所有配置一致）**：`dtype=auto(bf16), max_model_len=2048, gpu_memory_utilization=0.6, seed=0, enforce_eager=True, max_num_seqs=64, max_num_batched_tokens=4096, enable_prefix_caching=False` + env `VLLM_USE_FLASHINFER_SAMPLER=0`。**各配置仅改 3 个变量**（落盘 `outputs/m10/configs/*.json`）：

| 配置 | Linear | KV | attention_backend | 对照目的 |
|---|---|---|---|---|
| **C0** | BF16 | BF16 | FLASH_ATTN | 全局基线 |
| **C1** | 在线 FP8 **W8A8** | BF16 | FLASH_ATTN | C1−C0 = FP8 Linear 净影响（同后端）|
| **C0T** | BF16 | BF16 | TRITON_ATTN | C2 的**同后端基线**（全维度参与）|
| **C2** | BF16 | FP8(**未校准, scale=1.0**) | TRITON_ATTN | C2−C0T = FP8 KV 净影响（同后端）|
| **C2dyn** | BF16 | FP8(**动态 scale**) | TRITON_ATTN | vs C0T = 动态 scale KV 影响 |
- **归因**：`FP8 Linear=C1−C0`、`FP8 KV=C2−C0T`、`后端=C0T−C0`；KV 容量倍率用 **C2/C0T**（后端不同，不能用 C2/C0）。
- **sm_120 注记（如实）**：fp8 路径有告警 `SM 12.x requires CUDA >= 12.9`——不阻断运行，但 **fp8 在 sm_120 支持较新、kernel 可能未充分优化**，这与下文"fp8 decode 未提速"相关。

## 1. 源码机制（只读，真实调用链）
**在线 FP8 W8A8 Linear**（`quantization="fp8"` → `Fp8Config`，`activation_scheme="dynamic"`）：
- `Fp8LinearMethod.create_weights` 注册 `weight`(fp8_e4m3fn)、`weight_scale`(**per-tensor，shape [1] fp32**)、`input_scale`；`apply` 前向时对激活做**动态 per-token 量化**，调 `torch._scaled_mm`（选中 `CutlassFP8ScaledMMLinearKernel`），输出 bf16。
- **不是"先反量化成 bf16 再 GEMM"**，而是 **fp8×fp8 的 scaled MM，scale 作为 kernel 参数直接消费**。
- 量化对象 = **96 个 linear 权重**（24 层 ×{qkv_proj,o_proj,gate_up_proj,down_proj}）；**embed_tokens / lm_head（0.5B tie_word_embeddings）/ norm / bias 保持 bf16**。

**FP8 KV Cache**（`kv_cache_dtype="fp8"`）：KV tensor 以 fp8 存；写：`reshape_and_cache_flash(..., k_scale, v_scale)`（M4/M8 已见签名）；读：**TRITON_ATTN** 用 `_k_scale/_v_scale` 反量化后算注意力（bf16）。scale 来源：`Attention._k_scale/_v_scale` 默认 `torch.tensor(1.0)`；仅当 `cache_config.calculate_kv_scales=True` 时由 `Attention.calc_kv_scales`（`attention.py:569`）在运行时按源码逻辑动态计算（本笔记不推导公式，以源码为准）。
- **后端**：FLASH_ATTN **不支持 fp8 KV**；fp8 KV 候选 `[FLASHINFER, TRITON_ATTN]`，**FLASHINFER 在 sm_120 JIT 失败**（"requires sm75+"，根因 TARGET_CUDA_ARCHS 空）→ **必须用 TRITON_ATTN**（本版经 `LLM(attention_backend=...)` 指定；旧 `VLLM_ATTENTION_BACKEND` env 已失效）。

## 2. KV FP8 scale 审计（关键前置，实测）—— `configs/{C2,C2dyn}_perf.json`
| 配置 | calculate_kv_scales | 24 层 k/v scale |
|---|---|---|
| **C2** | False | **全 = 1.0（未校准）**，k_range/v_range=[1.0,1.0] |
| **C2dyn** | True | **动态非 1.0**：k_range≈[0.024,1.055]、v_range≈[0.0016,0.101]（按各层实际 K/V 幅值算）|
→ **C2 = 未校准（默认 scale=1.0）FP8 KV**；其质量结果**不代表**经 calibration/checkpoint/动态 scale 优化后的最佳 FP8 KV。C2dyn 展示动态 scale 的改善。

## 3. 显存（多来源，逐项标源）
**权重（Tensor 求和，storage 去重 tied）**：
| 配置 | 权重合计 | 明细 |
|---|---|---|
| C0/C0T/C2/C2dyn | **942.3 MiB**(bf16) | 全 bf16 |
| **C1** | **601.0 MiB** | bf16 259.8（embed/lm_head 259.7 + norm/bias）+ **fp8 341.25**（96 linear）|
- **C1 权重 942→601 MiB（−36%）**：linear 部分 682→341（半减），但 **embed/lm_head 260 MiB 保持 bf16**（占权重 ~28%）→ **不是理论 −50%**（§真实 Tensor 分类给出的解释）。
- KV tensor 字节：C0 8575.7 MiB(731,792 tok bf16)、C2 8543.7 MiB(**1,458,128 tok fp8**)——**总字节相近而 token ≈2×**（fp8 每 token 减半、块数≈翻倍）。C1 KV 8887 MiB / 758,384 tok（权重变小 → KV 池更大）。
- **KV 容量倍率 C2/C0T：tokens 1.993×（≈2×，干净）**，kv_tensor 字节 0.996×（≈1，印证"字节不变、容量翻倍"）。（注：早期 smoke 见 1.5× 是 gpu_mem_util 不同的假象，固定 COMMON 后为 ~2×。）

## 4. 质量（token 加权 PPL，固定语料 5,364 评分 token；中文 6 段+英文 6 段，长度 128–1024）
`prompt_logprobs=1` 取 ground-truth token logprob（`pl[i][ids[i]].logprob`，i≥1，首 token 无），`PPL=exp(Σ-logp / Σscored)`（**token 加权，非每段平均**）。

| 对照 | ΔPPL% | 结论 |
|---|---|---|
| C0 → 2.8184 (基线) | — | |
| **C0T−C0**（后端）| **+0.02%** | TRITON vs FLASH 对质量几乎无影响 → C0T 是干净基线 |
| **C1−C0**（FP8 Linear）| **+0.51%** | 在线 FP8 W8A8 质量损失很小 |
| **C2−C0T**（FP8 KV 未校准）| **+2.17%** | 未校准 fp8 KV 质量损失最大 |
| **C2dyn−C0T**（动态 scale KV）| **+1.72%** | 动态 scale 明显优于未校准（2.17→1.72）|
| C2−C0（总体参照）| +2.19% | 不用于单变量归因 |
- 语料因 tile 到目标长度、PPL 绝对值偏低，但**同一文本对所有配置一致，相对 ΔPPL 有效**。

## 5. 性能（进程内 proxy，全 eager；如实含反常）
prefill=`engine_first_step_latency`（median ms），decode=`steady_decode_step_latency`（seq_len 固定），warmup 8 + 重复 30。

**Prefill（median ms）**：
| len | C0(flash) | C1(fp8,flash) | C0T(triton) | C2(kvfp8,triton) |
|---|---|---|---|---|
| 128 | 7.48 | 9.44 | 9.39 | 10.19 |
| 512 | 7.54 | 10.02 | 9.36 | 10.76 |
| **2000** | **22.17** | **16.10** | 24.58 | 24.27 |

**Decode（median ms / 吞吐 tok/s）**：
| 场景 | C0(flash) | C1(fp8,flash) | C0T(triton) | C2(kvfp8,triton) |
|---|---|---|---|---|
| ctx64 b1 | 6.89 | 9.36 | 9.33 | 9.91 |
| ctx64 b64 | 8.48 / 7549 | 10.90 / 5874 | 10.41 / 6148 | 11.47 / 5578 |
| ctx1024 b1 | 6.97 | 9.46 | 8.53 | 10.06 |
| ctx1024 b8 | 7.71 | 9.60 | 9.57 | 10.78 |

**实测观察（诚实，含反常）**：
1. **fp8 Linear（C1 vs C0）在 decode 更慢**（ctx64 b1 6.89→9.36ms，+36%）：batch 小时是 GEMV（访存受限、计算量小），fp8 的**动态激活量化开销**盖过收益，且 sm_120 fp8 kernel 可能未优化。**但长 prefill 更快**（len2000 22.17→**16.10ms，−27%**）：prefill 是大 GEMM（计算受限），fp8 tensor-core GEMM 真正提速。→ **fp8 Linear 的加速只在计算受限（长 prefill）显现，decode/短序列反而变慢**。
2. **TRITON 后端本身比 FLASH 慢（C0T vs C0）**：decode ctx64 b1 6.89→9.33ms。故 fp8 KV 的比较必须用同后端基线 C0T（否则会把后端开销误算进 KV）。
3. **fp8 KV（C2 vs C0T）不提速**（ctx64 b1 9.33→9.91、ctx1024 b1 8.53→10.06，略慢）：fp8 KV 在 attention 读时需反量化，且本机 decode 未到 KV 带宽瓶颈（呼应 M7"decode 受 launch/CPU 主导"）→ **fp8 KV 的价值是显存/容量（~2×），不是本机 eager 下的速度**。
4. C2dyn 与 C2 性能相近（动态 scale 主要影响质量，不影响速度量级）。

## 6. 结论（诚实的量化权衡，本机 sm_120 / eager / 0.5B）
| 维度 | 在线 FP8 W8A8 Linear（C1）| FP8 KV Cache（C2 未校准 / C2dyn 动态）|
|---|---|---|
| **显存** | 权重 942→601 MiB（−36%，embed 保 bf16 故非 −50%）| KV 容量 ~2×（每 token 字节减半）|
| **速度** | 长 prefill −27%（计算受限受益）；decode +36%（**反而变慢**）| 不提速（略慢），价值在容量 |
| **质量(PPL)** | +0.51%（很小）| 未校准 +2.17% / 动态 scale +1.72% |
- **总体**：fp8 **主要收益是显存**（权重 −36% / KV 容量 ~2×），质量损失可接受（<2.2%，动态 KV scale 更小）；**速度非普遍收益**——只在计算受限的长 prefill 上 fp8 Linear 提速，decode 因动态激活量化/未优化 kernel/非带宽瓶颈反而变慢。**动态 KV scale 优于默认 scale=1.0**。
- **未校准 C2 的质量结论仅代表最简单 fp8 KV**，非校准后最佳。

## 7. 局限
- 单卡 / 单模型(0.5B) / eager；**结论不外推 compiled/CUDA Graph**（cudagraph 下 launch 开销被消除，fp8 相对表现可能不同）。
- 在线 fp8 = per-tensor 动态量化（非 AWQ/GPTQ 校准），代表"最简单 fp8"。
- 语料 tile → PPL 绝对值偏低（相对 ΔPPL 有效）；proxy 指标非在线 TTFT/TPOT。
- sm_120 fp8 支持较新（CUDA<12.9 告警），kernel 可能未充分优化——性能反常项已如实记录，不美化。
- 零 vLLM 源码改动。

## 附：复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
python scripts/m10_run_all.py            # C0/C1/C0T/C2/C2dyn 各独立进程 -> outputs/m10/{configs,memory,quality,raw}/
python scripts/m10_summarize.py          # -> summary.json / latency.csv / summary.csv
# 单配置：VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python scripts/m10_quant.py --config C2
```
逐样本 perf：`outputs/m10/raw/*_perf.jsonl`；质量逐样本：`outputs/m10/quality/*.json`；scale 审计：`configs/{C2,C2dyn}_perf.json` 的 `scale_audit`。
