# M18 · 实测结论（W4A16 量化：AWQ / GPTQ vs BF16）

> **状态**：**M18 T+E+O COMPLETE**（2026-08-13）。Gate 0 / 0-b / 0-c + 全矩阵 6 配置已执行；
> 不变量 `check_invariants.py --module m18` **11 PASS / 0 FAIL**。
> **边界**：单卡 RTX 5080(sm_120) / WSL2 / `enforce_eager` / in-process proxy 指标 / 单副本 /
> 激活 dtype 冻结为 bfloat16。**绝对值不外推**到 compiled、其他 GPU、其他模型或在线服务。
> 设计冻结见 [`quantization_w4a16.md`](quantization_w4a16.md)。

---

## 0. 一页结论

| # | 结论 | 证据 |
|---|---|---|
| 1 | **W4A16 省显存非常显著**：整体权重 0.5B **−53.6%**、1.5B **−62.8%** | `outputs/m18/memory.csv` |
| 2 | 但**整体降幅达不到 4bit 的理论 −75%**，因为 `embed`/`lm_head` **完全没被量化**；只看**可量化部分**则是 **−74.0%**（几乎就是理论值） | 同上，不变量 `m18_i5b` |
| 3 | **稀释程度随模型变小而变大**：embed 占基线权重 0.5B **27.6%** / 1.5B **15.1%** → 模型越小，量化的整体收益越被摊薄 | `memory.csv` |
| 4 | 省下的显存换成 KV 容量：**+5.9%（0.5B）/ +30.1%（1.5B）** | `memory.csv`、不变量 `m18_i6` |
| 5 | **速度上没有任何收益：28 个对照点里 0 个变快**，23 个显著变慢（Bootstrap 95% CI 排除 0），5 个 CI 含 0 | `outputs/m18/perf.csv` |
| 6 | **连长 prefill 也没受益**（0.5B n=2000 **+9.2%**、1.5B **+6.6%**），而 M7 已证该区间是**计算受限** → 慢的来源包含 Marlin GEMM 本身，不只是 CPU 提交受限的伪影 | `perf.csv` + M7 §gpu_frac 0.88 |
| 7 | **执行前假设 H5 被推翻**：我预期「模型越大越可能显现收益」，实测 1.5B 的 decode 惩罚**反而更大**（+28.6% vs 0.5B 的 +12.9%） | `perf.csv` |
| 8 | 质量代价：0.5B ΔPPL **+5.0%(AWQ) / +4.8%(GPTQ)**；1.5B **+1.4%(AWQ) / +4.4%(GPTQ)** | `outputs/m18/quality.csv` |
| 9 | **同 bits/同 group_size 下 AWQ 与 GPTQ 的差异不大且不一致**：1.5B 上 AWQ 质量明显更好，0.5B 上两者接近；速度上互有胜负 | `perf.csv`、`quality.csv` |

---

## 1. 配置与「只有一个变量」的保证

| 配置 | 模型 | 量化 | 超参 |
|---|---|---|---|
| `Q0` | `Qwen2.5-{0.5B,1.5B}-Instruct` | 无（BF16 基线） | — |
| `Q1` | `…-Instruct-AWQ` | `auto_awq` | 4bit / group 128 / **非对称（zero_point）** |
| `Q2` | `…-Instruct-GPTQ-Int4` | `auto_gptq` | 4bit / group 128 / **对称（sym, desc_act=false）** |

AWQ 与 GPTQ 的 `bits` 与 `group_size` **完全相同**（逐项读自各 checkpoint 的
`quantization_config`），差别只在量化算法与是否带 zero-point → 这是一次**受控**的 AWQ vs GPTQ 对照。

**写死的隐藏变量**（不变量已核对，见 §5）：
- **激活 dtype 显式冻结为 `bfloat16`**。这是本模块最容易被忽略的坑：量化 checkpoint 的
  `config.json` 声明 `torch_dtype: float16`，而 BF16 基线声明 `bfloat16`；若两边都用
  `dtype="auto"`，测出来的差异里会同时混着「4bit 量化」和「fp16 vs bf16 激活」两个变量。
- `kv_cache_dtype=auto`（**KV 不量化**，那是 M10 的正交轴）、attention backend 跨配置一致、
  `enforce_eager` / `enable_prefix_caching=False` / `async_scheduling=False` 等与 M10 的 `COMMON` 逐项相同。

**Q0 在本模块内重测，不引用 M10 的旧数字** —— M21 已实测本机存在跨会话性能漂移（根因未定），
跨会话相减会把环境漂移算进量化效应。6 个配置在同一会话内连续跑完，前后各落一次环境指纹
（`outputs/m18/env_fingerprint_{before,after}.json`）。

---

## 2. 显存：真实降幅 vs 理论降幅

| 配置 | 权重总量 | vs Q0 | **可量化部分** | vs Q0 | embed/lm_head | 占基线 | KV tokens | vs Q0 |
|---|---|---|---|---|---|---|---|---|
| 0.5b Q0 | 942.3 MiB | — | 682.6 MiB | — | 259.7 MiB | 27.6% | 731,792 | — |
| 0.5b Q1 (AWQ) | **437.1** | **−53.6%** | 177.4 | **−74.0%** | 259.7 | — | 775,104 | **+5.9%** |
| 0.5b Q2 (GPTQ) | 435.8 | −53.8% | 176.1 | −74.2% | 259.7 | — | 775,216 | +5.9% |
| 1.5b Q0 | 2944.4 MiB | — | 2499.3 MiB | — | 445.1 MiB | 15.1% | 233,856 | — |
| 1.5b Q1 (AWQ) | **1094.6** | **−62.8%** | 649.4 | **−74.0%** | 445.1 | — | 304,224 | **+30.1%** |
| 1.5b Q2 (GPTQ) | 1089.7 | −63.0% | 644.6 | −74.2% | 445.1 | — | 304,288 | +30.1% |

**机制（有证据，不是猜）**：不变量 `m18_i5b` 核对了 `embed`/`lm_head` 的字节数在三个配置里
**完全相同**（0.5B 恒为 259.7 MiB、1.5B 恒为 445.1 MiB）→ 它们根本没被量化。
把它们剔除后，可量化部分的降幅是 **−74.0%**，几乎就是 4bit vs 16bit 的理论值 −75%
（差的那点是新增的 scales/zeros：0.5B 5.3 MiB、1.5B 19.5 MiB）。

→ **可直接迁移的工程结论**：*报「量化省了多少显存」时必须说清分母*。
同一套 4bit 量化，在 1.5B 上省 62.8%、在 0.5B 上只省 53.6%，差别不在量化算法，
而在**未量化的 embedding 占比**。**模型越小，这个稀释越严重。**
这与 M10 的 FP8 现象同源（当时 −36% 而非理论 −50%），本模块把它量化成了可外推的规律。

---

## 3. 速度：28 个对照点，**0 个变快**

单位 ms（median，5 warmup + 15 measure），`rel%` 为相对同模型 Q0 的变化，**正数 = 更慢**；
`CI` 为 median difference 的 Bootstrap 95% CI 是否排除 0。

| 网格 | 0.5b AWQ | 0.5b GPTQ | 1.5b AWQ | 1.5b GPTQ |
|---|---|---|---|---|
| prefill n=128 | +16.2% ✅ | +16.6% ✅ | +35.1% ✅ | +4.2% ns |
| prefill n=512 | +14.2% ✅ | +13.5% ✅ | +6.1% ✅ | +0.5% ns |
| **prefill n=2000** | **+9.2%** ✅ | +5.6% ✅ | **+6.6%** ✅ | +10.3% ✅ |
| decode ctx64 b1 | +12.9% ns | +15.4% ns | **+28.6%** ✅ | +16.9% ✅ |
| decode ctx64 b64 | +15.6% ✅ | +14.9% ✅ | +12.4% ✅ | +2.3% ns |
| decode ctx1024 b1 | +20.6% ✅ | +26.4% ✅ | +20.1% ✅ | +31.4% ✅ |
| decode ctx1024 b8 | +19.5% ✅ | +14.8% ✅ | +30.1% ✅ | +28.3% ✅ |

（✅ = CI 排除 0；ns = CI 含 0，只能说**未观察到超出噪声的差异**，**不能**说「没有差异」。）

**28 个对照点中：23 个显著变慢、5 个 CI 含 0、变快的 0 个。**

### 为什么这不只是「M7 地板效应」的伪影
M7 已证本机 eager decode 是 **CPU 提交受限**（GPU 仅忙 23%），在那个区间省权重带宽本来就
不该有收益 —— 所以 decode 变慢**可以**被解释为「省不到、还多付 dequant」。
但 M7 同时证了 **长 prefill（len2000）是 GPU 计算受限**（gpu_frac 0.88）。
在这个计算受限区间，W4A16 **仍然更慢**（0.5B +9.2%、1.5B +6.6%，两者 CI 均排除 0）。
→ **慢的来源包含 Marlin W4A16 GEMM 本身**，不能全推给 CPU 提交地板。

**不声称更细的机制**：是「4bit 反量化回 16bit 的额外开销超过了省下的权重读取」，还是
「Marlin 在 sm_120 上未针对性优化」，或两者兼有 —— 本模块**没有做 kernel 级 profiling**，
不下这个判断。要定这个因，需要 M7 式的 `torch.profiler` 拆解（列为可选扩展）。

### H5 被数据推翻（如实记录）
设计冻结 §6 的 H5 写的是「**1.5B 比 0.5B 更容易显现收益**」，理由是固定开销占比更小。
实测**相反**：1.5B 的 decode 惩罚普遍**更大**（ctx64 b1：AWQ +28.6% vs 0.5B 的 +12.9%）。
加 1.5B 这一档的初衷是「控住 M7 地板效应这个混淆」，结果它没有把收益放出来，
反而说明**在本机这两个规模上，W4A16 的速度劣势都不是地板效应造成的**。

---

## 4. 质量：token 加权 PPL

| 配置 | PPL(overall) | ΔPPL% | ΔPPL% (zh) | ΔPPL% (en) |
|---|---|---|---|---|
| 0.5b Q0 | 2.8184 | — | — | — |
| 0.5b Q1 (AWQ) | 2.9593 | **+5.00%** | +3.79% | +6.21% |
| 0.5b Q2 (GPTQ) | 2.9535 | **+4.79%** | +3.58% | +6.02% |
| 1.5b Q0 | 2.4704 | — | — | — |
| 1.5b Q1 (AWQ) | 2.5039 | **+1.36%** | +1.30% | +1.42% |
| 1.5b Q2 (GPTQ) | 2.5800 | **+4.44%** | +6.29% | +2.62% |

- 评分 token 数六个配置**完全相同**（5,364，不变量 `m18_i9`），故 ΔPPL 可比。
- **语料有 tile，PPL 绝对值偏低**（M10 已注明），因此**只看相对 ΔPPL**，不解读绝对 PPL。
- **1.5B 上 AWQ 明显优于 GPTQ**（+1.36% vs +4.44%）；0.5B 上两者接近（+5.0% vs +4.8%）。
  **不外推**成「AWQ 普遍更好」——只有两个模型规模、一份语料、一次测量。
- 模型越大质量损失越小（1.5B AWQ 仅 +1.4%），与「大模型对量化更鲁棒」的常见说法方向一致，
  但两个点不足以称为趋势。

---

## 5. 不变量（`check_invariants.py --module m18`：11 PASS / 0 FAIL）

| 不变量 | 结果 |
|---|---|
| I1 六配置生效 dtype 均为 `torch.bfloat16` | ✅（量化 checkpoint 声明 float16，已被显式覆盖） |
| I2 attention backend 跨配置一致 | ✅ |
| I3 `kv_cache_dtype` 跨配置一致 | ✅（均 `auto`） |
| I5a W4A16 侧存在 int32 打包权重 | ✅ |
| I5b `embed`/`lm_head` 字节数跨配置相同 | ✅ → 支撑 §2 的机制解释 |
| I6 权重变小 ⇒ KV 容量不减 | ✅ |
| I7 每个网格点 15 个 measure 样本 | ✅ |
| I8 全部性能样本 `valid=True` | ✅（decode 每步 scheduled_tokens 与 running 数均符合预期） |
| I9 PPL 评分 token 数跨配置相同 | ✅ |

---

## 6. 与 M10 的关系（**定性对照，不做数字相减**）

样本量不同（M10 是 8+30，M18 是 5+15）、会话不同（M21 已证跨会话漂移），故**不相减**。
定性上：

| | M10 · FP8 W8A8（在线量化） | M18 · W4A16（离线 checkpoint） |
|---|---|---|
| 权重显存 | −36%（理论 −50%，被 embed 稀释） | **−53.6% / −62.8%**（理论 −75%，同样被 embed 稀释） |
| 长 prefill 速度 | **−27%（更快）** | **+5.6~+10.3%（更慢）** |
| decode 速度 | +36%（更慢） | +12.9~+31.4%（更慢） |
| 主要价值 | 显存 | **显存**（更多） |

→ 一个值得记住的对比：**FP8 至少在计算受限的长 prefill 上换来了真加速，W4A16 在本机连这一段都没换到。**
两者的共同点是：**decode 都更慢，量化的价值主要在显存与容量，而不是本机 eager 下的速度。**

---

## 7. 局限与未做

- ⬜ **未做 kernel 级 profiling**：因此「Marlin W4A16 为什么慢」只有现象与区间证据，
  **没有 kernel 时间线归因**。要定因需 M7 式 `torch.profiler` 拆解（可选扩展）。
- ⬜ **未做 W8A8-INT8**：本版本 Marlin 明确不支持（`auto_gptq.py` 内 assert
  "W8A8-INT8 is not supported by marlin kernel."）。**不为补表格伪造不支持的路径**。
- ⬜ 未做 W4A16 × FP8 KV 组合；未做 compiled / CUDA Graph；未做多卡；未做在线 serving 口径。
- ⬜ 未自己做量化校准（只用官方 checkpoint），故**无法区分**「算法差异」与「官方校准数据差异」。
- ⚠️ 质量只用一份内置 tile 语料的 PPL，**不是下游任务准确率**；ΔPPL 小不代表任务效果无损。
- ⚠️ 只有两个模型规模、一张卡、一种 kernel（Marlin）；**AWQ vs GPTQ 的胜负不可外推**。

---

## 附：复现

```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0

python scripts/m18_probe.py                      # Gate 0：sm_120 上 W4A16 kernel 可行性（零成本）
python scripts/m18_load_smoke.py --model Qwen/Qwen2.5-0.5B-Instruct-AWQ   # Gate 0-b
python scripts/m21_env_fingerprint.py --out outputs/m18/env_fingerprint_before.json
python scripts/m18_run_all.py                    # 全矩阵：2 模型 × 3 配置，串行、一进程一引擎
python scripts/m21_env_fingerprint.py --out outputs/m18/env_fingerprint_after.json
python scripts/m18_summarize.py                  # -> memory.csv / perf.csv / quality.csv（含 Bootstrap CI）
python scripts/check_invariants.py --module m18 --verbose   # 11 PASS
python scripts/m18_figures.py                    # -> figures/fig13_m18_w4a16.png
```
