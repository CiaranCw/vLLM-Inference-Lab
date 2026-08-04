# M17 · 实测结论（E1 Prefix Caching / E3 Chunked Prefill）

> **状态**：E1/E3 **已实测完成**（2026-08-04，GPU 空闲后跑）。设计/源码定位/不变量见同目录 `prefix_cache_and_chunked_prefill.md`；本文件只放**结果与结论**。
> **环境**：单卡 RTX 5080（sm_120）· WSL2 · Qwen2.5-0.5B-Instruct · vLLM `ee0da84`/v0.24.0 · `enforce_eager` · seed=0 / temperature=0 · warmup 5 + measure 15。
> **数据**：`outputs/m17/{prefix_cache_sweep.csv,chunked_prefill.csv}` + `outputs/m17/raw/*.jsonl`（逐样本）；日志 `logs/m17/*`。
> **边界**：绝对值限本配置，不外推 compiled / 大模型 / 跨机。

---

## 0. 一句话结论

**M6 的「省计算 ≠ 省延迟」只是短前缀现象。** 前缀足够长时，APC 的延迟收益真实且巨大：**P=2048 时 TTFT 从 23.98ms 降到 9.03ms（−62.4%，14.96ms ≈ 8.8σ）**；而 P≤512 时省下的 0.9–1.3ms 确实落在噪声内。**分界线由 M7 已证的 ~7ms CPU 提交地板决定**——prompt 不够长时，省下的计算被地板吃掉，看不见。

---

## 1. E1：长共享前缀 on/off（核心结果）

`outputs/m17/prefix_cache_sweep.csv`（median of 15 measure samples，单位 ms）

| P（共享前缀） | off（无 APC） | on-cold（未命中） | on-hot（命中） | **hot 省下** | off_sd | hot_sd | 超噪声？ |
|---:|---:|---:|---:|---:|---:|---:|:--:|
| 256 | 8.57 | 7.43 | 7.26 | 1.31 | 1.39 | 0.70 | ❌ no（阈值 2.09） |
| 512 | 8.34 | 8.20 | 7.44 | 0.90 | 1.06 | 0.86 | ❌ no（阈值 1.92） |
| **1024** | 12.87 | 12.23 | **7.85** | **5.02** | 0.42 | 0.46 | ✅ **YES**（阈值 0.88，≈3.5σ） |
| **2048** | 23.98 | 23.00 | **9.03** | **14.96** | 0.41 | 1.28 | ✅ **YES**（阈值 1.69，≈8.8σ） |

> 判据（保守）：`省下 > off_sd + hot_sd` 才算超噪声。

**相对降幅**：P=256 −15.3% / 512 −10.7% / **1024 −39.0%** / **2048 −62.4%**。

### 1.1 为什么短前缀看不出收益（与 M7 打通）
把 **off 侧**延迟随 P 的斜率算出来：

| 区间 | 延迟增量 | 每 token |
|---|---:|---:|
| 256 → 512 | **−0.23 ms** | ≈ **0**（−0.89 us/tok，即噪声） |
| 512 → 1024 | +4.53 ms | **8.85 us/tok** |
| 1024 → 2048 | +11.11 ms | **10.85 us/tok** |

- **≤512：斜率≈0** —— 延迟被 **M7 已证的 ~7ms CPU kernel 提交地板**主导，多算几百个 token 几乎不额外花时间。**此时省计算当然省不出延迟。**
- **≥512：进入计算受限区**（8.85 → 10.85 us/tok 线性增长）。**这时省掉的计算才直接变成省下的时间。**

**hot 侧几乎不随 P 增长**（7.26 → 7.44 → 7.85 → 9.03 ms）：因为命中后每次只算 **16 个尾 token**，无论前缀多长。→ hot 侧本身就贴在那条 ~7ms 地板上。

**这解释了 APC 的收益条件**：`省下的时间 ≈ (命中 token 数) × (该长度区的每 token 成本)`，且必须**大于地板噪声**才可观测。每命中 token 省时实测 1.75–7.30 us，与上面的 8.85–10.85 us/tok 同量级（差异来自 hot 侧仍需算尾 token + 固定开销）。

### 1.2 cold vs off：APC 开着但没命中，有没有额外代价？
| P | off | on-cold | 差 |
|---:|---:|---:|---:|
| 256 | 8.57 | 7.43 | **−1.13** |
| 512 | 8.34 | 8.20 | −0.15 |
| 1024 | 12.87 | 12.23 | −0.65 |
| 2048 | 23.98 | 23.00 | −0.98 |

**cold 全部 ≤ off（差在噪声内或轻微更快）** → **本配置下开启 APC 没有可测的额外开销**（hash 计算/查表成本埋在噪声里）。
> 注意这与 M6 的记录不同：M6 曾观察到「prefix ON 短 prompt 反略慢于 OFF」。本轮 4 个长度、每档 15 样本均未复现该反向差异——**判定 M6 那次是噪声**（M6 该项 stdev 1.03–1.06ms，而差异仅 +0.88ms，本就在噪声内）。这是本轮对旧结论的修正。

### 1.3 不变量核对（I1–I3 全过）
| 不变量 | 结果 |
|---|---|
| **I1** 命中对齐 `hit ≈ floor(P/B)*B`（B=16） | ✅ 256/512/1024/2048 全部精确等于 P（P 本就 16 对齐） |
| **I2** 省计算守恒 `hit == actual − scheduled` | ✅ 四档全部成立（如 P=2048：2064−16=2048） |
| **I3** off 基线 `hit == 0` | ✅ 四档全为 0 |

**命中后 `scheduled_tokens` 恒为 16**（= 未命中的尾部后缀长度 `E1_SUFFIX_LEN`）——直接证明「命中的前缀完全跳过了 prefill 计算」。

---

## 2. E3：Chunked Prefill

`outputs/m17/chunked_prefill.csv`（prompt_len = 2048）

| enable_chunked_prefill | max_num_batched_tokens | num_chunks | chunk_sizes | sum_scheduled | ttft_proxy_ms |
|:--:|---:|---:|---|---:|---:|
| **False** | 4096 | 1 | `[2048]` | 2048 | 129.28 |
| True | 512 | **4** | `[512, 512, 512, 512]` | 2048 | **148.02** |
| True | 1024 | **2** | `[1024, 1024]` | 2048 | 134.52 |
| True | 2048 | 1 | `[2048]` | 2048 | 128.10 |
| True | 4096 | 1 | `[2048]` | 2048 | 129.19 |

### 2.1 不变量 I4（分块守恒）✅
**每一行 `sum(chunk_sizes) == 2048 == prompt_len`**，且每块 ≤ 预算。分块数恰为 `ceil(2048 / 预算)`：512→4、1024→2、≥2048→1。

### 2.2 分块本身有成本（单请求场景）
预算 2048（不分块，128.10ms）→ 1024（2 块，134.52ms，**+5%**）→ 512（4 块，148.02ms，**+15.6%**）。
**原因**：每多一块就多一轮完整的 engine step（调度 + metadata 构造 + kernel 提交），而 **M7 已证每步有 ~7ms 级 CPU 提交开销**——分 4 块 ≈ 多付 3 次这个固定开销（实测 +19.9ms ≈ 3×6.6ms，与 M7 的地板量级吻合）。
**边界**：这是**单请求**测量。chunked prefill 的真实价值是**多请求并发**下让长 prefill 不独占整个 step、避免阻塞其他请求的 decode（降低 ITL 抖动）——本轮**未覆盖该场景**（见 §4 局限）。**不能据此说"chunked prefill 有害"**，只能说「单请求下它以少量 TTFT 换取调度灵活性」。

### 2.3 ⭐ 意外发现（比原设计更有价值）：不开 chunked prefill 时，预算被强制 ≥ max_model_len
原计划跑「chunked=off × 预算 {512,1024,2048}」作对照，**全部被 vLLM 配置校验拒绝**：

```
ValueError: max_num_batched_tokens (512) is smaller than max_model_len (4096).
This effectively limits the maximum sequence length to max_num_batched_tokens
and makes vLLM reject longer sequences.
```

源码确证 `vllm/config/scheduler.py:268-279`：
```python
if (self.max_num_batched_tokens < max_model_len
        and not self.enable_chunked_prefill):
    raise ValueError(...)
```

**这是 chunked prefill 存在意义的硬编码证明**：不分块时，一个 prompt 必须在**单步内**算完，所以 token 预算必须 ≥ 模型最大长度，否则长请求会被静默拒绝——上游宁愿在配置层直接报错。
→ 因此「chunked off + 小预算」这个对照**在当前 vLLM 上根本无法构造**（不是配置写错），off 侧只能用预算 4096。**这条约束是本轮实测才发现的，写设计时并不知道。**

---

## 3. 与 M3/M6/M7 的关系（三处修正/深化）

| 旧结论 | 本轮 |
|---|---|
| **M6**：prefix 命中省 94% 计算 token，但首 token 延迟改善**落在噪声内**（+0.03ms） | **深化**：该结论只在**短前缀**成立。M6 用 body=256（正落在 ~7ms 地板区）；本轮扫到 1024/2048，收益 **5.02/14.96ms，3.5σ/8.8σ 显著**。 |
| **M6**：prefix **ON 短 prompt 反略慢于 OFF** | **修正**：4 档 ×15 样本均未复现，cold 全部 ≤ off。判定 M6 那次是噪声（其 stdev ~1.05ms > 观测差 0.88ms）。 |
| **M7**：decode ~7ms 地板 = CPU 提交开销 | **复用为解释工具**：该地板同样存在于 prefill 短路径，正是它**掩盖了短前缀 APC 的收益**（off 侧 256→512 斜率≈0），也解释了 **E3 分块的每块固定成本**。 |

---

## 4. 局限（如实）
- **单请求**测量：E3 未做「长 prefill + 并发 decode」场景，故未能验证 chunked prefill 对 **ITL 抖动**的正面作用（其主要卖点）。→ 想覆盖需扩展脚本加并发请求，属后续工作。
- `enforce_eager` / 0.5B / 单卡 / sm_120；**不外推 compiled**（CUDA Graph 会改变每步固定开销，从而移动上面所有分界线）。
- **E2（block 生命周期 ref_cnt/驱逐内省）未做**：需先验证 `block_pool` 的内省 API 可读，本轮聚焦 E1/E3 主结论，E2 仍为可选 TODO。
- APC 的 `hit_tokens` 用 `actual − scheduled` 间接推断（M6 已验证可靠），未直接读 `block_pool` 的 hash 表。
- E3 的 `ttft_proxy_ms` 是「首请求从 add 到 drain 完」的进程内代理，非在线 TTFT（在线口径见 M12）。

---

## 5. 复现
```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0
# 每个 shard 一个独立进程（一次只建一个引擎；同进程内多引擎无法归还 KV 显存）
python scripts/m17_prefix_cache.py --shard e1-on
python scripts/m17_prefix_cache.py --shard e1-off
for ch in on; do for mbt in 512 1024 2048 4096; do
  python scripts/m17_prefix_cache.py --shard e3 --e3-chunked $ch --e3-mbt $mbt; done; done
python scripts/m17_prefix_cache.py --shard e3 --e3-chunked off --e3-mbt 4096   # off 侧预算必须 ≥ max_model_len
python scripts/m17_prefix_cache.py --shard summarize   # 不建引擎、不需 GPU
```

### 实测踩坑（诚实记录，全部已修）
1. **同进程内建多个引擎会 OOM**：原脚本在一个进程里循环建 on/off（E1）与 6 种配置（E3）；`del engine + torch.cuda.empty_cache()` **不足以**归还 vLLM 的 KV cache 显存，第二个引擎构建即报 `Free memory ... less than desired`（nvidia-smi 显示 12GB 空闲，但 vLLM 只看到 5.08GB）。
   → **修复**：改为 **M6 早已验证的模式——每配置一个独立进程**（`--shard` + 可选 `--e3-chunked/--e3-mbt`），shard 各自落 JSONL，最后 `--shard summarize` 汇总（不建引擎）。这是"写骨架时没跑过"的代价。
2. **chunked=off 无法配小预算**：见 §2.3，属上游硬约束，据实调整实验网格。
3. 冒烟测试进程退出瞬间与下一次构建竞争显存 → shard 之间加 `sleep`。
