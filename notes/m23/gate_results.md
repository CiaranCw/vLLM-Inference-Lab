# M23 · Gate 0 / 1 / 2 实测结果（2026-08-13）

> **结论：三个 Gate 全部 PASS。TRT-P（当前 PyTorch backend）在本机 sm_120 上真的能跑。**
> 证据：`outputs/m23/gate{0,1,2}.json`、`logs/m23/gate*.log`。
> 镜像：`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24`，digest `sha256:16a103b8b1b682d287e8043fc674d23fa52d5b5f2127da913bf6c0643db3a073`。

---

## Gate 0 · 容器与设备可见性　✅ PASS（13.6s）

| 项 | 实测 |
|---|---|
| `tensorrt_llm.__version__` | **1.3.0rc24** |
| 容器内 torch | **2.12.0a0+5aff3928d8.nv26.05** / CUDA **13.2** |
| 设备 | NVIDIA GeForce RTX 5080，**capability 120** |
| `torch.cuda.get_arch_list()` | 含 **`sm_120`** 与 `compute_120` |

---

## Gate 1 · 真实生成（决定性关卡）　✅ PASS（78.9s；加载 65.9s）

**判据红线（冻结稿 §4）**：import 成功 ≠ PASS，load 成功 ≠ PASS，GPU 可见 ≠ PASS ——
**必须真实 generate 出文本**。实测输出：

```text
prompt : "The capital of France is"
output : " Paris. It is the largest city in Europe and the third largest in the world"
```

**没有**出现此前上游 issue 里报告的 `invalid resource handle (GPT Attention Plugin)` / `CUDA ERROR 35`。

### ⚠️ 第一次尝试失败了，但根因是我的 harness，不是 sm_120

首版把 HF 缓存挂成 `:ro`，`huggingface_hub` 加载本地模型时仍要在缓存目录里**建锁文件**，于是报：

```text
OSError: [Errno 30] Read-only file system:
  /root/.cache/huggingface/hub/.locks/models--Qwen--Qwen2.5-0.5B-Instruct/....lock
```

**这是我脚本的缺陷，不是上游/硬件问题** —— 证据原样保留在
`outputs/m23/gate1_attempt1_readonly_mount.json`。
修法：挂载改 rw，同时设 `HF_HUB_OFFLINE=1` 硬性禁止联网下载（守住「不自动下载模型」这条规则）。
**这不属于冻结稿 §8 说的「Gate 失败 → 停止回报」情形**（那指的是上游/硬件不支持），
因为它既不换 tag、也不换硬件，只是修自己的挂载参数。

### 证据边界（冻结稿 §7 的表述纪律）

可以说：**`1.3.0rc24` + RTX 5080(sm_120) + WSL2 + 当前 driver + Qwen2.5-0.5B-Instruct + 默认配置，Gate 1 通过。**
**不能**扩大成「TensorRT-LLM 全面支持 RTX 5080」—— 只测了一个模型、一种配置、一个版本。

---

## Gate 2 · 对照可行性　✅ PASS（77.1s；加载 62.8s，生成 1.28s）

**能对齐的**（好消息）：

| 项 | 实测 |
|---|---|
| **接受 prompt token ids 输入** | ✅ `True` —— 可与 vLLM 用**完全相同的 token ids**，口径可严格对齐 |
| Shape A（in256 / out128）× 4 并发 | ✅ 跑通，输出长度**精确 128 × 4** |
| 生效配置可回读 | ✅ 通过 `llm.args`（`TorchLlmArgs`）读回 |
| **backend** | ✅ **`pytorch`** —— 确认走的是 TRT-P，不是 legacy engine |

**回读到的 TRT-P 默认配置**：

```text
backend                = pytorch
dtype                  = auto
max_batch_size         = 2048
max_num_tokens         = 8192
max_seq_len            = None
enable_chunked_prefill = False
tensor_parallel_size   = 1
kv_cache_config        = enable_block_reuse=True, free_gpu_memory_fraction=0.9, max_tokens=None, ...
cuda_graph_config      = batch_sizes=[1..32, 64, 128], max_batch_size=128     ← 默认开启
```

---

## Gate 2 的真正产出：**必须先解决的对齐缺口**

冻结稿要求「对不齐的项如实列出」。下面每一项都会实质影响结论：

| # | 项 | TRT-P 默认 | vLLM 侧（M18 口径） | 处理 |
|---|---|---|---|---|
| **A1** | **CUDA Graph** | **默认开启**（`cuda_graph_config` 有 batch_sizes 列表） | **`enforce_eager=True`（全程关闭）** | ✅ **所有者已决策**：改为 **2×2 execution-mode 对照**（冻结稿 v4 §5.0），OFF/ON 两条 lane 分别报告 |
| A2 | 激活 dtype | `auto` | 显式 `bfloat16` | 两侧显式冻结同一 dtype（M18 教训） |
| A3 | 显存预算 | `free_gpu_memory_fraction=0.9` | `gpu_memory_utilization=0.6` | 统一到同一值 |
| A4 | 批与 token 上限 | `max_batch_size=2048` / `max_num_tokens=8192` | `max_num_seqs=64` / `max_num_batched_tokens=4096` | 统一 |
| A5 | chunked prefill | `False` | M12/M20 显式 `True`，M18 未指定 | 统一并显式声明 |
| A6 | **KV block reuse** | **`enable_block_reuse=True`（≈ vLLM 的 prefix caching）** | M18 显式 `enable_prefix_caching=False` | 统一关闭，否则 APC 命中会污染对照（M17 已量化其影响） |
| **A7** | **显存分项不可得** | 主进程 `torch.cuda.memory_allocated()` 读到 **0.0 MiB**（executor 在独立 runtime 内） | vLLM 可遍历 tensor 分类统计 | **只能用 `nvidia-smi` 整卡口径** —— 与冻结稿 §5.3「total GPU memory 为 Primary，分项标 unavailable」完全吻合，**不得**用残差推导 |

### A1（CUDA Graph）为什么必须由所有者拍板

这是一个**没有唯一正确答案**的口径选择：

- **两侧都关**（TRT-P 也设 `cuda_graph_config=None`）：与本 lab 既有全部模块（M6/M7/M17/M18 都是 eager）口径一致、可对接，
  但**低估 TRT-P 的真实生产性能** —— CUDA Graph 正是它默认开启的优化。
- **两侧都开**：更接近生产默认，但 vLLM 侧要脱离 `enforce_eager`，
  与 M6–M18 的历史数字**不可比**，且本 lab 从未在 compiled 模式下下过结论。
- **各用各的默认**：反映"开箱即用"的真实差异，但那是 **stack 默认值之差**，
  **不能**称为引擎能力之差。

M7 已实测本机 CUDA Graph 影响巨大（eager→compiled 单步 7.44→2.50ms，提交次数 257→15），
所以这个选择**会主导结论**，不能由我替所有者定。

---

## 下一步（待决策后执行）

1. 所有者就 **A1** 定口径（建议见上）。
2. 按 A2–A6 写死两侧配置，实现 `scripts/m23_bench.py`（Primary metrics 优先：
   端到端延迟 / 吞吐 / 总 GPU 显存 / 冷启动；prefill/decode proxy 仅作解释性指标）。
3. 三点测量 `V-host` / `V-ctr` / `T-ctr`，同会话内跑完，前后各落环境指纹。


---

## 附：A1 决策落地 + V-O correctness 前置门（2026-08-13）

**所有者决策**：不二选一，改为 **2×2 execution-mode 对照**（冻结稿 v4 §5.0）：

```text
V-E / T-E ：Graph / optimization OFF   → 与 M6/M7/M17/M18 历史口径对接，控住 CUDA Graph 这个大混杂变量
V-O / T-O ：optimized ON               → 比较双方各自当前优化栈的实际性能
```

**表述纪律**：正式跨引擎比较只允许 `T-E vs V-E` 与 `T-O vs V-O`；
允许 `V-O vs V-E` / `T-O vs T-E` 作为同引擎 execution-mode ablation；
**禁止交叉 lane**（`T-E vs V-O` / `T-O vs V-E`）做 engine speedup；
**任一差值都不得称为「纯 engine intrinsic speedup」**；
四组都要回读并保存实际 execution/graph 配置，显存分别测量。
**不叫「eager vs compiled」** —— 两侧优化栈内容本就不同，那种叫法会暗示不存在的对应关系。

### V-O correctness smoke 结果　✅ **无硬失败**（证据 `outputs/m23/vo_correctness.json`）

| 项 | 结果 |
|---|---|
| 进程/输出/长度 | 6/6 正常，无崩溃、无空输出、长度精确 64 |
| **V-E 生效配置** | `cudagraph_mode=NONE`，`compilation_mode=0` |
| **V-O 生效配置** | `cudagraph_mode=FULL_AND_PIECEWISE`，`compilation_mode=3`，capture_sizes `[1,2,4,8,16,24,...]` |
| **exact-match（诊断项）** | **2/6 = 33.3%** |

**人工检视 4 处分歧**（自动判据查不出「乱码」，故必须人看）：**全部连贯、切题、无退化** ——
capital of France 两侧都答 Paris 只在 "third/second largest" 分叉；张量并行前 ~90 token 完全相同、
尾部各自通顺续写；fibonacci 两侧都产出可用代码只差提示字符串；北京仅选项顺序不同。

→ **判定**：属**贪心解码下的数值分歧**（与 M5/M19 先例一致），**不是正确性破坏**，
**optimized lane 可以继续**。但 **exact-match 2/6 必须随所有 V-O 结论一起呈现**，
且 V-O 与 V-E 的输出不可互相替代做正确性论证。


---

## V-ctr correctness preflight（2026-08-13）　✅ PASS

**为什么必须单独做**：host 上的 V-O smoke **不能替代** —— V-host 与 V-ctr 的 vLLM/software stack 不同
（host pinned `ee0da84` vs 容器内 **`0.24.0+092c4842.dev`**）。
输入复用 `outputs/m23/preflight_inputs.json` 里**同一份 6 条 token ids**（host 侧 tokenize 一次落盘，V/T 两侧共用）。

| 项 | 结果 |
|---|---|
| 硬失败 | **无**（4 次独立容器进程；非空、长度精确 64、**logprob 无 NaN/Inf**、无 runtime error） |
| **V-E 生效配置** | `cudagraph_mode=NONE`，`compilation_mode=0` |
| **V-O 生效配置** | `cudagraph_mode=FULL_AND_PIECEWISE`，`compilation_mode=3` |
| **V-E within-lane repeatability** | within-process **6/6**、across-process **6/6** |
| **V-O within-lane repeatability** | within-process **6/6**、across-process **6/6** |
| 跨 lane exact-match（诊断，**不阻塞**） | **1/6 = 16.7%**（host 侧为 2/6） |

**人工检视 5 处分歧**：全部连贯、切题、无退化 → **无 functional degradation**。

### repeatability 带来的一个更强判断

两条 lane **各自 across-process 都是 6/6 完全确定性**，
所以跨 lane 的 1/6 **不是随机抖动**，而是两条 execution path 之间**系统性、可复现的数值差异**。

→ **措辞纪律（随所有 V-O 结论保留）**：只能说「**未观察到明显 functional degradation**」；
**不得**升级成「两条 execution path 数值等价」——
本次证据恰恰表明它们**不等价**（各自确定性、但互不相同）。

### 两次 harness 缺陷（都不是引擎/硬件问题，如实记录）

1. **`IncompleteSnapshotError`**：容器内较新的 `huggingface_hub` 在 offline 模式下做完整快照校验，
   而本地缓存缺 3 个**纯文档**文件（`.gitattributes`/`LICENSE`/`README.md`），权重与 tokenizer 齐全。
   → 改为**直接指向容器内本地 snapshot 路径**，绕过 hub 解析：既不联网（守住「不自动下载模型」），
   又保证 V/T 两侧加载的是**同一份字节**。
2. **`JSONDecodeError: Extra data`**：vLLM 的 tqdm 进度条用 `\r` 刷新，捕获后与结果行粘连。
   → `use_tqdm=False` + `TQDM_DISABLE=1` + 解析改用 `raw_decode`。
