# M5 · 执行模式（本机实际路径验证）

> 目标：**在本机（RTX 5080 / sm_120 / torch 2.11+cu130 / vLLM ee0da84）上，用真实运行证明当前实际启用的执行模式**——
> `enforce_eager` 到底改了什么、有没有用 torch.compile、CUDA Graph 是 FULL / PIECEWISE / NONE、
> 图是怎么"捕获(capture)"和"回放(replay)"的、哪里做 padding、哪些部分留在图外（eager）。
> **边界**：M5 只看"执行路径"，不下性能结论（快慢留到 M6/M7）。
>
> 证据（全部来自本机真实运行，非猜测）：
> - `outputs/m5/eager_config.json`、`outputs/m5/compiled_config.json`（引擎**存活后读出的**解析配置）
> - `outputs/m5/eager.jsonl`、`outputs/m5/compiled.jsonl`（exec 组件 trace：capture / dispatch）
> - `logs/m5/eager.log`、`logs/m5/compiled.log`（启动日志：编译、捕获、显存）
> - 复现命令见文末。

---

## 0. 一句话结论

同一台机器、同一个模型，**执行模式完全由 `enforce_eager` 这一个开关切换**：

| | `enforce_eager=True`（当前基线） | `enforce_eager=False`（vLLM 默认 O2） |
|---|---|---|
| torch.compile | **不用**（mode=NONE） | **用**（mode=VLLM_COMPILE，后端 inductor） |
| CUDA Graph | **不用**（cudagraph_mode=NONE） | **用**（cudagraph_mode=FULL_AND_PIECEWISE） |
| 每步怎么跑 | 纯 eager，逐算子下发 | prefill 走 **PIECEWISE 图**、decode 走 **FULL 图** |
| 引擎初始化 | 快（无编译、无捕获） | 慢（编译 9.4s + 捕获 2s，本次 init≈26.6s） |
| 本机能否跑 | 能 | **能**（sm_120 上 torch.compile + CUDA Graph 均正常，未降级） |

> 换句话说：**基线之所以"没有图、没有编译"，不是硬件不支持，而是我们主动设了 `enforce_eager=True`。** 关掉它，本机默认就会启用"torch.compile + 混合 CUDA Graph"。

---

## 1. 先补概念：eager / torch.compile / CUDA Graph 是三件不同的事

初学最容易混。它们是**三层、可独立开关**的东西：

1. **eager（即时执行）**：PyTorch 默认方式。每个算子（matmul、softmax…）都由 Python 逐个下发给 GPU。灵活、好调试，但**每步都有 Python 开销和大量 kernel launch**。

2. **torch.compile（图编译）**：用 Dynamo 把 Python 前向"抓"成一张计算图，再交给 Inductor 生成/融合 kernel。**减少 Python 开销、融合算子**。它管的是"**怎么把模型编译成更高效的 kernel**"。

3. **CUDA Graph（图捕获/回放）**：把"一串 GPU 操作"录制成一张 CUDA 图，之后**一次 replay 就重放整串**，省掉每个 kernel 的 launch 开销。它管的是"**怎么把已经确定的一串 GPU 调用打包一次性发下去**"。

vLLM V1 里，(2) 和 (3) 常常一起用：先 torch.compile 把模型编好，再对若干"固定 batch 尺寸"把前向录成 CUDA 图。`enforce_eager=True` 就是**一键把 (2)(3) 全关**，退回纯 (1)。

### PIECEWISE vs FULL（vLLM 的两种 CUDA Graph 粒度）
- **FULL（整图）**：把**一整步前向（含 attention）**都录进一张图。要求这一步是"规整"的——即 **uniform decode**（batch 里每条都是 decode、每条只 1 个 token）。
- **PIECEWISE（分段图）**：把模型**按 attention 算子切开**，attention **留在图外用 eager 跑**，其余部分（各层的 Linear / norm / MLP）录成图。适合 prefill / 混合 batch 这种"每条长度不一（ragged）"的情况——因为变长 attention 不适合塞进固定形状的图。
- **FULL_AND_PIECEWISE（本机默认）**：两者都准备好，**运行时按 batch 形状自动选**：规整 decode → FULL，其它 → PIECEWISE。

> 为什么 attention 是切点？因为 prefill / 混合 batch 的 attention 是**变长（varlen）**的，序列长度每步都变，塞进"固定形状"的 CUDA 图很别扭；而 Linear/norm 这些对 token 数只是"行数不同"，padding 到固定尺寸就能复用图。所以 vLLM 把 attention 当作 piecewise 的**分割算子**踢出图外。

---

## 2. 配置是怎么被解析出来的（源码路径，可追溯）

不是我猜的，是从**存活引擎**里读出来的（`scripts/m5_exec_mode.py` 读 `llm.llm_engine.vllm_config.compilation_config`）。关键解析点：

- **默认优化级别 O2**：`vllm/config/vllm.py:364` → `optimization_level: OptimizationLevel = OptimizationLevel.O2`。
- **O2 → 编译 + 混合图**：`OPTIMIZATION_LEVEL_TO_CONFIG[O2]` 给出 `mode=VLLM_COMPILE`、`cudagraph_mode=FULL_AND_PIECEWISE`（`vllm/config/vllm.py:1162` 应用默认）。
- **`enforce_eager=True` 的"一票否决"**：`vllm/config/vllm.py:1061-1067`：
  ```python
  if self.model_config is not None and self.model_config.enforce_eager:
      logger.warning("Enforce eager set, disabling torch.compile and CUDAGraphs. "
                     "This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none")
      self.compilation_config.mode = CompilationMode.NONE
      self.compilation_config.cudagraph_mode = CUDAGraphMode.NONE
  ```
  → 这正是我们在 eager 日志里看到的那条 WARNING（见 `logs/m5/eager.log`）。
- **CUDAGraphMode 定义**：`vllm/config/compilation.py:53-103`（`NONE=0 / PIECEWISE=1 / FULL=2 / FULL_AND_PIECEWISE=(2,1)`）。
- **piecewise 的分割算子**：`vllm/config/compilation.py`（`_attention_ops` + `set_splitting_ops_for_v1`），实测 `splitting_ops` 里就是各种 attention 与 `unified_kv_cache_update`（见下）。

### 两配置的解析结果对照（实测 JSON）
| 字段 | eager | compiled(O2) |
|---|---|---|
| `optimization_level` | O2 | O2 |
| `model_config.enforce_eager` | **true** | false |
| `compilation.mode` | **NONE** | **VLLM_COMPILE** |
| `compilation.cudagraph_mode` | **NONE** | **FULL_AND_PIECEWISE** |
| `backend` | inductor（未生效，mode=NONE） | inductor |
| `custom_ops` | `["all"]` | `["none"]` |
| `ir_enable_torch_wrap` | false | true |
| `splitting_ops` | `[]` | attention 系列 + `unified_kv_cache_update`（16 个） |
| `cudagraph_capture_sizes` | `[]` | `[1,2,4,8,16,24,…,256,272,…,512]`（51 个） |
| `max_cudagraph_capture_size` | 0 | 512 |
| `cudagraph_num_of_warmups` | 0 | 1 |

> 注意 `custom_ops`：eager 用 `["all"]`（走 vLLM 自定义算子实现），compiled 用 `["none"]`（交给 Inductor 融合，尽量别用手写自定义 op）。这也是编译模式与 eager 数值有细微差异的原因之一（见 §5）。

---

## 3. compiled 模式的启动做了什么（真实日志时间线）

`logs/m5/compiled.log` 里按时间发生：

```
[core.py:258]      Initializing V1 LLM engine ... enforce_eager=False,
                   compilation_config={mode=VLLM_COMPILE, cudagraph_mode=FULL_AND_PIECEWISE, ...}
[gpu_model_runner:5442]  Model loading took 0.93 GiB, 1.73s
[backends.py:1089] Using cache directory: .../torch_compile_cache/... for vLLM's torch.compile
[backends.py:1148] Dynamo bytecode transform time: 3.09 s      # ← torch.compile 第 1 步：抓图
[backends.py:378]  Cache the graph of compile range (1, 8192) for later use
[backends.py:393]  Compiling a graph for compile range (1, 8192) takes 4.36 s   # ← Inductor 编译
[decorators.py:708] saved AOT compiled function to .../torch_aot_compile/...
[monitor.py:53]    torch.compile took 9.44 s in total          # ← 编译合计 9.44s
[gpu_worker.py:523] CUDA graph memory profiling is enabled ...
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 100%|████| 51/51    # ← 捕获 51 张 PIECEWISE
Capturing CUDA graphs (decode, FULL):                   100%|████| 35/35     # ← 捕获 35 张 FULL
[gpu_model_runner:6843] Graph capturing finished in 2 secs, took 0.22 GiB    # ← 捕获合计 2s / 0.22 GiB
```

拆开看这条链：
1. **torch.compile（编译）**：Dynamo 把前向抓成图（3.09s），Inductor 编译（4.36s），并**存到磁盘缓存**（下次可直接加载，省编译时间）。**只编译"一张按 attention 切过的图"**（compile range (1,8192)），因为 piecewise 用同一张编译图配不同 batch 尺寸。
2. **CUDA Graph 捕获（capture）**：对 `cudagraph_capture_sizes` 里的每个尺寸，把前向录成图。实测：
   - **51 张 PIECEWISE**（"mixed prefill-decode"，对应 `uniform_decode=False`）
   - **35 张 FULL**（"decode"，对应 `uniform_decode=True`）
   - 合计占显存 **0.22 GiB**，耗时约 2s。
3. 结果：引擎 init 总耗时约 **26.6s**（eager 无此开销）。

> **exec/capture 钩子实测**（`outputs/m5/compiled.jsonl`，均在 step=0 即启动阶段）：
> ```
> exec/capture runtime_mode=PIECEWISE num_tokens=512 uniform_decode=False
> ... (51 条 PIECEWISE)
> exec/capture runtime_mode=FULL      num_tokens=256 uniform_decode=True
> ... (35 条 FULL)
> ```
> 按 (mode,uniform) 计数：`{('PIECEWISE',False):51, ('FULL',True):35}`。**先捕获大尺寸再捕小尺寸**（大尺寸的显存池可被小尺寸复用，见 `capture_model` 注释）。

---

## 4. 运行时到底用哪张图（每步 dispatch 实测）—— 这是闭环关键

引擎每步在 `execute_model` 里调 `_determine_batch_execution_and_padding()`（`gpu_model_runner.py:4313`），
根据 batch 形状选运行时模式并把 token 数 padding 到某个"已捕获尺寸"。我们的 **exec/dispatch 钩子**卡在
`pad_attn = cudagraph_mode == CUDAGraphMode.FULL`（`gpu_model_runner.py:4366`）之后，记录了真实决策：

**eager（`outputs/m5/eager.jsonl`）** —— 每步都不走图：
```
step1  tokens 11→11 (pad0)  cudagraph_mode=NONE  uniform=False pad_attn=False   # prefill
step2  tokens  1→1  (pad0)  cudagraph_mode=NONE  uniform=False pad_attn=False   # decode
step3..8 同上（NONE）
```

**compiled（`outputs/m5/compiled.jsonl`）** —— prefill 与 decode 走不同图：
```
step1  tokens 11→16 (pad5)  cudagraph_mode=PIECEWISE  uniform=False pad_attn=False  # prefill：分段图
step2  tokens  1→1  (pad0)  cudagraph_mode=FULL       uniform=True  pad_attn=True   # decode：整图
step3..8 同上（FULL）
```

逐点解读（compiled）：
- **prefill（step1，11 token，非均匀）→ PIECEWISE**：token 数从 11 **padding 到 16**（`num_paddings=5`，16 是最近的已捕获尺寸）。`pad_attn=False` → **attention 不被 padding、在图外用 eager varlen 跑**；图里跑的是 Linear/norm/MLP 那些段。
- **decode（step2+，1 token，均匀）→ FULL**：1 恰好是已捕获尺寸，**无 padding**。`pad_attn=True` → **整步（含 attention）都在 CUDA 图里 replay**。
- 这正好对上 §1：**规整 decode 用 FULL，混合/prefill 用 PIECEWISE**，与配置 `FULL_AND_PIECEWISE` 一致。

> **padding 是干嘛的**：CUDA 图形状固定，只能对"录过的尺寸"回放。真实 batch 若是 11 个 token，就 pad 到录过的 16，用 16 尺寸的图跑，多出来的 5 个是填充位（结果丢弃）。这就是"**用固定尺寸的图服务可变 batch**"的代价与手段。

---

## 5. 一个必须如实记录的现象：compiled 与 eager 的输出**会分叉**

同一 prompt、贪心解码（temperature=0）、同一 seed，两模式输出 token：

```
eager   : [84648, 19479, 54851, 101053, 100751, 18493, 111390, 15946]
compiled: [84648, 19479, 54851, 101053,  18493, 102217, 100134, 104949]
                                  ↑ 前 4 个相同，第 5 个开始分叉
```

- **原因**：torch.compile（Inductor 融合、`custom_ops=["none"]` 换实现）+ CUDA Graph 会带来**浮点数值的细微差异**。贪心解码在某一步两个候选 token 概率非常接近时，细微差异就会让 argmax 选到不同 token，之后越走越远。
- **这不是 bug**，是"编译/图化改变数值"的正常表现；**不能把 eager 的逐 token 输出当作 compiled 的正确性判据**。真正的正确性应看整体质量（留待后续），而非逐 token 相等。
- 对本项目的影响：**E1 回归对齐只在"同一执行模式内"有效**（我们的 trace 钩子零副作用是在 eager 下用逐 token 一致验证的，见 §7）。跨模式对齐 token 不成立，属预期。

---

## 6. Python 层 ↔ 图/kernel 边界（把 M4 的边界再推进一层）

- M4 讲了"一次 attention = 两个 CUDA 算子（写 `reshape_and_cache_flash` / 读 `flash_attn_varlen_func`）"。
- M5 补上"**这些算子外面套了什么**"：
  - **eager**：Python 一个个下发所有算子（含那两个 attention 算子）。
  - **compiled + PIECEWISE**：Linear/norm/MLP 段被 torch.compile 编译并录成 CUDA 图；**attention（`splitting_ops` 里的 `unified_attention_with_output` 等）被切在图外**，仍走 eager varlen —— 所以 M4 的两个 attention 算子在 prefill 步依然是"图外的 Python 调用"。
  - **compiled + FULL（decode）**：**整步含 attention 一起录进图**，replay 时一次性重放，Python 每步只做"选图 + 填输入 + replay"。
- 因此 `splitting_ops` 就是"**图的边界线**"：线以内是可图化的规整算子，线以外（attention / kv_cache_update）是变长、需 eager 的部分。

---

## 7. 本轮改动与零副作用验证

- **新增只读钩子（component `exec`，`VLLM_LAB_TRACE` 门控，默认关）**，均在 `gpu_model_runner.py`，标 `# VLLM_INFERENCE_LAB_TRACE`：
  - `exec/dispatch`（`execute_model` 内，line ~4368）：每步的 `num_tokens_unpadded/padded、num_paddings、cudagraph_mode、uniform_decode、pad_attn`。
  - `exec/capture`（`_capture_cudagraphs` 内，line ~6913）：每张被捕获图的 `runtime_mode、num_tokens、uniform_decode`。
  - （另注：此前模块已有 `model_runner/model_forward` 钩子记录每步 `cudagraph_mode`，本次未改动它。）
- **备份**：`backups/m5/gpu_model_runner.py.pre_m5`。
- **回归（零副作用）**：不设 `VLLM_LAB_TRACE` 跑 eager 基线（`scripts/day1_baseline.py --max-tokens 16`），输出 token 与 Day1 基线 **逐一致（IDENTICAL）**：
  ```
  [84648,19479,54851,101053,100751,18493,111390,15946,99982,24360,60949,25511,32664,9370,99982,24360]
  ```
  证明新钩子在关闭时不改变执行语义。见 `outputs/m5/regress_eager.json`、`logs/m5/regress_eager.log`。

---

## 8. 结论清单（对应 ROADMAP M5 覆盖点）

- [x] **本机实际启用的执行模式**：默认（O2）= `VLLM_COMPILE + FULL_AND_PIECEWISE`；基线因 `enforce_eager=True` = `NONE + NONE`（纯 eager）。
- [x] **`enforce_eager` 改了什么**：`vllm/config/vllm.py:1061-1067` 强制 `mode=NONE、cudagraph_mode=NONE`，并打印 WARNING；实测确认。
- [x] **是否用 torch.compile**：compiled 模式用（Dynamo 3.09s + Inductor 4.36s，合计 9.44s，AOT 磁盘缓存）；eager 不用。
- [x] **CUDA Graph FULL / PIECEWISE / NONE**：eager=NONE；compiled=FULL_AND_PIECEWISE（捕获 51 PIECEWISE + 35 FULL，0.22 GiB）。
- [x] **capture / padding / replay / 非图区**：capture 见 §3；prefill padding 11→16 见 §4；FULL decode replay 见 §4；非图区 = `splitting_ops`（attention 等）见 §6。
- [x] **只对本机实际可运行的模式给证据**：sm_120 上 compiled **确实跑起来、未降级**（若未来出现降级/报错会另行如实记录）。
- [ ]（超出 M5）编译/图化带来的**性能收益**留到 **M6（Benchmark）/M7（Profiler）**；本笔记只证路径，不下快慢结论。

---

## 附：复现

```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate

# A) eager（当前基线）：mode=NONE / cudagraph=NONE / 纯 eager
VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_MAX_STEPS=20 VLLM_LAB_TRACE_COMPONENTS=exec \
VLLM_LAB_TRACE_JSONL=outputs/m5/eager.jsonl \
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m5_exec_mode.py --enforce-eager --max-tokens 8 \
  --out outputs/m5/eager_config.json 2>&1 | tee logs/m5/eager.log

# B) compiled（vLLM 默认 O2）：VLLM_COMPILE + FULL_AND_PIECEWISE
VLLM_LAB_TRACE=1 VLLM_LAB_TRACE_MAX_STEPS=20 VLLM_LAB_TRACE_COMPONENTS=exec \
VLLM_LAB_TRACE_JSONL=outputs/m5/compiled.jsonl \
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/m5_exec_mode.py --no-enforce-eager --max-tokens 8 \
  --out outputs/m5/compiled_config.json 2>&1 | tee logs/m5/compiled.log

# 回归（钩子零副作用，trace 关）：输出应与 Day1 基线逐 token 一致
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  python scripts/day1_baseline.py --max-tokens 16 --out outputs/m5/regress_eager.json
```
