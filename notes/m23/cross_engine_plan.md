# M23 · TensorRT-LLM 与跨引擎对照 —— **设计冻结 / DESIGN FROZEN**（v3，2026-08-13）

> **状态：🔒 DESIGN FROZEN。未执行任何实验、未拉取任何镜像、未启动任何 Gate。**
> 冻结之后**不得**为了让结论好看而改网格、改样本量、改判据或挑档位；
> 需要改动时必须先说明理由并重新走审查。
>
> **v2 修订（所有者指出的重大上游事实）**：上游已**移除 TensorRT engine backend**，
> PyTorch backend 成为**唯一**执行后端。v1 把「engine 构建 / tactic / 固定 runtime」当作当前 TRT-LLM 的
> 主执行路径 —— **那是错的**，已改写技术内核（见 §1、§2）。
>
> **v3 修订（本轮，冻结前的最后一次小范围修正）**：
> ① 修正 U4/L3 —— 被移除的是 **legacy TensorRT-engine Triton backend**，
> **不等于**「Triton 与当前 TRT-LLM 的集成全部消失」；L3 是**可行但本轮不批准**，**不是不可行**（§1、§6）。
> ② 收窄显存对照口径，禁止残差推导（§5.3）。
> ③ 引入 **Primary cross-engine metrics / Engine-internal explanatory metrics** 两层分级（§5.2）。
> ④ 冻结唯一镜像 tag 与授权状态（§6.1、§7）。
>
> 边界：单卡 RTX 5080(sm_120) / WSL2 / Qwen2.5-0.5B(+1.5B) / 单副本。

---

## 1. 上游事实（先摆证据，后定方案）

| # | 事实 | 出处 |
|---|---|---|
| **U1** | **TensorRT engine backend 已被移除，PyTorch 成为唯一执行后端**（AutoDeploy 建在 PyTorch backend 之上，仍保留） | 官方迁移指南 *Migration Guide: TensorRT Backend Removed* |
| **U2** | **v1.3.0rc20（2026-06-30）是最后一个支持 TensorRT backend 的版本** —— release notes 原文："This RC version will be the last one supporting the TensorRT backend, in the next version the TensorRT backend will be removed!" | GitHub release v1.3.0rc20 |
| **U3** | 随之移除的 API/工具：`LLM(backend="tensorrt")`（现抛 `ValueError`）、`TrtLlmArgs`（改用 `TorchLlmArgs`）、`tensorrt_llm._tensorrt_engine.LLM`、**`trtllm-build` / `trtllm-refit` / `trtllm-prune`**（**不再有 engine build 步骤，HF checkpoint 直接加载**）、每模型的 `convert_checkpoint.py`（**不再需要 checkpoint 转换**）、`tensorrt` pip 依赖（已不再安装） | 同迁移指南 |
| **U4** | **legacy TensorRT-engine 的 Triton backend 已被移除**（PR #15907，属同一分阶段移除序列的一步）。**⚠️ 边界**：该删除针对的是**服务旧 TensorRT engine / legacy TensorRT backend 的那条 Triton 路径**，**不等于** Triton 与**当前** TRT-LLM（PyTorch / LLM-API）的集成全部消失 —— `Triton Inference Server → 当前 TRT-LLM 集成 → LLM API / PyTorch backend → GPU` 这类路线仍然存在，只是需按**现行官方方式**重新确认接法 | PR #15918 描述里列出的移除步骤 |
| **U5** | 官方 README 现在自述 **"Architected on PyTorch"** | 迁移指南 / README |

### 这对 M23 意味着什么（技术内核改写）

**v1 的框架错在哪**：我把 M23 的内核写成「vLLM 的动态 PyTorch 执行 **vs** TRT-LLM 的 engine 构建/tactic/固定 runtime」。
按 U1/U3，**当前 TRT-LLM 根本没有 engine 构建步骤**，两边都是 PyTorch 运行时。这个对立面已经不存在了。

**v2 的框架**：M23 要分清**两条完全不同的对象**，且必须在文档里始终显式区分：

| 代号 | 对象 | 现状 | 在 M23 中的角色 |
|---|---|---|---|
| **TRT-P** | **当前 TRT-LLM（PyTorch backend）** | 上游唯一在演进的路径 | **对照实验的主角**：与 vLLM 同为 PyTorch 栈，比的是 **scheduler / kernel / runtime 成熟度与工程取舍**，不是「编译 vs 动态」 |
| **TRT-L** | **legacy TensorRT engine backend** | **已退役**，最后可用版本 **v1.3.0rc20** | **只作为 T（理解）**：engine 构建 / tactic 选择 / profile 机制，是理解「编译式推理栈」的活教材，但**它是一条上游已经放弃的路** |

→ **结论表述纪律**：任何涉及 engine build/tactic 的描述，**必须标注它属于 TRT-L（已退役）**，
不得写成「TRT-LLM 的执行方式」。反过来，性能对照的结论只能挂在 **TRT-P** 上。

**顺带修正 L3 的定位（v3 再修）**：v1 里 L3 是「Triton + TRT-LLM Backend」。按 U4，
**原 L3 所指的 legacy TensorRT-engine Triton 路线已失效**；
但**当前 Triton + TRT-LLM（PyTorch / LLM-API）integration 仍需按现行官方方式重新定义**，
而不是「当前版本上根本不可行」。

> **必须严格区分的两句话**（v2 在这里说过头了，v3 更正）：
> - ❌ 不可行 —— 会误导成「这条路已经堵死」；
> - ✅ **可行但本轮不做** —— 技术上存在现行集成方式，只是**边际价值待评估、本轮不批准**。

**L3 本轮不批准，留待 L1/L2 结论出来后重新评估其边际价值。**

---

## 2. 修订后的 M23 价值主张

### 2.0 冻结后的技术主线（M23 的核心问题）

> **既然当前 vLLM 和 TRT-LLM 都以 PyTorch runtime 为核心，
> 它们为什么仍然是两套不同的 LLM inference engine？**

后续实验围绕这个问题展开，研究维度限定为：

```text
Scheduler
KV management
Kernel / runtime optimization
Quantization support
Request adaptation
Memory behavior
Startup / deployment complexity
Observability
```

**明令弃用的旧对立**（已被 U1/U3 推翻，任何文档/结论中不得再用）：

```text
vLLM = dynamic PyTorch   vs   TRT-LLM = prebuilt TensorRT engine
```

### 2.1 为什么仍然值得做

即使没有了「编译 vs 动态」这个卖点，M23 仍然值得做，但**理由变了**，得说清楚：

1. **vLLM vs TRT-LLM 是产业界最常被问的那个对照**，而现在两者**同为 PyTorch 栈** ——
   差异落在调度、kernel 选型、量化支持、动态请求适配、可观测性上。这比「编译 vs 解释」更贴近真实选型问题。
2. **它是项目 D 唯一的阻塞项**（见 `notes/m26/portfolio.md`）。
3. **U1–U5 本身就是一条有价值的结论**：一个以 TensorRT 命名的库把 TensorRT backend 删了。
   能把这件事的**时间线、迁移影响（U3 那张表）、以及它对"编译式推理栈"路线的含义**讲清楚，
   本身就是很好的面试素材 —— 而且**这部分零成本、零风险**，不依赖 Gate 是否通过。

---

## 3. 硬性红线

1. **container-only**：**不在 host 建任何 TRT-LLM venv**。
   （所有者 2026-08-13 明确：独立 venv **只有在证明问题出在 container packaging 之后**才可申请。）
   理由仍然成立：`pip install tensorrt_llm` 有替换 torch 的记录（cu130→cu128），会摧毁 lab venv 与 M6–M22 的可复现性。
2. **一次只拉一个精确 tag**。失败时**保存完整 Gate 证据并回来上报**，
   **不得自动连续拉多个 20GB 级镜像**。
3. **不做源码构建**（~63GB / 30–60 分钟）；**不碰 NVFP4**（消费级 Blackwell 有已知数值问题）。
4. **不预授权租机时**。Gate 1 若在有针对性的版本上仍失败，**带证据回来再由所有者决策**。
5. 不与 M10/M18 的历史数字相减（跨会话漂移，M21 已实证）；对照在同一会话内重测。
6. **失败如实报告**，不把 TRT-LLM 的失败包装成「未深入」。

---

## 4. Gate 设计

### Gate 0 · 容器可用性（**只拉一个 tag**）
- 拉一个**精确 tag** 的 `nvcr.io/nvidia/tensorrt-llm/release:<tag>`；容器内 `import tensorrt_llm`、打印版本、确认 GPU 可见。
- **判据**：导入成功 + 看到 sm_120 设备。
- **失败**：保存 `docker logs` / 原始报错 / 镜像 digest → **回来上报**，不自行换 tag 连拉。

### Gate 1 · **真实推理**（决定性关卡）
- **TRT-P 路线**（推荐，见 §7）：用 `tensorrt_llm.LLM` 直接加载 **Qwen2.5-0.5B-Instruct**（本地已缓存）跑 greedy 生成。
  **注意：按 U3，这条路径没有 engine build，也不需要 checkpoint 转换。**
- **TRT-L 路线**（仅当选择研究 legacy 时）：tag 固定为能提供 legacy backend 的 **v1.3.0rc20**，
  走 `convert_checkpoint.py` → `trtllm-build` → 推理。
- **⭐ 判据红线（两条路线通用，冻结不改）**：

  ```text
  import 成功        ≠ PASS
  model load 成功    ≠ PASS
  GPU 可见           ≠ PASS

  必须真实 generate 出文本才算 PASS
  ```

  且无 `invalid resource handle` / `CUDA ERROR 35` 一类 sm_120 kernel 缺失报错。
  **理由**：sm_120 的 kernel / runtime 问题**可能只有在真正执行 forward / attention / decode 时才暴露** ——
  上游已有「构建成功、推理失败」的实测报告（F2）。
- **失败**：保存完整证据（原始报错、版本、digest、`nvidia-smi`、容器日志）→ **回来上报**（见 §8）。

### Gate 2 · 对照可行性
- 确认能对齐固定网格（in/out 长度、并发），并能从两侧**读回生效配置**（精度、batch 上限、KV 配置）。
- 对不齐的项**如实列出**，据此调整口径而不是硬凑。

---

## 5. 对照设计（Gate 全过后执行）

### 5.1 三点测量与**措辞纪律**

| 代号 | 运行位置 | 作用 |
|---|---|---|
| `V-host` | vLLM，host lab venv（M18 同配置） | 与既有模块可对接 |
| `V-ctr` | vLLM，`tritonserver:26.07-vllm-python-py3`（M20-C 已验证） | 容器内的 vLLM |
| `T-ctr` | TRT-P，NGC release 容器 | 被测引擎 |

- **跨引擎结论只用 `V-ctr` vs `T-ctr`**（两边都在容器里）。
- **⚠️ 措辞修正（v2）**：`V-host` vs `V-ctr` **不得称为「纯容器化开销」**。
  M20 已实测两侧 vLLM 版本不同（host `ee0da84` vs 容器内 `0.24.0+092c4842`），**软件栈本身就不同**；
  该差值是**「host 与 container 两套软件栈的综合差异」**，其中容器化只是成因之一。
  只有当两侧软件栈逐项一致时，才可以叫容器化开销 —— **本模块做不到，所以不这么叫**。
- **必须声明的不可消除差异**：两个容器的 CUDA / PyTorch / 驱动栈版本不同，vLLM 与 TRT-LLM 属不同软件世代。
  → 本对照是 **stack-vs-stack**，**不是** kernel 级 A/B。这句话要写进结论。

### 5.2 固定网格与**指标分级**（跑前冻结）

**网格**
- **模型**：Qwen2.5-0.5B-Instruct（主）；1.5B 为可选规模对照。均已在本地缓存，**无需下载**。
- **精度**：两侧统一（BF16 或 FP16，取决于 Gate 2 实测 TRT-P 在 sm_120 上稳定支持哪一种）。
  M18 的教训：dtype 不冻结就会混入第二个变量。
- **workload**：沿用 M12/M20 的 Shape A（in256/out128），greedy、`ignore_eos`、固定 seed。
- **样本量**：每点 5 warmup + 15 measure；median difference 的 **Bootstrap 95% CI（固定 seed）**。

#### ⭐ 指标分两级（v3 新增，这是本模块公平性的关键约束）

**A. Primary cross-engine metrics —— 只有这些能承担正式性能结论**

两侧**共同、且由外部定义**（不依赖任一 runtime 的内部 instrumentation）：

```text
1. end-to-end request latency
2. request throughput
3. output token throughput
4. total GPU memory
5. startup / cold-start time
```

**`V-ctr` vs `T-ctr` 的正式对照结论只能建立在这五项上。**

**B. Engine-internal explanatory metrics —— 只能解释现象，不能当结论**

```text
prefill proxy
decode proxy
internal scheduler timing
internal KV accounting
```

这些指标两侧的 **instrumentation boundary 未必一致**（各自在不同位置打点、包含的阶段不同），因此：
- ✅ **可以分别报告**；
- ✅ **可以用来解释** Primary 指标为什么呈现某种形态；
- ❌ **不得直接相减后称为「严格的 engine speedup」**。

> **纪律：同名指标 ≠ 自动同口径。**
> 这条直接沿用 M21 的教训 —— 服务端 `histogram_quantile` 的 P99 与客户端 exact percentile
> 名字都叫「P99」，实测中位偏差 +9.7%、最大 +68%，而且**无法归因到单一成因**；
> 当时能下的结论只有「两者不可互换」。跨引擎的内部指标同理，只会更严重。

**沿用既有命名纪律**：in-process proxy 指标**不叫 TTFT/TPOT**（那是 M12 的在线口径）。

### 5.3 要比的维度

| 维度 | 指标层级 | 能否实测 | 说明 |
|---|---|---|---|
| 安装/上手复杂度 | — | 🟢 记录真实步骤与耗时 | **U3 之后这条变了**：TRT-P 不再需要 checkpoint 转换与 engine build，上手成本大幅下降 —— 这个「变化本身」就是结论 |
| **启动 / 冷启动时间** | **Primary** | 🟢 | |
| **end-to-end request latency** | **Primary** | 🟢 | 固定网格 + Bootstrap CI |
| **request throughput / output token throughput** | **Primary** | 🟢 | |
| **total GPU memory** | **Primary** | 🟢 | 见下方「显存口径」 |
| prefill / decode proxy | Internal（解释用） | 🟢 | **不得相减称 engine speedup** |
| 量化能力 | — | 🟡 只做能力对照 | vLLM 侧已有 M10/M18；TRT-P 侧只验证「能否加载 INT4-AWQ」 |
| 动态请求适配 | — | 🟡 定性 | 见下方事实修正 |
| 可观测性 / 调试 | — | 🟡 定性 | 承接 M21 |

#### ⭐ 显存口径（v3 收窄：宁可少报，不制造假精度）

两个 runtime 的 allocator、workspace、CUDA context、reserve memory、temporary buffers 的口径**未必一致**，
因此**不承诺**「权重 / KV / 激活三项都能分开对照」：

| 项 | 承诺级别 |
|---|---|
| **total GPU memory** | **必须实测**，并作为**正式跨引擎指标**（Primary） |
| **weights / KV** | **仅当两侧 runtime 都能给出定义清晰、可验证的分项时**才分别报告；否则不报 |
| **activations / workspace / runtime buffers** | 若拿不到两侧同口径的可信分项，**明确标记 `unavailable` / `not directly comparable`** |

**明令禁止的做法**：

```text
activation = total − weights − KV
```

除非能**证明** residual 中没有混入 allocator reserve、CUDA context、workspace 等其他成分 ——
否则这只是把「不知道的部分」改个名字叫「激活」，属于制造假精度。

**⚠️ 事实修正（v2）**：v1 写的「TRT-LLM engine 的 `max_batch/max_seq_len` 是构建期**固定**的」**不准确**。
正确表述：**legacy TensorRT engine（TRT-L）的 build-time `max_batch_size` / `max_seq_len` 是
optimization profile 的【上界】，不是固定 shape；runtime 在 profile 范围内仍可动态执行**，
只是**超出上界需要重建 engine**。而 **TRT-P（当前路径）根本没有 build 步骤**，这条对比项在当前版本上已不适用 ——
若要讲，必须标明讲的是 TRT-L。

---

## 6. 分级交付（**本轮只批准 L1**）

| 级别 | 内容 | 状态 |
|---|---|---|
| **L1** | Gate 0–2 + **离线跨引擎对照**（`V-ctr` vs `T-ctr`，Shape A，Primary 指标 + 上手成本） | ✅ **已批准** |
| L2 | TRT-LLM 在线服务（`trtllm-serve`）+ M12 口径开环压测 | ⏸ **未批准**，待 L1 结论后再申请 |
| **L3** | **Triton + current TRT-LLM PyTorch/LLM-API integration** | ⏸ **技术上可行，但本轮不批准**；待 L1/L2 结果后再评估边际价值。（**注意：原 v1 所指的 legacy TensorRT-engine Triton 路线已失效，但这不等于当前集成不可行 —— 见 §1 U4**） |
| L4 | Transformers 朴素基线 | ⏸ **未批准**（可选） |
| ⬜ 不做 | 源码构建、NVFP4、多卡、FP8 全矩阵 | 见 §3 |

**另有一项零成本、不依赖 Gate 的产出**（建议无论 Gate 结果如何都做）：
把 **U1–U5 的时间线与迁移影响**整理成 `notes/m23/backend_migration.md` —— 这是 M23 的 T 部分，
不需要 GPU、不需要镜像，且即使 Gate 1 失败也**照样成立**。详见 §6.2。

### 6.1 授权状态（**冻结，2026-08-13**）

```text
L1：✅ 已批准
L2：⏸ 未批准
L3：⏸ 可行但未批准
L4：⏸ 未批准

唯一镜像：
nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24

container-only：      是
host TRT-LLM venv：   禁止
自动换 tag：           禁止
自动租机：             禁止
```

**Gate 0 / Gate 1 任意失败时的固定动作**：

```text
保存完整证据  →  停止  →  回报所有者
```

**不自行进入下一条补救路线**（不换 tag、不建 venv、不租机）。

### 6.2 `backend_migration.md`（独立 T 产出，设计不变）

必须清楚区分两条路径：

```text
TRT-L · legacy TensorRT engine backend
  → engine build
  → tactic
  → optimization profile
  → serialized engine

TRT-P · current PyTorch backend
  → HF checkpoint direct load
  → no trtllm-build
  → current experimental target
```

**明确用途边界**：**学习 TRT-L 是为了理解 TensorRT / AOT inference 的历史与机制，
不用于当前性能主结论。** 当前性能结论只挂在 TRT-P 上。

---

## 7. 唯一镜像 tag（**已冻结**）

```text
路线：TRT-P
Tag ：nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24
```

**选择理由**：

1. **TRT-P 是当前仍在演进的 PyTorch backend 主线**（U1/U5）。
2. **M23 的项目 D 与求职叙事应基于 current stack**，而不是已退役的 TRT-L ——
   对照一条被上游删除的后端，结论一出来就是过时的。
3. **TRT-L 的 engine build / tactic / optimization profile 机制保留在 `backend_migration.md` 的理论学习部分**（§6.2），
   不必为此消耗掉唯一的一次镜像拉取额度。
4. **`rc24` 是 Release Candidate**：sm_120 支持**仍必须通过 Gate 0/1 实测**，
   **不提前宣称兼容成功**。

> 镜像体积参考：NGC release 镜像 amd64 约 **20.5–20.7 GB**（压缩），磁盘现余 **537 GB**。
> 备查：`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc20` 是**最后一个**带 legacy TensorRT backend 的版本（U2），
> 本轮**不拉取**；仅在所有者另行授权研究 TRT-L 时才使用。

### ⚠️ 证据边界（失败时的表述纪律）

若 `rc24` 的 Gate 失败，**只能**表述为：

```text
rc24 + 当前 RTX 5080(sm_120) + WSL2 + 当前 driver / model / config 组合，Gate N 失败
```

**不得**扩大为：

```text
TensorRT-LLM 不支持 RTX 5080
```

—— 未测过的版本、驱动、模型与配置组合不在证据覆盖范围内。

---

## 8. Gate 1 失败时的处置（**不预授权任何花钱选项**）

1. **保存完整证据**：原始报错全文、TRT-LLM 版本与镜像 digest、`nvidia-smi`、容器日志、复现命令 →
   落盘 `outputs/m23/gate1_failure_*.json` + `logs/m23/*`。
2. **回来上报**，由所有者在以下选项间决策（**我不自行启动任何一项**）：
   - **A · 换一个有针对性的 tag**（每次仍只拉一个，需再次授权）
   - **B · 租机时**（单卡 H20 / A100 / L40S；M22 已有成熟的打包/回传/校验流程可复用）。
     若走这条，**vLLM 与 TRT-LLM 必须在同一张卡上同时测**，不能拿新卡的 TRT-LLM 比本机 sm_120 的 vLLM。
   - **C · 降级为 T+O**：只交付 §6 的 `backend_migration.md` + 构建/失败实录，
     E 部分如实标 ⚪，`notes/m26/portfolio.md` 里项目 D 的占位继续保留。

---

## 9. 产物

`notes/m23/{cross_engine_plan(本文→冻结),backend_migration,results}.md`、
`scripts/m23_{probe,bench,summarize,figures}.py`、
`outputs/m23/{gate*,raw/*.jsonl,*.csv,configs/*,env_fingerprint_*}`、`logs/m23/*`、
`figures/fig14_m23_cross_engine.png`、`check_invariants.py --module m23`；
完成后同步 `ROADMAP.md` / `notes/evidence_index.md` / `README.md` / `HANDOFF.md`，
并**回填 `notes/m26/portfolio.md` 中项目 D 的占位**。
