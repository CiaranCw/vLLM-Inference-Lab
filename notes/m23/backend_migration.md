# M23-T · TensorRT-LLM 的后端迁移：从 TensorRT engine 到 PyTorch runtime

> **这是 M23 的 T（理解）部分，零成本、不占 GPU、不依赖任何 Gate** ——
> 即使 Gate 0/1 在本机 sm_120 上失败，本文件的结论**照样成立**。
> **性质声明**：本文件是**对上游公开信息的整理与解读**，不含本机实测数据。
> 凡属解读的地方都会标「**解读**」，与事实分开。

---

## 0. 一句话

**一个以 TensorRT 命名的库，把 TensorRT backend 删了。**
从 v1.3.0rc20 之后，**PyTorch 成为 TensorRT-LLM 唯一的执行后端**，
`trtllm-build` / checkpoint 转换 / serialized engine 这一整套 AOT（ahead-of-time）流程**不再存在于主线**。

---

## 1. 事实与出处（先摆证据）

| # | 事实 | 出处 |
|---|---|---|
| U1 | **TensorRT engine backend 已被移除，PyTorch 成为唯一执行后端**；AutoDeploy（建于 PyTorch backend 之上）保留 | 官方 *Migration Guide: TensorRT Backend Removed* |
| U2 | **v1.3.0rc20（2026-06-30）是最后一个支持 TensorRT backend 的版本**。release notes 原文：<br>"This RC version will be the last one supporting the TensorRT backend, in the next version the TensorRT backend will be removed!" | GitHub release v1.3.0rc20 |
| U3 | 移除是**分阶段**进行的：examples(#15763) → docs(#15767) → tests(#15810) → **Triton C++ backend(#15907)** → python 模块与测试(#15918，自述为 step 1.5) | PR #15918 描述 |
| U4 | 官方 README 现自述 **"Architected on PyTorch"** | README / 迁移指南 |

### 1.1 迁移对照表（U3 之后，什么没了、换成什么）

| 移除项 | 替代 / 新行为 |
|---|---|
| `LLM(backend="tensorrt")` | 抛 `ValueError` —— PyTorch 是唯一后端，直接不传 `backend` |
| `TrtLlmArgs` | 改用 `TorchLlmArgs`（默认） |
| `tensorrt_llm._tensorrt_engine.LLM` | 改用 `tensorrt_llm.LLM` |
| **`trtllm-build` / `trtllm-refit` / `trtllm-prune`** | **没有 engine 构建步骤** —— HuggingFace checkpoint **直接加载** |
| **每模型的 `convert_checkpoint.py`** | **不需要 checkpoint 转换** |
| `--backend tensorrt`（CLI） | 省略，或显式 `--backend pytorch`（默认） |
| `tensorrt` pip 依赖 | **不再安装** |

> **边界（很重要，别读过头）**：U3 里被删的 Triton backend，指的是**服务 legacy TensorRT engine 的那条 Triton 路径**。
> 这**不等于** 「Triton 与当前 TRT-LLM 的集成全部消失」——
> `Triton Inference Server → 当前 TRT-LLM 集成 → LLM API / PyTorch backend → GPU` 这类路线**仍然存在**，
> 只是接法需按现行官方方式重新确认。**「不可行」与「可行但要换接法」必须分清。**

---

## 2. 两条路径的机制对照

| | **TRT-L**（legacy TensorRT engine backend） | **TRT-P**（当前 PyTorch backend） |
|---|---|---|
| 状态 | **已退役**，最后可用版本 v1.3.0rc20 | 上游唯一在演进的路径 |
| 模型准备 | HF checkpoint → `convert_checkpoint.py` → TRT-LLM checkpoint | **无转换**，HF checkpoint 直接加载 |
| 构建 | `trtllm-build` 产出 **serialized engine**（AOT 编译） | **无构建步骤** |
| 优化决策时机 | **构建期**：算子融合、tactic 选择、内核自动调优、精度决策都在这时定死 | **运行期**：kernel 选择、CUDA Graph 捕获等在运行时完成 |
| shape 处理 | **optimization profile**：`max_batch_size` / `max_seq_len` 等是 profile 的**上界** | 运行期动态 |
| 部署产物 | 与 **GPU 型号 / 软件版本绑定**的 engine 文件 | 权重 + 代码 |
| 换 shape / 换卡 | 超出 profile 上界或换 GPU → **需重建 engine** | 直接跑 |

### ⚠️ 一个必须纠正的常见误解（我自己 v1 规划里就写错过）

> ❌ 错误说法：「TRT 的 `max_batch_size` / `max_seq_len` 是**构建期固定的 shape**，runtime 不能变。」
>
> ✅ 正确说法：它们是 **optimization profile 的【上界】，不是固定 shape**。
> **runtime 在 profile 范围内仍可动态执行**（这正是 optimization profile 的用途）；
> 只有**超出上界**才需要重建 engine。

把「上界」说成「固定」，会把 AOT 路线描述得比实际更僵化 —— 这是评价一条技术路线时要避免的失真。

---

## 3. 这件事说明了什么（**解读**，与事实分开）

以下是解读，不是上游声明：

1. **被淘汰的不是「快」，而是一种「优化时机」的选择。**
   上游同期仍在 PyTorch 路径上推 NVFP4 / MXFP8、EAGLE3 投机解码、disaggregated serving 等重度优化工作。
   所以这不是「放弃性能」，而是**把优化从构建期搬到运行期**。

2. **AOT 编译在 LLM 推理上的代价越来越难摊平。** 可观察到的代价包括：
   engine 与 GPU 型号/版本绑定、换 shape 要重建、每个新模型要写 `convert_checkpoint.py`、
   新特性要在编译栈里重新实现一遍。而 LLM 领域模型与特性的迭代速度极快 ——
   **构建期锁定的东西越多，跟上迭代的成本越高**。

3. **对本 lab 的直接影响**：M23 原本设想的对立面「**vLLM = 动态 PyTorch** vs **TRT-LLM = 预编译 engine**」
   **已经不成立**。现在两者同为 PyTorch 运行时，真正的问题变成了：

   > **既然都以 PyTorch runtime 为核心，它们为什么仍然是两套不同的 LLM inference engine？**

   → 差异落在 **Scheduler / KV 管理 / kernel 与 runtime 优化 / 量化支持 / 请求适配 / 显存行为 /
   启动与部署复杂度 / 可观测性**，而不是「编译 vs 解释」。这也是 M23 实验部分的设计依据。

4. **一个可迁移的判断习惯**：看一个项目的技术路线，**看它删掉了什么比看它加了什么更有信息量**。
   README 从「engine」改成 "Architected on PyTorch"，比任何 benchmark 数字都更能说明重心迁移。

---

## 4. 这部分在 M23 里的定位（纪律）

- **学习 TRT-L 是为了理解 TensorRT / AOT inference 的历史与机制**，
  **不用于当前的性能主结论**。任何 engine build / tactic / profile 的描述，**必须标注属于 TRT-L（已退役）**。
- **性能对照只挂在 TRT-P 上**（见 `cross_engine_plan.md` §5）。
- 本文件**不含本机实测**；一旦 Gate 通过并产出数据，实测结论写进 `notes/m23/results.md`，两者不混。

---

## 5. 待 Gate 通过后可以回答、现在**还不能**回答的问题

如实列出，避免把「还没测」写成「已理解」：

- ⬜ TRT-P 在本机 sm_120 上**能否真正跑起来**（Gate 0/1 的目标）。
- ⬜ 同模型同硬件下，TRT-P 与 vLLM 在 Primary metrics（端到端延迟 / 吞吐 / 总显存 / 冷启动）上差多少。
- ⬜ 两者的量化支持差异在本机的实际表现。
- ⬜ TRT-P 的启动与部署复杂度相对 vLLM 的真实差距（U3 之后这块应该显著缩小了，但**没实测就不下结论**）。
