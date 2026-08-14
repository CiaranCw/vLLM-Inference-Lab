# M25 · 模型压缩（蒸馏 / 结构化剪枝 / 低秩）—— T

> **状态**：**T COMPLETE**（2026-08-14）。推理岗只需 T。**本轮明确不做 E**。
> **边界声明**：下列机制是通用知识，接到本 lab 时只作**解释框架**；
> 本机绝对值仍只适用于 RTX 5080 / sm_120 / Qwen2.5-0.5B / eager / in-process，**不外推**。

面试时能讲清三件事即可：**压缩的是什么、推理时为什么可能不加速、和量化/投机解码差在哪**。

---

## 0. 和本 lab 已有证据的关系（先立边界）

压缩的宣传口径几乎总是「参数少了 → FLOPs 少了 → 更快更省」。
本 lab 已经多次测到这个链条在 GPU 推理里会断：

| 已有证据 | 对压缩的含义 |
|---|---|
| **M7**：eager decode 单步 ~7ms 地板 = CPU kernel 提交，GPU 只忙 ~23% | 减 FLOPs 若不减 **launch 次数 / 图捕获形态**，墙钟可以几乎不动 |
| **M6**：prefix 命中省 94% 计算 token，配对延迟差落在噪声里 | 「省计算 ≠ 省延迟」——短 decode 被固定开销盖住 |
| **M9**：合并两处同型 H2D，机制成立，E2E 落噪声 | 机制正确也可以零收益；压缩实验必须报 wall-clock，不能只报 FLOPs |
| **M10 / M18**：FP8 长 prefill 真加速、decode 反而慢；W4A16 **28 个对照点 0 个变快** | 权重变小、连计算受限段都可能更慢（kernel 实现主导） |
| **M19**：投机解码优化的是串行 execution rounds，不是「少算就一定快」 | 另一条「改算法形态」的线；压缩改的是**权重本身** |

因此 M25 的 T 不回答「本机 0.5B 剪枝能快多少」——那需要 E，且很可能撞上同一块地板。
T 要回答的是：**三类方法各自改了计算图的哪一块，以及什么条件下 wall-clock 才有机会动**。

---

## 1. 知识蒸馏（Knowledge Distillation）

### 1.1 角色

- **Teacher**：大模型，质量高、推理贵。
- **Student**：小模型，部署对象。蒸馏的目标是让 student **在更小的容量里模仿 teacher 的行为**，
  而不是从零用 hard label 学任务。

Student 变小之后，推理成本下降来自 **student 实际变小的 architecture**
（更少层、更窄 hidden、更少 KV head 等），**不是**蒸馏损失函数本身，
也**不是**蒸馏会天然缩小词表或 context window。
词表通常与 teacher 对齐以便 logit KD；context limit 改的是 **workload / KV 容量 / attention 计算量**，
不是蒸馏算法的默认产物。蒸馏是**训练期**技术；推理期跑的是已经定下来的 student。

### 1.2 三种常见监督

| 类型 | 学生学什么 | 直觉 |
|---|---|---|
| **Logit / response KD** | teacher 的输出分布（通常带温度） | 软标签比 one-hot 含类间结构 |
| **Feature / hint KD** | 中间层隐状态或 attention map | 对齐内部表示，不只对齐最终词 |
| **Sequence KD** | teacher 的 token 序列（或 on-policy 自己的序列 + teacher 打分） | 更接近自回归生成；暴露偏差（exposure bias） |

**温度 T**：把 softmax 做平滑，`p ∝ exp(z/T)`。T>1 让 teacher 的长尾类也有梯度，
学生能学到「次优词之间的相对关系」。推理时 student **不用**这个温度（除非作为采样超参），
训练温度不会留在 serving 路径上。

### 1.3 怎样真正降推理成本

只有 **student 的 serving 图更便宜** 才算数：

- 更少层 / 更窄 hidden / 更少 KV head → 每 token 的 GEMM 与 KV 读写变小；
- 若只把 teacher 的知识蒸进**同结构** student（同层数、同宽度、同词表、同 `max_model_len`），
  推理成本几乎不变，变的是质量。
- 若另外把 student 的 context window 做小，那是在改 **可服务的 workload 与 KV/attention 规模**，
  应单独声明，不要算进「蒸馏带来的加速」。

接本 lab：即便 student 参数减半，**短 decode 仍可能停在 M7 的 CPU 提交地板上**。
要验证加速，必须在真实引擎里测 TTFT/ITL/吞吐，而不是只报参数量或 FLOPs。

**和量化的差别**：量化改数值精度与 kernel（M10/M18），模型结构通常不变；
蒸馏改的是**另一套更小的权重**。两者可叠加（先蒸小再量化），但变量必须分开报。

---

## 2. 剪枝（Pruning）

### 2.1 非结构化稀疏为什么不必然加速

逐权重置零（unstructured sparsity）**不会**让稠密 GEMM 自动少干活：
`W` 的 shape 没变时，GPU 仍按稠密 tensor core 路径乘完整矩阵，
零元素照样占带宽与指令。必须有**匹配的 sparse kernel / 硬件**
（例如 2:4 半结构化 + sparse tensor core），实际 kernel 工作量才会降。
**「90% 权重是零」≠「快 10 倍」**。这是面试高频点。

### 2.2 结构化剪枝：剪的是 GPU 能看见的维度

| 剪什么 | 计算图变化 | 加速条件 |
|---|---|---|
| **Head** | Q/K/V/O 的 head 维变小；GQA 时还要看 KV head | kernel 按新 shape 走稠密 GEMM |
| **Channel / hidden** | 中间 FFN 宽度变小 | 同上；需整除对齐（warp/tile） |
| **Layer / block** | 深度变小，串行层数下降 | 直接减 decode 步内的层循环；对 M7 地板更敏感（launch 次数下降） |

结构化的价值是：**留下来的是仍然稠密、仍然能走现有 kernel 的更小张量**。
这与 M18 的教训对称——换了一个「理论上更省」但不走高效 kernel 的实现，墙钟可以更差。

### 2.3 和 KV / 调度的耦合

剪 head 或层会改变：

- 每 token KV 字节（接 M3/M6：本机 ≈12 KiB/token 是 **未剪** 的 0.5B 实测）；
- 一层内的 kernel 个数（接 M7 提交数）。

所以剪枝报告不能只写「精度掉了多少」；推理岗要能说出 **KV 容量与 launch 形态变了没有**。

---

## 3. 低秩分解（Low-rank factorization）

### 3.1 在做什么

把一个大矩阵 \(W \in \mathbb{R}^{m \times n}\) 近似成 \(W \approx AB\)，
\(A \in \mathbb{R}^{m \times r}\)、\(B \in \mathbb{R}^{r \times n}\)，\(r \ll \min(m,n)\)。
经典来源是 SVD：截断到前 r 个奇异值。

一次大 GEMM \(xW\) 变成两次小 GEMM \( (xA) B \)。
参数量从 \(mn\) 降到 \(r(m+n)\)。当 \(r < \frac{mn}{m+n}\) 时存储下降。

### 3.2 秩与误差

截断 SVD 在 Frobenius 范数下最优，但：

- 语言模型的下游误差 **不是** Frobenius 误差；
- **逐层可压缩性是 model-dependent 的**，不能先验说某一类层（embed / lm_head / out-proj 等）「通常更难压」；
- r 太小 → 质量崩；r 太大 → 两次 GEMM 的 launch 与中间激活把收益吃掉。

### 3.3 「两小 GEMM」是计算节省与额外开销的 trade-off

一个大 GEMM \(xW\) 变成 \( (xA)\,B \) 之后，算术量可降，但同时多了：

- **至少一次额外 kernel launch**（接 M7：eager decode 上提交次数是一等公民）；
- **中间激活** \(xA\) 的分配与读写。

墙钟变快还是变慢，取决于节省的计算是否盖过这两项。这不是「只有 compute-bound 才可能快」的绝对条件：
memory-bound 路径上若两次小 GEMM 的访存合计更少，也可能更快；compute-bound 路径上若中间结果把带宽打满，也可能更慢。

**CUDA Graph ≠ kernel fusion**。Graph 只降低 **CPU submission overhead**（把一串 launch 收成一次回放），
并不把两个 GEMM 融成一个 kernel，也不消灭中间激活。融合要另走 compiler / epilogue fusion；
有 Graph 只能说提交税变小，不能说计算图变简单。

短 decode 上这个 trade-off 很像 M9：局部更「干净」，E2E 仍可能看不见。

### 3.4 LoRA ≠ 低秩压缩（必须分开讲）

| | **低秩分解压缩** | **LoRA / Adapter** |
|---|---|---|
| 作用对象 | **已有稠密权重 W**，用 AB 替换它来**减小 W** | 在 W **旁边**加 \(\Delta W = BA\)，**W 仍在** |
| 推理图 | 服务时可以只保留 A、B（若真正替换） | 默认是 \(xW + xBA\)，参数与计算都是**加法** |
| 目的 | 压缩部署 | 参数高效微调 |
| 常见误区 | 把「模型是 LoRA 训的」说成「已经低秩压缩了」 | 合并（merge）进 W 后，推理形态回到稠密 W，**没有**压缩收益 |

一句话：**LoRA 是训练期的低秩更新；低秩分解是部署期的权重近似。** 二者线性代数长得像，工程目标相反。

---

## 4. 三类方法怎么放在推理系统里讲

```text
训练期改权重 ──蒸馏──► 更小的 student ──► 推理图更小（层/宽/KV）
                 剪枝──► 更小的稠密 shape 或稀疏 kernel
                 低秩──► 大 GEMM → 两小 GEMM（可能多 launch）

推理期改数值 ──量化（M10/M18）──► 同结构，换 dtype/kernel
推理期改算法 ──投机解码（M19）──► 同权重，少 execution rounds
```

面试 30 秒：**压缩改模型；量化改精度；投机解码改解码过程。** 三者都可以「看起来省了」，
但本 lab 的方法学要求同一句话后面跟上：**在真实引擎、同 dtype、有 warmup 的 wall-clock 里验证过没有。**

---

## 5. 本轮明确不做

- [ ] 下载蒸馏/剪枝/低秩后的 checkpoint（需所有者授权）
- [ ] 在 vLLM 里对压缩模型做 E2E benchmark
- [ ] 自做 SVD / 剪枝 / 蒸馏训练
- [ ] 把 LoRA 合并后的模型当成「低秩压缩完成」

若以后做 E，最低可证伪设计（**未执行，不是承诺**）：
同模型族、同引擎、同 shape，对比 dense vs 压缩 student 的 decode 单步与长 prefill；
先验假设是「短 decode 大概率撞 M7 地板，长 prefill 才看得到 FLOPs」。
数据推翻假设时改结论。

---

## 6. 可引用的外部锚点（T 级，非本仓库数据）

- Distillation：Hinton et al., *Distilling the Knowledge in a Neural Network*（温度与 soft target）。
- 非结构化稀疏不加速：GPU 稠密 tensor core 主导；半结构化 2:4 才是硬件路径。
- LoRA：Hu et al. 2021；与 SVD 截断部署不是同一件事。

这些是概念锚点，**不是**本 lab 的实测。本 lab 能拿来当旁证的只有 §0 那张表。
