# M24 · Context Parallel / Ring Attention / Prefill-Decode Disaggregation —— T + DCP E

> **状态**：**T COMPLETE**（2026-08-14）。**DCP E+O+H COMPLETE**（2026-08-16 · 4×H20 / 1.5B）。**Ring Attention / P-D E 未执行**。
> **版本基准**：`../vllm-src` @ `ee0da84` = tag **v0.24.0**。下文 `file:line` 均对本树实读；
> 本工作树含所有者中文学习注释，行号相对干净上游会偏。
> **DCP 实测数字**只适用于 4×H20 / v0.24.0 wheel / 1.5B / eager / in-process，见 [`results.md`](results.md)；
> 不得与 5080 或 M22 7B TP 相减。
>
> 面试一句：**在 vLLM 里看到 CP 配置 ≠ 掌握 Ring Attention。**

---

## 0. 三件事先拆开

| 名字 | 切什么 | 本树落地程度 | 典型通信 |
|---|---|---|---|
| **Prefill CP（PCP）** | 长 prompt 的 **序列维** | 配置项在；官方写两种策略 **under active development**；attention impl 默认 `supports_pcp=False`，全仓 **无** `supports_pcp = True` | 按策略：KV gather，或环形传 KV |
| **Decode CP（DCP）** | decode 的 **KV 沿 T 维分片** | **已支持**（MLA 与 GQA）；复用 TP 的 GPU，**不增加** `world_size` | **Q all-gather + LSE/output 合并**（默认 `ag_rs`） |
| **Ring Attention** | 分块序列、环形传 KV、online softmax | **论文算法**。官方 PCP 文档把它列为「partial Q, partial KV」的候选。**本树 DCP ≠ Ring Attention** | 每 hop 传一块 KV |
| **P-D Disaggregation** | prefill / decode **拆成两个实例** | experimental；连接器在 `vllm/distributed/kv_transfer/`；生产级靠第三方 | 跨实例搬 **整段 KV** |

M22 已测的是 **TP（head/权重维）**。CP 切的是 **上下文/序列维**。两者可以叠，但不是同一件事。

---

## 1. Context Parallel

### 1.1 官方把 Prefill CP 和 Decode CP 拆开

`docs/serving/context_parallel_deployment.md`：

- 长 prefill：用序列切分摊 TTFT。
- 长 decode：KV 太大或 TP 把 KV **按 head 切完后仍在复制**，再沿 **T** 切以减 duplication、腾 batch。

Prefill 两种策略（**原文：Both approaches are under active development**）：

1. **Partial Q, full KV**：各卡算自己那截 QKV，再 **gather 全量 KV**，每卡只算自己那截 Q 的 attention。适合「KV 还能放得下、只想摊 prefill」。
2. **Partial Q, partial KV**：KV 也放不下 → 文档点名 **ring-attention**（arxiv 2310.01889），逐块 send/recv KV。

Decode：先 TP 沿 **H_kv** 切；`H` 用尽后继续加 TP 只会 **复制** KV。这时加 `-dcp`，沿 **T** 再切。
**`dcp` 不增加要启动的 GPU 数**，只是把已有 TP 组再划成 DCP 组；`dcp ∈ [1, tp/H_kv]`。

### 1.2 配置：谁占 GPU、谁不占

`vllm/config/parallel.py`：

| 字段 | 行 | 对 GPU 数的影响 |
|---|---|---|
| `prefill_context_parallel_size` | `:124` | **乘进** `world_size`（`:793-797` = PP × TP × **PCP**） |
| `decode_context_parallel_size` | `:339-342` | **不改** `world_size`；「reuse the GPUs of TP group」；要求 `tp % dcp == 0`（`:498-507`） |
| `cp_kv_cache_interleave_size` | `:359-372` | 沿 T 交错存放；`dcp_kv_cache_interleave_size`（`:344-348`）已弃用、等 PCP 完整后删除 |
| `dcp_comm_backend` | `:351-357` | 默认 `"ag_rs"`（AllGather + ReduceScatter）；`"a2a"` 把部分 output+LSE 做 all-to-all |

GQA 上限在 `vllm/config/model.py:1184-1207`（非 MLA）：

- `tp > H_kv`
- `dcp ≤ tp / H_kv`
- `H_q / H_kv` 必须能被 `dcp` 整除

进程组：`get_dcp_group` / `get_pcp_group`（`distributed/parallel_state.py:1357`、`:1405`）。

### 1.3 KV 怎么按 T 交错

官方归功于 Moonshot 的 interleave（文档另引 arxiv 2507.07120）：未来 decode token 也能自然落到各 rank。

对 `total_cp_rank = pcp_rank * dcp_world_size + dcp_rank`、
`total_cp_world_size = pcp * dcp`（`parallel.py:361-366`）：

- `interleave_size=1`：token `i` 在 rank `i % total_cp_world_size`
- `interleave_size=block_size`：先填满 (rank i, block j) 再填 (rank i+1, block j)

Scheduler 侧把逻辑 block 放大，使一次分配覆盖所有 CP rank 的物理片段：

- `v1/core/kv_cache_utils.py:635-641`：单 KV group 时 `scheduler_block_size = cache_config.block_size * dcp * pcp`
- `v1/core/single_type_kv_cache_manager.py:70-73`：manager 的 `block_size *= dcp * pcp`

Hybrid 多 group + CP>1 直接拒（`kv_cache_utils.py:643-647`）。

### 1.4 Slot mapping（钉死本树，不要和 M13 的 main 混）

`v1/worker/block_table.py:160-173` **本 pinned 树**仍把 PCP 算进去：

```text
total_cp_world_size = pcp_world_size * dcp_world_size
total_cp_rank       = pcp_rank * dcp_world_size + dcp_rank
```

传给 Triton kernel 的是 `TOTAL_CP_WORLD_SIZE` / `TOTAL_CP_RANK` / `CP_KV_CACHE_INTERLEAVE_SIZE`。

**版本漂移（M13 PR 在最新 main 上核对过）**：上游后来把 slot mapping 改成 **只按 DCP 分片**（PCP 不再参与）。
那是 **另一棵树** 的事实。本模块 T 只陈述 **ee0da84 = pcp×dcp**。CPU fallback 仍 `assert TOTAL_CP_WORLD_SIZE == 1`（`vllm/utils/cpu_triton_utils.py`），CP 语义只在 GPU kernel 里。

M8 的 CP world=2/4 用例测的是 **这套 interleave 的 slot 公式**，不是 Ring Attention。

### 1.5 DCP decode 路径（这就是「看到 CP 配置」实际跑的东西）

FlashAttention：`dcp_world_size > 1` 走 `_forward_with_dcp`（`v1/attention/backends/flash_attn.py:836-849`、`:1019+`）：

1. **`all_gather(query, dim=1)`**（`:1041`）——各 rank 补齐 Q。
2. 在 **本地 KV shard** 上跑 `flash_attn_varlen_func`，**要 LSE**（`:1052-1074`）。
3. 用 `dcp_combine` 合成全局 softmax：默认 `cp_lse_ag_out_rs`，`dcp_comm_backend=a2a` 时 `dcp_a2a_lse_reduce`（`:692-698`、`:1076-1080`）。
4. 再对 **本步新 Q/K/V** 做一次（通常很短的）query-side attention（`:1087+`）。

DCP>1 时 cascade attention 关掉（`:1260-1262`）。

兼容性门禁 `v1/worker/cp_utils.py:14-44`：

- DCP>1 → impl 必须 `need_to_return_lse_for_decode`（由 `can_return_lse_for_decode and dcp>1` 推出，`v1/attention/backend.py:804-806`）。FA 设了 `can_return_lse_for_decode = True`（`flash_attn.py:630`）。
- PCP>1 → impl 必须 `supports_pcp`。基类默认 **False**（`backend.py:753`），**本树没有任何 backend 把它改成 True**。
  因此：配置里能写 `prefill_context_parallel_size>1`，但现有 attention impl 会在兼容性检查处断言失败。
  这与官方「PCP under active development」一致，**不是漏读**。

### 1.6 DCP 性能模型（未测，只作面试框架）

记 `T` = 上下文 token 数，`d` = dcp。相对「TP 已切完 H_kv、KV 仍整段复制」：

```text
每步 ≈  T_Q_allgather          # payload 小：decode 每请求 ~1 token × heads × dim
      + T_local_attn(T/d)      # 本地 KV 是 1/d
      + T_LSE_combine          # ag_rs：三次 NCCL 量级；a2a：文档写 MLA 可降到两次
      + 非 attention 层        # 仍按 TP，不按 dcp 再切
```

- **省的是 KV 容量**（复制因子从 `tp/H_kv` 往 1 收），从而 **batch / 并发** 可能上去——这是吞吐侧的主收益假说。
- **付的是每层额外通信**。Q 很小，主导往往是 LSE/output 同步延迟，不是带宽。
- 小 batch、短上下文：很像 M22 的「小 batch TP 更慢」——固定通信税 > 省下的本地 attention。
- **这不是 Ring**：没有「KV 块绕环走 N−1 hop」。把 DCP 讲成 Ring Attention 是错的。

PCP 若将来走「partial Q, full KV gather」：通信是 **整段 KV all-gather**，prefill 算力按 Q 分摊；KV 显存峰值并不降。
若走 Ring：见 §2。本树 **没有** 可引用的 PCP 实现行号。

---

## 2. Ring Attention（理论；本树未实现为 DCP）

论文：Liu et al., *Ring Attention with Blockwise Transformers for Near-Infinite Context*，arxiv **2310.01889**（2023）。
vLLM 官方 PCP 文档把它列为第二种 prefill 策略的例子，并写明 **尚未完成**。

### 2.1 标准 attention 的数据依赖

完整 `softmax(Q Kᵀ) V` 需要每个 query 看到（因果约束内的）全部 key/value。
把序列切成 N 块后，单卡上的 `Q_i, K_i, V_i` **不够**算 `Q_i` 的正确输出。

### 2.2 环形传 KV + online softmax

- 每卡常驻自己的 `Q_i` 和当前这块 `KV`。
- 沿 ring 把 `KV` 传给下一卡、从上一卡收下一块。
- **Online softmax**（Milakov & Gimelshein 的 running m/lse 归约）：每看到一块 KV 就更新分子和 log-sum-exp，最后才归一化。块顺序不影响数值（因果 mask 除外）。
- **重叠**：传下一块 KV 的同时算当前块。重叠失败则 N−1 次 hop 的延迟直接露出来。
- **因果**：块要按时间序处理，使 `Q_i` 不会先看到未来的 `K_j`；实现上通常靠块调度而不是把整幅因果矩阵物化。

### 2.3 性能模型

```text
T_ring ≈ Σ_hops max( T_compute(Q_i, KV_chunk), T_comm(KV_chunk) )  +  收尾归一化
```

- 本地计算随 chunk 变小；**总算术量不降**（每个 Q 仍要扫完全部允许的 K），变的是 **峰值激活 / 峰值 KV**。
- 收益条件：chunk 的计算 ≥ 一次 KV hop，否则变成通信墙。
- 和 DCP 的对比一句话：**Ring 搬的是 KV 块；本树 DCP decode 搬的是 Q 和 LSE。**

### 2.4 本 lab 明确不声称的

- 未跑多卡 Ring，未读第三方 ring-attn CUDA 实现并对照 Nsight。
- 未把「配置了 `-dcp`」写成「会走 Ring」。
- 路径「论文 → PyTorch reference → 开源实现 → 多卡 → Nsight」留在 ROADMAP，属 **E/H，未执行**。

---

## 3. Prefill-Decode Disaggregation

### 3.1 官方动机（以及一句硬话）

`docs/features/disagg_prefill.md`：

1. **分别调 TTFT 和 ITL**：prefill 实例与 decode 实例可用不同 TP/PP。
2. **控尾部 ITL**：colocated 时，一条正在 decode 的请求可能被同引擎插入的 prefill 拉长尾延迟。
   Chunked prefill 也能瞄这个目标，但 **chunk size 很难调对**；文档认为 P-D 更稳。

**原文：Disaggregated prefill DOES NOT improve throughput。** 优化的是延迟分解与尾部，不是总 token/s。

本 lab 已有 **同机 colocated** 旁证，不是 P-D 实测：M17 E3-conc 里，长 prefill 插入同批时 decode 最坏 ITL **22.87→10.51 ms（−54%）**（chunked vs 一次打满），median/p95 的 CI 含 0。
那是「把 prefill 切碎」对尾部 ITL 的效果。P-D 走得更远：prefill **根本不进** decode 实例的 batch。两者机制不同，数字 **不得相减**。

### 3.2 源码路径

两个 vLLM 实例 + connector。全部在 `vllm/distributed/kv_transfer/`。

角色（`vllm/config/kv_transfer.py:11-48`）：

- `kv_producer` / `kv_consumer` / `kv_both`
- `kv_rank`：注释写典型 0=prefill、1=decode；**「Currently only 1P1D is supported.」**

工厂注册（`kv_connector/factory.py:152-242`，lazy import）：
ExampleConnector、ExampleHiddenStatesConnector、LMCacheConnectorV1、LMCacheMPConnector、
NixlConnector / NixlPull / NixlPush、MultiConnector、MoRIIOConnector、OffloadingConnector、
DecodeBenchConnector、MooncakeConnector / MooncakeStoreConnector、FlexKVConnectorV1、
SimpleCPUOffloadConnector、HF3FSKVConnector。

文档还区分三层抽象：Connector / LookupBuffer（`insert` 非阻塞、`drop_select` 阻塞）/ Pipe（单向 `send_tensor`/`recv_tensor`）。

Scheduler connector 与 Worker connector **分开构造**（`factory.py:67-75`）：调度进程只排 transfer op，worker 在 forward 里执行。

Worker 生命周期（`v1/worker/kv_connector_model_runner_mixin.py:86-100`）：

1. `bind_connector_metadata`
2. **`start_load_kv`（forward 前）**
3. 模型 forward
4. **`wait_for_save`（forward 后）**

逐层：attention 包装器在算完后 `save_kv_layer`（`model_executor/layers/attention/kv_transfer_utils.py:50-58`）。

官方定性：P-D **experimental**；生产级连接器依赖第三方，vLLM 收 PR。

### 3.3 性能模型（break-even；未测）

colocated 一步 decode 的墙钟里，混进长 prefill 时：

```text
ITL_colo ≈ T_decode_step + T_prefill_interference   # 尾部尤其被插入的 prefill 拉高
```

P-D：

```text
TTFT_pd  ≈ T_prefill_isolated + T_KV_transfer + T_decode_setup
ITL_pd   ≈ T_decode_step_isolated     # 理想：decode 实例不再插入 prefill
```

**KV 传输量**（数量级，用 M6 的本机 0.5B 口径 **≈12 KiB/token**，**不是** P-D 实测）：

```text
bytes ≈ P × kv_bytes_per_token
# 例：P=2048 → ~24 MiB / 请求（仅 KV；不含激活）
```

NVLink 上 24 MiB 往往不是墙；跨节点以太网 + 排队则可能把 TTFT 吃掉。
Break-even：**若 `T_KV_transfer + decode setup` 大于 colocated 省下的 prefill 干扰，P-D 不赚。**
官方已写明不提吞吐，所以验收应是 **TTFT / 尾部 ITL / SLO 达标率**，不要拿 token/s 当成功线。

还要付：两套权重（或至少两套引擎状态）、路由、backpressure、失败重试。`kv_rank` 注释限制当前是 **1P1D**。

### 3.4 本轮明确不做 / 已做边界（E）

- [x] 多卡 **DCP**（4×H20 / 1.5B / TP=4 / dcp∈{1,2}）——见 [`results.md`](results.md)
- [x] PCP **负例**（4 卡 TP=4×PCP=2 → `world_size=8`，引擎未建成；未走到 `supports_pcp`）
- [ ] PCP 作为性能实验（本树无 `supports_pcp=True`）
- [ ] 双实例 P-D（任何 connector）——P3 未跑
- [ ] Ring Attention 实现或 Nsight
- [ ] 单卡「假 P-D」（两个进程抢一张卡）——即使得出数字，也 **不得** 写成落地

---

## 4. 和已有模块怎么接（面试 30 秒）

| 已有证据 | 接到 M24 |
|---|---|
| **M22 TP** | TP 切权重/head；DCP 在 TP 把 H_kv 切尽之后沿 **T** 再切 KV。小 batch 通信税的故事同构，对象不同。 |
| **M17 E3-conc** | colocated 长 prefill 会抬 decode **尾部** ITL。P-D 与 chunked prefill 都瞄这个问题；P-D 用隔离，chunked 用切块。 |
| **M6 KV ≈12 KiB/token** | 只给 P-D 传输量级估算；0.5B / 未剪枝 / 本机。 |
| **M7 CPU 提交地板** | 额外 NCCL / transfer kernel 会抬 decode 固定开销；短上下文 DCP 很可能看不见容量收益。 |
| **M13 / M8 slot mapping** | CP interleave 改的是 **哪颗 token 落在哪张卡的哪个 slot**；钉死本树 `pcp×dcp`，不要背最新 main 的 DCP-only。 |

---

## 5. 未执行 / 已执行

- [x] E：多卡 DCP（KV 复制因子 2.0×、decode step 六档 DCP=2 更慢且 CI 排除 0、通信 trace）——[`results.md`](results.md)
- [x] E：PCP 负例（本 4 卡机 `world_size` 先拒；不是性能数字）
- [ ] E：PCP 正性能（等 impl `supports_pcp=True` 或换支持 PCP 的 backend/版本）
- [ ] E：Ring Attention（论文实现或第三方，Nsight 看重叠）
- [ ] E：P-D 双实例（TTFT / 尾部 ITL / 传输延迟；不把吞吐当成功线）
- [ ] H：跨节点 KV、多 P 多 D、故障与 backpressure

Ring / P-D / 跨节点不因 DCP 已跑而勾选。
