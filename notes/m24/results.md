# M24 · 实测结论（DCP=1 vs 2 · D1 KV · D2 decode · P1 通信 · P2 PCP 负例）

> **状态**：**DCP 主线 T+E+O+H COMPLETE**（数据采集 2026-08-16 · 4×H20；本机收口 2026-08-16）。
> 理论 / 源码路径见 [`context_parallel_ring_disagg.md`](context_parallel_ring_disagg.md)。
> 冻结矩阵见 [`M24_H20_PLAN.md`](M24_H20_PLAN.md)。远端观察（不含结论数字）见 [`../../logs/m24/REMOTE_NOTES.md`](../../logs/m24/REMOTE_NOTES.md)。
>
> **指标口径（勿与 M12 混淆）**：延迟均为 **in-process engine** 路径；
> 字段名用 `steady_step_ms_*` / `in_process_first_token_latency_ms_*`，**不是**在线 TTFT/ITL。
>
> **适用边界**：4× NVIDIA H20（NV18）/ driver 535.247.01 / vLLM **0.24.0 wheel** /
> `Qwen2.5-1.5B-Instruct` / TP=4 固定 / `enforce_eager=True` / `async_scheduling=False` /
> in-process / `temperature=0` / `VLLM_ENABLE_V1_MULTIPROCESSING=0`。
> **所有绝对值仅限本配置**。不得与 RTX 5080 / 0.5B / M22 的 7B TP 数字相减。
>
> **未做（仍为 ⬜）**：Ring Attention 实现或 Nsight；P-D 双实例（P3 未跑）；
> PCP 作为性能实验（本版本未实现，P2 只做负例）。

## 环境

| 项 | 值 |
|---|---|
| vLLM | **0.24.0** wheel（`/root/m24-venv/.../site-packages/vllm`；本机 lab pin 仍为 `ee0da84`） |
| 模型 | 本地目录 `Qwen2.5-1.5B-Instruct`（H_q=12 / H_kv=2；GQA 允许 TP=4 DCP=2） |
| GPU | 4× H20，97871 MiB/卡，NV18，开跑前无计算进程 |
| torch / CUDA userspace / Python | 2.11.0+cu130 / cuda-compat-13-0_580.95.05（用户态解包，同 M22）/ 3.11.15 |
| glibc | 2.28。`llguidance` 为 **1.7.6+m24stub**（官方 wheel 要 manylinux_2_31；M24 未用 guided decoding） |
| 采样 | 每格 **5 warmup + 15 measure**；汇总只用 measure 且 `success=True` |
| 统计 | median difference 的 **Bootstrap 95% CI**（10000，seed=0），见 `h20_formal_decode_effect.csv` |
| 正式 tag | `h20_formal` |
| 不变量 | `check_invariants.py --module m24` → **13 PASS / 0 FAIL / 0 PENDING** |

三层证据（传入 → `effective_cfg` 回读 → 不变量）已对齐：两档 `decode_context_parallel_size` 分别为 1 与 2，其余冻结项相同（TP=4、PCP=1、`mp`、eager、mbt=8192、max_model_len=16384）。

---

## 0. 一句话结论

1. **看到 CP 配置 ≠ 掌握 Ring Attention。** 本树 DCP decode = Q-allgather + 本地 KV shard + LSE 合并。P1 chrome trace 里的 `AllGather_RING_LL` 是 **NCCL 集合通信算法名**，不是环形传 KV。
2. **DCP=2 把 KV 可服务 token 正好扩成 2.0×**（5901936 → 11803584），权重不变——复制因子 2→1，**没有多卡**。
3. **本网格 6 档稳态 decode 单步全部是 DCP=2 更慢**（+10.6～+13.8 ms），Bootstrap CI **全部排除 0**。长上下文（8192）也没有出现「step 变快」。冻结假说里「长上下文有机会更快」**未被这组数据支持**。
4. **PCP=2 在 4 卡 TP=4 上建不起来**：失败点是 `world_size=8 > 4 GPU` 的 `ValidationError`，**没走到** `supports_pcp`。与 T 笔记「PCP 乘进 world_size」一致。
5. **P-D 未跑**（冻结可选）。不得把 ExampleConnector 或未做的 P-D 写成落地。

---

## 1. D1 · 显存 / KV（`h20_formal_memory_kv.csv`）

观测：`collective_rpc(callable)`，4 ranks 均 `ok=True`。

| DCP | n_ranks | Σ weight_bytes | weight MiB | KV tokens | vs DCP=1 | num_kv_heads_per_rank | dcp_world_size（per-rank） |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 3324039168 | 3170.05 | **5 901 936** | 1.00× | 1 | 全 rank = 1 |
| 2 | 4 | 3324039168 | 3170.05 | **11 803 584** | **2.00×** | 1 | rank 映射 `(0,0),(1,1),(2,0),(3,1)` |

- 权重两档 **逐字节相同**（DCP 不切权重）。
- KV token 容量 **恰好 2.0×**：TP=4 / H_kv=2 时 DCP=1 仍复制 KV；DCP=2 去掉复制。`num_gpu_blocks` 几乎不变（368871 vs 368862），变的是每 token 占用的复制份数。
- 这是计划里的 **expected observation**，不是「DCP 让 decode 变快」的证据。

---

## 2. D2 · 稳态 decode 单步（主指标）

数据：`outputs/m24/h20_formal_decode.csv`（每格 n=15）、效应：`h20_formal_decode_effect.csv`。
180 个 measure：输出长度均为 32、`steady_batch_stable=True`、scheduler 内省 180/180。

`median_dcp2_minus_dcp1_ms`：**正值 = DCP=2 更慢**。

| prompt_len | batch | DCP=1 med (ms) | DCP=2 med (ms) | Δ (DCP2−DCP1) | Bootstrap 95% CI | 排除 0？ |
|---:|---:|---:|---:|---:|---|:--:|
| 1024 | 1 | 12.1157 | 22.7563 | **+10.6406** | [10.5449, 11.1437] | ✅ |
| 1024 | 8 | 12.2488 | 25.9767 | **+13.7279** | [13.6531, 13.8903] | ✅ |
| 4096 | 1 | 12.0414 | 23.8265 | **+11.7851** | [11.5185, 11.9021] | ✅ |
| 4096 | 8 | 12.3077 | 25.9336 | **+13.6259** | [13.3040, 13.8719] | ✅ |
| 8192 | 1 | 12.1128 | 23.8469 | **+11.7341** | [11.6082, 11.8132] | ✅ |
| 8192 | 8 | 12.3265 | 26.1144 | **+13.7879** | [13.5707, 14.1188] | ✅ |

相对变慢（点估计）：batch=1 约 **+88～+98%**，batch=8 约 **+111～+112%**。六档 **全部 CI 排除 0**。

**和冻结假说的对齐**：短上下文 / 小 batch「DCP=2 可能更慢」——本数据支持。  
长上下文「本地 KV 为 1/2，step 有机会更快或持平」——在 P∈{1024,4096,8192}、batch∈{1,8}、1.5B eager **未观察到**；不能外推「再加长就会翻盘」，也不能把「未观察到收益」写成「DCP 没有容量价值」（D1 的 2.0× token 是另一口径）。

`in_process_first_token_latency_ms` 同样是 DCP=2 更高（例如 P=1024 batch=1：13.15→27.00 ms）。这不是在线 TTFT，也不单独当主结论。

---

## 3. P1 · 通信证据（DCP=2）

产物：`outputs/m24/traces/h20_formal_dcp2/profiler_out_{0..3}.txt` + 4 份 chrome trace（体积大，gitignore；kernel 名以 `profiler_out` 入库）。

H20 首跑只传 `ProfilerConfig`、**不** `start_profile()` → 无 trace（rc=1）。现场只改 profile 分支后重跑；P0 raw 未动。本机已把 `start_profile`/`stop_profile` 补进 `scripts/m24_dcp.py`。

profile 样本自身 `success=False`：8 token → 稳态 step 仅 4（&lt;5），与 smoke 同一阈值，**不进** formal 汇总。

chrome trace（rank0）实际出现的名字（不预设、只记录）：

| 名称 | 约出现次数 | 解读 |
|---|---:|---|
| `vllm::all_gather` | 456 | DCP 的 Q-allgather 路径 |
| `nccl:_all_gather_base` / `AllGather_RING_LL` | 457 / 456 | NCCL AllGather；**RING_LL = NCCL 算法，不是 Ring Attention** |
| `vllm::reduce_scatter` | 224 | 与默认 `dcp_comm_backend=ag_rs` 一致 |
| `_correct_attn_cp_out_kernel` | 224 | CP 输出校正 |
| `vllm::merge_attn_states_kernel` / `FlashAttnFwdCombine` | 224 / 448 | LSE / 注意力状态合并 |

找不到环形传 KV block 的证据。标 observation：**通信路径支持「这是 DCP，不是 Ring Attention」**。

---

## 4. P2 · PCP 负例

`outputs/m24/raw/h20_formal_pcp_negative.json`：`built_engine=false`，`error_type=ValidationError`。

原始失败（节选）：

```text
World size (8) is larger than the number of available GPUs (4) in this node.
```

脚本当时把「符合预期」写成命中 `supports_pcp`，故落盘 `ok_as_negative=false`。  
**引擎确实没建成**；失败点比断言更早——PCP=2 乘进 `world_size`（TP4×PCP2=8）。这与 T 笔记一致，**不是**「PCP 意外可用」。  
未改已落盘 json。本机脚本已把 `World size` 也识别为有效负例，避免下次编排器口径含糊。

4 卡上把 TP 改成 2 再去撞 `supports_pcp` = 换自变量，冻结矩阵禁止。PCP 性能实验仍为 ⬜。

---

## 5. 明确未做

- [ ] **P3 P-D smoke**（冻结可选；官方不提吞吐；ExampleConnector ≠ 生产 NIXL）
- [ ] Ring Attention 实现 / Nsight
- [ ] 加长 prompt / 加大 batch 寻找 DCP=2 翻盘点（改冻结网格）
- [ ] 与 M22 7B TP 或 5080 数字做差

---

## 6. 面试 30 秒

「vLLM 这棵树的 Decode Context Parallel 不是 Ring Attention。我在 4×H20 上只改 `decode_context_parallel_size`：KV token 容量精确变成 2 倍，但 1.5B eager 的稳态 decode 单步六档都显著更慢（CI 排除 0）。通信 trace 看到的是 all-gather / reduce-scatter 和 LSE 合并；NCCL 名字里的 RING_LL 不要讲成环形 Attention。PCP 在 4 卡 TP=4 上连 world_size 都过不了。P-D 我没跑，也不在共享目录连接器上报加速。」
