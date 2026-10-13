# M28 限制与未做项

> 与 [`results.md`](results.md) 配套。这里列出的每一条，都意味着对应结论不能外推。

## 1. 配置边界

- 单卡 RTX 5080（sm_120，16 GB）/ WSL2 / Qwen2.5-0.5B-Instruct / `enforce_eager` / vLLM `ee0da84`（editable，`VLLM_USE_PRECOMPILED=1`）。
- **CPU 层是 pageable 内存**：`CPUOffloadingSpec` 用 `torch.zeros(pin_memory=PIN_MEMORY)` 分配，WSL2 上 `PIN_MEMORY` 跟随
  `VLLM_WSL2_ENABLE_PIN_MEMORY`（默认 0，计划 v2 强制 unset）。load 约 3.8–4.0 GB/s、store 约 3.1 GB/s 都是 pageable 下的值，
  不代表 pinned 内存、原生 Linux 或 PCIe 5.0 服务器上的带宽。
- 0.5B 模型每 token KV 只有 12288 B，prefill 也很快。"P=1024 时 CPU 命中不划算"这个盈亏点与模型尺寸强相关：更大的模型重算更贵、每 token KV 也更大，盈亏点会移动，方向无法从本实验推出。
- A 组是**串行**在线请求（一次一个），测的是单请求路径延迟，不是吞吐，也不是并发下 store/load 与计算的争用。
- B 组负载固定为 8 条在途请求（4 流式 + 4 非流式），客户端是本仓库的 `http.client` 实现，不是 OpenAI SDK 或真实网关。

## 2. 统计边界

- A 组每格 n=15，报 median + Bootstrap 95% CI（固定 seed）。跨 arm 的对照（offload vs base）来自不同 server 进程，是 unpaired，
  会把进程间差异算进去；1–2 ms 级的差应视为接近噪声。
- evict arm 只报 median，没有计算 cpu_evicted − cold 的 CI。
- B 组每档每轮 n=3，**不估计概率**。B3 两轮分开报告，不合并；第一轮 raw 的挂住请求只有 `missing` 标记（runner 缺陷），
  其"挂住"判断依据的是独立来源（90 s 时 API 存活、日志 "Waiting for connections to close"、清理时 rc −9），而不是请求记录本身。

## 3. 未做的事（未执行，不代表结论成立或不成立）

- [ ] **profiler**：没有对 offload.cold 的额外 39 / 69 ms 做 nsys / torch profiler。"store 卡在 chunk 之间"只是假设。
- [ ] pinned CPU 层（`VLLM_WSL2_ENABLE_PIN_MEMORY=1`）对照——属于改变冻结配置，未做。
- [ ] 并发负载下的 offload 收益与开销。
- [ ] tiering spec / mmap + `cudaHostRegister` 路径——本实验只覆盖 `CPUOffloadingSpec`。
- [ ] B2 SIGSTOP / SIGCONT：所有者选择不做 → **SKIPPED**。
- [ ] B3 t=2 档 EngineDeadError 是否传到在途请求（2/6 传到、4/6 没有）的源码级原因**未定位**。
- [ ] 外部监督器（systemd / k8s）对 B1"API 以 rc 0 退出"的实际反应——只是推断，未实测。
- [ ] `ee0da84` 之外的 vLLM 版本、uvicorn 0.51.0 之外的版本。uvicorn 无 graceful 超时这一点依赖 vLLM 不设置 `timeout_graceful_shutdown`，上游若修改就会变化。

## 4. 产品源码与环境

- `../vllm-src` 有 22 个脏文件（所有者的中文学习注释 + lab 钩子），M28 前后不变。`m28_source_audit.py` 用 AST 比对证明
  M28 路径上的 lab 钩子在默认环境下惰性，但**钩子的固定 Python 开销没有单独测量**，所以绝对延迟不等于上游 wheel 的延迟。
- 第一轮 B3 的 5 次强制清理（runner 当时先 SIGKILL 了 API 和 resource_tracker）在 `/dev/shm` 留下 5 个 `sem.mp-*` 信号量文件
  （`oce92guz`、`mjeod_5h`、`uvel949z`、`mt1elx_i`、`3g9tarul`）。修复清理顺序后第二轮 0 新增。这 5 个文件在 2026-10-13 经所有者同意后删除（删除前确认没有任何进程映射或打开它们）。
- 每个 A server 首次请求期间都出现一次推理期 Triton JIT（`_compute_slot_mapping_kernel`），发生在 server warmup 阶段，不在 measure 内。
