# M28 面试表达（KV Offloading + Serving 可靠性）

> 所有数字来自 [`results.md`](results.md)，边界：单卡 RTX 5080 / WSL2 / Qwen2.5-0.5B / eager / 串行在线请求 / pageable CPU 层。
> 讲的时候先说边界，再说数字。

## 30 秒版

"我在 vLLM 原生的 OffloadingConnector 上，把 GPU 命中、GPU 被挤掉后从 CPU 取回、完全重算这三条路径逐样本用 Prometheus 计数器证明出来，
并和无 cache 重算逐 token 对照。CPU 往返不改变任何输出，从 4K 前缀开始比重算快，16K 时快约 3.5 倍，但 1K 时反而慢 0.7 ms；
开 connector 后冷请求在长前缀上也会多付 39–69 ms。可靠性那边，我发现 SIGTERM 加 `shutdown-timeout=0` 或给得不够时，
API 进程可能无限期挂住——根因是 uvicorn 的 graceful shutdown 没有超时，而在途请求永远收不到结束信号。"

## 三个可以展开的点

1. **路径必须被证明，不能靠"应该命中了"**
   每个样本同时看请求级 `cached_tokens` 和 Prometheus 的 `local/external prefix cache hits` 增量，两个来源逐条相等；
   CPU 命中时 `load_bytes / 命中 token` 恒为 12288 B，正好等于这个模型每 token 的 KV 字节数（24 层 × K/V × 2 head × 64 × bf16）。

2. **offload 不是免费的，盈亏点要实测**
   - CPU 命中比 GPU 命中慢 +6 → +58 ms（随前缀线性），与 load 时间对得上。
   - P=1024 时比重算还慢（+0.70 ms，CI [0.53, 2.09]）：小模型重算太便宜。
   - store 侧：connector 让冷请求在 P≥8192 时多 39 / 69 ms，与 store 耗时 32 / 65 ms 量级接近。我的假设是多 chunk prefill 时 store 卡在 chunk 之间，
     但**没跑 profiler，所以只说假设**。
   - 这台机器 CPU 层是 pageable 内存（WSL2 默认不 pin），带宽 3–4 GB/s，换到 pinned / 服务器上数字会变。

3. **停机语义：drain 保护在途请求，但不保护"进程会退出"这件事**
   - `shutdown-timeout=60`（剩余工作约 10 s）：两轮 6/6 在途请求正常完成，API 退出码 0。
   - `=0` 或 `=2`：SIGTERM 与 SIGKILL 几乎同时到 EngineCore（process manager 的 terminate → 立即 kill_process_tree），
     如果 abort 结果没来得及发回、API 也没收到 EngineDeadError，在途请求就永远在等。uvicorn `timeout_graceful_shutdown` 默认 None、vLLM 没设，
     所以 API 一直打印 "Waiting for connections to close"。t=0 两轮 6 次挂住 5 次，t=2 两轮 6 次挂住 4 次——n 太小，不说概率，只说两种结局都能复现。
   - 有一次例外，恰好印证机制：8 条请求都 abort 了，只剩一个探针连接，它 30 s 客户端超时一断，API 就在 0.3 s 后退出。
   - 无论哪个档位，监听口都在 SIGTERM 后约 0.1 s 关闭；drain 期间已建立连接的新请求收到的是 **HTTP 200 + `finish_reason=abort`**，客户端必须检查 finish_reason。

## B1（和 M21 对比着讲）

M21 杀 API 父进程，留下持显存的孤儿 EngineCore；M28 杀 EngineCore 子进程，API 在 0.5 s 内把在途请求全部 500 结束，**然后自己以退出码 0 退出**，
没有停在 503 的状态。推论（未实测）：按"非零退出才重启"配置的监督器不会拉起它。

## 方法学上值得说的

- 一个进程一个引擎；A/B 不同时跑 GPU；每个注入目标在发信号前重新核对身份（sid、ppid、进程名）。
- validator 有 15 个注入反例，必须全部检出才算可信。
- B3 第一轮 runner 把挂住的请求记成了 `missing`，汇总脚本直接崩了。我修了记录逻辑，按同一冻结矩阵重跑，第一轮原样保留、不合并——
  两轮在挂住比例上并不一样（t=0：3/3 vs 2/3），如果只挑一轮讲，结论会被说偏。
- 计划冻结后发现模型 `generation_config.json` 悄悄覆盖了采样参数（`repetition_penalty=1.1`），在正式数据之前升级到 v2，v1 冒烟数据另存、不进统计。

## 被追问时的诚实边界

- "为什么不用 pinned？" —— 冻结配置要和 WSL2 默认一致；pinned 对照属于新实验，没做。
- "39/69 ms 到底卡在哪？" —— 没跑 profiler，不知道，只有 store 耗时的量级吻合。
- "t=2 为什么有时能收到 EngineDeadError？" —— 没定位到源码级原因。
