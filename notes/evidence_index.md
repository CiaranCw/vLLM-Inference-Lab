# 证据索引（evidence index）

> 每个模块的**结论 → 对应证据文件**，方便随时核查。（持续维护：每完成一模块在此登记。）

| 模块 | 关键结论 | 证据文件 |
|---|---|---|
| M0 环境/基础设施 | 固定 commit ee0da84 / Qwen2.5-0.5B；editable 指向 vllm-src；统一 trace 工具；trace-off 零副作用 | `vllm_dev_environment.md`、`logs/day1/{vllm_collect_env,nvidia_smi_before}.txt`、`vllm-src/vllm/_lab_trace.py`、各 `logs/*/E1_regress*` |
| M1 主链路 | LLM.generate→…→Attention→Sampling 全链；Prefill/Decode shape 对照；backend=FLASH_ATTN/FA2 | `notes/day1/{baseline_analysis,runtime_trace}.md`、`outputs/day1/*`、`logs/day1/*` |
| M2 Continuous Batching | 同批 decode+prefill 混跑；max_num_seqs 排队；chunked prefill；E5 预算共享/槽位争用 | `notes/day2/{continuous_batching,e5_blocked_decode_vs_prefill}.md`、`outputs/day2/*` |
| M3 KV/Block | block_table 逻辑→物理；prefix 命中(hash/ref_cnt/共享/对齐)；释放→复用；抢占重算；K5 尾部优先淘汰 | `notes/day3/{kv_cache_block_management,k5_eviction_order}.md`、`outputs/day3/*` |
| M4 Attention Backend | SchedulerOutput→prepare_inputs→attention 层→KV 写(slot_mapping)→读(block_table) 闭环；GQA 14/2/64；kv_cache T(34801,2,16,2,64) | `notes/m4/m4_attention_backend.md`、`outputs/m4/m4_closed_loop.jsonl`、`logs/m4/m4_closed_loop.log` |
| M5 执行模式 | enforce_eager 一票否决 mode/cudagraph=NONE；默认 O2=VLLM_COMPILE+FULL_AND_PIECEWISE；捕获 51 PIECEWISE+35 FULL；prefill→PIECEWISE(pad 11→16,attn 图外) / decode→FULL(pad_attn=True)；sm_120 未降级；compiled 与 eager 输出第 5 token 起分叉(编译数值差异) | `notes/m5/execution_mode.md`、`outputs/m5/{eager,compiled}_config.json`、`outputs/m5/{eager,compiled}.jsonl`、`logs/m5/{eager,compiled}.log` |
| M6 引擎 Benchmark | 进程内闭环测量：decode 单步 latency 近平/吞吐近线性(compiled 全 batch 快2.5–2.85×,逐 batch 1–64 实测全走 FULL)；prefill ≤512 有 ~7ms 开销地板,compiled 仅 ≤128(PIECEWISE)提速、≥256 超捕获上限128 回退 NONE≈eager；prefix 命中省94%计算token 但配对差在噪声内(+0.03ms)、ON 短prompt 反略慢于 OFF(省计算≠省延迟)；KV≈12KiB/token(实测=理论),eager池73.2万/compiled72.2万token,最大并发357×是容量等价值(运行受max_num_seqs=64限) | `notes/m6/benchmark.md`、`outputs/m6/baseline.csv`、`outputs/m6/raw/*.jsonl`、`outputs/m6/config/{*_meta.json,diag_modes_*.jsonl}`、`outputs/m6/summary/summary.json`、`logs/m6/*.log` |
| M7 Profiler 定位瓶颈 | torch.profiler 拆 step：**decode 地板=CPU launch**(eager 257 launches/步、cudaLaunchKernel 1.59ms 为头号CPU、gpu_frac0.52；compiled FULL 图→14 launches、wall 6.66→2.47ms)；**长prefill=GPU计算受限**(gpu_frac1.75、cutlass tensor-core GEMM+flash attn 主导；compiled 砍 launch 161→62 但 wall 不变);batch↑ launch 数不变(257→279)、权重读取被摊销;nsys 在WSL无GPU kernel追踪(如实记录,改用torch profiler) | `notes/m7/profiling.md`、`outputs/m7/profile_{eager,compiled}.json`、`outputs/m7/traces/*`(未入库,可复现)、`logs/m7/*.log` |
