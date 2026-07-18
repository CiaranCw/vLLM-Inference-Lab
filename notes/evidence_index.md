# 证据索引（evidence index）

> 每个模块的**结论 → 对应证据文件**，方便随时核查。（持续维护：每完成一模块在此登记。）

| 模块 | 关键结论 | 证据文件 |
|---|---|---|
| M0 环境/基础设施 | 固定 commit ee0da84 / Qwen2.5-0.5B；editable 指向 vllm-src；统一 trace 工具；trace-off 零副作用 | `vllm_dev_environment.md`、`logs/day1/{vllm_collect_env,nvidia_smi_before}.txt`、`vllm-src/vllm/_lab_trace.py`、各 `logs/*/E1_regress*` |
| M1 主链路 | LLM.generate→…→Attention→Sampling 全链；Prefill/Decode shape 对照；backend=FLASH_ATTN/FA2 | `notes/day1/{baseline_analysis,runtime_trace}.md`、`outputs/day1/*`、`logs/day1/*` |
| M2 Continuous Batching | 同批 decode+prefill 混跑；max_num_seqs 排队；chunked prefill；E5 预算共享/槽位争用 | `notes/day2/{continuous_batching,e5_blocked_decode_vs_prefill}.md`、`outputs/day2/*` |
| M3 KV/Block | block_table 逻辑→物理；prefix 命中(hash/ref_cnt/共享/对齐)；释放→复用；抢占重算；K5 尾部优先淘汰 | `notes/day3/{kv_cache_block_management,k5_eviction_order}.md`、`outputs/day3/*` |
| M4 Attention Backend | SchedulerOutput→prepare_inputs→attention 层→KV 写(slot_mapping)→读(block_table) 闭环；GQA 14/2/64；kv_cache T(34801,2,16,2,64) | `notes/m4_attention_backend.md`、`outputs/m4/m4_closed_loop.jsonl`、`logs/m4/m4_closed_loop.log` |
