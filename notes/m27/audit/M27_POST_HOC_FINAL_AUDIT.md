# M27 Post-hoc Full Independent Final Audit

- 审计对象：`/home/ciaran/m27_results`（只读，未做任何写入）
- 审计目录：`/home/ciaran/m27_audit_work/20261006T084013Z/`
- 被审声明：`M27_H20_FORMAL_COMPLETE`（final commit `c4db4561`，archive `m27_h20_final_20261006T003354Z.tar.zst`）
- 本轮只做离线工作：checksum / tar / git / parser / summarize / bootstrap / invariants / figures。没有运行任何 vLLM 服务或 GPU 实验。
- **结论：`M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`**

## 1. 结论分层

| 维度 | 判定 | 依据 |
|---|---|---|
| 科学有效性 | PASS | 冻结设计未变；同实例 fresh；同成本预算正确；没有混用前机数据；负结果保留 |
| 执行有效性 | PASS | 18 个 effective config 与冻结设计一致；E0–E4 矩阵完整；0 失败；没有隐藏 fallback |
| 制品完整性 | PASS | archive SHA 与 manifest 一致；zstd -t 通过；路径安全；secret 扫描 0 命中。archive.sha256 未随包收到，但可逐位重建 |
| 可复现性 | PASS | 9 个 summarize 派生文件逐位相同；log_evidence 逐位相同；3 张图逐像素相同；77/77 headline；6 个 CI 逐位复现 |
| 文档精度 | 需修正 | 计费附注；live doc 中残留过时状态；RQ3/RQ4 措辞；校准判据说明；invariant 首次失败日志未保留 |
| 仓库集成 | 需所有者决策 | 提交者为 root@TENCENT64.site；11 个超过 100 MB 的 raw 文件被 git 跟踪，无法推送到 GitHub |

## 2. 收到的制品
- `m27_h20_final_20261006T003354Z.tar.zst`：2,465,709,981 B，SHA256 `fb1664ca…33ad6`，与 manifest 一致。
- `M27_RESULT_MANIFEST.tsv`：522 B，CRLF。
- `FINAL_REPORT.md`：收到 11,294 B（CRLF，89 个 CR）。转成 LF 后为 11,206 B，SHA256 `a6433830…`，与 manifest 完全一致。差异来自传输过程中的换行规范化。
- `archive.sha256`：**未收到**。用 `printf '%s  %s\n'` 重建后为 105 B，SHA256 `0a820e3f…`，与 manifest 一致，`sha256sum -c` OK。
- tar 共 4811 条（4309 个文件、502 个目录、0 个链接），没有绝对路径，也没有 `..`。

## 3. Git / 源码链
- `423575cd → 884503dc → c4db4561`，两步 `merge-base --is-ancestor` 都是 YES。工作区干净，index 无差异，无 tag。
- 所有代码改动都在 `884503d`（shutdown timeout、proxy `force_close`、按 arrival horizon 计算校准速率、warmup 有界、新增 4 条 invariant），`c4db456` 没有改任何脚本。
- `FROZEN` 在 anchor、final 和 PLAN JSON 三处一致；`REQUIRED_ENV`、PLAN、RUNBOOK 和 repro_manifest 的 diff 都是 0。
- 偏差全部记录在 `execution_record.md` §8–§10。
- `c4db456` 覆盖了 70 个文件。其中 46 个前机 raw/config 原样保存在 `backup_*`；10 个前机 E0–E2 server log 只保存在 git `884503d` 中。

## 4. 冻结设计与执行
- 18 个 effective config：模型为 Qwen2.5-7B bf16，TP=PP=DP=1，eager，HND，FLASH_ATTN，APC 关，chunked 开，mbt 与 arm 一致。requested、effective、server_info 和日志四个来源一致。
- Gate 0：fresh preflight 在 TASK_START 之后；8×H20 97871 MiB，两两 NV18；driver 535.247.01，torch 2.11.0+cu130，vLLM 0.24.0，NIXL 1.2.0。
- 隔离：`summarize raws()` 用非递归 glob，排除了 `backup_*`；`e3_` 不会匹配 `e3cal_`。所有 active raw 的首个时间戳都晚于 TASK_START（1791202460）。

## 5. 原始数据到派生结果的重算
- **E0**：PD 5/5；D external = P local = 10240；D local = 0；NIXL 5 次、0 失败；fault probe 返回 HTTP 502 `fail_closed`，D prompt Δ = 0。
- **E1**：五档 bytes 都恰好等于 57344×L；xfer 中位 6.914→208.242 ms；GB/s 4.25→4.51，但 2048 档为 4.86，并不单调。
- **E2**：总计 320 条、measure 240 条，全部 overlap_valid。注入后 D token 最少 250 条（要求 ≥16）。
  - 8K：PD 105.05 ms，colo1_mbt8192 886.03 ms，差值 +780.98 ms，CI [780.51, 781.36]。
  - p=0：差值 −1.59 ms，CI [−1.95, −0.93]。
- **E3**：180/180 run，78,685 条请求，0 失败，rates SHA `9d5e…`。
- **E4**：108/108 run，130,428 条请求，0 失败，rates SHA `5cd7…`。
- **schedule**：独立重建后，SHA、速率和逐请求字段全部一致。跨 arm 的 160,769 对 prompt 无差异。
- **PD 计数器**：D local = 0，D ext = Σprompt，NIXL 失败 0；所有 arm 均为 `force_close=True`，没有非 200 响应，也没有 kv_reject。
- **headline**：77/77 PASS（见 `headline_validation.tsv`）。2304 个 per-run 值与 online CSV 0 差异；6 个 cluster bootstrap CI 逐位复现。
- **离线重建**：见 `rebuild_comparison.tsv`。

## 6. Invariants
- 审计时在副本中用冻结 env 重跑：M27 35/0/0，全仓 187/0/0。
- 对照组不设 env：185 PASS / 2 FAIL，失败项正是 live env 检查，说明这两项检查确实在看运行时环境。
- 当前实例上首次 2 FAIL 的那次运行日志没有保留，只在 ledger 中有文字记录。最终日志只有汇总行，没有命令、env 和时间戳。

## 7. final_evidence_audit.json
- 73 个 key 全部为 `true`，另有一个 `details` 汇总。
- `scripts/` 里没有生成器，每一项也没有附带证据，所以它本身**不能单独作为证据**。
- 但这 73 项覆盖的每一类内容（fresh、done、config、overlap、online、共享 schedule、校准新鲜度、不混用前机数据、图）都已被本审计独立复核通过。

## 8. 时间与计费
- 起点 TASK_START 为 `2026-10-05T12:14:20Z`。
- 最后一条 ledger 记录：00:32:49Z，12.308 h。
- FINAL_REPORT 快照：12.3619 h，98.89 GPU·h。
- Agent 最终回复约为 12.442 h，即 99.54 GPU·h。所有者观察到的 746 min（12.433 h）与之一致。
- 168 GPU·h ÷ 8 = 21 h，即额度应在约 `2026-10-06T09:14Z` 耗尽。实验结束后机器仍被占用，空转约 8.56 h，合计约 68.5 GPU·h。
- 建议在 results.md、execution_record.md 和 FINAL_REPORT 中加入以下附注：

> Formal M27 execution and closure completed in approximately 12.44 h, corresponding to approximately 99.5 GPU·h under 8-GPU whole-instance accounting. The H20 instance subsequently remained allocated after the experiment had completed, and the refreshed 168 GPU·h quota was eventually exhausted. The post-completion allocation (~8.56 h, ~68.5 GPU·h) is not part of the formal experiment workload, but it is part of the provider-side total resource cost.

  中文版：正式 M27 执行与收尾约 12.44 h，按 8 卡整机计约 99.5 GPU·h；实验结束后实例仍保持分配，refreshed 168 GPU·h 额度最终耗尽。结束后的约 8.56 h / 68.5 GPU·h 不属于正式实验负载，但属于云端实际资源成本。

## 9. 解读审查（RQ1–RQ4）
- **RQ1（E2）**：只能证明机制，因为对比的是 1 GPU 和 2 GPU。文档各处都已注明这一边界。p=0 时 PD 反而略差（−1.59 ms），这个结果被保留了。
- **RQ2（E1）**：bytes 随 context 线性增长，且恰好等于 57344×L。GB/s 在 2048 档达到峰值，不是单调变化。
- **RQ3（E3）**：PD 的 ITL P99 更低，但 TTFT 和 achieved rate 更差，"不存在 PD 全面胜出"的表述正确。
  - 不过 headline 的对比对象是**未调优**的 colo2_mbt8192（balanced ITL 424.6 ms）。调优后的 colo2_mbt2048 为 220.3 ms，PD 为 102.5 ms：差距缩小约一半，方向不变。
- **RQ4（E4）**：在 90% 负载下，colo4 的 TTFT P99 在全部 3 种 shape、所有 seed 上都优于每一种 PD ratio，CI 均排除 0。
  - 原文"decode-heavy 的 1P:3D 减小 ITL tail"与表格不一致：d_heavy 下 3P1D 为 47.5 ms，低于 1P3D 的 53.0 ms。
- **E5 / P1**：E5 是可选项，跳过有数字依据；P1 没有执行，文档也没有声称证明了 overlap。

## 10. Findings
见 `findings.tsv`：CRITICAL 0，MAJOR 0，MINOR 7（F-M01…F-M07），INFO 10。

## 11. 需求矩阵
见 `requirement_matrix.tsv`：PASS 41，FAIL 8（全部对应 MINOR finding），UNVERIFIED 2。

## 12. 审计产物
`received_files.{tsv,sha256}`、`archive_file_list.txt`、`audit_commands.log`、`config_audit.txt`、`recompute_e0_e2.json`、`online_audit.json`、`headline_validation.tsv`、`headline_online.tsv`、`online_median_of_seeds.tsv`、`rebuild_comparison.tsv`、`requirement_matrix.tsv`、`findings.tsv`、`invariants_*_auditor.log`，以及审计脚本 `audit_*.py`、`write_tables.py`。
