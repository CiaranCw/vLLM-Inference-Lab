# M27 · 最终收口附录（post-hoc corrections，2026-10-06）

本文件分两层：**第一层是 H20 交付的原始证据**，原样保留、不改写；**第二层是之后独立审计提出的修正**，逐条说明改了什么、改在哪里。修正只涉及文档措辞、证据存放和可复现性，**不改变任何 raw、派生数字或实验结论**。M27 的科学状态仍是 `M27_H20_FORMAL_COMPLETE`（M27_E = COMPLETE、M27_H = COMPLETE；E0–E4 COMPLETE，E5 可选跳过，P1 profiler 未做）。

## 第一层 · 原始证据（不改写）

| 对象 | 身份 |
|---|---|
| final archive | `m27_h20_final_20261006T003354Z.tar.zst`，2,465,709,981 B，SHA256 `fb1664ca9506efb75bb784ab19a8b65186c6eabedf69127805e8e72b29633ad6`（与 `M27_RESULT_MANIFEST.tsv` 一致） |
| archive 内 H20 提交 | runtime `884503dccf4d8108326e34cabac0ff36c1fb8823`、final `c4db4561f20b8c777ee2e63545b89e7220a4b089`（见 [`H20_COMMIT_PROVENANCE.md`](H20_COMMIT_PROVENANCE.md)） |
| `FINAL_REPORT.md`（H20 执行方的最终报告，不入本仓库） | manifest 记录 11,206 B、SHA256 `a64338307f4b191a3e6fcc8665bd71ec69e429c5237a576fc13f15ce0a1b3fab`。收到的副本为 11,294 B、SHA256 `9ab7b540bd89f363e7a525f9312d792472075adce84e165a6978e70e6afa3b37`：传输中 89 个 LF 变成 CRLF，且末尾换行丢失；去掉 CR 并补回末尾换行后正好是 manifest 的 11,206 B / `a6433830…`，内容未变 |
| 仓库内 H20 收口时的证据 | `outputs/m27/final_evidence_audit.json`（73/73 PASS，SHA256 `6e130886afcd61a3dfd03a7a8d5c6b596e585a334f7580f4bbcb5c7e47fd3949`）、`logs/m27/invariants_{m27,full}_final.log`（M27 35、全仓 187 PASS）、`outputs/m27/wall_clock_ledger.csv` 等，均未改动 |

FINAL_REPORT 是当时的陈述。下面的修正是事后补充，不代表「它当时就是这样写的」。

## 第二层 · Post-hoc 审计修正

独立审计结论 **`M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`**：CRITICAL 0 / MAJOR 0 / MINOR 7 / INFO 10；9 个派生文件重建逐位一致、3 张图逐像素一致、77/77 headline、6 个 CI 逐位复现。审计原件见 [`audit/`](audit/)。

| ID | 问题 | 处理 | 位置 | 状态 |
|---|---|---|---|---|
| F-M01 | 计费只写了正式执行段；「refreshed 168 GPU·h」一句有歧义 | 补计费附注：正式执行与收尾约 12.44 h ≈ 99.5 GPU·h；之后实例仍保持分配，168 GPU·h 额度耗尽，其中约 8.56 h ≈ 68.5 GPU·h 是完成后的空闲分配，不是正式负载。注明 12.44 h 的出处（H20 执行方最终回复 12.442 h、所有者观察 746 min；ledger 最后一行 12.308 h）。原 ledger 未改 | `results.md`「计费与资源」、`execution_record.md` §11、`ROADMAP.md` | CLOSED |
| F-M02 | README / evidence_index / execution_record 仍有「E/H NOT EXECUTED」等过时状态；execution_record §5 校准表为空 | 当前状态改为 COMPLETE；历史行明确标为「历史」，不删原文；§5 用 rates 文件原样回填，§4、§6 回填当前实例 | `README.md`、`HANDOFF.md`、`ROADMAP.md`、`notes/evidence_index.md`、`execution_record.md`、`notes/m27/README.md` | CLOSED |
| F-M03 | RQ3 只拿未调优的 colo2_mbt8192 做对照；RQ4「decode-heavy 1P:3D 减小 ITL tail」不对 | RQ3 拆成冻结基线（PD vs colo2_mbt8192）与调优后对照（PD vs 实测最好的 colo2 mbt，最好的 mbt 随指标而变）；balanced ITL 差距从约 4.1× 缩到约 2.1×，方向不变。RQ4 改为：decode-heavy 下 ITL tail 最低的是 3P:1D（47.5 ms < 1P:3D 53.0 ms），2P:2D 的 TTFT 在 PD 中最好。数字从 `h20_formal_e{3,4}_online.csv` 重新取 | `results.md` RQ3/RQ4、`README.md`、`ROADMAP.md`、`notes/evidence_index.md` | CLOSED |
| F-M04 | 校准修正后吞吐判据在 0 失败时不起作用，未说明 | 写明：e3cal 39 级、e4cal 36 级中 achieved 都等于 offered，每级是否通过都与 TTFT 判据一致，C_cal 实际由 TTFT 判据决定；这是判据局限，不改 raw 与结果 | `results.md`「指标口径与统计局限」、`execution_record.md` §5/§11 | CLOSED |
| F-M05 | 首次全仓 invariants 的 2 FAIL 原始输出未保留；final invariant 日志没有命令/环境/时间；前机 10 个 server log 在树中被覆盖 | 如实记录该日志未完整保留（原因：调用端缺两个冻结环境变量；补上后重跑，M27 35 / 全仓 187 PASS），**不补造日志**；审计方在无这两个变量时复现出相同的 2 个 env FAIL。集成后的复验日志开头写入命令、环境变量和 UTC 时间。前机 10 个 server log 原样保存在 `884503dc` 与规范化提交 N1 中，写明取回方法 | `execution_record.md` §11、`logs/m27/invariants_final_integration_{m27,full}.log`、[`H20_COMMIT_PROVENANCE.md`](H20_COMMIT_PROVENANCE.md) §3 | CLOSED |
| F-M06 | H20 提交为 root@TENCENT64.site + 英文消息；11 个 >100MB raw 无法推送 GitHub | 在 `f41f04b5` 之上按内容重放为 CiaranCw 提交（中文消息、无 AI trailer），原 SHA 永久记录；11 个超大 raw 改为外部证据身份（逐文件 size/SHA256 manifest + 逐路径 `.gitignore` + 身份不变量），不引入 LFS；仓库可达对象中没有 >100MB 的 | [`H20_COMMIT_PROVENANCE.md`](H20_COMMIT_PROVENANCE.md)、[`H20_RAW_ARCHIVE_MANIFEST.tsv`](H20_RAW_ARCHIVE_MANIFEST.tsv)、`.gitignore`、`scripts/check_invariants.py` | CLOSED |
| F-M07 | `final_evidence_audit.json` 没有生成器，只有 73 个布尔值 | 新增离线生成器 `scripts/m27_final_evidence_audit.py`，从 raw/derived/configs/logs 重算同名同序 73 项并附证据，73/73 PASS，结果在 `outputs/m27/final_evidence_audit_regenerated.json`；原 JSON 不动；不变量 `m27_final_evidence_audit_regenerable` 要求重算结果与落盘一致。反向测试 5/5 检出 | `scripts/m27_final_evidence_audit.py`、`scripts/check_invariants.py`、`config/repro_manifest.yaml` | CLOSED |

INFO 项中需要在文档里说明的也已补上：E1 等效带宽不是单调的，2048 档达到峰值 4.86 GB/s（F-I05，`results.md` RQ2）；E3/E4 CI 的局限，以及 effect 表点估计与 headline 是不同统计量（F-I06，`results.md`，只披露，不替换冻结 CI）；achieved rate 包含排空时间（F-I10）；93,690 是四舍五入值（F-I04）。F-I07（archive 中 raw 存了两份）和 F-I08（archive 的 `.git` 中有 31 个 dangling commit）只与 archive 本身有关，archive 不修改；这些对象没有进入本仓库。F-I03（IDE shim 路径出现在 PYTHONPATH）没有影响 vLLM 导入的证据，只作记录。

## 收口状态

- M27：**COMPLETE**；E0–E4 COMPLETE，为最终范围；E5 SKIPPED（冻结设计中可选的 8-GPU 配比，未执行、不再补跑）；P1 NOT RUN。
- 规模：运行环境为单节点 8×H20；正式实验覆盖 1/2/4 GPU（E0–E3 按 arm 1–2 GPU，P:D ratio 实验 E4 每 arm 4 GPU），最大正式并发 4 GPU，没有 8-GPU P-D / scaling 实验。
- Post-hoc 审计：`M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`；7 条 MINOR 修正 **CLOSED**。
- 适用边界不变：单节点 8×H20 环境、正式实验最多 4 GPU / Qwen2.5-7B bf16 / TP1 / eager / HND / vLLM 0.24.0 / NIXL 1.2.0 / 实例 `ins-a8qac2ve` / 这组 workload 与负载；不外推到 8 GPU、跨机、其他模型、其他 TP 或 compiled 模式。
