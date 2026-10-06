# M27 · H20 提交来源与规范化记录

H20 上的两个原始提交**真实存在**，完整保存在 reviewed final archive 内的仓库里；本仓库主线上的是它们的**规范化重放**。New commits preserve the reviewed content/evidence, while normalizing repository author/message policy and excluding oversized raw payloads from ordinary Git history.

## 1. 原始 H20 提交（永久保留的 provenance）

| | runtime 提交 | final 提交 |
|---|---|---|
| SHA | `884503dccf4d8108326e34cabac0ff36c1fb8823` | `c4db4561f20b8c777ee2e63545b89e7220a4b089` |
| parent | `423575cd73559825ea0595b94f608742714adb86`（pre-H20 anchor） | `884503dc…` |
| tree | `bfe8e50e4b8c03ef91d6b35fa61347c410c4ad5e` | `b660a327fbea326ccae4023db125b1ce9837309c` |
| author / committer | `root <root@TENCENT64.site>`，2026-10-05T20:17:45+08:00 | `root <root@TENCENT64.site>`，2026-10-06T08:32:55+08:00 |
| message | `fix: recover M27 H20 runtime corrections and partial evidence` | `feat: complete same-instance M27 H20 E0-E4 evidence and audit` |
| 内容 | 运行时修复 + 前一实例 Gate0/E0–E2 中途证据；282 files | 当前实例 `ins-a8qac2ve` fresh Gate0→E4 正式证据与审计；301 files |
| 位置 | archive `m27_h20_final_20261006T003354Z.tar.zst`（2,465,709,981 B，SHA256 `fb1664ca9506efb75bb784ab19a8b65186c6eabedf69127805e8e72b29633ad6`）内的 `vLLM-Inference-Lab/.git` | 同左 |

两者没有在原 archive 中被改写，也没有被 fetch 进本仓库（所以本仓库的对象库里没有那 11 个超大 blob）。不合规的只有两点：提交者是 `root@TENCENT64.site`（违反 [`AGENTS.md`](../../AGENTS.md) §2 只允许 CiaranCw），消息是英文；另外 11 个单文件 >100MB 的 raw 让该历史无法推送到 GitHub。

## 2. 拓扑与规范化提交

```text
423575cd ── 884503dc ── c4db4561                 （H20 原历史，只在 archive 中）
    │
    └── f41f04b5 ── N1 ── N2 ── N3 ── N4          （本仓库 main；全部 CiaranCw）
```

`f41f04b5` 是 H20 期间本地主线上的文档清理提交（从当前版本移除两份个人备考文档），与 H20 分支是兄弟关系。规范化不是对 H20 原提交做 cherry-pick 或 rebase，而是**在 `f41f04b5` 之上按内容逐个重放**：从 archive 仓库导出 `423575cd..884503dc`、`884503dc..c4db4561` 的二进制补丁（排除超大 raw 路径），用 `git apply --index` 应用；三个两边都改过的文件（`README.md`、`ROADMAP.md`、`notes/evidence_index.md`）用 `--3way` 合并，无冲突。

| 规范化提交 | 对应 | author = committer | 说明 |
|---|---|---|---|
| N1 `aec5f5320e6184771f1ac4a60b223c1a379bb485` | `884503dc` | CiaranCw <1399538830@qq.com> | 重放 runtime 修复与前机中途证据；排除 1 个超大 raw |
| N2 `cc1f06bd92855facf536d15042f965e2dc2667d1` | `c4db4561` | CiaranCw <1399538830@qq.com> | 重放同实例 Gate0→E4 正式证据；再排除 10 个超大 raw |
| N3 `c9e317f5b71d3fdb84a133219315f59e5700842a` | （新增） | CiaranCw <1399538830@qq.com> | 外部 raw manifest + 不变量；`scripts/m27_final_evidence_audit.py` 离线生成器 |
| N4（包含本文件的提交，`git log -1 --format=%H -- notes/m27/H20_COMMIT_PROVENANCE.md`） | （新增） | CiaranCw <1399538830@qq.com> | post-hoc 审计材料入库、文档修正、本 provenance 文档 |

## 3. 内容关系（逐 blob 核对）

用 `git ls-tree -r` 比较规范化提交与对应原提交的全部路径和 blob ID：

- **N1 vs `884503dc`**：只有 11 条路径不同。其中 10 条是 `f41f04b5` 本来就改过的文件（`.cursor/rules/project-rules.mdc`、`.gitignore`、`AGENTS.md`、`README.md`、`ROADMAP.md`、`notes/evidence_index.md`、`notes/m26/{README,portfolio}.md`，以及从当前版本移除的 `notes/m16/interview_and_presentation.md`、`notes/m26/interview_second_stage.md`），另 1 条是被排除的超大 raw `outputs/m27/raw/backup_20261004T225246/raw/h20_formal_e3_pd_1p1d.jsonl`。其余 H20 文件 blob 全部相同。
- **N2 vs `c4db4561`**：只有 21 条路径不同，即同样那 10 条 `f41f04b5` 路径，加上全部 11 个超大 raw。所有脚本、派生 CSV/JSON、configs、logs、图、其余 raw 的 blob 全部相同。对 `README.md` / `ROADMAP.md` / `notes/evidence_index.md` 三个合并文件，「N2 相对 `c4db4561`」的差异恰好等于「`f41f04b5` 相对 `423575cd`」的差异，即两边改动都完整保留。
- 因此 N2 的科学内容与审计过的 `c4db4561` 完全一致：post-hoc 审计的 9 个派生文件逐位重建、3 张图逐像素一致、77/77 headline、6 个 CI 的结论直接适用于 N2。
- 前一实例的 10 个 E0–E2 server log（`logs/m27/h20_formal/` 下、被 `c4db4561` 用当前实例版本覆盖的那些）在 N1 中与 `884503dc` 逐 blob 相同，可用 `git show aec5f532:logs/m27/h20_formal/<name>.log` 取回。

## 4. 超大 raw 的排除策略

- 11 个单文件 >100MB 的 formal raw（合计 2,712,916,405 B；最大 382,435,923 B）**不进入普通 Git 历史**，本轮不引入 Git LFS。它们在 `.gitignore` 中逐个路径列出（没有用 `outputs/**`、`*.jsonl` 这类宽规则）。其余 raw（含 E0–E2、校准、lifecycle、E3/E4 `_proxy.jsonl` 等）仍在 Git 中。
- 逐文件身份见 [`H20_RAW_ARCHIVE_MANIFEST.tsv`](H20_RAW_ARCHIVE_MANIFEST.tsv)：`size_bytes`、`sha256`（从 archive 直接解出的文件计算）、`archive_sha256`、`repository_policy = external_evidence_not_git_tracked`，以及 H20 原 Git blob ID 与首次出现的原提交。size 与 blob 均和 `884503dc` / `c4db4561` 的 tree 对上。
- 拿到 archive 后的恢复方式：解包，把上述 11 个路径拷回仓库同一相对路径，然后运行 `python scripts/check_invariants.py --module m27`；`m27_external_raw_identity` 会逐文件核对 size + SHA256，`m27_final_evidence_audit_regenerable` 会离线重算 73 项。raw 不在本地时这两项报 PENDING，不会被静默记为通过。
- 本仓库任何可达对象都不超过 100MB（以 `git rev-list --objects --all | git cat-file --batch-check` 核对）。

## 5. 审计结论

独立 post-hoc 审计：**`M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`**（CRITICAL 0 / MAJOR 0 / MINOR 7 / INFO 10）。仓库集成相关的是 F-M06（root 署名 + 英文消息 + 超大 raw），由本文件所述的规范化与排除处理；其余各条的闭环见 [`FINAL_CLOSURE_ADDENDUM.md`](FINAL_CLOSURE_ADDENDUM.md)，审计原件见 [`audit/`](audit/)。

规范化不改变任何实验结论，M27 的科学状态仍是 `M27_H20_FORMAL_COMPLETE`（M27_E = COMPLETE、M27_H = COMPLETE）。
