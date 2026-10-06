# M27 · 独立 post-hoc 最终审计（2026-10-06）

**结论：`M27_FINAL_AUDIT_PASS_WITH_DOCUMENTATION_CORRECTION`**（CRITICAL 0 / MAJOR 0 / MINOR 7 / INFO 10）。
审计对象是 H20 交付的 final archive `m27_h20_final_20261006T003354Z.tar.zst`（2,465,709,981 B，SHA256 `fb1664ca9506efb75bb784ab19a8b65186c6eabedf69127805e8e72b29633ad6`）及其 `FINAL_REPORT.md` / `M27_RESULT_MANIFEST.tsv`。审计在本地 CPU 上完成，没有运行任何 GPU 实验或 `vllm serve`。审计报告中的「8×H20」指运行环境（单节点 8 卡、GPU0–7 拓扑已核实）和整机计费口径；被审计的正式实验最多用到 4 GPU（E4 每 arm 4 GPU），没有 8-GPU 实验。

本目录文件是审计产物的**原样副本**，未做改动；审计报告里出现的 `/home/ciaran/...` 是审计当时的本地位置，不是证据的永久地址（证据身份以 archive 名称 + 大小 + SHA256 为准）。

| 文件 | 内容 | SHA256 |
|---|---|---|
| [`M27_POST_HOC_FINAL_AUDIT.md`](M27_POST_HOC_FINAL_AUDIT.md) | 审计报告：范围、方法、发现、结论 | `d388e5e5471d1d20daf2ec996d32238afeb11abefee05d92100a7113af20046b` |
| [`requirement_matrix.tsv`](requirement_matrix.tsv) | 冻结要求逐条核对 | `5aaff0ccb922a10851309bb701cd8861c3a7018858250bc333a916011453b5c1` |
| [`headline_validation.tsv`](headline_validation.tsv) | 77 个 headline 数字从 raw/derived 复算 | `e2c32771844d8953d0fcd01604d9f34b62c7772c01648773d94e8bc9451d409b` |
| [`rebuild_comparison.tsv`](rebuild_comparison.tsv) | 9 个派生文件与 3 张图的重建对比（逐位 / 逐像素一致） | `d4df7d734d337a7f0011776513dffbbf9d6569b4bc1668a356238d1248d52dfa` |
| [`findings.tsv`](findings.tsv) | 全部发现（F-M01…F-M07 为 MINOR） | `56970abcf4c5a713ccffc68f2b77e951ffa529032c64417195ada19a20217c9c` |

各条 MINOR 的处理与闭环状态见 [`../FINAL_CLOSURE_ADDENDUM.md`](../FINAL_CLOSURE_ADDENDUM.md)。
