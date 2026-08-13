# M18 · 模块入口　✅ **T+E+O COMPLETE**（2026-08-13）

| 文件 | 内容 |
|---|---|
| [`quantization_w4a16.md`](quantization_w4a16.md) | **设计冻结**：与 M10 的分工、Gate 0/0-b/0-c 结果、配置矩阵、写死的隐藏变量、不变量、执行前假设、不做的事 |
| [`results.md`](results.md) | **E 结论**：显存/速度/质量三轴，含「28 个对照点 0 个变快」与 H5 被推翻的记录 |

脚本：`scripts/m18_{probe,load_smoke,quant,run_all,summarize,figures}.py`
数据：`outputs/m18/`（`memory.csv` / `perf.csv` / `quality.csv` / `raw/*.jsonl` / `configs/` / `gate0*`）
日志：`logs/m18/` · 图：`figures/fig13_m18_w4a16.png`
不变量：`python scripts/check_invariants.py --module m18`（11 PASS / 0 FAIL）
