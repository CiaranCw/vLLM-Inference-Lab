# M23 · 模块入口　✅ **L1 FROZEN**（2026-08-14）

| 文件 | 内容 | 状态 |
|---|---|---|
| [`cross_engine_plan.md`](cross_engine_plan.md) | **设计冻结 v5**：2×2 execution-mode、比较纪律、consistency 口径 | 🔒 冻结 |
| [`backend_migration.md`](backend_migration.md) | **T 产出**：TensorRT backend 移除；TRT-P vs TRT-L | ✅ |
| [`gate_results.md`](gate_results.md) | Gate 0/1/2 + A1 决策 + V-O / V-ctr 前置门 | ✅ |
| [`results.md`](results.md) | **E 结论**：2×2 性能 + 跨引擎 generation consistency | ✅ **L1 FROZEN** |

脚本：`scripts/m23_*.py`　数据：`outputs/m23/`　图：`figures/fig14_m23_cross_engine.png`
镜像：`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24`（TRT-P）· `tritonserver:26.07-vllm-python-py3`（V-ctr）

---

## 授权与冻结状态

```text
L1（Gate 0–2 + 离线 2×2 + generation consistency）：✅ 已完成并冻结
L2 / L3 / L4                                        ：⬜ 明确不做（L3 可行但未批准）
kernel profiling                                    ：⬜ 本轮暂不追加

container-only：是      host TRT-LLM venv：禁止
自动换 tag：禁止        自动租机：禁止
```

**比较纪律（v5）**：正式跨引擎只允许 `T-E vs V-E`、`T-O vs V-O`；
允许同引擎 `V-O vs V-E` / `T-O vs T-E` ablation；**禁止交叉 lane 做 engine speedup**。

---

## 已完成

- [x] 设计冻结 v5（含比较纪律收窄 + consistency 口径）
- [x] `backend_migration.md`（T）
- [x] Gate 0/1/2 PASS
- [x] V-ctr correctness preflight PASS（各 lane 6/6 确定性；V-O vs V-E 1/6）
- [x] L1 2×2 性能对照（两条 lane 下 vLLM 均更快，CI 全排除 0）
- [x] 跨引擎 generation consistency（`T-E vs V-E` 2/6、`T-O vs V-O` 1/6；mismatch = numerical divergence）
- [ ] L2 / L3 / L4 —— **明确不做**

复现命令见 [`results.md`](results.md) 末尾；不变量：`python scripts/check_invariants.py --module m23`。
