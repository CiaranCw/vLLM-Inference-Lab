# M21 · 模块入口　✅ **T+E+O COMPLETE**（2026-08-12）

| 文件 | 内容 |
|---|---|
| [`monitoring_and_capacity.md`](monitoring_and_capacity.md) | 设计冻结：5 个工程问题、监控栈方案、E1–E6 设计、不变量、不做的事、Gate 0 实测结论 |
| [`results.md`](results.md) | **E 结论**（含容量阶梯、瓶颈签名、监控开销 A/B、故障演练、与 M12 差异的排查） |

栈的用法与踩坑：[`../../monitoring/README.md`](../../monitoring/README.md)

脚本：`scripts/m21_{host_exporter,gate1,capacity_sweep,validate,bottleneck,capacity_users,fault_drills,observability_cost,env_fingerprint,figures}.py`
配置：`monitoring/{docker-compose.yml,prometheus.yml,alerts.yml,grafana/**}`
数据：`outputs/m21/` · 日志：`logs/m21/` · 图：`figures/fig11_m21_capacity.png`、`figures/fig12_m21_bottleneck_signature.png`
不变量：`python scripts/check_invariants.py --module m21`（14 PASS / 0 FAIL）
