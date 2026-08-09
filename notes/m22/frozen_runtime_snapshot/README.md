# Frozen runtime snapshot（与 M22 Freeze 一致）

工作区根目录的 `scripts/check_invariants.py` / `config/repro_manifest.yaml` 可能在 Freeze 之后因 M19 等文档工作产生 **SHA 漂移**。

本目录保存 **runtime bundle / Freeze Manifest 记录的版本**：

| 文件 | Freeze sha256 |
|---|---|
| `check_invariants.py` | `25f95f12f5792ae9d116e2ae779f1e9d6f8e62c1f563458f9ff919312a8fedb9` |
| `repro_manifest.yaml` | `fc60892e30da05f84f58afb149f9dbd59e0f87188672b797c7d1a27fd057491f` |
| `m22_runtime_bundle.tar.gz.sha256` | bundle = `ebc3c7d58b8ef63d747f4bfc59434fcee7dff840345f6e7ae57092e5226286b1` |

H20 正式执行前（若未解压 runtime bundle）：

```bash
cp notes/m22/frozen_runtime_snapshot/check_invariants.py scripts/check_invariants.py
cp notes/m22/frozen_runtime_snapshot/repro_manifest.yaml config/repro_manifest.yaml
```

**不要**用本目录覆盖的意图去「更新」Freeze；若需改正式执行代码，必须走 Freeze 纪律重冻。
