# M22 · H20 就绪确认清单（诚实核对）

> 核对日：以执行前最新一次为准。任一项 ❌ 则**不得**宣称「已完全就绪」。

## A. 冻结与范围

| # | 项 | 状态 |
|---|---|---|
| A1 | Freeze ID `FROZEN_FOR_H20_2026-08-09_20-00_CST` 存在 | |
| A2 | 窗口 20:00–~00:00 Asia/Shanghai；不提前登录 | |
| A3 | 范围仅 TP+通信；PP/DP/EP 明确不做 | |
| A4 | `m22_*.py` / Runbook / Plan 的 sha256 与 Freeze 一致 | 见下方自动核 |

## B. 代码可达性（GitHub 小文件）

| # | 项 | 状态 |
|---|---|---|
| B1 | `notes/m22/REMOTE_AGENT_PROMPT.md` 已写入仓库 | |
| B2 | `notes/m22/README.md` 导航可用 | |
| B3 | `scripts/m22_*` 已在 Git 可推集合中 | |
| B4 | `frozen_runtime_snapshot/` 含 Freeze 版 invariants/repro | |
| B5 | 已 `git push` 到操作者可拉的 remote（**推送后勾选**） | |

## C. 大文件（仅拖拽，不进 Git）

| # | 项 | 状态 |
|---|---|---|
| C1 | Offline Pack 7B 模型 READY + sha256 manifest | |
| C2 | Offline Pack 1.5B fallback READY | |
| C3 | `m22_runtime_bundle.tar.gz` sha256=`ebc3c7d5…286b1` | |
| C4 | wheel 仅为 candidate（不保证） | |
| C5 | 操作者知道如何拖到 `$REPO_ROOT/models` 等 | |

## D. 远端 Agent 可执行性

| # | 项 | 状态 |
|---|---|---|
| D1 | Prompt 含 Stage 顺序、门禁、降级、成功线 | |
| D2 | 文档无「必须用本机绝对路径才能跑」的硬依赖 | |
| D3 | 产物回传规则（pull+checksum；GitHub 只收小文件）已写明 | |
| D4 | 已知漂移：工作区 `check_invariants.py` / `repro_manifest.yaml` ≠ Freeze → 已用 snapshot/bundle 方案 | |

## E. 仍无法在本清单闭环的项（必须如实保留）

| # | 项 | 说明 |
|---|---|---|
| E1 | H20 SSH / 内网客户端连通 | **需窗口开始后由操作者验证** |
| E2 | 远端 GitHub 是否可达 | **需现场测**；不可达则走拖拽 bundle+模型 |
| E3 | 远端 Python/CUDA 与 vLLM 0.24.0 wheel 兼容 | **需 Stage 5 验证**；失败则换官方容器或中止，不编译 |
| E4 | 4×H20 NVLink 拓扑是否支持 custom AR | **需 `nvidia-smi topo -m`**；PCIe 回落 NCCL 是合法观察 |
| E5 | 正式 E/H 数据 | **尚未产生**；在数据回传校验前不得标完成 |

## 自动核（操作者本机、仓库根执行）

```bash
python3 - <<'PY'
import json, hashlib
from pathlib import Path
man=json.loads(Path('notes/m22/M22_EXECUTION_FREEZE_MANIFEST.json').read_text())
bad=[]
for art in man['code_artifacts']:
    p=Path(art['path'])
    if art['path'] in ('scripts/check_invariants.py','config/repro_manifest.yaml'):
        # 允许工作区漂移；核对 snapshot
        continue
    if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest()!=art['sha256']:
        bad.append(art['path'])
snap=Path('notes/m22/frozen_runtime_snapshot')
for name, expect in [
    ('check_invariants.py','25f95f12f5792ae9d116e2ae779f1e9d6f8e62c1f563458f9ff919312a8fedb9'),
    ('repro_manifest.yaml','fc60892e30da05f84f58afb149f9dbd59e0f87188672b797c7d1a27fd057491f'),
]:
    h=hashlib.sha256((snap/name).read_bytes()).hexdigest()
    if h!=expect: bad.append(f'snapshot:{name}')
print('FAIL' if bad else 'PASS', bad)
PY
```
