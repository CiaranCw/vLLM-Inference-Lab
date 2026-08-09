# M22 Execution Freeze Manifest

> **执行后状态（2026-08-09）**：正式矩阵 `h20_formal` 已在 4×H20 跑完并回传；结论见 [`results.md`](results.md)。  
> 下文保留**上机前冻结快照**（矩阵 / bundle hash / 文件指纹），供审计；其中列出的远端 Prompt / Runbook 等运营文件已在收口时删除，学习入口改为 [`README.md`](README.md)。

**Freeze ID:** `FROZEN_FOR_H20_2026-08-09_20-00_CST`

- Freeze time (local): `2026-08-08T12:54:44+08:00`
- Window: **2026-08-09 20:00 Asia/Shanghai** → ~2026-08-10 00:00 Asia/Shanghai (4h)
- CP_ENV_MODEL hard gate: **2026-08-09 21:00 Asia/Shanghai**
- Status at freeze: **FROZEN_FOR_H20_2026-08-09_20-00_CST** / READY_FOR_H20_EXECUTION
- Status after window: **TP E/H COMPLETE**（见 results.md；PP/DP/EP 仍 ⬜）

## Project

| Item | Value |
|---|---|
| lab HEAD | `8b20a122b6ce34e5a885cacfb0894b76dd51d3fe` |
| lab dirty files | 31 (uncommitted WIP OK; freeze records hashes of M22 files) |
| vllm-src HEAD | `ee0da84ab9e04ac7610e28580af62c365e898389` |
| vllm-src dirty count | 22 (baseline expected 22) |

## Frozen experiment

- Scope: TP + collective communication only
- D1 / D2 / D3 / P1: unchanged from frozen matrix
- Model priority: `Qwen/Qwen2.5-7B-Instruct`
- Fallback: `Qwen/Qwen2.5-1.5B-Instruct` (time/network only)
- TP fallback: TP=4 unstable → TP=1/2
- Hard stop: T+3:25

## Runtime bundle

- Path: `/home/ciaran/m22_offline/bundles/m22_runtime_bundle.tar.gz`
- Size: 83070 bytes (~81.1 KiB)
- SHA256: `ebc3c7d58b8ef63d747f4bfc59434fcee7dff840345f6e7ae57092e5226286b1`
- Clean-room: **PASS** (`/home/ciaran/m22_offline/manifests/bundle_cleanroom_20260808_125404.log`)

## Models

### 7B primary
- Path: `/home/ciaran/m22_offline/models/Qwen2.5-7B-Instruct`
- Size: 15242807272 bytes / 15 files
- Config: heads=28, kv=4, hidden=3584
- Manifest: `/home/ciaran/m22_offline/manifests/qwen2.5-7b.sha256`
- Readiness: **READY**

### 1.5B fallback
- Path: `/home/ciaran/m22_offline/models/Qwen2.5-1.5B-Instruct`
- Size: 3098955668 bytes / 7 files
- Config: heads=12, kv=2 （TP=4 有 KV replication）
- Manifest: `/home/ciaran/m22_offline/manifests/qwen2.5-1.5b.sha256`
- Readiness: **READY**

## Wheelhouse

- Candidate only: `vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl`
- SHA256: `2d2831aeba311292250df0132dbc4d8e9f42c654536eaec48e6fe58acb1822cf`
- **不保证**与未知 H20 Python/CUDA/PyTorch/glibc 兼容
- 源码编译：**禁止**作为正常 fallback

## Code artifacts (sha256)

| path | size | sha256 |
|---|---:|---|
| `scripts/m22_tensor_parallel.py` | 44465 | `079c8706cef771c4ab3dea2692d58050e3fca0f8a8110a75ede0d377834aa929` |
| `scripts/m22_run_all.py` | 8128 | `1d11074e5bd1e7acb2a7a8159e1c180137351c3d18e8e76ef02b3b4827593e6e` |
| `scripts/m22_preflight.sh` | 6315 | `4627b3bdad68deefc005c2618d6a4973f6fa7a1dc2eb18dc96671cb109966f3e` |
| `scripts/m22_nccl_smoke.py` | 5922 | `8253402d3981426b444fcc41d6851ebfc0ab9f3bc250f5263a9e570dc19e04e0` |
| `scripts/m22_trace_analyze.py` | 9046 | `e33d4a158a1c67c4abdae4874cda100524ca204c1833a061e6e8990991071668` |
| `scripts/m22_figures.py` | 6463 | `25cea1e605652f1c4e150a7dc88a7f659cabea78e96a7478386aad44beb9e4e1` |
| `scripts/check_invariants.py` | 63912 | `25f95f12f5792ae9d116e2ae779f1e9d6f8e62c1f563458f9ff919312a8fedb9` |
| `scripts/reproduce.py` | 14585 | `3f73f3a3e06344fe97cb9721b2faee7b451c10bab7b4b06f86bfdcb2ff7c4507` |
| `config/repro_manifest.yaml` | 25386 | `fc60892e30da05f84f58afb149f9dbd59e0f87188672b797c7d1a27fd057491f` |
| `notes/m22/H20_RUNBOOK.md` | 23907 | `c487edf9533da61991816c24764c3e01f4590bc9d1a69f2ef27f03cca6ca6e7a` |
| `notes/m22/tensor_parallel.md` | 21112 | `2086f738b411a278ad85c9ffb4bb07f2b21ead1a583f9678799ce5c332ed33a6` |
| `MULTI_GPU_EXPERIMENT_PLAN.md` | 24670 | `b26d80b43061e550cf47256b7724158dcb0f44880b863b7c9592f63333d93842` |

## Validation

- Local TP=1 preflight: prior READY
- Invariants `--module m22`: recognizes module; PENDING without H20 data
- Bundle clean-room: PASS
- Trace analyzer synthetic: PASS
- Model local_files_only smoke (7B + 1.5B): PASS
- Full per-file SHA256 for both models: YES

## Discipline

Freeze 之后**不再修改** M22 正式执行代码。若发现 blocker 必须改：
1. 记录原因；2. 更新 SHA256；3. 重跑受影响测试；4. 重新生成本 manifest。

# **M22 FROZEN — 窗口已执行完毕（2026-08-09）；见 results.md**

---

## Handoff amendment（2026-08-09 · 不改执行矩阵）

上机前曾为内部客户端准备运营文档；**收口后已删除**（Prompt / 拖拽 / GitHub 私有访问说明等）。  
学习入口：[`README.md`](README.md) · [`results.md`](results.md) · [`execution_record.md`](execution_record.md)。

仍保留：`notes/m22/frozen_runtime_snapshot/`（Freeze 版 invariants / repro）。  
本文件上表 `code_artifacts` 指纹仍供审计；其中 `H20_RUNBOOK.md` 已不在工作树。

上表 Offline 路径中的 `/home/ciaran/m22_offline/...` 仅为**冻结时操作者本机快照**。

