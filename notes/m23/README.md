# M23 · 模块入口　✅ **L1 COMPLETE**（2026-08-13）

| 文件 | 内容 | 状态 |
|---|---|---|
| [`cross_engine_plan.md`](cross_engine_plan.md) | **设计冻结 v3**：上游事实、TRT-P/TRT-L 区分、Gate 设计、Primary/Internal 指标分层、显存口径、授权状态 | 🔒 冻结 |
| [`backend_migration.md`](backend_migration.md) | **T 产出**：TensorRT backend 移除的时间线、迁移对照表、TRT-L vs TRT-P 机制对照、解读 | ✅ **已完成（零成本，不依赖 Gate）** |
| [`gate_results.md`](gate_results.md) | **Gate 0/1/2 实测结果** + 必须先解决的对齐缺口（A1–A7） | ✅ **已完成** |
| [`results.md`](results.md) | **E 结论**：2×2 execution-mode 四组正式对照 | ✅ **已完成** |

脚本：[`../../scripts/m23_gate.py`](../../scripts/m23_gate.py)（Gate 0/1 执行器，**默认 dry-run**）
数据：`outputs/m23/gate{0,1,2}.json` · 日志：`logs/m23/gate*.log`
镜像：`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24`（digest `sha256:16a103b8…`，已拉取，本地 54.8GB）

---

## 当前授权状态（冻结稿 §7）

```text
L1（Gate 0–2 + 离线跨引擎对照）：✅ 已批准，但**等所有者指令再开始**
L2 / L3 / L4                    ：⏸ 未批准（L3 可行但未批准，不是不可行）

唯一镜像：nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24   （尚未拉取）
container-only：是      host TRT-LLM venv：禁止
自动换 tag：禁止        自动租机：禁止
```

Gate 0/1 任意失败 → **保存完整证据 → 停止 → 回报所有者**，不自行进入下一补救路线。

---

## 执行清单（**待所有者下令后**逐条执行）

```bash
cd /home/ciaran/projects/vllm-inference-lab && source .venv/bin/activate

# 0) 前置：确认 GPU 空闲（本机 16GB，他人任务在跑时不得抢占）
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

# 1) dry-run 复核将要执行的命令（不拉镜像、不碰 GPU）
python scripts/m23_gate.py --gate 0

# 2) 拉取唯一授权镜像（约 20.5–20.7 GB；失败不得自行换 tag）
~/bin/docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc24

# 3) Gate 0：import + 版本 + GPU 可见性
python scripts/m23_gate.py --gate 0 --run

# 4) Gate 1：HF checkpoint 直接加载 + **真实生成**（红线：不生成文本不算 PASS）
python scripts/m23_gate.py --gate 1 --run

# 5) 实验前后各落一次环境指纹（复用 M21 成果）
python scripts/m21_env_fingerprint.py --out outputs/m23/env_fingerprint_before.json
```

**脚本内置的安全闸**（已验证非空转）：GPU 已用显存 > 4000 MiB 时 `--run` **直接拒绝启动**，
避免抢占他人任务；需覆盖时必须显式 `--allow-busy-gpu`。

---

## 已完成的准备工作（不占 GPU、不拉镜像）

- [x] 设计冻结 v3（含所有者指出的 U4/L3、显存口径、指标分层三项修订）
- [x] `backend_migration.md`（T 产出，即使 Gate 失败也成立）
- [x] `scripts/m23_gate.py`：Gate 0/1 执行器，默认 dry-run，两个 Gate 的 dry-run 均已自检通过
- [x] GPU 安全闸验证（当前 14.1 GB 占用 → 判定「拒绝启动」）
- [x] ROADMAP 同步到 v3 / DESIGN FROZEN，并收窄 L3 表述为「可行但未批准」
- [x] **Gate 0 PASS**：TRT-LLM 1.3.0rc24 导入正常，识别 sm_120
- [x] **Gate 1 PASS**：HF checkpoint 直接加载 + **真实生成**连贯文本（无 sm_120 kernel 缺失报错）
- [x] **Gate 2 PASS**：接受 prompt token ids（可与 vLLM 严格对齐）、Shape A 跑通、生效配置可回读
- [x] **A1 口径决策**：改为 2×2 execution-mode（冻结稿 v4 §5.0）
- [x] **V-ctr correctness preflight PASS**：每条 lane 各自 across-process 6/6 确定性；跨 lane 1/6（limitation）
- [x] **L1 离线跨引擎对照完成**：四组 V-E/V-O/T-E/T-O，两条 lane 下 vLLM 均更快（CI 全排除 0）
- [ ] L2 在线服务 / L3 Triton 集成 / L4 Transformers 基线 —— **未批准**
