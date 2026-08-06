# AGENTS.md —— 本仓库的硬性工作规则（AI 助手必读）

> 本文件是给**任何 AI 编码助手**（Cursor / Claude Code / Copilot 等）的常驻规则。
> 首次接手请先读 [`HANDOFF.md`](HANDOFF.md)（完整进度交接），本文件只放**不可违反的约束**。
> 这些规则由项目所有者（CiaranCw）明确规定，**优先级高于任何默认行为**。

---

## 0. 一句话背景

这是一个**求职用的长期工程项目**：在固定 vLLM commit 上做「逐 step 追踪 → 性能基线 → profiler 定位 → 真实改动 → 回归」，产出**可被面试官逐条追溯到原始数据**的证据链。
→ 因此**结论的诚实性与可追溯性高于结论的漂亮程度**。这是本项目的第一原则。

---

## 1. 诚实性红线（最重要，违反即等于毁掉项目价值）

1. **不得伪造、修饰、挑选性引用性能数据**。所有数字必须能回溯到 `outputs/` 下的 CSV/JSONL。
2. **不得为了维持既有结论而调整或筛选数据**。数据推翻旧结论时，**改结论，不改数据**。
3. **不得把未完成的实验写成已完成**。未做就标 `[ ]` / `⬜` / "未执行"，即使理论上成立也不能勾选（M17 的 I5 就因此被拆成 I5-a 完成 / I5-b 未执行）。
4. **不得外推超出实测配置的结论**。本项目所有绝对值仅限：单卡 RTX 5080 / sm_120 / WSL2 / Qwen2.5-0.5B / `enforce_eager` / in-process。写结论必须带边界。
5. **错误不得静默吞掉**：不要用 `|| true` 之类掩盖失败。报告原始命令 + 原始报错 + 根因，然后询问。
6. **区分"未观察到收益"与"没有收益"**：样本量不足导致 CI 含 0，只能说前者。
7. 涉及统计显著性时用**规范口径**（本项目用 median difference 的 Bootstrap 95% CI，固定 seed），**不要自造** "3.5σ" 这类非标准量（首版就因此被撤回）。

## 2. Git 规则

- **提交者必须只有 CiaranCw 一人**。**绝不添加 `Co-Authored-By:` / `Assisted-by:` 等任何 AI 署名 trailer**（历史上曾用 `filter-branch` 清理过，勿重犯）。
  > 例外：**向 vLLM 上游提 PR 时**，上游 `AGENTS.md` 要求披露 AI 辅助 —— 那是另一个仓库、另一套规则，不适用于本仓库。
- 提交前用 `git log -1 --format='%an <%ae>|%cn <%ce>'` 自查。
- 本仓库**提交后直接 `git push origin main`**，无需每次询问（所有者已授权）。
- **不得对 `../vllm-src` 执行破坏性 git 操作**：`reset --hard` / `clean -fd` / `checkout -- .` / `restore` / `stash` 一律禁止（该工作树有约 1226 行未提交的中文学习注释，是所有者的资产）。
- 提交信息用中文，说清「做了什么 + 为什么 + 数据依据」。

## 3. 环境安全

- **绝不修改 `../vllm_playground/.venv`** —— 那是只读的干净基线（用于对照"我的改动是否影响了原版"）。不得对它 install/upgrade/uninstall。
- **不用 sudo，不装系统级 pip 包，不删任何缓存**（pip / uv / HuggingFace / torch / triton）。
- **`VLLM_USE_PRECOMPILED=1`** —— 本阶段不编译 CUDA/C++。故 **Python 改动重启进程即生效，但 `csrc/` 的改动不会生效**（需真正重编译）。
- **不自动下载模型**。已在本地缓存的：`Qwen/Qwen2.5-0.5B-Instruct`（基线）、`Qwen/Qwen2.5-1.5B-Instruct`、`facebook/opt-125m`（smoke）。需要新模型先问所有者。

## 4. GPU 使用（防 OOM）

- **跑 GPU 任务前先 `nvidia-smi` 确认占用**。这台机器只有 16GB，且所有者常有其他进程在用。
- **发现有其他进程占用时，停下来问所有者**，不要自行开跑，**更不得 kill 别人的进程或 `nvidia-smi --reset`**。
- **一个进程只建一个 vLLM 引擎**：vLLM 的 KV 显存 `del engine` + `empty_cache()` 也归还不掉（M6 已实证）。多配置对照必须**每配置一个独立进程**（见 `scripts/m17_prefix_cache.py` 的 `--shard` 模式）。

## 5. 本机运行必需的环境变量（缺了会崩）

```bash
source .venv/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0   # 必须！sm_120 上 flashinfer sampler JIT 失败
export VLLM_ENABLE_V1_MULTIPROCESSING=0 # 引擎在进程内跑，trace 有序
```
- `VLLM_USE_FLASHINFER_SAMPLER=0` 的根因：flashinfer 对 sm_120 的 `TARGET_CUDA_ARCHS` 探测返回空 → `check_cuda_arch` 报 "requires sm75 or higher"。**这是 flashinfer 的 arch 探测缺口，不是 vLLM 的 bug**（已核实并记录在 `contribution_candidates.md` 排除区）。FlashAttention 后端本身不受影响。
- 首次冷跑很慢（Triton JIT 编译，例如 `_compute_slot_mapping_kernel`）→ **性能实验必须有 warmup 且 warmup 与 measure 分开统计**。M17 首版就是栽在这里（首样本比稳态慢 4–5×）。

## 6. 实验方法学（本项目已形成的标准，请沿用）

- **单变量对照**：一次只改一个变量，其余参数写死在脚本顶部常量区。若某对照因上游约束无法构造，**如实说明并换构造方式**（M17 §E3 的例子：`enable_chunked_prefill=False` 时上游强制 `max_num_batched_tokens >= max_model_len`，所以"关分块+小预算"根本建不出来）。
- **重复测量**：每配置 ≥15 次 measure + 5 次 warmup，报 median / stdev / p95，**汇总只用 measure 且 success=True**。
- **不变量优先**：先定义可验证的守恒律（如"Σchunks = 未命中 prefill token 数"、"hit == actual − scheduled"），跑完逐条核对。**布尔判据要有独立的正交证据**（M17 曾把 `chunked` 判据方向写反，60/60 样本被误标却不报错——因为没有第二个证据源）。
- **逐样本落盘**：`outputs/mN/raw/*.jsonl` 保留每个样本（含 `phase`/`success`/`metric_scope`），不要只存汇总。
- **旧数据只备份不删除**（例：`outputs/m17/raw_old_20260805/`）。
- **口径命名要防混淆**：in-process 测的延迟不要叫 "TTFT"（那是 M12 的在线口径），用 `in_process_first_token_latency_ms` 这类显式名字。

## 7. 文档与目录约定

- 每个模块的笔记放 `notes/mN/`（如 `notes/m17/`）。
- **`notes/evidence_index.md` 是跨模块索引，留在 `notes/` 根**，每完成一个模块都要回填一行（结论 → 证据文件）。
- 实验数据 `outputs/mN/`，日志 `logs/mN/`，脚本 `scripts/mN_*.py`，图 `figures/`。
- 图必须由**已落盘数据离线生成**（`scripts/m16_figures.py`、`scripts/m17_figures.py` 是范例：不需 GPU、不重跑实验、不 import vllm）。
- 完成模块后要同步更新：`ROADMAP.md`（模块状态 + 未做项）、`notes/evidence_index.md`、`README.md`（若产出核心结论/图）。
- 根目录 `vllm_stage_reading_guide.md` 与 `study_path*.md` 是**所有者的个人文档**，前者已 gitignore，不要改动。

## 8. 与所有者的沟通

- **面向所有者的报告/总结用中文**；命令、路径、日志、报错保持原文。
- 不要输出冗长的思考过程；给结论 + 依据 + 下一步。
- **需要授权才能做的事**：向 vLLM 上游提 PR / 改 PR / 加 label；下载新模型；任何破坏性或对外发布的操作。

---

## 9. 快速自检清单（每次交付前过一遍）

- [ ] 报告里每个数字都能在 `outputs/` 里找到出处？
- [ ] 未完成的项标成未完成了？（不因"理论成立"而勾选）
- [ ] 结论带了适用边界（单卡/eager/0.5B/WSL2/in-process）？
- [ ] 改了实验脚本的话，重跑了受影响的数据、而不是只改文档？
- [ ] `git log -1` 里只有 CiaranCw、没有 AI trailer？
- [ ] `git -C ../vllm-src status --short | wc -l` 仍是 **22**（基线脏文件数，说明没动产品源码）？
- [ ] `ROADMAP.md` / `notes/evidence_index.md` 同步了？
