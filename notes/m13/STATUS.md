# M13 断点快照（RESUME / 交接）

> **一句话状态**：M13 首次上游 PR = **候选 C1（为 `compute_slot_mapping` 补单元测试）**。研究+实现+GPU 验证+自审已完成，**测试草稿 26/26 通过、ruff 全绿**，**尚未提交上游 PR**——**停在「提 PR 之前」**，等用户 go/no-go 与署名决策。为切换到别的任务而暂停。
>
> **最后更新**：M12 收口之后（2026-07-28 当轮）。恢复时先读本文件，再读 `notes/m13/pr_slot_mapping_test.md`（完整研究）。

---

## 1. 现在进度到哪了（阶段）
研究 ✅ → 实现草稿 ✅ → **GPU 验证 ✅（26/26）** → 自我审查+补强 ✅ → **⏸ 停在提 PR 之前** → （待办）干净工作树 + 完整 pre-commit + DCO + 建 PR。

lab 仓库相关提交均已 push 到 origin/main：
- `567ccd5` 研究阶段（选定 C1）
- `4b65d44` 定稿草稿（止于 GPU 验证前）
- `e805b42` GPU 验证 24/24
- `0366a3a` 审查补强 → **26/26**（最新）

## 2. 要做的事（本质）
为 vLLM `vllm/v1/worker/block_table.py` 的 `BlockTable.compute_slot_mapping` /
`_compute_slot_mapping_kernel` 补**单元测试**（当前 `tests/` 零覆盖）。
**策略**：全新写 `tests/v1/worker/test_block_table.py`，用当前 API + 独立 numpy oracle +
手算 golden，**引用并取代 stale PR #30683**（#30683 写在旧 API `compute_slot_mapping(req_indices, positions)` 上、且死于 pre-commit 未过）。

## 3. 关键产物（都在本仓库）
- `notes/m13/pr_draft/test_block_table.py` — **已验证的测试**（26 例：2 golden + 7 非CP边界 + 16 DCP4/PCP2 interleave{1,16}×8rank + 1 退化）。**这就是要提交给上游的文件**，落点 `tests/v1/worker/test_block_table.py`。
- `notes/m13/pr_draft/PR_DESCRIPTION.md` — PR 描述草稿（Purpose / Test Plan / 真实 Test Result / 取代 #30683 / DCO+署名注意）。
- `notes/m13/pr_slot_mapping_test.md` — 完整研究留痕（kernel 语义拆解、API 漂移、构造机制、覆盖矩阵）。
- `contribution_candidates.md`（仓库根）— C1 条目含前车与策略。

## 4. 如何立刻重跑验证（已确认可用）
```bash
# 规范检查（必须在 vllm-src 内跑，才用 vLLM 自己的 ruff/isort 配置；仓库外跑会误报 I001）
cp notes/m13/pr_draft/test_block_table.py /home/ciaran/projects/vllm-src/tests/v1/worker/test_block_table.py
source /home/ciaran/projects/vllm-inference-lab/.venv/bin/activate
cd /home/ciaran/projects/vllm-src && ruff check tests/v1/worker/test_block_table.py && ruff format --check tests/v1/worker/test_block_table.py

# 跑测试（放 /tmp 跑以绕开需额外依赖 tblib 的 tests/conftest；测试不依赖任何 conftest fixture）
mkdir -p /tmp/m13v && cp notes/m13/pr_draft/test_block_table.py /tmp/m13v/ && cd /tmp/m13v
VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 python -m pytest test_block_table.py -v
# 预期：26 passed（需 CUDA；GPU 占用时先问用户，避免 OOM）

# 跑完清理临时副本，勿在 vllm-src 留未跟踪文件
rm -rf /tmp/m13v && rm -f /home/ciaran/projects/vllm-src/tests/v1/worker/test_block_table.py
```

## 5. 环境 / 工作树注意（恢复时务必知道）
- lab venv `/.venv` 已装 `pytest`(9.1.1) + `ruff`(0.16.0)（本轮所装）。它可编辑安装到 `../vllm-src`。
- **`../vllm-src` 工作树是脏的**（~22 文件带中文学习标注 + 未跟踪 `_lab_trace.py`）。**提 PR 不能从这个树切分支**——须另做干净 fork/clone。
- pinned：vllm-src HEAD = v0.24.0 / `ee0da84`（committed）。kernel 语义在当前 main 未变（已核）。
- **GPU 使用前先问用户**（曾因占用而暂停，避免 OOM）。

## 6. 待用户决策（阻塞提 PR）
1. **go/no-go**：是否进入提 PR 准备。
2. **署名口径**：vLLM 要求披露 AI 辅助并加 `Co-authored-by:`——与本仓库 no-coauthor 约定（[[vllm-lab-no-coauthor-trailer]]）不同规则，须用户定。

## 7. 恢复后下一步动作清单（放行后）
1. 干净 checkout：fork `vllm-project/vllm` → clone 到独立目录（或 `git worktree`），基于最新 `main`。
2. 放入 `tests/v1/worker/test_block_table.py`（= 本仓库 pr_draft 的文件）。
3. 装 `pre-commit`，`pre-commit run --all-files`（mypy/codespell 等）全绿——**#30683 的死因，硬门槛**。
4. 可选：装 `requirements/test.txt` 后在 CI 式环境跑 `pytest tests/v1/worker/test_block_table.py`（需 tblib 等）。
5. `git commit -s`（DCO sign-off）→ push fork → `gh pr create`（描述用 PR_DESCRIPTION.md，引用/致谢 #30683 作者 pisceskkk）。
6. **建 PR 前再停一次**给用户看最终 diff/描述。
7. 回应 review、迭代。目标＝完整经历流程，非必须合入。

## 8. 关键 gotchas（省得再踩）
- ruff 配置：**在 vllm-src 内跑**（否则默认配置误报 import 排序 I001）。
- pytest：**放 /tmp 跑**绕开 `tests/conftest`（否则缺 tblib 报错）。
- oracle 循环论证隐患：已加 2 例手算 golden 化解（勿删）。
- 用例 `prefix_offset`：positions 到 89 需 ≥6 个 block，否则 numpy oracle 越界（已修为 6 个）。
- CP 参数：构造 BlockTable 后**直接设 `bt.pcp_world_size/pcp_rank/dcp_world_size/dcp_rank`** 即可（无需 mock 分布式组）。
- cuda 上须 `commit_block_table(num_reqs)` 把 block_table 从 np 拷到 gpu（#30683 用 cpu 才免此步）。
