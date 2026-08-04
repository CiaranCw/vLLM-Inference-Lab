# M13 · 提交人亲自复现命令（REPRO_COMMANDS）

> 全部命令请**你本人执行**并亲眼确认输出。环境遵循 vLLM `AGENTS.md`：只用 `uv` / `.venv/bin/python`，**不使用系统 python3 或裸 pip**。
> 标注「⛔ 仅在人工审查全部完成后执行」的命令，**本轮我一律未执行**。

## 1. 进入干净工作树
```bash
cd /home/ciaran/projects/vllm-m13-upstream
git branch --show-current      # 期望: m13/block-table-slot-mapping-tests
git rev-parse HEAD             # 期望: 72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0
```

## 2. 查看改动（逐行审查主入口）
```bash
git status --short                              # 期望: 只有 tests/v1/worker/test_block_table.py
git diff --stat HEAD                            # 期望: 1 file changed, 347 insertions(+)
git diff HEAD -- tests/v1/worker/test_block_table.py | less   # ← 逐行读
```
对照产品源码（建议并排）：
```bash
sed -n '182,215p' vllm/v1/worker/block_table.py   # compute_slot_mapping 调用侧
sed -n '379,443p' vllm/v1/worker/block_table.py   # Triton kernel 全文
```

## 3. 确认最新 main（重复检查，勿省）
```bash
git fetch origin main
git rev-parse origin/main       # 若与 HEAD 不同，需重新 port + 重跑验证
git log -1 --format=fuller origin/main
```

## 4. 运行测试
```bash
.venv/bin/python -m pytest tests/v1/worker/test_block_table.py --collect-only    # 期望: 18 tests collected
.venv/bin/python -m pytest tests/v1/worker/test_block_table.py -v                # 期望: 18 passed
```
GPU 安全（跑前确认无他人进程；**不要 kill 任何进程**）：
```bash
nvidia-smi
```

## 5. 运行 Pre-commit 与 Ruff
```bash
.venv/bin/pre-commit run --files tests/v1/worker/test_block_table.py   # 期望: 全部 Passed
.venv/bin/python -m ruff check tests/v1/worker/test_block_table.py     # 期望: All checks passed!
.venv/bin/python -m ruff format --check tests/v1/worker/test_block_table.py  # 期望: 1 file already formatted
git diff --check                                                        # 期望: 无输出
```

## 6. 确认产品代码零改动
```bash
git diff --name-only HEAD -- vllm/ csrc/     # 期望: 无输出
grep -n "map_to_kernel_blocks" tests/v1/worker/test_block_table.py   # 期望: 无输出（Oracle 独立）
```

## 7. 最后一次查重（AGENTS.md 要求）
```bash
gh pr list --repo vllm-project/vllm --state open --search "compute_slot_mapping"
gh pr list --repo vllm-project/vllm --state open --search "block table slot mapping"
gh pr list --repo vllm-project/vllm --state open --search "test_block_table"
gh pr view 30683 --repo vllm-project/vllm --json state,mergedAt
```

## 8. 确认 Git 身份（DCO）
```bash
git config user.name       # 当前: CiaranCw
git config user.email      # 当前: CiaranCw@users.noreply.github.com
```
→ **由你确认这就是你要用于 DCO sign-off 的身份**（我不代你确认）。

## 9. ⛔ 仅在人工审查全部完成后执行
```bash
# 9a. 创建 commit（需 -s 以满足 DCO；AI 归属 trailer 按 AGENTS.md 与你的决定）
git add tests/v1/worker/test_block_table.py
git commit -s -m "[Core] Add unit tests for BlockTable slot mapping"

# 9b. 推到你自己的 fork（先确认 fork 地址；本轮未添加任何 fork remote）
git remote add fork <your-fork-url>
git push fork m13/block-table-slot-mapping-tests

# 9c. 创建 PR（Body 取自 notes/m13/pr_draft/PR_DESCRIPTION.md 的英文正文，
#     并先把「submitter will review」改成真实完成时、AI 工具名按实际填写）
gh pr create --repo vllm-project/vllm \
  --title "[Core] Add unit tests for BlockTable slot mapping" \
  --body-file <你的 PR body 文件>
```

## 10. 参考：本轮验证时的环境
```
upstream/main : 72cd5424da80a4a9caa3f42fd65bc0b94e61cbf0
vllm          : 0.1.dev1+gcb8104839（precompiled 安装时的基线，代码已 port 到 72cd542）
python        : 3.12.13
torch         : 2.13.0+cu130
triton        : 3.7.1
ruff          : 0.14.0（与 .pre-commit-config.yaml 的 rev 一致）
GPU           : NVIDIA GeForce RTX 5080, sm_120 (12,0), driver 610.62
```
