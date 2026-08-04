# M13 · 提交人亲自复现命令（REPRO_COMMANDS）

> 全部命令请**你本人执行**并亲眼确认输出。环境遵循 vLLM `AGENTS.md`：只用 `uv` / `.venv/bin/python`，**不使用系统 python3 或裸 pip**。
> 标注「⛔ 仅在人工审查全部完成后执行」的命令，**本轮我一律未执行**。
>
> 最终基线：upstream/main **`0b1c151cbcacd12e207f2a91e4ca9f5b6961ec8a`**
> 最终测试：**384 行 / 18 tests**，SHA256 `b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`

## 1. 进入干净工作树
```bash
cd /home/ciaran/projects/vllm-m13-upstream
git branch --show-current      # 期望: m13/block-table-slot-mapping-tests
git rev-parse HEAD             # 期望: 0b1c151cbcacd12e207f2a91e4ca9f5b6961ec8a
sha256sum tests/v1/worker/test_block_table.py   # 期望: b0a15b41...
```

## 2. 查看改动（逐行审查主入口）
```bash
git status --short                        # 期望: 只有 tests/v1/worker/test_block_table.py
git diff --stat HEAD                      # 期望: 1 file changed, 384 insertions(+)
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
nvidia-smi     # 跑前确认无他人重要进程；不要 kill 任何进程
.venv/bin/python -m pytest tests/v1/worker/test_block_table.py --collect-only   # 期望: 18 tests collected
.venv/bin/python -m pytest tests/v1/worker/test_block_table.py -v               # 期望: 18 passed
```

## 5. 运行 Pre-commit 与 Ruff
```bash
.venv/bin/pre-commit run --files tests/v1/worker/test_block_table.py   # 期望: 全部 Passed
.venv/bin/python -m ruff check tests/v1/worker/test_block_table.py     # 期望: All checks passed!
.venv/bin/python -m ruff format --check tests/v1/worker/test_block_table.py  # 期望: 1 file already formatted
git diff --check HEAD                                                  # 期望: 无输出
```

## 6. 确认产品代码零改动 + Oracle 独立
```bash
git diff --name-only HEAD -- vllm/ csrc/     # 期望: 无输出
grep -n "map_to_kernel_blocks(" tests/v1/worker/test_block_table.py   # 期望: 无输出（仅 docstring 提名字）
```

## 7. （可选）自行复跑变异测试
```bash
cp vllm/v1/worker/block_table.py /tmp/bt_backup.py
# 手动在 map_to_kernel_blocks 的返回表达式末尾加 "+ 1"
.venv/bin/python -m pytest tests/v1/worker/test_block_table.py -k hybrid -q   # 期望: 3 failed
cp /tmp/bt_backup.py vllm/v1/worker/block_table.py     # 必须恢复
git diff --name-only HEAD -- vllm/                     # 必须无输出
.venv/bin/python -m pytest tests/v1/worker/test_block_table.py -q             # 期望: 18 passed
```

## 8. 最后一次查重（AGENTS.md 要求）
```bash
gh pr list --repo vllm-project/vllm --state open --search "compute_slot_mapping"
gh pr list --repo vllm-project/vllm --state open --search "block table slot mapping"
gh pr list --repo vllm-project/vllm --state open --search "test_block_table"
gh pr list --repo vllm-project/vllm --state open --search "DCP slot mapping"
gh pr list --repo vllm-project/vllm --state open --search "hybrid block table"
gh pr view 30683 --repo vllm-project/vllm --json state,mergedAt
```

## 9. 确认 Git 身份（DCO）
```bash
git config user.name       # 当前: CiaranCw
git config user.email      # 当前: CiaranCw@users.noreply.github.com
```
→ **由你确认这就是你要用于 DCO sign-off 的身份**（我不代你确认）。

## 10. ⛔ 仅在人工审查全部完成后执行
> 前置条件：`FINAL_PRE_PR_CHECKLIST.md` 全部勾选完毕。

```bash
# 10a. 创建 commit（-s 满足 DCO；AI 归属 trailer 按 AGENTS.md 与你的决定）
git add tests/v1/worker/test_block_table.py
git commit -s -m "[Core] Add unit tests for BlockTable slot mapping"

# 10b. 推到你自己的 fork（本轮未添加任何 fork remote）
git remote add fork <your-fork-url>
git push fork m13/block-table-slot-mapping-tests

# 10c. 创建 PR（Body 取自 notes/m13/pr_draft/PR_DESCRIPTION.md 的英文正文；
#      先把 AI 披露段的未来时改成真实完成时、并按实际填写工具名）
gh pr create --repo vllm-project/vllm \
  --title "[Core] Add unit tests for BlockTable slot mapping" \
  --body-file <你的 PR body 文件>
```

## 11. 参考：本轮验证环境
```
upstream/main : 0b1c151cbcacd12e207f2a91e4ca9f5b6961ec8a
python        : 3.12.13
torch         : 2.13.0+cu130
triton        : 3.7.1
ruff          : 0.14.0（与 .pre-commit-config.yaml rev 一致）
GPU           : NVIDIA GeForce RTX 5080, sm_120 (12,0), driver 610.62
```
