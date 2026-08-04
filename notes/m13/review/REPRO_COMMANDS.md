# M13 · 提交人亲自复现命令（REPRO_COMMANDS）

> 全部命令请**你本人执行**并亲眼确认输出。环境遵循 vLLM `AGENTS.md`：只用 `uv` / `.venv/bin/python`，**不使用系统 python3 或裸 pip**。
> 标注「⛔ 仅在人工审查全部完成后执行」的命令，**本轮我一律未执行**。
>
> 最终基线：upstream/main **`a1657a0235a77ce85f8d348c83ff4adb29918b71`**（本轮 fetch 实测）
> 最终测试：**384 行 / 18 tests**，SHA256 `b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`

## 1. 进入干净工作树
```bash
cd /home/ciaran/projects/vllm-m13-upstream
git branch --show-current      # 期望: m13/block-table-slot-mapping-tests
git rev-parse HEAD             # 期望: a1657a0235a77ce85f8d348c83ff4adb29918b71
sha256sum tests/v1/worker/test_block_table_slot_mapping.py   # 期望: b0a15b41...
```

## 2. 查看改动（逐行审查主入口）
```bash
git status --short                        # 期望: 只有 tests/v1/worker/test_block_table_slot_mapping.py
git diff --stat HEAD                      # 期望: 1 file changed, 384 insertions(+)
git diff HEAD -- tests/v1/worker/test_block_table_slot_mapping.py | less   # ← 逐行读
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
.venv/bin/python -m pytest tests/v1/worker/test_block_table_slot_mapping.py --collect-only   # 期望: 18 tests collected
.venv/bin/python -m pytest tests/v1/worker/test_block_table_slot_mapping.py -v               # 期望: 18 passed
```

## 5. 运行 Pre-commit 与 Ruff
```bash
.venv/bin/pre-commit run --files tests/v1/worker/test_block_table_slot_mapping.py   # 期望: 全部 Passed
.venv/bin/python -m ruff check tests/v1/worker/test_block_table_slot_mapping.py     # 期望: All checks passed!
.venv/bin/python -m ruff format --check tests/v1/worker/test_block_table_slot_mapping.py  # 期望: 1 file already formatted
git diff --check HEAD                                                  # 期望: 无输出
```

## 6. 确认产品代码零改动 + Oracle 独立
```bash
git diff --name-only HEAD -- vllm/ csrc/     # 期望: 无输出
grep -n "map_to_kernel_blocks(" tests/v1/worker/test_block_table_slot_mapping.py   # 期望: 无输出（仅 docstring 提名字）
```

## 7. （可选）自行复跑变异测试
```bash
cp vllm/v1/worker/block_table.py /tmp/bt_backup.py
# 手动在 map_to_kernel_blocks 的返回表达式末尾加 "+ 1"
.venv/bin/python -m pytest tests/v1/worker/test_block_table_slot_mapping.py -k hybrid -q   # 期望: 3 failed
cp /tmp/bt_backup.py vllm/v1/worker/block_table.py     # 必须恢复
git diff --name-only HEAD -- vllm/                     # 必须无输出
.venv/bin/python -m pytest tests/v1/worker/test_block_table_slot_mapping.py -q             # 期望: 18 passed
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

## 10. ⛔ 仅在人工审查全部完成后执行（模板，Agent 未执行）

> 前置条件：`FINAL_PRE_PR_CHECKLIST.md` **全部勾选完毕**。以下命令由**你本人**执行。

### 10.1 DCO 身份检查（必须你本人确认这些值属于你）
```bash
git config user.name
git config user.email
```
`git commit -s` 会用这两个值生成 `Signed-off-by:`。**Agent 不代你确认。**

### 10.2 Commit 模板
```bash
git add tests/v1/worker/test_block_table_slot_mapping.py

git commit -s \
  -m "[Core] Add unit tests for BlockTable slot mapping" \
  -m "Assisted-by: <actual AI tool/model>"
```
- `Assisted-by:` **只保留你实际使用过的工具**；未使用的不要写。
- 若你实际用过多个工具，可加多条 `-m "Assisted-by: ..."`。
- **Agent 不猜测、不填写任何 AI 工具名称。**

### 10.3 Push 模板
```bash
git push -u <fork-remote> m13/block-table-slot-mapping-tests
```
- `<fork-remote>` 需你先添加（本轮**未添加任何 fork remote**）：
  `git remote add <fork-remote> <your-fork-url>`
- **Agent 不猜测 fork 地址或 GitHub 用户名。**

### 10.4 PR 创建模板
```bash
gh pr create \
  --repo vllm-project/vllm \
  --base main \
  --head <github-user>:m13/block-table-slot-mapping-tests \
  --title "[Core] Add unit tests for BlockTable slot mapping" \
  --body-file <final-english-pr-body>
```

**`<final-english-pr-body>` 必须只包含英文上游正文**（取自 `pr_draft/PR_DESCRIPTION.md` 的「PR Body」段），**不得包含**：
- 中文内部状态说明；
- 本地路径（`/home/ciaran/...`）；
- Lab Notes 引用；
- 未完成的 Checklist；
- 无关环境失败（shellcheck SC2317、WSL2 UVA）。

提交前把 AI 披露段的未来时改成**真实完成时**，并填入实际工具名。

### 10.5 创建后
在 GitHub 页面再核对一次最终 Diff 与 PR Body 是否与本地一致。

## 11. 参考：本轮验证环境
```
upstream/main : a1657a0235a77ce85f8d348c83ff4adb29918b71
test file     : 384 lines, sha256 b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb
tests         : 18 passed, 14 warnings in 5.46s (agent automated run)
python        : 3.12.13
torch         : 2.13.0+cu130
triton        : 3.7.1
ruff          : 0.14.0（与 .pre-commit-config.yaml rev 一致）
GPU           : NVIDIA GeForce RTX 5080, sm_120 (12,0), driver 610.62
```
