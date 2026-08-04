# M13 · 提交人最后必须亲自执行的命令（HUMAN_FINAL_COMMANDS）

> 这是**最小命令集**。请你本人逐条执行并亲眼确认输出。
> **我（Agent）不会执行 commit / push / PR**。
>
> 基线：upstream/main **`a1657a0235a77ce85f8d348c83ff4adb29918b71`**
> 测试文件：384 行 / 18 tests / sha256 `b0a15b41c12edd9084295a9f88ef24b56b78d338a004c761f974872142a6b2eb`

```bash
cd /home/ciaran/projects/vllm-m13-upstream

# 1) 确认基线与改动范围
git log -1 --oneline
git status --short                       # 期望：只有 tests/v1/worker/test_block_table.py

# 2) 逐行审查最终 Diff（这一步不可省略、不可替代）
git diff HEAD -- tests/v1/worker/test_block_table.py | less

# 3) 亲自运行目标测试
.venv/bin/python -m pytest tests/v1/worker/test_block_table.py -v

# 4) 亲自运行 changed-file pre-commit
.venv/bin/pre-commit run --files tests/v1/worker/test_block_table.py

# 5) 确认无空白错误、且产品代码零改动
git diff --check HEAD
git diff --name-only HEAD -- vllm/ csrc/     # 期望：无输出

# 6) 确认 Git 身份（用于 DCO sign-off）——必须由你本人确认这是你的身份
git config user.name
git config user.email
```

## 配套阅读（建议顺序）
1. `MANUAL_GOLDEN_CASES.md` —— 手算复核 3 类 Golden（含 DCP 33→96、Hybrid 48→304、Tile 1023→1039）
2. `LINE_BY_LINE_REVIEW.md` —— 按行号逐段讲解（16 节）
3. `FINAL_PRE_PR_CHECKLIST.md` —— 逐项勾选（**只能你勾**）

---

## 人工确认全部完成后才执行（命令模板见 `REPRO_COMMANDS.md` §10）
```
# git add / git commit -s / git push <fork> / gh pr create
# —— 本文件刻意不提供可直接粘贴执行的一键脚本，避免误触。
```
