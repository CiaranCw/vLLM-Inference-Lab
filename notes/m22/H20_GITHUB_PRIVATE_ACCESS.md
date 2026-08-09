# M22 · H20 访问私有 GitHub（拉取 + 正确身份提交）

> 仓库私有时，H20 **不能**直接匿名 `git clone`。须先完成认证，再拉代码。  
> **密钥 / PAT 只放在远端用户目录，禁止写进仓库、禁止写进 Agent Prompt 正文。**

假定：

- GitHub 仓库：`github.com/CiaranCw/vLLM-Inference-Lab`（私有）
- 远端项目根：`$REPO_ROOT`
- Offline Pack：`$REPO_ROOT/m22_offline/`（整目录拖入，与 Git 互补）
- 提交身份必须是 **CiaranCw \<1399538830@qq.com\>**（本仓库 AGENTS：提交者姓名为 CiaranCw；禁止 AI `Co-authored-by` trailer）

---

## 总流程（建议顺序）

```text
1. H20 上生成 SSH key（或准备 PAT）
2. 把公钥 / PAT 加到 GitHub（本机浏览器操作）
3. 远端测通：ssh -T git@github.com
4. git config --global user.name / user.email（CiaranCw / 1399538830@qq.com）
5. git clone 私有仓 → $REPO_ROOT
6. 拖入 m22_offline（若尚未拖）
7. 再开实验；需要回传小摘要时再 commit/push
```

GitHub 网页操作在**你自己的浏览器**完成；H20 上只做密钥生成与 clone。

---

## 方案 A · SSH（推荐）

### A1. 在 H20 生成密钥（每人/每机一把，勿复用本机私钥文件上传）

```bash
# 在 H20 tmux 或登录 shell
mkdir -p ~/.ssh && chmod 700 ~/.ssh
ssh-keygen -t ed25519 -C "h20-m22-$(hostname)-$(whoami)" -f ~/.ssh/id_ed25519_github -N ""
chmod 600 ~/.ssh/id_ed25519_github
chmod 644 ~/.ssh/id_ed25519_github.pub
cat ~/.ssh/id_ed25519_github.pub
```

把打印出的 **一整行公钥** 复制到本机。

可选：仅对 github.com 使用该钥匙：

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
chmod 600 ~/.ssh/config
```

### A2. 本机浏览器：加到 GitHub

任选其一：

| 方式 | 路径 | 权限 |
|---|---|---|
| **用户 SSH key** | GitHub → Settings → SSH and GPG keys → New SSH key | 可访问你有权限的私有仓 |
| **Deploy key**（只读更安全） | 打开 `vLLM-Inference-Lab` → Settings → Deploy keys → Add | **Allow write** 仅当需要从 H20 `git push` |

若 Agent/你需要在 H20 **push** 小摘要：用户 SSH key，或 Deploy key 勾 write。  
若 **只 pull**：Deploy key 只读即可。

### A3. H20 测通

```bash
ssh -T git@github.com
# 成功会看到：Hi <user>! You've successfully authenticated...
```

若超时/被墙：记录报错，改用方案 B，或走「仅拖拽 + bundle、不用 Git」降级（见文末）。

### A4. Clone

```bash
# 例：放到工作目录下
cd ~
git clone git@github.com:CiaranCw/vLLM-Inference-Lab.git
cd vLLM-Inference-Lab
export REPO_ROOT="$PWD"
```

已有目录时：

```bash
cd "$REPO_ROOT"
git remote -v
# 若还是 https:// 且 401：改成 SSH
git remote set-url origin git@github.com:CiaranCw/vLLM-Inference-Lab.git
git pull --ff-only origin main
```

---

## 方案 B · HTTPS + Personal Access Token（SSH 不通时）

### B1. 本机浏览器创建 PAT

GitHub → Settings → Developer settings → Personal access tokens  
（Fine-grained：选中 `vLLM-Inference-Lab`，Permissions → Contents: **Read**；若需 push 再开 **Read and write**）

**不要**把 PAT 写进仓库或 Prompt。只在 H20 上用一次或放进 credential helper。

### B2. H20 clone

```bash
git clone https://github.com/CiaranCw/vLLM-Inference-Lab.git
# Username: 你的 GitHub 用户名
# Password: 粘贴 PAT（不是登录密码）
```

少交互可（注意历史记录风险）：

```bash
git clone "https://<GITHUB_USER>:<PAT>@github.com/CiaranCw/vLLM-Inference-Lab.git"
# 用完后立刻：改 remote 去掉 token；或删 shell history
cd vLLM-Inference-Lab
git remote set-url origin https://github.com/CiaranCw/vLLM-Inference-Lab.git
```

---

## 提交身份（两套都要配）

在 H20（建议 global，避免漏配）：

```bash
git config --global user.name "CiaranCw"
git config --global user.email "1399538830@qq.com"
# 可选：禁止本机乱改
git config --global commit.gpgsign false

# 自检
git config --global --get user.name
git config --global --get user.email
# 期望：CiaranCw / 1399538830@qq.com
```

每次 commit 后强制自检（本仓库规则）：

```bash
git log -1 --format='%an <%ae>|%cn <%ce>|%B'
# 作者/提交者必须是 CiaranCw <1399538830@qq.com>；正文不得出现 Co-authored-by / Assisted-by / cursoragent
```

若工具自动注入 `Co-authored-by: Cursor`：**不要 push**；用干净 message 重写后再推（或只在本机整理后推）。

---

## 与 Offline Pack 的配合

```bash
export REPO_ROOT="$PWD"                 # clone 后的项目根
export OFFLINE="$REPO_ROOT/m22_offline" # 整目录拖入后应存在

# 缺 m22_offline → 停下让操作者拖完整目录
# 有 Git + m22_offline → 按 REMOTE_AGENT_PROMPT：从 $OFFLINE/bundles 解压 Frozen 代码
```

| 来源 | 用途 |
|---|---|
| `git clone` / `pull` | 文档、Agent Prompt、脚本树 |
| `$OFFLINE` | 模型、wheel、与 Freeze 一致的 runtime bundle |

---

## 从 H20 提交 / 推送（可选，且严格限小文件）

1. **先**完成本机 Final Pull + `sha256sum -c`（大数据优先走拖拽/rsync，不走 Git）。  
2. 仅 stage 小文件：`notes/m22/results.md`、小 CSV、短日志摘要等。  
3. 推送前：

```bash
find . -type f -size +1M | head -50    # 不得包含模型/whl/formal tar
git status
git commit -m "M22 H20：……（做了什么 + 数据依据）"
git log -1 --format='%an <%ae>|%B'     # 确认身份与无 AI trailer
git push origin main
```

**禁止** push：`m22_offline/`、`*.safetensors`、`*.whl`、`m22_h20_formal_*.tar.gz`、大体量 traces。

---

## 故障速查

| 现象 | 处理 |
|---|---|
| `Permission denied (publickey)` | 公钥未加到 GitHub / `IdentityFile` 不对 / 用了错 key |
| `Repository not found` | 私有仓无权限，或 remote URL 用户名/仓库名错 |
| HTTPS `Authentication failed` | 应用 PAT，不是账户密码；PAT 权限不足 |
| `ssh: connect to host github.com port 22` 超时 | 内网拦 22：试 `~/.ssh/config` 里 `HostName ssh.github.com` + `Port 443`，或改方案 B |
| clone 成功但 push 拒 | Deploy key 只读，或 PAT 无 write |

SSH over 443 示例：

```bash
cat >> ~/.ssh/config <<'EOF'
Host github.com
  HostName ssh.github.com
  User git
  Port 443
  IdentityFile ~/.ssh/id_ed25519_github
  IdentitiesOnly yes
EOF
ssh -T git@github.com
```

---

## 降级（GitHub 始终不可达）

1. 本机 `git archive` / 打包**不含** `m22_offline` 的小文件树，拖到 H20 当 `$REPO_ROOT`。  
2. 仍拖完整 `m22_offline`。  
3. 结果只走本机 pull + checksum；**不在 H20 上强行配置 Git**。  
4. 文档回填在本机完成后再 push。

---

## 给内部客户端 Agent 的一句话

先确认 `ssh -T git@github.com` 或 HTTPS PAT 已通，且 `user.name/email` 为 CiaranCw；再 `clone/pull`。**不要**把 PAT/私钥写进笔记或命令历史可提交文件；缺认证时报 **BLOCKED**，等操作者完成本文 A2/B1。
