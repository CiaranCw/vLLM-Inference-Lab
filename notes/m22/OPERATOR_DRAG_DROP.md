# M22 · 操作者拖拽清单（本机 → H20）

> **现行做法：整目录拖入 `m22_offline`**（不要拆文件零散传）。  
> 上传代价可忽略；**不要**把该目录推进 GitHub。

## 本机源

```text
<操作者本机>/m22_offline/          # 即 Offline Fallback Pack 根
  bundles/
  models/
  manifests/
  wheelhouse/
```

（本机常见位置仅供操作者自己找文件：家目录下的 `m22_offline`；**文档与 Agent 不得写死该绝对路径**。）

## 拖到 H20

```text
$REPO_ROOT/m22_offline/            # 与项目根并列的一整个目录
```

即：先（或同时）准备好 git clone 的 `$REPO_ROOT`，再把整个 `m22_offline` 拖进项目根，使远端出现：

```text
$REPO_ROOT/
  notes/m22/ ...
  scripts/m22_*.py ...
  m22_offline/
    bundles/m22_runtime_bundle.tar.gz
    bundles/m22_runtime_bundle.tar.gz.sha256
    models/Qwen2.5-7B-Instruct/
    models/Qwen2.5-1.5B-Instruct/
    manifests/
    wheelhouse/
```

然后在远端：

```bash
export REPO_ROOT="$PWD"            # 在项目根
export OFFLINE="$REPO_ROOT/m22_offline"
export MODEL_PATH="$OFFLINE/models/Qwen2.5-7B-Instruct"
```

Agent 会从 `$OFFLINE/bundles/` 解压 Frozen runtime bundle，并从 `$OFFLINE/models/` 读权重。

## 可选补充

| 内容 | 说明 |
|---|---|
| GitHub clone/pull `$REPO_ROOT` | 文档 + Agent Prompt；与 `m22_offline` **互补** |
| 仅缺 Git 时 | 至少保证 `m22_offline/bundles` 可解压出 scripts |

## 不要做

- 把 `m22_offline`（含模型/wheel）commit / push 到 GitHub  
- 只拖部分子目录却声称 Pack 完整（缺 `bundles` 或 `models` 会 BLOCKED）  
- 正式结果只丢 GitHub、不做本机 `sha256sum -c`

## 回传（H20 → 本机）

优先本机 pull（Runbook Stage 3/13）。GitHub 只回收小摘要。
