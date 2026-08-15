# M24 Execution Freeze（上机前快照）

**Freeze ID:** `FROZEN_FOR_H20_M24_2026-08-15`  
**Copy-ready path:** `/home/ciaran/m24_h20_transfer/`

| 项 | 值 |
|---|---|
| 正式模型 | `Qwen2.5-1.5B-Instruct` 扁平目录 |
| `model.safetensors` sha256 | `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 大小 | 3087467144 bytes |
| 矩阵 | TP=4 固定，dcp∈{1,2}，见 `M24_H20_PLAN.md` |
| 远端提示 | `REMOTE_AGENT_PROMPT.md` |
| 7B / 0.5B | **禁止**（GQA 算术不合法） |

拷到 H20 后不要再改脚本；若必须改，在 `logs/m24/REMOTE_NOTES.md` 写明 diff 原因。
