#!/usr/bin/env bash
# M24 · 远端 H20 自检（只读、不下载、不建 vLLM 引擎）
#
#   bash scripts/m24_preflight.sh /path/to/Qwen2.5-1.5B-Instruct 2>&1 | tee logs/m24/preflight.log
#
set -uo pipefail

MODEL="${1:-${M24_MODEL:-}}"
FAIL=0
ok()   { printf '  [ ok ] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=1; }
hdr()  { printf '\n===== %s =====\n' "$*"; }

hdr "0. 身份"
echo "  date     : $(date -Iseconds)"
echo "  hostname : $(hostname)"
echo "  user     : $(whoami)"
echo "  pwd      : $(pwd)"
echo "  python   : $(command -v python3 || echo MISSING)"

hdr "1. tmux / screen"
if [ -n "${TMUX:-}" ]; then ok "tmux $TMUX"
elif [ -n "${STY:-}" ]; then ok "screen $STY"
else warn "不在 tmux/screen —— SSH 一断实验就没了。先: tmux new -s m24"
fi

hdr "2. GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "无 nvidia-smi"
else
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
             --format=csv 2>&1 | sed 's/^/  /'
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  echo "  可见 GPU 数: $NGPU"
  [ "$NGPU" -ge 4 ] && ok "≥4 张 GPU" || bad "只看到 $NGPU 张（DCP 矩阵需要 TP=4）"
  echo "  --- 其他进程 ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/  /'
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]' || true)
  if [ "${APPS:-0}" -gt 0 ]; then
    warn "GPU 上已有计算进程。不要 kill。停下问所有者。"
  fi
  nvidia-smi topo -m 2>&1 | sed 's/^/  /'
fi

hdr "3. 环境变量"
V="${VLLM_ENABLE_V1_MULTIPROCESSING:-<unset>}"
if [ "$V" = "0" ]; then ok "VLLM_ENABLE_V1_MULTIPROCESSING=0"
else bad "VLLM_ENABLE_V1_MULTIPROCESSING=$V —— 必须为 0"
fi
echo "  VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-<unset>}"
echo "  HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-<unset>}  (建议 1)"
echo "  M24_MODEL=${M24_MODEL:-<unset>}"

hdr "4. Python / torch / vLLM"
python3 - <<'PY' 2>&1 | sed 's/^/  /'
import os, sys
print("python", sys.version.replace("\n"," "))
try:
    import torch
    print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available(), "ndev", torch.cuda.device_count())
except Exception as e:
    print("torch FAIL", e)
try:
    import vllm
    print("vllm", vllm.__version__, "path", os.path.realpath(vllm.__file__))
except Exception as e:
    print("vllm FAIL", e)
PY

hdr "5. 模型目录（必须本地、禁止联网下）"
if [ -z "$MODEL" ]; then
  bad "未给出模型路径。用法: bash scripts/m24_preflight.sh /path/to/Qwen2.5-1.5B-Instruct"
elif [ ! -d "$MODEL" ]; then
  bad "模型目录不存在: $MODEL"
else
  echo "  MODEL=$MODEL"
  [ -f "$MODEL/config.json" ] && ok "config.json" || bad "缺 config.json"
  if [ -f "$MODEL/model.safetensors" ] || ls "$MODEL"/model-*.safetensors >/dev/null 2>&1; then
    ok "safetensors 权重存在"
  else
    bad "没有 model.safetensors"
  fi
  python3 - "$MODEL" <<'PY' 2>&1 | sed 's/^/  /'
import json, os, sys
p = sys.argv[1]
c = json.load(open(os.path.join(p, "config.json"), encoding="utf-8"))
hq, hkv = c["num_attention_heads"], c["num_key_value_heads"]
print("H_q", hq, "H_kv", hkv, "q/kv", hq // hkv)
tp, dcp = 4, 2
legal = tp > hkv and dcp <= tp // hkv and (hq // hkv) % dcp == 0 and tp % dcp == 0
print("TP=4 DCP=2 legal?", legal)
if not legal:
    print("FAIL: 这个 config 不能跑正式矩阵（不要改成 7B/0.5B）")
    sys.exit(1)
PY
  if [ $? -ne 0 ]; then bad "GQA 算术不合法"; else ok "GQA 允许 TP=4 DCP=2"; fi
fi

hdr "6. 脚本"
for s in scripts/m24_dcp.py scripts/m24_run_all.py scripts/m24_pcp_negative.py scripts/check_invariants.py; do
  if [ -f "$s" ]; then
    python3 -m py_compile "$s" && ok "py_compile $s" || bad "语法错误 $s"
  else
    bad "缺 $s"
  fi
done

hdr "7. 磁盘"
df -h . 2>&1 | sed 's/^/  /'

hdr "8. 结论"
if [ "$FAIL" -eq 0 ]; then
  echo "  >>> 关键项通过。下一步: python3 scripts/m24_run_all.py --tag h20_formal --transfer-test"
else
  echo "  >>> 有 [FAIL]。先停下报告，不要开 formal。"
fi
exit "$FAIL"
