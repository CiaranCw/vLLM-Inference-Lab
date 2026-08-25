#!/usr/bin/env bash
# M22 · 远端 H20 上机自检（只读、快、不下载模型）
#
# 目标：SSH 上机后一条命令在几分钟内回答「这台机器能不能跑我们的实验」。
# 刻意不做任何耗时的大文件下载；只做连通性探测（HEAD 请求级别）。
#
#   bash scripts/m22_preflight.sh 2>&1 | tee logs/m22/preflight.log
#
# 退出码：0 = 全部关键项通过；1 = 有关键项失败（详见输出的 [FAIL] 行）。
# 关键项 = 会让 TP 实验根本跑不起来的项。非关键项只 [WARN]，不影响退出码。

set -uo pipefail

FAIL=0
ok()   { printf '  [ ok ] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=1; }
hdr()  { printf '\n===== %s =====\n' "$*"; }

hdr "0. 基本身份"
echo "  date        : $(date -Iseconds)"
echo "  hostname    : $(hostname)"
echo "  user        : $(whoami)"
echo "  pwd         : $(pwd)"
echo "  uname       : $(uname -srmo)"
if [ -r /etc/os-release ]; then . /etc/os-release; echo "  os          : ${PRETTY_NAME:-unknown}"; fi
echo "  cpu cores   : $(nproc 2>/dev/null || echo '?')"
echo "  mem total   : $(awk '/MemTotal/{printf "%.1f GiB", $2/1048576}' /proc/meminfo 2>/dev/null || echo '?')"

hdr "1. 是否在持久会话中（SSH 断线保护）"
if [ -n "${TMUX:-}" ]; then
  ok "已在 tmux 会话中（TMUX=$TMUX）"
elif [ -n "${STY:-}" ]; then
  ok "已在 screen 会话中（STY=$STY）"
else
  warn "不在 tmux/screen 中 —— SSH 一断实验就没了。强烈建议先: tmux new -s m22"
fi

hdr "2. GPU / 驱动"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "找不到 nvidia-smi"
else
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version \
             --format=csv 2>&1 | sed 's/^/  /'
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  echo "  可见 GPU 数: $NGPU"
  [ "$NGPU" -ge 4 ] && ok "≥4 张 GPU 可见" || bad "只看到 $NGPU 张 GPU（TP=4 需要 4 张）"
  echo "  --- 其他进程占用（非空即代表机器不独占）---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/  /'
  echo "  --- CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES:-<unset>} ---"
fi

hdr "3. GPU 互联拓扑（决定 custom all-reduce 能否启用）"
# custom_all_reduce.py:149-167 —— world_size>2 时要求 is_fully_connected（NVLink 一跳）
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi topo -m 2>&1 | sed 's/^/  /'
  echo "  ↑ 关注 GPU0-3 之间是否为 NV# / NVLink。若为 PHB/SYS(PCIe)，TP=4 的"
  echo "    custom all-reduce 很可能被禁用而回落到 NCCL —— 这会显著改变通信路径与结论。"
else
  bad "无 nvidia-smi，无法读拓扑"
fi

hdr "4. Python / PyTorch / CUDA"
command -v python3 >/dev/null 2>&1 && echo "  python3: $(python3 -V 2>&1)" || bad "无 python3"
python3 - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import torch
    print("torch        :", torch.__version__)
    print("torch.cuda   :", torch.version.cuda)
    print("is_available :", torch.cuda.is_available())
    print("device_count :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} cc={p.major}.{p.minor} mem={p.total_memory/2**30:.1f}GiB")
    print("nccl available:", torch.distributed.is_nccl_available())
    try:
        print("nccl version :", torch.cuda.nccl.version())
    except Exception as e:
        print("nccl version : (n/a)", e)
    print("dist available:", torch.distributed.is_available())
except ImportError as e:
    print("torch 未安装:", e)
PY

hdr "5. vLLM"
python3 - <<'PY' 2>&1 | sed 's/^/  /'
try:
    import vllm, os
    print("vllm         :", vllm.__version__)
    print("vllm path    :", os.path.realpath(vllm.__file__))
except ImportError as e:
    print("vllm 未安装:", e)
PY

hdr "6. 必需环境变量（缺了会静默改变进程拓扑）"
# llm_engine.py:224-226：该变量为真会强制覆盖 enable_multiprocessing=False
V="${VLLM_ENABLE_V1_MULTIPROCESSING:-<unset>}"
if [ "$V" = "0" ]; then
  ok "VLLM_ENABLE_V1_MULTIPROCESSING=0"
else
  bad "VLLM_ENABLE_V1_MULTIPROCESSING=$V —— 必须为 0，否则 EngineCore 跑到独立进程，逐 step 内省全部失效"
fi
echo "  VLLM_USE_FLASHINFER_SAMPLER = ${VLLM_USE_FLASHINFER_SAMPLER:-<unset>}  (sm_90 上非必需，但保持与既有实验一致)"
echo "  VLLM_LAB_TRACE              = ${VLLM_LAB_TRACE:-<unset>}   (须未设置或 0)"
echo "  VLLM_LAB_M9_COALESCE        = ${VLLM_LAB_M9_COALESCE:-<unset>}   (须未设置或 0)"
echo "  HF_HOME                     = ${HF_HOME:-<unset>}"
echo "  HF_ENDPOINT                 = ${HF_ENDPOINT:-<unset>}"

hdr "7. 磁盘空间"
df -h . 2>&1 | sed 's/^/  /'
AVAIL_G=$(df -BG --output=avail . 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -n "${AVAIL_G:-}" ]; then
  [ "$AVAIL_G" -ge 40 ] && ok "可用 ${AVAIL_G}G（7B BF16 约需 16G + 余量）" \
                        || warn "可用仅 ${AVAIL_G}G —— 7B 可能放不下，考虑降级 1.5B"
fi
echo "  --- HF 缓存现状（判断是否已有模型，避免重复下载）---"
HFH="${HF_HOME:-$HOME/.cache/huggingface}"
if [ -d "$HFH/hub" ]; then
  du -sh "$HFH/hub"/models--* 2>/dev/null | sed 's/^/  /' || echo "  (空)"
else
  echo "  (无 $HFH/hub)"
fi

hdr "8. 网络可达性（只做 HEAD 探测，不下载模型）"
probe() {
  local name="$1" url="$2"
  if curl -sS -I -m 8 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null | grep -qE '^(2|3)'; then
    ok "$name 可达 ($url)"
  else
    warn "$name 不可达或超时 ($url)"
  fi
}
if command -v curl >/dev/null 2>&1; then
  probe "HuggingFace"  "https://huggingface.co"
  probe "HF CDN"       "https://cdn-lfs.huggingface.co"
  probe "hf-mirror"    "https://hf-mirror.com"
  probe "PyPI"         "https://pypi.org/simple/"
else
  warn "无 curl，跳过网络探测"
fi

hdr "9. 结论"
if [ "$FAIL" -eq 0 ]; then
  echo "  >>> 关键项全部通过。下一步："
  echo "      1) python3 scripts/m22_nccl_smoke.py --world-size 4"
  echo "      2) 确认模型可用后再跑 TP smoke"
else
  echo "  >>> 有关键项失败（见上面的 [FAIL]）。先停下报告，不要盲目往下跑。"
fi
exit "$FAIL"
