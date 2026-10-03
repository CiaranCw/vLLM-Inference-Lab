#!/usr/bin/env bash
# M27 · H20 Gate 0 自检（只读、不下载、不建 vLLM 引擎、不改 driver/CUDA）
#
#   export M27_MODEL=/root/m22_offline/models/Qwen2.5-7B-Instruct
#   # 可选：M27_MODEL_SHA256 默认是仓库内冻结的 notes/m27/model_artifact/qwen2.5-7b-instruct.sha256；
#   #       若另行指定，内容必须与它逐字节相同
#   mkdir -p logs/m27 && bash scripts/m27_preflight.sh h20_formal 2>&1 | tee logs/m27/preflight_h20_formal.log
#
# 全部关键项 PASS 才会写 outputs/m27/configs/{preflight,gpu_map}_<tag>.json；
# m27_run_all.py 的 formal 门禁要求本机存在 result=PASS 的 preflight 记录。
set -uo pipefail

TAG="${1:-h20_formal}"
MODEL="${M27_MODEL:-}"
FROZEN_SUMS="notes/m27/model_artifact/qwen2.5-7b-instruct.sha256"
SUMS="${M27_MODEL_SHA256:-$FROZEN_SUMS}"
PY="${PYTHON:-python3}"
FAIL=0
FAILS=()
ok()   { printf '  [ ok ] %s\n' "$*"; }
warn() { printf '  [WARN] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; FAIL=1; FAILS+=("$*"); }
hdr()  { printf '\n===== %s =====\n' "$*"; }

case "$TAG" in h20_formal|h20_smoke|local_preflight) ;; *) echo "tag 必须是 h20_formal|h20_smoke|local_preflight"; exit 2;; esac
[ -f scripts/m27_common.py ] || { echo "必须在仓库根目录运行（找不到 scripts/m27_common.py）"; exit 2; }

hdr "0. 身份"
echo "  timestamp : $(date -Iseconds)"
echo "  hostname  : $(hostname)"
echo "  uname     : $(uname -a)"
echo "  user/pwd  : $(whoami) $(pwd)"
echo "  python    : $(command -v "$PY") ($("$PY" -V 2>&1))"
echo "  glibc     : $(ldd --version 2>&1 | head -1)"
if [ -n "${TMUX:-}" ] || [ -n "${STY:-}" ]; then ok "在 tmux/screen 内"; else warn "不在 tmux/screen —— 先: tmux new -s m27"; fi

hdr "1. GPU / driver / 进程"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "无 nvidia-smi"
  NGPU=0
else
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version --format=csv 2>&1 | sed 's/^/  /'
  NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  echo "  可见 GPU 数: $NGPU"
  [ "$NGPU" -ge 2 ] && ok "≥2 张（E0–E3 可行）" || bad "只看到 $NGPU 张；E0–E3 需要 ≥2"
  [ "$NGPU" -ge 4 ] && ok "≥4 张（E4 候选，最终由 NV 互联决定）" || warn "<4 张：E4 不执行"
  [ "$NGPU" -ge 8 ] && ok "8 张可见（E5 候选，需 Gate 8 拓扑）" || warn "<8 张：E5 不执行"
  if [ "$TAG" != "local_preflight" ]; then
    NON_H20=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -vc H20)
    [ "$NON_H20" -eq 0 ] && ok "全部为 H20" || bad "存在非 H20 GPU（tag=$TAG 只能在 H20 上跑）"
  fi
  echo "  --- compute apps ---"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv 2>&1 | sed 's/^/  /'
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c '[0-9]')
  [ "${APPS:-0}" -eq 0 ] && ok "无其他 GPU 进程" || bad "GPU 上已有 $APPS 个计算进程 —— 不 kill，停下问所有者"
  echo "  --- topo ---"
  nvidia-smi topo -m 2>&1 | sed 's/^/  /'
  echo "  --- nvlink status (仅记录) ---"
  nvidia-smi nvlink --status 2>&1 | head -40 | sed 's/^/  /'
fi
echo "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"

hdr "2. Python / torch / vLLM / NIXL 版本"
"$PY" - <<'PY' 2>&1 | sed 's/^/  /'
import os, sys
from importlib.metadata import version, PackageNotFoundError
def v(p):
    try: return version(p)
    except PackageNotFoundError: return None
want = {"python": "3.11", "torch": "2.11.0+cu130", "vllm": "0.24.0", "nixl": "1.2.0", "nixl-cu13": "1.2.0"}
got = {"python": ".".join(map(str, sys.version_info[:2])), "torch": v("torch"), "vllm": v("vllm"),
       "nixl": v("nixl"), "nixl-cu13": v("nixl-cu13"), "nixl-cu12": v("nixl-cu12")}
bad = []
for k, w in want.items():
    g = got.get(k)
    print(f"{k:10s} want={w:14s} got={g}")
    if g != w:
        bad.append(k)
try:
    import torch
    print("torch.version.cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(),
          "device_count", torch.cuda.device_count())
    if torch.version.cuda != "13.0":
        bad.append("torch.cuda")
except Exception as e:
    print("torch import FAIL", e); bad.append("torch_import")
try:
    import vllm
    print("vllm file", os.path.realpath(vllm.__file__))
except Exception as e:
    print("vllm import FAIL", e); bad.append("vllm_import")
print("nixl-cu12 installed (不需要，记录)", got["nixl-cu12"])
print("RESULT", "FAIL " + ",".join(bad) if bad else "PASS")
sys.exit(1 if bad else 0)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] && ok "版本与冻结值一致" || bad "版本不符（见上）"

hdr "3. NIXL 导入 / UCX agent 冒烟（不注册显存、不建引擎）"
"$PY" - <<'PY' 2>&1 | sed 's/^/  /'
import os, sys, glob
bad = []
try:
    import nixl
    from nixl._api import nixl_agent, nixl_agent_config
    from nixl._bindings import nixlXferTelemetry  # noqa: F401
    print("nixl backend module:", type(nixl).__name__, getattr(nixl, "__file__", None))
    import nixl_cu13
    libdir = os.path.join(os.path.dirname(os.path.dirname(nixl_cu13.__file__)), "nixl_cu13.libs")
    print("ucx plugins:", sorted(os.path.basename(p) for p in glob.glob(os.path.join(libdir, "ucx", "libuct_*.so"))))
    ag = nixl_agent("m27-preflight", nixl_agent_config(backends=["UCX"]))
    print("nixl_agent(UCX) created:", ag is not None)
except Exception as e:
    print("NIXL FAIL", type(e).__name__, e); bad.append("nixl")
try:
    from vllm.distributed.nixl_utils import NixlWrapper
    print("vllm NixlWrapper:", NixlWrapper)
    if NixlWrapper is None:
        bad.append("vllm_nixl_wrapper_none")
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import NixlConnector  # noqa: F401
    print("vllm NixlConnector import ok")
except Exception as e:
    print("vLLM NIXL import FAIL", type(e).__name__, e); bad.append("vllm_nixl")
print("UCX env:", {k: v for k, v in os.environ.items() if k.startswith("UCX_")} or "<none>")
print("RESULT", "FAIL " + ",".join(bad) if bad else "PASS")
sys.exit(1 if bad else 0)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] && ok "NIXL / vLLM NixlConnector 可导入，UCX agent 可创建" || bad "NIXL 导入或 UCX agent 失败"

hdr "4. 环境变量（服务进程由 launcher 统一注入；这里只记录 shell 现状）"
for k in VLLM_USE_FLASHINFER_SAMPLER VLLM_ENABLE_V1_MULTIPROCESSING HF_HUB_OFFLINE TRANSFORMERS_OFFLINE \
         VLLM_KV_CACHE_LAYOUT VLLM_NIXL_SIDE_CHANNEL_HOST VLLM_NIXL_SIDE_CHANNEL_PORT CUDA_VISIBLE_DEVICES \
         PYTHONPATH M27_MODEL M27_MODEL_SHA256; do
  echo "  $k=${!k:-<unset>}"
done
env | grep -E '^(UCX_|NIXL_)' | sed 's/^/  /'
[ -z "${CUDA_VISIBLE_DEVICES:-}" ] && ok "shell 未设 CUDA_VISIBLE_DEVICES（由 launcher 逐实例设置）" || warn "shell 设了 CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES（launcher 会覆盖；GPU 编号以 gpu_map 为准）"

hdr "5. 模型（本地 M22 artifact，禁止联网下载）"
if [ -z "$MODEL" ] || [ ! -d "$MODEL" ]; then
  bad "M27_MODEL 未设置或目录不存在: '$MODEL'"
else
  echo "  M27_MODEL=$MODEL"
  du -sb "$MODEL" 2>/dev/null | sed 's/^/  size_bytes /'
  ls "$MODEL" | sed 's/^/  /'
  "$PY" - "$MODEL" <<'PY' 2>&1 | sed 's/^/  /'
import json, os, sys
sys.path.insert(0, "scripts")
import m27_common as C
p = sys.argv[1]
c = json.load(open(os.path.join(p, "config.json")))
m = C.FROZEN["model"]
got = {"num_hidden_layers": c.get("num_hidden_layers"), "num_attention_heads": c.get("num_attention_heads"),
       "num_key_value_heads": c.get("num_key_value_heads"), "hidden_size": c.get("hidden_size"),
       "head_dim": c.get("hidden_size", 0) // max(c.get("num_attention_heads", 1), 1),
       "torch_dtype": c.get("torch_dtype"), "max_position_embeddings": c.get("max_position_embeddings")}
print("arch", c.get("architectures"), "torch_dtype", c.get("torch_dtype"),
      "max_position_embeddings", c.get("max_position_embeddings"), "rope_scaling", c.get("rope_scaling"),
      "use_sliding_window", c.get("use_sliding_window"), "quantization_config", c.get("quantization_config"))
bad = [k for k, v in got.items() if k != "max_position_embeddings" and v != m[k]]
print("arch fields", got, "mismatch", bad)
print("kv_bytes_per_token(theory)", 2 * got["num_hidden_layers"] * got["num_key_value_heads"] * got["head_dim"] * 2)
mpe = c.get("max_position_embeddings") or 0
print("max_model_len decision:", "32768 (frozen)" if mpe >= 32768 else "LEGAL FALLBACK 16384 / max prompt 8192")
for f in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
    print(f, "present" if os.path.exists(os.path.join(p, f)) else "MISSING")
if c.get("quantization_config"):
    bad.append("quantized_checkpoint")
if not os.path.exists(os.path.join(p, "tokenizer.json")):
    bad.append("tokenizer")
sys.exit(1 if bad else 0)
PY
  [ "${PIPESTATUS[0]}" -eq 0 ] && ok "config / tokenizer 与冻结架构一致" || bad "config 或 tokenizer 与冻结值不符"
  want_sha="$("$PY" -c 'import sys; sys.path.insert(0, "scripts"); import m27_common as C; print(C.FROZEN["model"]["identity_manifest_sha256"])')"
  if [ -f "$SUMS" ]; then
    SUMS="$(cd "$(dirname "$SUMS")" && pwd)/$(basename "$SUMS")"
    got_sha="$(sha256sum "$SUMS" | cut -d' ' -f1)"
    echo "  checksum manifest: $SUMS (sha256 $got_sha)"
    if [ "$got_sha" != "$want_sha" ]; then
      bad "checksum manifest ≠ 冻结的身份 manifest（期望 sha256 $want_sha）"
    elif (cd "$MODEL" && sha256sum -c "$SUMS" 2>&1 | sed 's/^/  /'; exit "${PIPESTATUS[0]}"); then
      ok "sha256sum -c 全部 OK（与冻结的身份 manifest 一致）"
    else
      bad "模型 checksum 不符"
    fi
  else
    bad "checksum manifest 不存在: '$SUMS'（默认 $FROZEN_SUMS）"
  fi
fi

hdr "6. 端口（proxy 8300 / HTTP 8400-8407 / NIXL side channel 5600-5607）"
"$PY" - <<'PY' 2>&1 | sed 's/^/  /'
import socket, sys
busy = []
for p in [8300] + list(range(8400, 8408)) + list(range(5600, 5608)):
    s = socket.socket()
    if s.connect_ex(("127.0.0.1", p)) == 0:
        busy.append(p)
    s.close()
print("busy ports:", busy or "none")
sys.exit(1 if busy else 0)
PY
[ "${PIPESTATUS[0]}" -eq 0 ] && ok "端口全部空闲" || bad "有端口被占用（不 kill 别人的进程）"

hdr "7. 脚本"
for s in scripts/m27_common.py scripts/m27_launch.py scripts/m27_proxy.py scripts/m27_workload.py \
         scripts/m27_run_all.py scripts/m27_summarize.py scripts/m27_trace_analyze.py scripts/m27_figures.py \
         scripts/check_invariants.py; do
  if [ -f "$s" ]; then "$PY" -m py_compile "$s" && ok "py_compile $s" || bad "语法错误 $s"; else bad "缺 $s"; fi
done

hdr "8. 磁盘"
df -h . 2>&1 | sed 's/^/  /'
[ -n "$MODEL" ] && [ -d "$MODEL" ] && df -h "$MODEL" 2>&1 | sed 's/^/  /'
AVAIL_KB=$(df -Pk . | awk 'NR==2{print $4}')
[ "${AVAIL_KB:-0}" -ge 20971520 ] && ok "工作目录可用 ≥20 GiB" || warn "工作目录可用 <20 GiB（raw + logs 预计数 GB）"

hdr "9. 拓扑 / GPU 映射（Gate 0）"
if [ "$FAIL" -eq 0 ]; then
  if "$PY" scripts/m27_run_all.py --tag "$TAG" --gpu-map --model "$MODEL" 2>&1 | sed 's/^/  /'; [ "${PIPESTATUS[0]}" -eq 0 ]; then
    ok "gpu_map 已写入 outputs/m27/configs/gpu_map_${TAG}.json"
  else
    bad "NV 互联 GPU <2 或模型 config 缺 max_position_embeddings"
  fi
else
  warn "前面已有 FAIL，跳过 gpu_map"
fi

hdr "10. 结论"
RESULT=$([ "$FAIL" -eq 0 ] && echo PASS || echo FAIL)
"$PY" - "$TAG" "$RESULT" "${FAILS[@]+"${FAILS[@]}"}" <<'PY'
import sys, json, os, socket, time
sys.path.insert(0, "scripts")
import m27_common as C
tag, result, fails = sys.argv[1], sys.argv[2], sys.argv[3:]
C.save_json(os.path.join(C.CONFIGS, f"preflight_{tag}.json"), {
    "tag": tag, "result": result, "fails": fails, "hostname": socket.gethostname(),
    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "env_fingerprint": C.env_fingerprint(),
    "frozen_plan_version": C.FROZEN["plan_version"]})
print(f"  preflight record -> outputs/m27/configs/preflight_{tag}.json (result={result})")
PY
if [ "$FAIL" -eq 0 ]; then
  echo "  >>> Gate 0 PASS。下一步: $PY scripts/m27_run_all.py --tag $TAG --transfer-test，然后 --stage e0"
else
  echo "  >>> Gate 0 FAIL：${FAILS[*]}"
  echo "  >>> 保留本日志，停下报告，不要开 formal。"
fi
exit "$FAIL"
