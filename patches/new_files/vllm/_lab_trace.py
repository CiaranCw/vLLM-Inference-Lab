"""vLLM Inference Lab —— 统一运行时 Trace 工具（学习用，默认完全静默）。

设计约束（务必保持）：
- 仅当环境变量 VLLM_LAB_TRACE=1 时才输出；默认关闭时零副作用、零输出。
- 只打印 shape / dtype / 标量，绝不打印完整 tensor 值。
- 不修改任何函数返回值、不吞掉业务异常、不改变调度或执行逻辑。
- 工具自身的异常一律吞掉（trace 不得影响推理正确性）。
- 支持：全局 step 计数（由 scheduler 递增）、限制最大 step 数、只追踪指定 layer、按 key 去重。

环境变量：
    VLLM_LAB_TRACE            = 0/1     默认 0（关闭）
    VLLM_LAB_TRACE_MAX_STEPS  = int     默认 5，只追踪前 N 个调度 step
    VLLM_LAB_TRACE_LAYER      = int     默认 0，attention 只追踪该 layer

输出格式（写 stderr，带 flush）：
    [VLLM_LAB_TRACE] pid=<pid> step=<n> <component>/<event> k1=v1 k2=v2 ...
"""

from __future__ import annotations

import os
import sys

_PREFIX = "[VLLM_LAB_TRACE]"

# 全局调度 step 计数（在 Scheduler.schedule 中通过 bump_step() 递增）
_step: int = 0
# once_key 去重集合
_seen: set = set()

# 实验性：被「阻塞」的 request_id 集合（供 scheduler 的实验性 skip 钩子读取）。
# 仅在 VLLM_LAB_BLOCK=1 时才对调度产生影响；默认空集 + 默认关闭 = 零影响。
_BLOCKED: set = set()


def _truthy(name: str, default: str = "0") -> bool:
    try:
        return bool(int(os.environ.get(name, default)))
    except Exception:
        return False


def enabled() -> bool:
    return _truthy("VLLM_LAB_TRACE", "0")


def _max_steps() -> int:
    try:
        return int(os.environ.get("VLLM_LAB_TRACE_MAX_STEPS", "5"))
    except Exception:
        return 5


def _target_layer() -> int:
    try:
        return int(os.environ.get("VLLM_LAB_TRACE_LAYER", "0"))
    except Exception:
        return 0


def bump_step() -> int:
    """在每次 Scheduler.schedule() 开始时调用，推进全局 step。"""
    global _step
    _step += 1
    return _step


def current_step() -> int:
    return _step


# ---- 实验性「按请求 block」API（默认关闭；仅 E5 类实验使用）----
def block_enabled() -> bool:
    """实验性 skip 钩子总开关。未设 VLLM_LAB_BLOCK=1 时永远返回 False。"""
    return _truthy("VLLM_LAB_BLOCK", "0")


def set_blocked(ids) -> None:
    """由 driver 在进程内调用：把这些 request_id 标记为本步应被 scheduler 跳过。"""
    global _BLOCKED
    _BLOCKED = set(ids)


def clear_blocked() -> None:
    global _BLOCKED
    _BLOCKED = set()


def blocked_ids() -> set:
    """scheduler 钩子调用：仅当 VLLM_LAB_BLOCK=1 时返回被阻塞集合，否则空集。"""
    if not block_enabled():
        return set()
    return _BLOCKED


def layer_index(layer_name: str | None) -> int | None:
    """从 'model.layers.7.self_attn.attn' 之类的名字中解析出层号。"""
    if not layer_name:
        return None
    parts = str(layer_name).split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except Exception:
                return None
    return None


def _shape_of(v):
    """把 tensor-like 转成安全摘要，绝不打印数值。"""
    try:
        import torch  # local import; 避免任何 import 期副作用

        if isinstance(v, torch.Tensor):
            return f"T{tuple(v.shape)}:{str(v.dtype).replace('torch.', '')}"
    except Exception:
        pass
    return v


def _fmt(fields: dict) -> str:
    out = []
    for k, v in fields.items():
        out.append(f"{k}={_shape_of(v)}")
    return " ".join(out)


def _components() -> set:
    """VLLM_LAB_TRACE_COMPONENTS=scheduler,model_runner → 只发这些 component；
    未设置/为空 → 不过滤。"""
    raw = os.environ.get("VLLM_LAB_TRACE_COMPONENTS", "").strip()
    if not raw:
        return set()
    return {c.strip() for c in raw.split(",") if c.strip()}


def _jsonl_path():
    """VLLM_LAB_TRACE_JSONL=<path> → 事件同时以 JSON 行写入该文件。"""
    p = os.environ.get("VLLM_LAB_TRACE_JSONL", "").strip()
    return p or None


def _jsonable(fields: dict) -> dict:
    """把 fields 转成 JSON 可序列化：tensor→shape 字符串，其余尽量原样、否则 str()。"""
    import json as _json

    out = {}
    for k, v in (fields or {}).items():
        s = _shape_of(v)
        try:
            _json.dumps(s)
            out[k] = s
        except Exception:
            out[k] = str(s)
    return out


def trace_event(
    component: str,
    event: str,
    fields: dict | None = None,
    *,
    layer=None,
    once_key: str | None = None,
) -> None:
    """输出一条 trace。默认关闭；开启后受 step 上限 / layer / once_key 限制。

    layer 可为 int 或 layer_name 字符串；为 None 时不做 layer 过滤。
    """
    try:
        if not enabled():
            return
        if _step > _max_steps():
            return
        comps = _components()
        if comps and component not in comps:
            return
        if layer is not None:
            idx = layer if isinstance(layer, int) else layer_index(layer)
            if idx is None or idx != _target_layer():
                return
        if once_key is not None:
            if once_key in _seen:
                return
            _seen.add(once_key)

        msg = f"{_PREFIX} pid={os.getpid()} step={_step} {component}/{event}"
        if fields:
            msg += " " + _fmt(fields)
        print(msg, file=sys.stderr, flush=True)

        jp = _jsonl_path()
        if jp:
            import json as _json

            rec = {
                "pid": os.getpid(),
                "step": _step,
                "component": component,
                "event": event,
                "fields": _jsonable(fields or {}),
            }
            with open(jp, "a", encoding="utf-8") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        # trace 绝不能影响推理
        pass
