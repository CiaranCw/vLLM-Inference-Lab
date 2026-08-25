#!/usr/bin/env python3
"""M18 Gate 0 · W4A16(AWQ/GPTQ) 在本机 sm_120 上的可行性探针。

零成本：不下载任何模型、不建 vLLM 引擎、不跑 benchmark。
只做三件事：
  1. 读设备 capability 与量化方法的准入门槛；
  2. 直接调用 `choose_mp_linear_kernel()`，看真实会选中哪个 kernel（或拿到失败原因）；
  3. 用极小张量真跑一次选中 kernel 的底层 op —— 这是唯一能证明
     「预编译 wheel 里确实有 sm_120 cubin」的方法（VLLM_USE_PRECOMPILED=1，
     本项目不自行编译 CUDA，所以「源码支持」≠「本机二进制可用」）。

背景（为什么必须先探）：M10 已有先例 —— FP8 KV 在纸面上支持，实测却必须换 `TRITON_ATTN`
后端才能跑（FLASH_ATTN 不支持 fp8 KV、FLASHINFER 在 sm_120 上 JIT 失败）。
所以在向所有者申请下载量化模型之前，先用本脚本确认本机真的跑得动。

用法：python scripts/m18_probe.py
"""

import json
import os
import time
import traceback

import torch

OUT = "outputs/m18"

# 典型 W4A16 层配置：Qwen2.5 系列的 MLP 投影量级，group_size=128 是 AWQ/GPTQ 常见默认
LAYER_IN, LAYER_OUT = 4864, 896
GROUP_SIZE = 128


def probe_device():
    cap = torch.cuda.get_device_capability(0)
    return {
        "device_name": torch.cuda.get_device_name(0),
        "capability_tuple": list(cap),
        "capability_int": cap[0] * 10 + cap[1],
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "sm120_in_torch_arch_list": any("120" in a for a in torch.cuda.get_arch_list()),
    }


def probe_quant_configs():
    """各量化方法声明的最低 capability（纸面门槛）。"""
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    cap = torch.cuda.get_device_capability(0)
    cap_int = cap[0] * 10 + cap[1]
    out = {}
    for name, cls in (("AutoAWQ", AutoAWQConfig), ("AutoGPTQ", AutoGPTQConfig)):
        try:
            m = cls.get_min_capability()
            out[name] = {"min_capability": m, "device_passes_gate": cap_int >= m}
        except Exception as e:
            out[name] = {"error": f"{type(e).__name__}: {e}"}
    return out


def probe_kernel_choice():
    """真正的关键一步：让 vLLM 自己在本机选 kernel。"""
    from vllm.model_executor.kernels.linear import choose_mp_linear_kernel
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.scalar_type import scalar_types

    results = {}
    # AWQ: uint4 带 zero point；GPTQ(sym): uint4b8 不带 zero point
    cases = {
        "awq_w4a16_g128": dict(weight_type=scalar_types.uint4, zero_points=True,
                               has_g_idx=False),
        "gptq_w4a16_g128_sym": dict(weight_type=scalar_types.uint4b8, zero_points=False,
                                    has_g_idx=False),
        "gptq_w4a16_g128_desc_act": dict(weight_type=scalar_types.uint4b8,
                                         zero_points=False, has_g_idx=True),
    }
    for name, kw in cases.items():
        cfg = MPLinearLayerConfig(
            full_weight_shape=(LAYER_IN, LAYER_OUT),
            partition_weight_shape=(LAYER_IN, LAYER_OUT),
            act_type=torch.bfloat16,
            group_size=GROUP_SIZE,
            **kw,
        )
        try:
            kernel = choose_mp_linear_kernel(cfg)
            results[name] = {
                "chosen_kernel": kernel.__name__,
                "kernel_min_capability": kernel.get_min_capability(),
                "ok": True,
            }
        except Exception as e:
            results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return results


def probe_marlin_ops():
    """检查 Marlin 相关 op 是否存在于已安装的二进制中，并用极小张量真跑。

    `hasattr` 只能证明符号存在；只有真的 launch 一次，才知道 wheel 里有没有
    针对 sm_120 编译的 cubin（否则会报 "no kernel image is available for execution"）。
    repack 与 GEMM 必须分别验证：它们是两个独立的 kernel，可能各自做 arch 门控。
    """
    out = {"ops_present": {}, "smoke": {}}
    # 注意：本版本 GEMM 叫 marlin_gemm（不是旧名 gptq_marlin_gemm）；
    # torch.ops 是惰性命名空间，dir() 列不出来，必须逐个 getattr 探测。
    for op in ("marlin_gemm", "gptq_marlin_gemm", "gptq_marlin_repack",
               "awq_marlin_repack", "machete_mm", "gptq_gemm", "awq_gemm"):
        try:
            getattr(torch.ops._C, op)
            out["ops_present"][op] = True
        except AttributeError:
            out["ops_present"][op] = False

    # --- 真跑 1：AWQ marlin repack ---
    k, n, num_bits = 256, 128, 4
    pack_factor = 32 // num_bits
    packed = None
    try:
        qweight = torch.randint(
            0, 2**31 - 1, (k, n // pack_factor), dtype=torch.int32, device="cuda"
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        packed = torch.ops._C.awq_marlin_repack(qweight, k, n, num_bits, False)
        torch.cuda.synchronize()
        out["smoke"]["awq_marlin_repack"] = {
            "ok": True,
            "out_shape": list(packed.shape),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
        }
    except Exception as e:
        out["smoke"]["awq_marlin_repack"] = {
            "ok": False, "error": f"{type(e).__name__}: {e}",
            "traceback_tail": traceback.format_exc().strip().splitlines()[-3:],
        }

    # --- 真跑 2：marlin_gemm（决定性证据；张量用 vLLM 自己的 helper 构造）---
    try:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new,
            marlin_permute_scales,
        )
        from vllm.scalar_type import scalar_types

        if packed is None:
            raise RuntimeError("repack 未成功，无法构造 GEMM 输入")
        m, group_size = 16, 128
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        scales = torch.randn(
            (k // group_size, n), dtype=torch.bfloat16, device="cuda"
        ).abs() * 0.01
        scales = marlin_permute_scales(scales, k, n, group_size)
        workspace = marlin_make_workspace_new(a.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = torch.ops._C.marlin_gemm(
            a, None, packed, None, scales, None, None, None, None, None,
            workspace, scalar_types.uint4b8.id, m, n, k,
            True, False, True, False,
        )
        torch.cuda.synchronize()
        out["smoke"]["marlin_gemm"] = {
            "ok": True,
            "out_shape": list(c.shape),
            "finite": bool(torch.isfinite(c).all().item()),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 3),
            "note": "只验证 kernel 能在 sm_120 上 launch 并产出有限值；未做数值正确性比对",
        }
    except Exception as e:
        out["smoke"]["marlin_gemm"] = {
            "ok": False, "error": f"{type(e).__name__}: {e}",
            "traceback_tail": traceback.format_exc().strip().splitlines()[-3:],
        }
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {"probed_at": time.strftime("%F %T%z"), "purpose": "M18 W4A16 可行性 Gate 0"}

    report["device"] = probe_device()
    print("[m18] device:", json.dumps(report["device"], ensure_ascii=False))

    report["quant_config_gates"] = probe_quant_configs()
    print("[m18] 纸面门槛:", json.dumps(report["quant_config_gates"], ensure_ascii=False))

    report["kernel_choice"] = probe_kernel_choice()
    print("[m18] kernel 选择:")
    for k, v in report["kernel_choice"].items():
        print(f"    {k}: {json.dumps(v, ensure_ascii=False)}")

    report["marlin_ops"] = probe_marlin_ops()
    print("[m18] 二进制 op:", json.dumps(report["marlin_ops"]["ops_present"], ensure_ascii=False))
    print("[m18] kernel smoke:", json.dumps(report["marlin_ops"]["smoke"], ensure_ascii=False))

    chosen = {k: v.get("chosen_kernel") for k, v in report["kernel_choice"].items()}
    smoke = report["marlin_ops"]["smoke"]
    smoke_ok = all(smoke.get(k, {}).get("ok") for k in ("awq_marlin_repack", "marlin_gemm"))
    report["verdict"] = {
        "all_cases_have_kernel": all(v.get("ok") for v in report["kernel_choice"].values()),
        "chosen_kernels": chosen,
        "binary_kernel_launch_ok": smoke_ok,
        "smoke_detail": {k: smoke.get(k, {}).get("ok") for k in smoke},
        "conclusion": (
            "GO：本机能选出 W4A16 kernel 且二进制可在 sm_120 上真实 launch"
            if all(v.get("ok") for v in report["kernel_choice"].values()) and smoke_ok
            else "NO-GO / 需进一步排查：见 kernel_choice 与 smoke 的原始报错"
        ),
    }
    path = os.path.join(OUT, "gate0_w4a16_probe.json")
    with open(path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n[m18] 结论:", report["verdict"]["conclusion"])
    print("[m18] ->", path)


if __name__ == "__main__":
    main()
