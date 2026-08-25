"""M22 · 独立的 torch.distributed / NCCL 冒烟测试（不 import vllm）。

用途：在启动 vLLM TP 之前，用最小代价把故障域切开——

    · 本脚本失败  → 是机器/驱动/NCCL/拓扑的问题；
    · 本脚本通过而 vLLM TP 失败 → 是 vLLM 配置或实现的问题。

没有这一步，TP 起不来时会分不清该查哪边，而 4 小时机时经不起这种消耗。

    python scripts/m22_nccl_smoke.py --world-size 4
    python scripts/m22_nccl_smoke.py --world-size 2      # 降级排查

退出码 0 = 通过。所有 rank 的 all-reduce 结果都会逐一校验数值，
不是「跑完没报错就算过」。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time


def _worker(rank: int, world_size: int, sizes_mib: list[float], out_path: str):
    import torch
    import torch.distributed as dist

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=120),
    )
    dev = torch.device(f"cuda:{rank}")
    rec = {
        "rank": rank,
        "world_size": world_size,
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev),
        "capability": list(torch.cuda.get_device_capability(dev)),
        "backend": dist.get_backend(),
        "cases": [],
        "ok": True,
        "err": "",
    }
    try:
        # 正确性：每个 rank 填 (rank+1)，all-reduce SUM 后每个元素应为 Σ(1..world_size)
        expect = world_size * (world_size + 1) // 2
        for mib in sizes_mib:
            n = max(1, int(mib * 1024 * 1024 / 2))  # bf16 = 2 bytes
            t = torch.full((n,), float(rank + 1), dtype=torch.bfloat16, device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(dev)
            got = float(t[0].item())
            correct = abs(got - expect) < 1e-3
            # 计时：warmup 后取多次的中位数
            for _ in range(3):
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
            torch.cuda.synchronize(dev)
            dist.barrier()
            ts = []
            for _ in range(10):
                t.fill_(float(rank + 1))
                torch.cuda.synchronize(dev)
                t0 = time.perf_counter()
                dist.all_reduce(t, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(dev)
                ts.append((time.perf_counter() - t0) * 1000)
            ts.sort()
            rec["cases"].append(
                {
                    "payload_MiB": mib,
                    "numel": n,
                    "expected": expect,
                    "got": got,
                    "correct": correct,
                    "allreduce_ms_median": round(ts[len(ts) // 2], 4),
                    "allreduce_ms_min": round(ts[0], 4),
                }
            )
            if not correct:
                rec["ok"] = False
                rec["err"] += f"payload {mib}MiB 数值错误: got={got} expect={expect}; "
    except Exception as e:  # noqa: BLE001
        rec["ok"] = False
        rec["err"] += repr(e)
    finally:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001, S110
            pass
    with open(f"{out_path}.rank{rank}", "w") as f:
        json.dump(rec, f, ensure_ascii=False)


def main():
    p = argparse.ArgumentParser(description="独立 NCCL 冒烟（不 import vllm）")
    p.add_argument("--world-size", type=int, default=4)
    p.add_argument(
        "--sizes-mib",
        default="0.03,0.25,2,8,32",
        help="all-reduce payload（MiB）。刻意跨越 vLLM custom all-reduce 的 8MiB 门槛"
        "（custom_all_reduce.py:230-243），便于观察不同 size 的表现差异",
    )
    p.add_argument("--out", default="outputs/m22/nccl_smoke")
    a = p.parse_args()

    import torch
    import torch.multiprocessing as mp

    n = torch.cuda.device_count()
    if n < a.world_size:
        print(f"[nccl_smoke][FAIL] 只有 {n} 张 GPU，要求 {a.world_size}")
        return 1
    sizes = [float(x) for x in a.sizes_mib.split(",")]
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    print(f"[nccl_smoke] world_size={a.world_size} payloads(MiB)={sizes}")

    t0 = time.perf_counter()
    mp.spawn(_worker, args=(a.world_size, sizes, a.out), nprocs=a.world_size, join=True)
    took = time.perf_counter() - t0

    recs = []
    for r in range(a.world_size):
        pth = f"{a.out}.rank{r}"
        if not os.path.exists(pth):
            print(f"[nccl_smoke][FAIL] rank {r} 没有产出结果")
            return 1
        with open(pth) as f:
            recs.append(json.load(f))
        os.remove(pth)
    with open(f"{a.out}.json", "w") as f:
        json.dump(
            {"elapsed_s": round(took, 2), "ranks": recs},
            f,
            ensure_ascii=False,
            indent=2,
        )

    allok = all(r["ok"] for r in recs)
    print(
        f"\n{'rank':<5} {'device':<8} {'name':<24} {'payload(MiB)':>12} {'ms':>9} {'ok':>5}"
    )
    for r in recs:
        for c in r["cases"]:
            print(
                f"{r['rank']:<5} {r['device']:<8} {r['device_name'][:24]:<24} "
                f"{c['payload_MiB']:>12} {c['allreduce_ms_median']:>9} {c['correct']!s:>5}"
            )
    print(f"\n[nccl_smoke] elapsed={took:.1f}s -> {a.out}.json")
    if allok:
        print("[nccl_smoke][PASS] 所有 rank 的 all-reduce 数值校验通过")
        return 0
    for r in recs:
        if not r["ok"]:
            print(f"[nccl_smoke][FAIL] rank {r['rank']}: {r['err']}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
