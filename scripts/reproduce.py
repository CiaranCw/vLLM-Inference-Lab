"""统一复现入口（M0 收口产物之一）。

把散落在各模块 notes 里的复现命令收敛到 `config/repro_manifest.yaml`，并提供四种模式：

    python scripts/reproduce.py --list
    python scripts/reproduce.py --module m17 --dry-run       # 只打印，不建引擎、不写文件
    python scripts/reproduce.py --module m17 --verify-only   # 查环境 + 查产物 + 跑不变量
    python scripts/reproduce.py --module m17 --run --confirm-gpu

安全设计（对应 AGENTS.md 的硬规则）：
  - `--run` 默认只跑 status != done 的步骤，不会重跑并覆盖已核验的数据；
    要重跑必须显式加 `--include-done`（会打印将被覆盖的文件并要求 `--confirm-overwrite`）。
  - 任何 `creates_engine=true` 的步骤都单独起一个子进程（vLLM 的 KV 显存
    `del engine` + `empty_cache()` 归还不掉，同进程建第二个引擎会失败）。
  - `--run` 涉及 GPU 时必须显式 `--confirm-gpu`；本工具不自动判定 GPU 是否空闲，
    因为 WSL2 下 `nvidia-smi --query-compute-apps` 可能看不到真实占用进程。
  - 子进程失败立即停止，不用 `|| true` 之类掩盖。
  - 本工具不做任何 git 操作（不 commit / 不 push）。
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import subprocess
import sys
import time

import yaml  # PyYAML 随 vLLM 进入本 venv，非新增依赖

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "config", "repro_manifest.yaml")
RUN_LOG_DIR = os.path.join(ROOT, "logs", "repro")

# `--run` 的默认范围。刻意不含 `optional`：可选扩展必须用 `--only <id>` 显式点名
# 才会执行，避免"顺手把可选项也跑了"，也避免它们让主线看起来没做完。
RUNNABLE_STATUS = {"implemented", "in_progress"}
OPTIONAL_STATUS = {"optional"}


def load_manifest():
    with open(MANIFEST) as f:
        return yaml.safe_load(f)


def _p(rel):
    return os.path.join(ROOT, rel)


def steps_of(mod):
    return mod.get("steps", []) or []


# --------------------------------------------------------------------------- #
# --list
# --------------------------------------------------------------------------- #
def cmd_list(man):
    print(f"manifest : {os.path.relpath(MANIFEST, ROOT)} (version {man['version']})")
    pj = man["project"]
    print(f"vLLM     : {pj['vllm_commit']} ({pj['vllm_version']})  model={pj['model']}")
    print(f"platform : {pj['platform']}  enforce_eager={pj['enforce_eager']}\n")
    print(f"{'module':<8} {'GPU':<5} {'status':<12} {'steps':<7} title")
    print("-" * 78)
    for name, mod in man["modules"].items():
        st = steps_of(mod)
        pend = len([s for s in st if s.get("status") in RUNNABLE_STATUS])
        gpu = "yes" if mod.get("requires_gpu") else "no"
        print(
            f"{name:<8} {gpu:<5} {mod.get('status', '?'):<12} "
            f"{len(st):<7} {mod.get('title', '')}"
            + (f"   [{pend} 步待执行]" if pend else "")
        )
    # 这里混着三类：明确未做（optional_todo）、可选但已跑（optional）、以及已完成的遗留登记（done）。
    # 每项都带自己的 status，不要靠标题去推断执行状态。
    print("\n开放登记项与已知缺口（状态以每项的 status 为准，如实标注）：")
    any_open = False
    for name, mod in man["modules"].items():
        for item in mod.get("open_items", []) or []:
            any_open = True
            print(f"  - {name} · {item['id']}: {item['status']} —— {item['desc']}")
        for gap in mod.get("known_gaps", []) or []:
            any_open = True
            print(f"  - {name} · known_gap: {gap}")
    if not any_open:
        print("  （无）")
    return 0


# --------------------------------------------------------------------------- #
# --dry-run
# --------------------------------------------------------------------------- #
def cmd_dry_run(man, module, only):
    mod = man["modules"][module]
    env = man["env"]
    print(f"# module: {module} —— {mod.get('title', '')}")
    print("# DRY-RUN：以下命令**不会被执行**；不建引擎、不写文件、不需要 GPU。\n")
    print("cd " + ROOT)
    print("source .venv/bin/activate")
    for k, v in env["required"].items():
        print(f"export {k}={v}")
    for k in env["must_be_unset_or_zero"]:
        print(f"unset {k}   # 或设为 0")
    print()
    for s in steps_of(mod):
        if only and s["id"] not in only:
            continue
        tag = s.get("status", "?")
        gpu = "GPU" if s.get("requires_gpu") else "cpu"
        print(f"# [{tag:<11}] [{gpu}] {s['id']}: {s.get('desc', '').strip()}")
        if s.get("requires_gpu"):
            print("#   ↑ 跑前先 nvidia-smi 确认 GPU 空闲；一个进程一个引擎")
        print(shlex.join(s["cmd"]))
        outs = (s.get("outputs") or []) + (s.get("optional_outputs") or [])
        for o in outs:
            exists = "已存在" if os.path.exists(_p(o)) else "尚未产生"
            print(f"#   -> {o}  ({exists})")
        if s.get("log"):
            print(f"#   log: {s['log']}")
        print()
    shown = [s for s in steps_of(mod) if not only or s["id"] in only]
    n_gpu = len([s for s in shown if s.get("creates_engine")])
    n_gpu_pending = len(
        [
            s
            for s in shown
            if s.get("creates_engine") and s.get("status") in RUNNABLE_STATUS
        ]
    )
    print(
        f"# 上列共 {len(shown)} 步、其中 {n_gpu} 步会创建 vLLM 引擎（每步一个独立进程）。"
    )
    print(
        f"# `--run` 默认只跑 status!=done 的步骤 → 实际将创建 {n_gpu_pending} 个引擎进程。"
    )
    return 0


# --------------------------------------------------------------------------- #
# --verify-only
# --------------------------------------------------------------------------- #
def check_env(man):
    ok = True
    env = man["env"]
    print("== 环境变量 ==")
    for k, v in env["required"].items():
        got = os.environ.get(k)
        good = got == str(v)
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {k}={got!r}（期望 {v!r}）")
    for k in env["must_be_unset_or_zero"]:
        got = os.environ.get(k)
        good = got in (None, "", "0")
        ok &= good
        print(f"  [{'ok' if good else 'FAIL'}] {k}={got!r}（须未设置或为 0）")
    return ok


def check_vllm_src(man):
    """只读核对固定的 vLLM 源码 commit —— 不做任何写操作。"""
    src = os.path.normpath(os.path.join(ROOT, man["project"]["vllm_src"]))
    print("== vLLM 源码 ==")
    if not os.path.isdir(src):
        print(f"  [FAIL] {src} 不存在")
        return False
    try:
        head = subprocess.run(
            ["git", "-C", src, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [FAIL] 无法读取 HEAD: {e}")
        return False
    want = man["project"]["vllm_commit"]
    good = head.startswith(want)
    print(f"  [{'ok' if good else 'FAIL'}] HEAD={head[:12]}（期望 {want}）")
    return good


def check_artifacts(man, module):
    mod = man["modules"][module]
    print("== 产物 ==")
    missing_done, pending = [], []
    for s in steps_of(mod):
        for o in s.get("outputs") or []:
            if os.path.exists(_p(o)):
                continue
            if s.get("status") == "done":
                missing_done.append((s["id"], o))
            else:
                pending.append((s["id"], s.get("status"), o))
    for f in man["modules"][module].get("artifacts", {}).get("required_files", []):
        if not os.path.exists(_p(f)):
            missing_done.append(("artifacts", f))
    if missing_done:
        for sid, o in missing_done:
            print(f"  [FAIL] {sid}: 标记为 done 但产物缺失 -> {o}")
    if pending:
        for sid, st, o in pending:
            tag = "OPTIONAL" if st in OPTIONAL_STATUS else "PENDING"
            print(f"  [{tag}] {sid} ({st}): 尚未产生 -> {o}")
    if not missing_done and not pending:
        print("  [ok] 全部已登记产物均存在")
    elif not missing_done:
        print(
            "  [ok] 主线步骤的产物齐全"
            "（上列 OPTIONAL 为可选扩展，未执行不影响主线结论）"
        )
    return not missing_done


def cmd_verify_only(man, module):
    ok = True
    ok &= check_env(man)
    ok &= check_vllm_src(man)
    ok &= check_artifacts(man, module)
    print("\n== 正式不变量检查（scripts/check_invariants.py）==", flush=True)
    sys.stdout.flush()  # 子进程直接写 fd，不 flush 会导致输出顺序错乱
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "scripts", "check_invariants.py"),
            "--module",
            module,
        ],
        cwd=ROOT,
        check=False,
    )
    ok &= r.returncode == 0
    print(f"\n结论：{'PASS' if ok else 'FAIL'}（--verify-only 不执行任何实验）")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
# --run
# --------------------------------------------------------------------------- #
def cmd_run(man, module, only, include_done, confirm_gpu, confirm_overwrite):
    mod = man["modules"][module]
    sel = []
    for s in steps_of(mod):
        if only and s["id"] not in only:
            continue
        if not only and not include_done and s.get("status") not in RUNNABLE_STATUS:
            continue
        sel.append(s)
    if not sel:
        print(
            "没有需要执行的步骤（默认跳过 status=done；用 --include-done 可强制重跑）。"
        )
        return 0

    needs_gpu = [s for s in sel if s.get("requires_gpu")]
    if needs_gpu and not confirm_gpu:
        print("以下步骤需要 GPU，必须显式加 --confirm-gpu 才会执行：")
        for s in needs_gpu:
            print(f"  - {s['id']}")
        print(
            "\n跑之前请先人工执行 `nvidia-smi` 确认 GPU 空闲。"
            "本工具不代替这一步：WSL2 下 --query-compute-apps 可能看不到真实占用。"
        )
        return 2

    will_overwrite = [
        (s["id"], o)
        for s in sel
        for o in (s.get("outputs") or [])
        if os.path.exists(_p(o))
    ]
    if will_overwrite and not confirm_overwrite:
        print("以下已存在的产物会被覆盖，必须显式加 --confirm-overwrite：")
        for sid, o in will_overwrite:
            print(f"  - {sid}: {o}")
        print("\n（旧数据只备份不删除：请先自行备份到 outputs/<mod>/raw_old_<date>/）")
        return 2

    os.makedirs(RUN_LOG_DIR, exist_ok=True)
    stamp = dt.datetime.now(dt.UTC).astimezone().strftime("%Y%m%dT%H%M%S")
    meta = {
        "started_at": stamp,
        "module": module,
        "steps": [s["id"] for s in sel],
        "env": {
            k: os.environ.get(k)
            for k in list(man["env"]["required"]) + man["env"]["must_be_unset_or_zero"]
        },
        "lab_head": _git(["git", "-C", ROOT, "rev-parse", "HEAD"]),
        "vllm_src_head": _git(
            [
                "git",
                "-C",
                os.path.join(ROOT, man["project"]["vllm_src"]),
                "rev-parse",
                "HEAD",
            ]
        ),
        "results": [],
    }
    meta_path = os.path.join(RUN_LOG_DIR, f"run_{module}_{stamp}.json")

    rc = 0
    for s in sel:
        print(f"\n=== [{s['id']}] {shlex.join(s['cmd'])} ===", flush=True)
        t0 = time.monotonic()
        # 一步一个子进程：creates_engine 的步骤必须彼此隔离
        proc = subprocess.run(s["cmd"], cwd=ROOT, check=False)
        took = time.monotonic() - t0
        meta["results"].append(
            {
                "id": s["id"],
                "cmd": s["cmd"],
                "returncode": proc.returncode,
                "seconds": round(took, 2),
            }
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if proc.returncode != 0:
            print(
                f"\n[STOP] 步骤 {s['id']} 失败（returncode={proc.returncode}），"
                "立即停止，不继续后续步骤。"
            )
            rc = proc.returncode
            break
    print(f"\n运行元数据已写入 {os.path.relpath(meta_path, ROOT)}")
    print("提醒：本工具不做 git commit / push。")
    return rc


def _git(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 —— 元数据缺失不应中断实验
        return None


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="统一复现入口（读 config/repro_manifest.yaml）"
    )
    p.add_argument("--module", help="模块名，如 m17")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true", help="列出模块、GPU 要求与状态")
    g.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    g.add_argument("--verify-only", action="store_true", help="查环境/产物 + 跑不变量")
    g.add_argument("--run", action="store_true", help="真正执行（需 --confirm-gpu）")
    p.add_argument("--only", action="append", help="只处理指定 step id（可重复）")
    p.add_argument(
        "--include-done", action="store_true", help="允许重跑 status=done 的步骤"
    )
    p.add_argument("--confirm-gpu", action="store_true", help="确认已人工检查 GPU 空闲")
    p.add_argument(
        "--confirm-overwrite", action="store_true", help="确认允许覆盖已有产物"
    )
    a = p.parse_args()

    man = load_manifest()
    if a.list or not (a.dry_run or a.verify_only or a.run):
        return cmd_list(man)
    if not a.module:
        p.error("--dry-run / --verify-only / --run 需要 --module")
    if a.module not in man["modules"]:
        p.error(f"未知模块 {a.module}（可选：{', '.join(man['modules'])}）")

    if a.dry_run:
        return cmd_dry_run(man, a.module, set(a.only or []))
    if a.verify_only:
        return cmd_verify_only(man, a.module)
    return cmd_run(
        man,
        a.module,
        set(a.only or []),
        a.include_done,
        a.confirm_gpu,
        a.confirm_overwrite,
    )


if __name__ == "__main__":
    sys.exit(main())
