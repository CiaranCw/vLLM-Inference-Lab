#!/usr/bin/env python3
"""M28 · ../vllm-src 脏文件的 AST 语义比对（只读：git show HEAD:<file> vs 工作树文件）。

对每个已修改的 .py：去掉所有「裸字符串语句」（docstring / 中文注释块），再比较 ast.unparse；
仍不同的，列出差异中出现的 _lab_trace / VLLM_LAB_* / M9 门控标识，供人工核对它们在默认环境下是否惰性。
不改 ../vllm-src 的任何东西，不跑 git 写操作。

用法：python scripts/m28_source_audit.py [--out outputs/m28/m28_vllm_src_ast_audit.json]
"""

import argparse
import ast
import difflib
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m28_common as C  # noqa: E402

GATES = re.compile(r"_lab_trace|VLLM_LAB_\w+|lab_trace\.\w+|bump_step|blocked_ids|trace_event|M9_COALESCE|coalesce")
M28_PATHS = {
    "A_offload": ["vllm/v1/core/sched/scheduler.py", "vllm/v1/core/kv_cache_manager.py", "vllm/v1/core/block_pool.py",
                  "vllm/v1/worker/gpu_model_runner.py", "vllm/v1/attention/backends/flash_attn.py"],
    "B_lifecycle": ["vllm/v1/engine/core.py", "vllm/v1/engine/core_client.py"],
}


class StripStr(ast.NodeTransformer):
    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if isinstance(body, list):
                kept = [n for n in body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                                                and isinstance(n.value.value, str))]
                setattr(node, field, kept or [ast.Pass()])
        return node


def norm(src):
    return ast.unparse(StripStr().visit(ast.parse(src)))


def git(args):
    return subprocess.check_output(["git"] + args, cwd=C.VLLM_SRC, text=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(C.OUT, "m28_vllm_src_ast_audit.json"))
    a = ap.parse_args()
    head = git(["rev-parse", "HEAD"]).strip()
    status = [line for line in git(["status", "--short"]).splitlines() if line.strip()]
    files = []
    for line in status:
        st, path = line[:2].strip(), line[3:]
        rec = {"path": path, "status": st}
        if st == "M" and path.endswith(".py"):
            old = git(["show", f"HEAD:{path}"])
            with open(os.path.join(C.VLLM_SRC, path)) as f:
                new = f.read()
            a_old, a_new = norm(old), norm(new)
            rec["semantic_equal"] = a_old == a_new
            if not rec["semantic_equal"]:
                diff = [d for d in difflib.unified_diff(a_old.splitlines(), a_new.splitlines(), lineterm="", n=0)
                        if d.startswith(("+", "-")) and not d.startswith(("+++", "---"))]
                rec["changed_lines"] = len(diff)
                rec["gate_tokens"] = sorted(set(m for d in diff for m in GATES.findall(d)))
                rec["ungated_changed_lines"] = [d[:160] for d in diff if not GATES.search(d)][:40]
        files.append(rec)
    out = {"vllm_src_head": head, "dirty_count": len(status), "files": files,
           "m28_relevant": {k: [f for f in files if f["path"] in v] for k, v in M28_PATHS.items()},
           "offloading_files_dirty": [f["path"] for f in files if "offload" in f["path"] or "kv_connector" in f["path"]]}
    C.save_json(a.out, out)
    for f in files:
        tag = "≡" if f.get("semantic_equal") else ("Δ" if "semantic_equal" in f else f["status"])
        extra = f" gates={f.get('gate_tokens')} ungated={len(f.get('ungated_changed_lines', []))}" if tag == "Δ" else ""
        print(f"  {tag} {f['path']}{extra}")
    print(f"HEAD={head[:9]} dirty={len(status)} offloading_files_dirty={out['offloading_files_dirty']} → {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
