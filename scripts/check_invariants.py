"""正式不变量 / 回归检查（M0 收口产物之一）——不需 GPU、不 import vllm。

把此前散落在各模块 notes 里的「跑完逐条核对」固化成可执行断言，覆盖三类核查：

  1. 逐样本不变量：直接读 `outputs/mN/raw/*.jsonl`（权威证据），核对守恒律与字段语义；
  2. CSV ↔ JSONL 一致性：汇总 CSV 的 n 必须等于 JSONL 中 `phase=measure and success=True`
     的样本数——防止「改了脚本只重跑汇总、没重跑实验」；
  3. 文档数字可溯源：把 `notes/m17/results.md` 主结果表里的每个数字反向核对回 CSV。
     M17 审计的教训是「自动核对只覆盖表格、没覆盖正文叙述句」，故这里同时做正向核对
     （由 CSV 生成的数字串必须出现在文档中，含正文关键句）。

三种结论，语义严格区分（诚实性红线）：
  PASS    —— 已执行且通过
  FAIL    —— 已执行但违反 → 非零退出
  PENDING —— 实验尚未执行（产物不存在且 manifest 标注为 implemented）→ 不算通过，也不算失败

用法：
    python scripts/check_invariants.py                # 检查 manifest 里登记的全部模块
    python scripts/check_invariants.py --module m17
    python scripts/check_invariants.py --verbose      # 逐条打印 PASS 明细
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import py_compile
import re
import statistics
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCK_SIZE = 16  # vLLM KV block size（命中以完整 block 为单位）
E1_PREFIX_LENS = [256, 512, 1024, 2048]
E1_SUFFIX_LEN = 16
E1_EXPECT_N = 15
E3_PROMPT_LEN = 2048
E3_EXPECT_N = 15
E3_EXPECTED_STEPS = {512: 4, 1024: 2, 2048: 1, 4096: 1}
CONC_EXPECT_N = 15
I5B_PREFIX_LEN = 2048
I5B_SUFFIX_LEN = 16
I5B_MAX_TOKENS = [1, 8, 32, 128]
I5B_EXPECT_N = 15
CORR_EXPECT_N = 3


# --------------------------------------------------------------------------- #
# 结果收集
# --------------------------------------------------------------------------- #
class Report:
    def __init__(self):
        self.rows: list[
            tuple[str, str, str, str]
        ] = []  # (status, module, rule, detail)

    def ok(self, module, rule, detail=""):
        self.rows.append(("PASS", module, rule, detail))

    def fail(self, module, rule, detail):
        self.rows.append(("FAIL", module, rule, detail))

    def pending(self, module, rule, detail):
        self.rows.append(("PENDING", module, rule, detail))

    def check(self, module, rule, cond, detail_fail, detail_ok=""):
        if cond:
            self.ok(module, rule, detail_ok)
        else:
            self.fail(module, rule, detail_fail)
        return cond

    @property
    def failures(self):
        return [r for r in self.rows if r[0] == "FAIL"]

    @property
    def pendings(self):
        return [r for r in self.rows if r[0] == "PENDING"]


# --------------------------------------------------------------------------- #
# 读取工具
# --------------------------------------------------------------------------- #
def _p(*parts):
    return os.path.join(ROOT, *parts)


def load_jsonl(rel):
    path = _p(rel)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_json(rel):
    path = _p(rel)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_csv(rel):
    path = _p(rel)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return list(csv.DictReader(f))


def measure_rows(rows):
    """汇总口径：只用 phase=measure 且 success=True。"""
    return [r for r in rows if r.get("phase") == "measure" and r.get("success") is True]


def failed_measure(rows):
    """measure 阶段里 success=False 的样本——必须显式暴露，不能被静默丢弃。"""
    return [
        r for r in rows if r.get("phase") == "measure" and r.get("success") is not True
    ]


def boot_ci_median_diff(a, b, iters=10000, seed=0):
    """与 scripts/m17_prefix_cache.py 完全同算法、同种子，故结果可逐位复现。"""
    import random

    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        ra = [a[rng.randrange(len(a))] for _ in range(len(a))]
        rb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(statistics.median(ra) - statistics.median(rb))
    diffs.sort()
    return (
        statistics.median(a) - statistics.median(b),
        diffs[int(0.025 * (iters - 1))],
        diffs[int(0.975 * (iters - 1))],
    )


# --------------------------------------------------------------------------- #
# Markdown 表格解析（用于文档数字的反向核对）
# --------------------------------------------------------------------------- #
def _norm_num(s):
    """把文档里的数字单元格标准化成 float。

    注意：文档用的是 U+2212 MINUS SIGN「−」而不是 ASCII '-'，直接 float() 会炸。
    """
    s = s.replace("*", "").replace("`", "").strip()
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    m = re.match(r"^[+]?(-?\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else None


def _decimals(s):
    """文档单元格里写了几位小数（决定该处应有的容差）。"""
    s = s.replace("*", "").replace("`", "").strip()
    s = s.replace("\u2212", "-").replace("\u2013", "-")
    m = re.match(r"^[+]?-?\d+(?:\.(\d+))?", s)
    return len(m.group(1)) if (m and m.group(1)) else 0


def _render_candidates(v, decimals):
    """一个数在给定小数位下可接受的书写形式。

    浮点二进制表示会让 .5 边界两可（例如 58.45 的 f-string 给 '58.4'，而人手写
    通常是 '58.5'）——两者都是对同一 CSV 值的正确舍入，故都接受，避免把书写差异
    误报成数据对不上。
    """
    scale = 10**decimals
    half_up = (int(v * scale + (0.5 if v >= 0 else -0.5))) / scale
    return {f"{v:.{decimals}f}", f"{half_up:.{decimals}f}"}


def _num_matches(doc_cell, csv_value):
    """判据：文档数字必须等于 CSV 数字在文档自身精度下的正确舍入。

    不同列的书写精度不同（例如 E1 表的延迟写 2 位小数、相对降幅写 1 位小数），
    统一用固定容差会把「正确的四舍五入」误判成「数字对不上」。
    """
    v = _norm_num(doc_cell)
    if v is None:
        return False
    tol = 0.5 * (10 ** -_decimals(doc_cell)) + 1e-9
    return abs(v - csv_value) <= tol


def parse_md_tables(text):
    """返回文中所有表格：[(header_cells, [body_row_cells, ...]), ...]。"""
    tables, block = [], []
    for line in text.splitlines():
        if line.lstrip().startswith("|"):
            block.append(line)
        else:
            if len(block) >= 3:
                tables.append(block)
            block = []
    if len(block) >= 3:
        tables.append(block)

    out = []
    for block in tables:
        cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in block]
        header = cells[0]
        body = [c for c in cells[2:] if not all(x.startswith("---") for x in c)]
        out.append((header, body))
    return out


def find_table(tables, *anchors):
    """按表头包含的关键词定位一张表。"""
    for header, body in tables:
        joined = " ".join(header)
        if all(a in joined for a in anchors):
            return body
    return None


# --------------------------------------------------------------------------- #
# M17 · E1（APC 长共享前缀）
# --------------------------------------------------------------------------- #
def check_m17_e1(rep):
    M = "m17/E1"
    on = load_jsonl("outputs/m17/raw/e1_on.jsonl")
    off = load_jsonl("outputs/m17/raw/e1_off.jsonl")
    if on is None or off is None:
        rep.pending(M, "artifacts", "outputs/m17/raw/e1_{on,off}.jsonl 不存在")
        return
    allrows = on + off
    bad = failed_measure(allrows)
    rep.check(
        M,
        "no_failed_measure_sample",
        not bad,
        f"{len(bad)} 个 measure 样本 success!=True，首个: {bad[0] if bad else ''}",
    )
    rows = measure_rows(allrows)

    # I1 命中对齐：hot 侧 hit == floor(P/B)*B
    viol = [
        r
        for r in rows
        if r["kind"] == "hot"
        and r["hit_tokens"] != (r["prefix_len"] // BLOCK_SIZE) * BLOCK_SIZE
    ]
    rep.check(
        M,
        "e1_i1_hit_alignment",
        not viol,
        f"{len(viol)} 个 hot 样本 hit != floor(P/{BLOCK_SIZE})*{BLOCK_SIZE}，"
        f"首个: P={viol[0]['prefix_len']} hit={viol[0]['hit_tokens']}"
        if viol
        else "",
        f"{len([r for r in rows if r['kind'] == 'hot'])} 个 hot 样本全部对齐",
    )

    # I2 省计算守恒：hit == actual - scheduled
    viol = [
        r
        for r in rows
        if r["hit_tokens"] != r["actual_prompt_tokens"] - r["scheduled_prompt_tokens"]
    ]
    rep.check(
        M,
        "e1_i2_hit_conservation",
        not viol,
        f"{len(viol)} 个样本 hit != actual - scheduled，首个: {viol[0] if viol else ''}",
        f"{len(rows)}/{len(rows)} 个样本满足",
    )

    # I3 APC off → hit == 0
    offr = [r for r in rows if r["prefix_caching"] == "off"]
    viol = [r for r in offr if r["hit_tokens"] != 0]
    rep.check(
        M,
        "e1_i3_off_no_hit",
        not viol,
        f"{len(viol)} 个 off 样本 hit != 0",
        f"{len(offr)} 个 off 样本 hit 恒为 0",
    )

    # hot 字段语义：命中 / 单 prefill step / 只算 16 个尾 token
    hot = [r for r in rows if r["kind"] == "hot"]
    viol = [
        r
        for r in hot
        if not (
            r.get("prefix_cache_hit") is True
            and r.get("prefill_step_count") == 1
            and r.get("scheduled_prompt_tokens") == E1_SUFFIX_LEN
        )
    ]
    rep.check(
        M,
        "e1_hot_fields",
        not viol,
        f"{len(viol)} 个 hot 样本字段不符（要求 prefix_cache_hit=True / "
        f"prefill_step_count=1 / scheduled={E1_SUFFIX_LEN}），首个: {viol[0] if viol else ''}",
        f"{len(hot)} 个 hot 样本字段全部正确",
    )

    # cold / off 字段语义：未命中 / 单 prefill step
    co = [r for r in rows if r["kind"] in ("cold", "off")]
    viol = [
        r
        for r in co
        if not (r.get("prefix_cache_hit") is False and r.get("prefill_step_count") == 1)
    ]
    rep.check(
        M,
        "e1_cold_off_fields",
        not viol,
        f"{len(viol)} 个 cold/off 样本字段不符，首个: {viol[0] if viol else ''}",
        f"{len(co)} 个 cold/off 样本字段全部正确",
    )

    # 每组 measure 样本数 = 15，且 warmup 确实被排除在汇总之外
    groups = {}
    for r in rows:
        groups.setdefault((r["prefix_caching"], r["prefix_len"], r["kind"]), 0)
        groups[(r["prefix_caching"], r["prefix_len"], r["kind"])] += 1
    badn = {k: v for k, v in groups.items() if v != E1_EXPECT_N}
    rep.check(
        M,
        "e1_sample_counts",
        not badn and len(groups) == 12,
        f"分组样本数异常（期望 12 组 × {E1_EXPECT_N}）：实际 {len(groups)} 组，越界组 {badn}",
        f"12 组 × {E1_EXPECT_N} measure 样本",
    )
    n_warm = len([r for r in allrows if r.get("phase") == "warmup"])
    rep.check(
        M,
        "e1_warmup_excluded",
        n_warm > 0 and len(rows) < len(allrows),
        f"未见 warmup 样本（warmup={n_warm}，total={len(allrows)}），无法证明 warmup 已分离",
        f"warmup {n_warm} 条已落盘且不进汇总（measure {len(rows)}/{len(allrows)}）",
    )


# --------------------------------------------------------------------------- #
# M17 · E3-single（长 prompt 分块）
# --------------------------------------------------------------------------- #
def check_m17_e3_single(rep):
    M = "m17/E3-single"
    rows_all, missing = [], []
    for tag, mbt in (
        ("on", 512),
        ("on", 1024),
        ("on", 2048),
        ("on", 4096),
        ("off", 4096),
    ):
        rel = f"outputs/m17/raw/e3_single_{tag}_{mbt}.jsonl"
        got = load_jsonl(rel)
        if got is None:
            missing.append(rel)
        else:
            rows_all.extend(got)
    if missing:
        rep.pending(M, "artifacts", f"缺少 {missing}")
        return

    bad = failed_measure(rows_all)
    rep.check(
        M,
        "no_failed_measure_sample",
        not bad,
        f"{len(bad)} 个 measure 样本 success!=True，首个: {bad[0] if bad else ''}",
    )
    rows = measure_rows(rows_all)

    # I4 分块守恒：Σchunks == prompt_len，每块 ≤ 预算，step 数 == chunk 数
    viol = [r for r in rows if sum(r["chunk_sizes"]) != r["prompt_len"]]
    rep.check(
        M,
        "e3_single_chunk_conservation",
        not viol,
        f"{len(viol)} 个样本 Σchunks != prompt_len，首个: {viol[0] if viol else ''}",
        f"{len(rows)} 个样本 Σchunks 恒为 {E3_PROMPT_LEN}",
    )
    viol = [
        r
        for r in rows
        if any(c > r["max_num_batched_tokens"] for c in r["chunk_sizes"])
    ]
    rep.check(
        M,
        "e3_single_chunk_within_budget",
        not viol,
        f"{len(viol)} 个样本存在超预算的 chunk，首个: {viol[0] if viol else ''}",
    )
    viol = [r for r in rows if r["prefill_step_count"] != len(r["chunk_sizes"])]
    rep.check(
        M,
        "e3_single_step_count_matches",
        not viol,
        f"{len(viol)} 个样本 prefill_step_count != len(chunk_sizes)",
    )

    # 预算 → 步数：512→4、1024→2、≥2048→1
    viol = []
    for r in rows:
        want = E3_EXPECTED_STEPS.get(r["max_num_batched_tokens"])
        if want is not None and r["prefill_step_count"] != want:
            viol.append((r["max_num_batched_tokens"], r["prefill_step_count"], want))
    rep.check(
        M,
        "e3_single_step_counts",
        not viol,
        f"步数与预算不符（512→4/1024→2/≥2048→1）：{viol[:3]}",
        "512→4、1024→2、2048→1、4096→1 全部符合",
    )

    groups = {}
    for r in rows:
        k = (r["enable_chunked_prefill"], r["max_num_batched_tokens"])
        groups[k] = groups.get(k, 0) + 1
    badn = {k: v for k, v in groups.items() if v != E3_EXPECT_N}
    rep.check(
        M,
        "e3_single_sample_counts",
        not badn and len(groups) == 5,
        f"分组样本数异常（期望 5 组 × {E3_EXPECT_N}）：实际 {len(groups)} 组，越界 {badn}",
        f"5 组 × {E3_EXPECT_N} measure 样本",
    )


# --------------------------------------------------------------------------- #
# M17 · E3-concurrency
# --------------------------------------------------------------------------- #
def check_m17_e3_conc(rep):
    M = "m17/E3-conc"
    sides = {}
    for mode in ("chunked", "nochunk"):
        got = load_jsonl(f"outputs/m17/raw/e3_conc_{mode}.jsonl")
        if got is None:
            rep.pending(M, "artifacts", f"缺少 outputs/m17/raw/e3_conc_{mode}.jsonl")
            return
        sides[mode] = got

    allrows = sides["chunked"] + sides["nochunk"]
    bad = failed_measure(allrows)
    rep.check(
        M,
        "no_failed_measure_sample",
        not bad,
        f"{len(bad)} 个 measure 样本 success!=True，首个: {bad[0] if bad else ''}",
    )

    expect_shape = {"chunked": [512, 512, 512, 512], "nochunk": [2048]}
    itl_counts = {}
    for mode, rows_all in sides.items():
        rows = measure_rows(rows_all)
        rep.check(
            M,
            f"e3_conc_sample_count[{mode}]",
            len(rows) == CONC_EXPECT_N,
            f"{mode} 侧 measure 样本 {len(rows)} != {CONC_EXPECT_N}",
            f"{mode} 侧 {len(rows)} 个 measure 样本",
        )
        viol = [r for r in rows if r["long_prefill_chunk_sizes"] != expect_shape[mode]]
        rep.check(
            M,
            f"e3_conc_chunk_shape[{mode}]",
            not viol,
            f"{mode} 侧长 prefill 形状不是 {expect_shape[mode]}，"
            f"首个: {viol[0]['long_prefill_chunk_sizes'] if viol else ''}",
            f"{mode} 侧长 prefill 恒为 {expect_shape[mode]}",
        )
        viol = [r for r in rows if sum(r["long_prefill_chunk_sizes"]) != E3_PROMPT_LEN]
        rep.check(
            M,
            f"e3_conc_chunk_conservation[{mode}]",
            not viol,
            f"{mode} 侧 Σchunks != {E3_PROMPT_LEN}",
        )
        counts = {r["decode_itl_samples"] for r in rows}
        itl_counts[mode] = counts
        rep.check(
            M,
            f"e3_conc_itl_count_stable[{mode}]",
            len(counts) == 1,
            f"{mode} 侧各 repeat 的 ITL 样本数不一致：{sorted(counts)}",
            f"{mode} 侧各 repeat ITL 样本数恒为 {counts}",
        )
    # 两侧样本数必须相同，否则 ITL 分布不可公平比较
    rep.check(
        M,
        "e3_conc_itl_sample_consistency",
        itl_counts.get("chunked") == itl_counts.get("nochunk"),
        f"两侧 ITL 样本数不同：chunked={itl_counts.get('chunked')} "
        f"nochunk={itl_counts.get('nochunk')} → 分布不可公平比较",
        f"两侧 ITL 样本数相同（{itl_counts.get('chunked')}）",
    )


# --------------------------------------------------------------------------- #
# M17 · P0 E3-correctness（chunked vs nochunk 生成一致性）
# --------------------------------------------------------------------------- #
def check_m17_e3_correctness(rep):
    M = "m17/E3-correctness"
    sides = {}
    for mode in ("chunked", "nochunk"):
        got = load_jsonl(f"outputs/m17/raw/e3_correctness_{mode}.jsonl")
        if got is None:
            rep.pending(
                M,
                "artifacts",
                f"outputs/m17/raw/e3_correctness_{mode}.jsonl 不存在 —— "
                "E3 no-chunk correctness 属**可选扩展**，脚本已实现但未执行；"
                "不阻塞主线（M17 核心实验与 M19 均已完成）",
            )
            return
        sides[mode] = got

    idx = {
        mode: {(r["scenario"], r["case"], r["repeat"]): r for r in measure_rows(rows)}
        for mode, rows in sides.items()
    }
    keys = sorted(set(idx["chunked"]) & set(idx["nochunk"]))
    rep.check(
        M,
        "e3_correctness_pairs_exist",
        bool(keys) and set(idx["chunked"]) == set(idx["nochunk"]),
        f"两侧 case 不成对：chunked={sorted(set(idx['chunked']) - set(idx['nochunk']))} "
        f"nochunk={sorted(set(idx['nochunk']) - set(idx['chunked']))}",
        f"{len(keys)} 对 case 成对",
    )
    if not keys:
        return

    bad_prompt, bad_ids, bad_hash, bad_len, bad_conserve = [], [], [], [], []
    for k in keys:
        c, n = idx["chunked"][k], idx["nochunk"][k]
        if c["prompt_hash"] != n["prompt_hash"]:
            bad_prompt.append(k)
        if c["output_token_ids"] != n["output_token_ids"]:
            bad_ids.append(k)
        if c["output_token_ids_hash"] != n["output_token_ids_hash"]:
            bad_hash.append(k)
        if not (
            c["actual_output_tokens"]
            == n["actual_output_tokens"]
            == c["max_tokens_requested"]
        ):
            bad_len.append(k)
        if not (
            c["sum_prefill_scheduled"] == n["sum_prefill_scheduled"] == c["prompt_len"]
        ):
            bad_conserve.append(k)

    rep.check(
        M,
        "e3_correctness_prompt_hash",
        not bad_prompt,
        f"{len(bad_prompt)} 对 case 的 prompt_hash 不同：{bad_prompt[:3]}",
    )
    rep.check(
        M,
        "e3_correctness_output_token_ids",
        not bad_ids,
        f"{len(bad_ids)} 对 case 的 output_token_ids 不同：{bad_ids[:3]}",
        f"{len(keys)} 对 case 的完整输出 token 序列逐一相同",
    )
    rep.check(
        M,
        "e3_correctness_output_hash",
        not bad_hash,
        f"{len(bad_hash)} 对 case 的 output hash 不同：{bad_hash[:3]}",
    )
    rep.check(
        M,
        "e3_correctness_output_len",
        not bad_len,
        f"{len(bad_len)} 对 case 的输出长度不符：{bad_len[:3]}",
    )
    rep.check(
        M,
        "e3_correctness_chunk_conservation",
        not bad_conserve,
        f"{len(bad_conserve)} 对 case 的 prefill 守恒不成立：{bad_conserve[:3]}",
    )

    for mode, rows in sides.items():
        n = len(measure_rows(rows))
        want = CORR_EXPECT_N * 3  # 3 个 case（A / B-D / B-P）× repeats
        rep.check(
            M,
            f"e3_correctness_sample_count[{mode}]",
            n == want,
            f"{mode} 侧 measure 样本 {n} != {want}",
            f"{mode} 侧 {n} 个 measure 样本",
        )
        shape = [512, 512, 512, 512] if mode == "chunked" else [2048]
        viol = [
            r
            for r in measure_rows(rows)
            if r["prompt_len"] == E3_PROMPT_LEN and r["chunk_sizes"] != shape
        ]
        rep.check(
            M,
            f"e3_correctness_chunk_shape[{mode}]",
            not viol,
            f"{mode} 侧长 prefill 形状不是 {shape}，"
            f"首个: {viol[0]['chunk_sizes'] if viol else ''}",
        )


# --------------------------------------------------------------------------- #
# M17 · P2 I5-b（输出长度）
# --------------------------------------------------------------------------- #
def check_m17_i5b(rep):
    M = "m17/I5-b"
    on = load_jsonl("outputs/m17/raw/e1_output_len_on.jsonl")
    off = load_jsonl("outputs/m17/raw/e1_output_len_off.jsonl")
    if on is None or off is None:
        rep.pending(
            M,
            "artifacts",
            "outputs/m17/raw/e1_output_len_{on,off}.jsonl 不存在 —— "
            "I5-b 属**可选扩展**，脚本已实现但未执行；不阻塞主线",
        )
        return

    allrows = on + off
    bad = failed_measure(allrows)
    rep.check(
        M,
        "no_failed_measure_sample",
        not bad,
        f"{len(bad)} 个 measure 样本 success!=True，首个: {bad[0] if bad else ''}",
    )
    rows = measure_rows(allrows)

    viol = [r for r in rows if r["actual_output_tokens"] != r["max_tokens_requested"]]
    rep.check(
        M,
        "i5b_output_len_exact",
        not viol,
        f"{len(viol)} 个样本 actual_output_tokens != max_tokens_requested，"
        f"首个: {viol[0] if viol else ''}",
        f"{len(rows)} 个样本输出长度精确等于请求的 max_tokens",
    )

    hot = [r for r in rows if r["kind"] == "hot"]
    viol = [
        r
        for r in hot
        if not (
            r["hit_tokens"] == I5B_PREFIX_LEN
            and r["scheduled_prompt_tokens"] == I5B_SUFFIX_LEN
            and r["prefix_cache_hit"] is True
            and r["prefill_step_count"] == 1
        )
    ]
    rep.check(
        M,
        "i5b_hot_fields",
        not viol,
        f"{len(viol)} 个 hot 样本不符（要求 hit={I5B_PREFIX_LEN} / "
        f"scheduled={I5B_SUFFIX_LEN} / prefix_cache_hit=True / prefill_step_count=1）",
        f"{len(hot)} 个 hot 样本字段全部正确",
    )

    offr = [r for r in rows if r["kind"] == "off"]
    want_sched = I5B_PREFIX_LEN + I5B_SUFFIX_LEN
    viol = [
        r
        for r in offr
        if not (
            r["hit_tokens"] == 0
            and r["scheduled_prompt_tokens"] == want_sched
            and r["prefix_cache_hit"] is False
        )
    ]
    rep.check(
        M,
        "i5b_off_fields",
        not viol,
        f"{len(viol)} 个 off 样本不符（要求 hit=0 / scheduled={want_sched} / "
        "prefix_cache_hit=False）",
        f"{len(offr)} 个 off 样本字段全部正确",
    )

    # cold/hot 前缀相同、后缀不同；off 与 hot 的 prompt 完全相同
    by_key = {}
    for r in rows:
        by_key.setdefault((r["max_tokens_requested"], r["repeat"]), {})[r["kind"]] = r
    bad_pref, bad_sfx, bad_offhot = [], [], []
    for k, d in by_key.items():
        if "cold" in d and "hot" in d:
            if d["cold"]["prefix_hash"] != d["hot"]["prefix_hash"]:
                bad_pref.append(k)
            if d["cold"]["suffix_hash"] == d["hot"]["suffix_hash"]:
                bad_sfx.append(k)
        if (
            "off" in d
            and "hot" in d
            and (d["off"]["prompt_hash"] != d["hot"]["prompt_hash"])
        ):
            bad_offhot.append(k)
    rep.check(
        M,
        "i5b_cold_hot_same_prefix",
        not bad_pref,
        f"{len(bad_pref)} 组 cold/hot 前缀不同：{bad_pref[:3]}",
    )
    rep.check(
        M,
        "i5b_cold_hot_diff_suffix",
        not bad_sfx,
        f"{len(bad_sfx)} 组 cold/hot 后缀相同（会完全命中）：{bad_sfx[:3]}",
    )
    rep.check(
        M,
        "i5b_off_hot_same_prompt",
        not bad_offhot,
        f"{len(bad_offhot)} 组 off/hot 的 prompt 不同（对照不成立）：{bad_offhot[:3]}",
        f"{len(by_key)} 组 off/hot prompt hash 完全相同",
    )

    viol = [r for r in rows if not r.get("output_token_ids_hash")]
    rep.check(
        M,
        "i5b_output_hash_present",
        not viol,
        f"{len(viol)} 个样本缺 output_token_ids_hash",
    )
    viol = [
        r
        for r in rows
        if (r["in_process_first_token_latency_ms"] or 0) < 0
        or (r["request_completion_latency_ms"] or 0) < 0
        or (r["post_first_token_latency_ms"] or 0) < 0
    ]
    rep.check(M, "i5b_latency_non_negative", not viol, f"{len(viol)} 个样本存在负延迟")

    # TPOT 定义：只有 1 个输出 token 时必须为 null
    viol = [
        r for r in rows if (r["actual_output_tokens"] == 1) != (r["tpot_ms"] is None)
    ]
    rep.check(
        M,
        "i5b_tpot_definition",
        not viol,
        f"{len(viol)} 个样本 TPOT 定义不符（out==1 时必须为 null）",
    )

    groups = {}
    for r in rows:
        k = (r["prefix_caching"], r["kind"], r["max_tokens_requested"])
        groups[k] = groups.get(k, 0) + 1
    want_groups = len(I5B_MAX_TOKENS) * 3  # on-cold / on-hot / off-off
    badn = {k: v for k, v in groups.items() if v != I5B_EXPECT_N}
    rep.check(
        M,
        "i5b_sample_counts",
        not badn and len(groups) == want_groups,
        f"分组样本数异常（期望 {want_groups} 组 × {I5B_EXPECT_N}）："
        f"实际 {len(groups)} 组，越界 {badn}",
        f"{want_groups} 组 × {I5B_EXPECT_N} measure 样本",
    )


# --------------------------------------------------------------------------- #
# CSV ↔ JSONL 一致性
# --------------------------------------------------------------------------- #
def check_m17_csv_vs_jsonl(rep):
    M = "m17/CSV↔JSONL"

    on = load_jsonl("outputs/m17/raw/e1_on.jsonl")
    off = load_jsonl("outputs/m17/raw/e1_off.jsonl")
    sweep = load_csv("outputs/m17/prefix_cache_sweep.csv")
    if on and off and sweep:
        rows = measure_rows(on + off)
        counts = {}
        for r in rows:
            k = (r["prefix_caching"], str(r["prefix_len"]), r["kind"])
            counts[k] = counts.get(k, 0) + 1
        bad = []
        for r in sweep:
            k = (r["prefix_caching"], r["prefix_len"], r["kind"])
            if counts.get(k) != int(r["n"]):
                bad.append((k, r["n"], counts.get(k)))
        rep.check(
            M,
            "sweep_csv_n_matches_jsonl",
            not bad,
            f"prefix_cache_sweep.csv 的 n 与 JSONL 有效样本数不符：{bad[:3]}",
            f"{len(sweep)} 行 n 全部等于 JSONL measure 样本数",
        )

        # 重算 median / 差值 / CI，验证 CSV 确实由 JSONL 生成
        lat = {}
        for r in rows:
            lat.setdefault(
                (r["prefix_caching"], r["prefix_len"], r["kind"]), []
            ).append(r["in_process_first_token_latency_ms"])
        effect = load_csv("outputs/m17/prefix_cache_effect.csv")
        if effect:
            bad = []
            for r in effect:
                P = int(r["prefix_len"])
                a, b = lat.get(("off", P, "off")), lat.get(("on", P, "hot"))
                if not a or not b:
                    bad.append((P, "缺少样本"))
                    continue
                d, lo, hi = boot_ci_median_diff(a, b)
                for name, got, want in (
                    ("off_median", float(r["off_median_ms"]), statistics.median(a)),
                    ("hot_median", float(r["hot_median_ms"]), statistics.median(b)),
                    ("diff", float(r["median_diff_ms"]), d),
                    ("ci_lo", float(r["ci95_lo_ms"]), lo),
                    ("ci_hi", float(r["ci95_hi_ms"]), hi),
                ):
                    if abs(got - want) > 1e-3:
                        bad.append((P, name, got, round(want, 4)))
            rep.check(
                M,
                "effect_csv_recomputable",
                not bad,
                f"prefix_cache_effect.csv 无法由 JSONL 重算：{bad[:4]}",
                "4 档的 median / 差值 / Bootstrap CI 均可由 JSONL 逐位重算",
            )
    else:
        rep.pending(M, "e1_csv", "E1 的 CSV 或 JSONL 缺失")

    e3 = []
    ok = True
    for tag, mbt in (
        ("on", 512),
        ("on", 1024),
        ("on", 2048),
        ("on", 4096),
        ("off", 4096),
    ):
        got = load_jsonl(f"outputs/m17/raw/e3_single_{tag}_{mbt}.jsonl")
        if got is None:
            ok = False
        else:
            e3.extend(got)
    cp = load_csv("outputs/m17/chunked_prefill.csv")
    if ok and cp:
        counts = {}
        for r in measure_rows(e3):
            k = (str(r["enable_chunked_prefill"]), str(r["max_num_batched_tokens"]))
            counts[k] = counts.get(k, 0) + 1
        bad = [
            (k, r["n"], counts.get(k))
            for r in cp
            for k in [(r["enable_chunked_prefill"], r["max_num_batched_tokens"])]
            if counts.get(k) != int(r["n"])
        ]
        rep.check(
            M,
            "chunked_prefill_csv_n_matches_jsonl",
            not bad,
            f"chunked_prefill.csv 的 n 与 JSONL 不符：{bad[:3]}",
            f"{len(cp)} 行 n 全部一致",
        )

    conc_ok, conc = True, {}
    for mode in ("chunked", "nochunk"):
        got = load_jsonl(f"outputs/m17/raw/e3_conc_{mode}.jsonl")
        if got is None:
            conc_ok = False
        else:
            conc[mode] = measure_rows(got)
    cc = load_csv("outputs/m17/chunked_prefill_concurrency.csv")
    if conc_ok and cc:
        bad = [
            (r["mode"], r["n"], len(conc.get(r["mode"], [])))
            for r in cc
            if len(conc.get(r["mode"], [])) != int(r["n"])
        ]
        rep.check(
            M,
            "conc_csv_n_matches_jsonl",
            not bad,
            f"chunked_prefill_concurrency.csv 的 n 与 JSONL 不符：{bad}",
            f"{len(cc)} 行 n 全部一致",
        )

    # 新增实验的 CSV（未跑则 PENDING）
    for rel, dep in (
        (
            "outputs/m17/e3_correctness.csv",
            "outputs/m17/raw/e3_correctness_chunked.jsonl",
        ),
        ("outputs/m17/apc_output_length.csv", "outputs/m17/raw/e1_output_len_on.jsonl"),
    ):
        if not os.path.exists(_p(dep)):
            rep.pending(M, f"csv[{os.path.basename(rel)}]", f"依赖的 {dep} 尚未产生")
        elif not os.path.exists(_p(rel)):
            rep.fail(
                M,
                f"csv[{os.path.basename(rel)}]",
                f"JSONL 已存在但 {rel} 缺失（未汇总？）",
            )


# --------------------------------------------------------------------------- #
# 文档数字可溯源（正向 + 反向）
# --------------------------------------------------------------------------- #
def check_m17_doc_numbers(rep):
    M = "m17/文档溯源"
    doc_rel = "notes/m17/results.md"
    doc_path = _p(doc_rel)
    if not os.path.exists(doc_path):
        rep.pending(M, "doc", f"{doc_rel} 不存在")
        return
    with open(doc_path) as f:
        text = f.read()
    tables = parse_md_tables(text)

    effect = load_csv("outputs/m17/prefix_cache_effect.csv")
    sweep = load_csv("outputs/m17/prefix_cache_sweep.csv")
    cp = load_csv("outputs/m17/chunked_prefill.csv")
    cc = load_csv("outputs/m17/chunked_prefill_concurrency.csv")
    if not (effect and sweep and cp and cc):
        rep.pending(M, "csv", "M17 汇总 CSV 不全，无法核对文档")
        return

    # ---- 反向：E1 主表的每个数字单元格都必须能从 CSV 重算 ----
    body = find_table(tables, "off median", "hot median", "Bootstrap 95% CI")
    if body is None:
        rep.fail(M, "e1_table_found", f"{doc_rel} 中找不到 E1 主结果表（表头已漂移？）")
    else:
        eff_by_p = {int(r["prefix_len"]): r for r in effect}
        swp = {(r["prefix_caching"], int(r["prefix_len"]), r["kind"]): r for r in sweep}
        bad = []
        for cells in body:
            P = _norm_num(cells[0])
            if P is None or int(P) not in eff_by_p:
                bad.append((cells[0], "无法定位到 CSV 行"))
                continue
            P = int(P)
            e = eff_by_p[P]
            ci = cells[4].replace("*", "").strip().strip("[]")
            lo_s, hi_s = (ci.split(",") + ["", ""])[:2]
            for label, cell, csv_v in (
                ("off median", cells[1], float(e["off_median_ms"])),
                ("hot median", cells[2], float(e["hot_median_ms"])),
                ("median 差", cells[3], float(e["median_diff_ms"])),
                ("CI lo", lo_s, float(e["ci95_lo_ms"])),
                ("CI hi", hi_s, float(e["ci95_hi_ms"])),
                ("相对降幅", cells[6], float(e["rel_reduction_pct"])),
                ("off sd", cells[7], float(swp[("off", P, "off")]["lat_stdev_ms"])),
                ("hot sd", cells[8], float(swp[("on", P, "hot")]["lat_stdev_ms"])),
            ):
                if not _num_matches(cell, csv_v):
                    bad.append((P, label, _norm_num(cell), round(csv_v, 4)))
        rep.check(
            M,
            "e1_table_numbers_traceable",
            not bad,
            f"E1 主表存在无法溯源到 CSV 的数字：{bad[:5]}",
            f"E1 主表 {len(body)} 行 × 8 个数字全部可由 CSV 重算",
        )

    # ---- 反向：E3-single 表 ----
    body = find_table(tables, "chunked", "mbt", "chunk_sizes")
    if body is None:
        rep.fail(M, "e3_single_table_found", f"{doc_rel} 中找不到 E3-single 表")
    else:
        cp_by = {
            (r["enable_chunked_prefill"], int(r["max_num_batched_tokens"])): r
            for r in cp
        }
        bad = []
        for cells in body:
            key = (cells[0].replace("*", "").strip(), int(_norm_num(cells[1])))
            if key not in cp_by:
                bad.append((key, "无法定位到 CSV 行"))
                continue
            r = cp_by[key]
            for label, cell, csv_v in (
                ("steps", cells[2], float(r["prefill_step_count"])),
                ("Σ", cells[4], float(r["sum_prefill_scheduled"])),
                ("median", cells[5], float(r["ttft_median_ms"])),
                ("sd", cells[6], float(r["ttft_stdev_ms"])),
                ("p95", cells[7], float(r["ttft_p95_ms"])),
                ("min", cells[8], float(r["ttft_min_ms"])),
                ("max", cells[9], float(r["ttft_max_ms"])),
            ):
                if not _num_matches(cell, csv_v):
                    bad.append((key, label, _norm_num(cell), round(csv_v, 4)))
        rep.check(
            M,
            "e3_single_table_numbers_traceable",
            not bad,
            f"E3-single 表存在无法溯源的数字：{bad[:5]}",
            f"E3-single 表 {len(body)} 行全部可由 CSV 重算",
        )

    # ---- 正向：由 CSV 生成关键数字串，必须出现在文档中（含正文叙述句） ----
    # M17 审计的原教训是「自动核对只覆盖表格、漏了正文叙述句」，故这一步专门做
    # 「从 CSV 算出来的数 → 必须在正文里找得到」这个方向。
    c = {r["mode"]: r for r in cc}
    must_appear = []
    for mode in ("chunked", "nochunk"):
        must_appear.append(
            (f"E3-conc {mode} itl_max", float(c[mode]["itl_max_median_ms"]), 2, "")
        )
        must_appear.append(
            (
                f"E3-conc {mode} 长 prefill TTFT",
                float(c[mode]["long_prefill_ttft_median_ms"]),
                2,
                "",
            )
        )
    for r in effect:
        if r["ci_excludes_zero"] == "yes":
            must_appear.append(
                (
                    f"E1 P={r['prefix_len']} 相对降幅",
                    float(r["rel_reduction_pct"]),
                    1,
                    "%",
                )
            )
    missing = [
        (k, f"{v:.{d}f}{sfx}")
        for k, v, d, sfx in must_appear
        if not any(f"{cand}{sfx}" in text for cand in _render_candidates(v, d))
    ]
    rep.check(
        M,
        "doc_key_numbers_present",
        not missing,
        f"以下由 CSV 算出的关键数字在 {doc_rel} 中找不到（文档与数据脱节）：{missing}",
        f"{len(must_appear)} 个由 CSV 算出的关键数字均出现在文档中",
    )

    # 覆盖范围声明：不把部分覆盖说成全覆盖（M17 审计的原教训）
    rep.ok(
        M,
        "coverage_note",
        "覆盖：E1 主表 + E3-single 表的全部数字（反向）、E3-conc 与 E1 显著档的关键数字（正向）。"
        "**未覆盖**：§1.2 斜率、§5 对照表等正文推导值 —— 如实声明，勿当全覆盖",
    )


# --------------------------------------------------------------------------- #
# 环境不变量
# --------------------------------------------------------------------------- #
def check_env(rep):
    M = "env"
    for k, want in (
        ("VLLM_USE_FLASHINFER_SAMPLER", "0"),
        ("VLLM_ENABLE_V1_MULTIPROCESSING", "0"),
    ):
        got = os.environ.get(k)
        rep.check(
            M,
            f"env[{k}]",
            got == want,
            f"{k}={got!r}，期望 {want!r}（缺了会在 sm_120 上崩，见 AGENTS.md §5）",
            f"{k}={want}",
        )
    for k in ("VLLM_LAB_M9_COALESCE", "VLLM_LAB_TRACE"):
        got = os.environ.get(k)
        rep.check(
            M,
            f"env[{k}]",
            got in (None, "", "0"),
            f"{k}={got!r} —— 实验性门控必须关闭，否则测的不是基线语义",
            f"{k} 未设置或为 0",
        )


# --------------------------------------------------------------------------- #
# M19 · Speculative Decoding（baseline vs N-gram）
# --------------------------------------------------------------------------- #
M19_EXPECT_N = 15
M19_OUTPUT_LEN = 128
M19_BATCHES = [1, 4, 16]
M19_WORKLOADS = ["high_repeat", "low_repeat"]
M19_K = 3  # num_speculative_tokens


def check_m19_samples(rep):
    M = "m19/samples"
    base = load_jsonl("outputs/m19/raw/m19_baseline.jsonl")
    ng = load_jsonl("outputs/m19/raw/m19_ngram.jsonl")
    if base is None or ng is None:
        rep.pending(M, "artifacts", "outputs/m19/raw/m19_{baseline,ngram}.jsonl 不存在")
        return
    allrows = base + ng
    bad = failed_measure(allrows)
    rep.check(
        M,
        "no_failed_measure_sample",
        not bad,
        f"{len(bad)} 个 measure 样本 success!=True，首个: "
        f"{bad[0].get('invalid_reason') if bad else ''}",
    )
    rows = measure_rows(allrows)

    # 输出长度必须精确等于配置（ignore_eos=True）
    viol = [
        r
        for r in rows
        if any(n != r["max_tokens_requested"] for n in r["actual_output_tokens"])
        or len(r["actual_output_tokens"]) != r["batch_size"]
    ]
    rep.check(
        M,
        "m19_output_len_exact",
        not viol,
        f"{len(viol)} 个样本输出长度/请求数不符，首个: "
        f"{viol[0]['actual_output_tokens'] if viol else ''}",
        f"{len(rows)} 个样本的每个请求都恰好产出 {M19_OUTPUT_LEN} 个 token",
    )

    # baseline 侧不得出现任何投机计数
    bl = [r for r in rows if r["config"] == "baseline"]
    viol = [
        r
        for r in bl
        if r["proposed_tokens"] != 0
        or r["accepted_tokens"] != 0
        or r["num_drafts"] != 0
        or r["acceptance_rate"] is not None
    ]
    rep.check(
        M,
        "m19_baseline_no_spec_counts",
        not viol,
        f"{len(viol)} 个 baseline 样本出现了投机计数（对照被污染）",
        f"{len(bl)} 个 baseline 样本的 proposed/accepted/drafts 恒为 0",
    )
    # baseline 每步恰好 1 个输出 token（即 steps ≈ 输出长度）
    viol = [r for r in bl if not 0.99 <= r["steps_per_output_token"] <= 1.02]
    rep.check(
        M,
        "m19_baseline_one_token_per_step",
        not viol,
        f"{len(viol)} 个 baseline 样本的 steps/token 偏离 1.0，"
        f"首个: {viol[0]['steps_per_output_token'] if viol else ''}",
        "baseline 每个 engine step 恰好产出 1 个 token（steps/token≈1.0）",
    )

    # ngram 侧的计数守恒
    ng_rows = [r for r in rows if r["config"] == "ngram"]
    viol = [r for r in ng_rows if not 0 <= r["accepted_tokens"] <= r["proposed_tokens"]]
    rep.check(
        M,
        "m19_accepted_within_proposed",
        not viol,
        f"{len(viol)} 个样本违反 0 <= accepted <= proposed",
        f"{len(ng_rows)} 个 ngram 样本满足 0 <= accepted <= proposed",
    )
    viol = [
        r
        for r in ng_rows
        if not (r["acceptance_rate"] is None or 0 <= r["acceptance_rate"] <= 1)
    ]
    rep.check(
        M,
        "m19_acceptance_rate_range",
        not viol,
        f"{len(viol)} 个样本的 acceptance_rate 越界 [0,1]",
    )
    # proposed == num_drafts * k（每次出 draft 都提 k 个 token）
    viol = [r for r in ng_rows if r["proposed_tokens"] != r["num_drafts"] * M19_K]
    rep.check(
        M,
        "m19_proposed_equals_drafts_times_k",
        not viol,
        f"{len(viol)} 个样本 proposed != num_drafts * {M19_K}，"
        f"首个: proposed={viol[0]['proposed_tokens']} drafts={viol[0]['num_drafts']}"
        if viol
        else "",
        f"proposed 恒等于 num_drafts × k（k={M19_K}）",
    )
    # 逐位置计数之和必须等于总数（守恒律）
    viol = [
        r
        for r in ng_rows
        if sum(r["accepted_per_pos"]) != r["accepted_tokens"]
        or sum(r["draft_per_pos"]) != r["proposed_tokens"]
    ]
    rep.check(
        M,
        "m19_per_pos_conservation",
        not viol,
        f"{len(viol)} 个样本的逐位置计数之和与总数不符",
        "逐位置 accepted/draft 之和 == 总 accepted/proposed",
    )
    # 投机必须真的开着，且配置就是冻结的那组
    viol = [
        r
        for r in ng_rows
        if not (
            r["effective_cfg"]["speculative_enabled"]
            and r["effective_cfg"]["spec_method"] == "ngram"
            and r["effective_cfg"]["num_speculative_tokens"] == M19_K
            and r["effective_cfg"]["prompt_lookup_min"] == 3
            and r["effective_cfg"]["prompt_lookup_max"] == 5
        )
    ]
    rep.check(
        M,
        "m19_ngram_cfg_effective",
        not viol,
        f"{len(viol)} 个 ngram 样本的生效配置与冻结值不符（上游可能改写了默认值）",
        "生效配置 = method=ngram / k=3 / prompt_lookup 3..5（从引擎读回，"
        "而非「我传了什么」）",
    )

    # 两侧除 speculative 外必须逐字相同——否则不是单变量对照
    keys = [
        "async_scheduling",
        "enable_chunked_prefill",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_prefix_caching",
        "enforce_eager",
    ]
    bcfg = bl[0]["effective_cfg"] if bl else {}
    ncfg = ng_rows[0]["effective_cfg"] if ng_rows else {}
    diff = [k for k in keys if bcfg.get(k) != ncfg.get(k)]
    rep.check(
        M,
        "m19_single_variable",
        not diff,
        f"两侧引擎配置存在差异（非单变量）：{[(k, bcfg.get(k), ncfg.get(k)) for k in diff]}",
        "两侧仅 speculative_config 不同，其余生效配置逐字相同",
    )

    # 两侧 prompt 必须相同
    idx = {}
    for r in rows:
        idx.setdefault((r["workload"], r["batch_size"], r["repeat"]), {})[
            r["config"]
        ] = r
    bad_p = [
        k
        for k, d in idx.items()
        if "baseline" in d
        and "ngram" in d
        and d["baseline"]["batch_prompt_hash"] != d["ngram"]["batch_prompt_hash"]
    ]
    rep.check(
        M,
        "m19_same_prompt_both_sides",
        not bad_p,
        f"{len(bad_p)} 组两侧 prompt hash 不同（对照不成立）：{bad_p[:3]}",
        f"{len(idx)} 组两侧 prompt hash 完全相同",
    )

    # 分组样本数
    groups = {}
    for r in rows:
        k = (r["config"], r["workload"], r["batch_size"])
        groups[k] = groups.get(k, 0) + 1
    want = 2 * len(M19_WORKLOADS) * len(M19_BATCHES)
    badn = {k: v for k, v in groups.items() if v != M19_EXPECT_N}
    rep.check(
        M,
        "m19_sample_counts",
        not badn and len(groups) == want,
        f"分组样本数异常（期望 {want} 组 × {M19_EXPECT_N}）："
        f"实际 {len(groups)} 组，越界 {badn}",
        f"{want} 组 × {M19_EXPECT_N} measure 样本",
    )
    n_warm = len([r for r in allrows if r.get("phase") == "warmup"])
    rep.check(
        M,
        "m19_warmup_excluded",
        n_warm > 0 and len(rows) < len(allrows),
        f"未见 warmup 样本（warmup={n_warm}, total={len(allrows)}）",
        f"warmup {n_warm} 条已落盘且不进汇总（measure {len(rows)}/{len(allrows)}）",
    )


def check_m19_csv_vs_jsonl(rep):
    M = "m19/CSV↔JSONL"
    base = load_jsonl("outputs/m19/raw/m19_baseline.jsonl")
    ng = load_jsonl("outputs/m19/raw/m19_ngram.jsonl")
    summary = load_csv("outputs/m19/spec_decode_summary.csv")
    if base is None or ng is None or summary is None:
        rep.pending(M, "artifacts", "M19 的 JSONL 或汇总 CSV 缺失")
        return
    rows = measure_rows(base + ng)
    counts = {}
    for r in rows:
        k = (r["config"], r["workload"], str(r["batch_size"]))
        counts[k] = counts.get(k, 0) + 1
    bad = [
        (k, r["n"], counts.get(k))
        for r in summary
        for k in [(r["config"], r["workload"], r["batch_size"])]
        if counts.get(k) != int(r["n"])
    ]
    rep.check(
        M,
        "m19_summary_n_matches_jsonl",
        not bad,
        f"spec_decode_summary.csv 的 n 与 JSONL 有效样本数不符：{bad[:3]}",
        f"{len(summary)} 行 n 全部一致",
    )

    # 效应量 CSV 的 median / CI 必须能由 JSONL 逐位重算
    effect = load_csv("outputs/m19/spec_decode_effect.csv")
    if effect:
        lat = {}
        for r in rows:
            lat.setdefault((r["config"], r["workload"], r["batch_size"]), []).append(
                r["request_completion_latency_ms"]
            )
        bad = []
        for r in effect:
            wl, b = r["workload"], int(r["batch_size"])
            a = lat.get(("baseline", wl, b))
            c = lat.get(("ngram", wl, b))
            if not a or not c:
                bad.append((wl, b, "缺少样本"))
                continue
            d, lo, hi = boot_ci_median_diff(a, c)
            for name, got, wantv in (
                (
                    "base_median",
                    float(r["base_completion_median_ms"]),
                    statistics.median(a),
                ),
                (
                    "ngram_median",
                    float(r["ngram_completion_median_ms"]),
                    statistics.median(c),
                ),
                ("diff", float(r["completion_median_diff_ms"]), d),
                ("ci_lo", float(r["completion_ci95_lo_ms"]), lo),
                ("ci_hi", float(r["completion_ci95_hi_ms"]), hi),
            ):
                if abs(got - wantv) > 1e-3:
                    bad.append((wl, b, name, got, round(wantv, 4)))
        rep.check(
            M,
            "m19_effect_csv_recomputable",
            not bad,
            f"spec_decode_effect.csv 无法由 JSONL 重算：{bad[:4]}",
            "6 组的 median / 差值 / Bootstrap CI 均可由 JSONL 逐位重算",
        )


def check_m19_equivalence(rep):
    """输出一致性：如实记录，不静默忽略差异。

    这里刻意不把「baseline 与 ngram 输出必须逐 token 相同」写成硬性 FAIL：
    `batch_shape_control.csv` 已证明baseline 自己换个 batch 组成就会在同一位置分歧，
    说明分歧来自批形状引起的浮点差异而非投机解码。因此硬判据只保留两条真正的功能红线
    ——输出长度必须一致、分歧不得出现在第 0 位；一致率本身只做记录与展示。
    """
    M = "m19/输出一致性"
    detail = load_csv("outputs/m19/output_equivalence_detail.csv")
    ctrl = load_csv("outputs/m19/batch_shape_control.csv")
    if detail is None or ctrl is None:
        rep.pending(M, "artifacts", "M19 一致性 probe 的 CSV 缺失（probe 未执行）")
        return

    bad_len = [r for r in detail if r["base_len"] != r["ngram_len"]]
    rep.check(
        M,
        "m19_equiv_same_length",
        not bad_len,
        f"{len(bad_len)} 个请求两侧输出长度不同（功能问题，非数值差异）",
        f"{len(detail)} 个请求两侧输出长度全部一致",
    )
    at_zero = [
        r
        for r in detail
        if r["exact_match"] == "False" and r["first_divergence_index"] == "0"
    ]
    rep.check(
        M,
        "m19_equiv_no_divergence_at_zero",
        not at_zero,
        f"{len(at_zero)} 个请求在第 0 个 token 就分歧（指向实现问题，而非数值差异）",
        "没有任何请求在第 0 个 token 分歧",
    )
    # 可复现性：同配置跨 repeat 必须完全一致，否则"数值差异"的解释站不住
    non_repro = [
        r
        for r in ctrl
        if r["batch_a"] == r["batch_b"]
        and int(r["n_identical"]) != int(r["n_compared"])
    ]
    rep.check(
        M,
        "m19_within_config_reproducible",
        not non_repro,
        f"{len(non_repro)} 个配置在同一配置内跨 repeat 输出不一致（存在不可控随机性）",
        "所有配置在同一配置内跨 repeat 输出完全一致（排除随机性解释）",
    )
    n_match = sum(1 for r in detail if r["exact_match"] == "True")
    baseline_self_div = [
        r
        for r in ctrl
        if r["config"] == "baseline"
        and r["batch_a"] != r["batch_b"]
        and int(r["n_identical"]) != int(r["n_compared"])
    ]
    rep.ok(
        M,
        "m19_equiv_record",
        f"baseline↔ngram 逐 token 完全一致 {n_match}/{len(detail)}；"
        f"正交对照中 **baseline 自身**仅换 batch 组成就出现分歧的配置有 "
        f"{len(baseline_self_div)} 个 → 与 batch-shape numerical sensitivity 一致；"
        "现有证据未发现明确 Spec Decode correctness bug"
        "（详见 notes/m19/results.md §5；非严格逐例排除）",
    )


# --------------------------------------------------------------------------- #
# M22 · Tensor Parallel
# --------------------------------------------------------------------------- #
M22_TAG = "h20_formal"  # 正式数据的标签；local_preflight 是 5080 预演，不参与核对


def check_m22(rep):
    """M22 不变量。

    刻意区分两类（这是本模块设计上的一个明确判断）：
      · correctness invariant —— 违反即实验无效，FAIL；
      · expected observation  —— 理论预期，实测不符时记录并要求解释，不自动判失败。
    例如「每卡权重 ≈ 1/k」就不是 invariant：embedding/lm_head 是否切分随实现而变
    （M10 已见过「非 −50%」的先例），KV head 不足 TP 时还会被复制
    （`vllm/config/model.py:1259-1270`）。把理论预期伪装成 invariant 会逼出假结论。
    """
    M = "m22"
    rows = []
    d = _p("outputs/m22/raw")
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if n.startswith(f"{M22_TAG}_tp") and n.endswith("_run.jsonl"):
                rows.extend(load_jsonl(f"outputs/m22/raw/{n}") or [])
    if not rows:
        rep.pending(
            M,
            "artifacts",
            f"outputs/m22/raw/{M22_TAG}_tp*_run.jsonl 不存在 —— "
            "M22 多卡实验尚未执行（脚本已就绪）。当前 M22 等级仅 T+O，不得标 E/H",
        )
        return

    bad = failed_measure(rows)
    rep.check(
        M,
        "m22_no_failed_measure_sample",
        not bad,
        f"{len(bad)} 个 measure 样本 success!=True，首个原因: "
        f"{bad[0].get('invalid_reason') if bad else ''}",
    )
    meas = measure_rows(rows)
    tps = sorted({r["tp"] for r in rows})

    # ---- 硬 invariant：单变量 ----
    d2 = [r for r in meas if r["experiment"] == "D2_decode"]
    d3 = [r for r in meas if r["experiment"] == "D3_prefill"]
    cfgs = {r["tp"]: r["effective_cfg"] for r in rows if "effective_cfg" in r}
    keys = [
        "async_scheduling",
        "enable_chunked_prefill",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_prefix_caching",
        "enforce_eager",
        "max_model_len",
        "dtype",
        "model",
        "block_size",
    ]
    diff = []
    if len(cfgs) > 1:
        ref_tp = min(cfgs)
        ref = cfgs[ref_tp]
        for tp, c in cfgs.items():
            for k in keys:
                if c.get(k) != ref.get(k):
                    diff.append((k, ref_tp, ref.get(k), tp, c.get(k)))
    rep.check(
        M,
        "m22_single_variable",
        not diff,
        f"TP 档之间存在非 TP 的配置差异（单变量不成立）：{diff[:4]}",
        f"{len(cfgs)} 档 TP 除并行度外生效配置逐字相同",
    )
    # executor backend 必然随 TP 变化 —— 不是错误，但必须显式记录
    be = {tp: c.get("distributed_executor_backend") for tp, c in cfgs.items()}
    rep.ok(
        M,
        "m22_executor_backend_note",
        f"executor backend 随 TP 变化（{be}）——这是 TP 的固有伴随变量，"
        "无法消除，结论中必须声明",
    )
    for tp, c in cfgs.items():
        rep.check(
            M,
            f"m22_tp_effective[{tp}]",
            c["tensor_parallel_size"] == c["tp_requested"],
            f"TP={tp} 请求值与生效值不符：{c['tp_requested']} vs {c['tensor_parallel_size']}",
        )
        rep.check(
            M,
            f"m22_async_sched_off[{tp}]",
            c["async_scheduling"] is False,
            f"TP={tp} 的 async_scheduling={c['async_scheduling']}，必须为 False"
            "（默认 None 会因 executor 类不同而自动取不同值，config/vllm.py:938）",
        )

    # ---- 硬 invariant：D2 口径 ----
    viol = [
        r
        for r in d2
        if any(n != r["max_tokens_requested"] for n in r["actual_output_tokens"])
        or len(r["actual_output_tokens"]) != r["batch_size"]
    ]
    rep.check(
        M,
        "m22_d2_output_len_exact",
        not viol,
        f"{len(viol)} 个 D2 样本的输出长度/请求数不符",
        f"{len(d2)} 个 D2 样本每请求输出长度精确",
    )
    viol = [r for r in d2 if r.get("steady_batch_stable") is not True]
    rep.check(
        M,
        "m22_d2_steady_batch_stable",
        not viol,
        f"{len(viol)} 个 D2 样本在稳态窗口内 batch 不恒定 —— "
        "测到的可能是 batch 尾部收缩而非纯 decode scaling",
        f"{len(d2)} 个 D2 样本稳态窗口内 batch 恒定（已排除 tail effect）",
    )
    viol = [r for r in d2 if (r.get("n_steady_steps") or 0) < 5]
    rep.check(
        M,
        "m22_d2_enough_steady_steps",
        not viol,
        f"{len(viol)} 个 D2 样本的稳态 step < 5 个，中位数不可靠",
    )
    viol = [r for r in d2 + d3 if not r.get("scheduler_introspection_ok", True)]
    rep.check(
        M,
        "m22_scheduler_introspection",
        not viol,
        f"{len(viol)} 个样本拿不到 scheduler 内省 —— 稳态判定退化，"
        "请确认 VLLM_ENABLE_V1_MULTIPROCESSING=0（llm_engine.py:224-226）",
        "逐 step scheduler 内省在所有 TP 档均可用",
    )

    # ---- 硬 invariant：D3 边界 ----
    # 覆盖全部样本（含 warmup）：prompt_len >= max_model_len 属配置合法性问题，与 phase 无关
    # （input_processor.py:423-432 在 add_request 时就 ValueError），只查 measure 会漏掉 warmup。
    viol = [
        r
        for r in rows
        if r.get("experiment") == "D3_prefill"
        and r["prompt_len"] >= (cfgs.get(r["tp"], {}).get("max_model_len") or 0)
    ]
    rep.check(
        M,
        "m22_d3_prompt_within_limit",
        not viol,
        f"{len(viol)} 个 D3 样本 prompt_len >= max_model_len —— "
        "input_processor.py:423-432 会直接 ValueError",
    )

    # ---- 硬 invariant：样本数与 warmup ----
    g = {}
    for r in d2:
        g[(r["tp"], r["batch_size"])] = g.get((r["tp"], r["batch_size"]), 0) + 1
    n_expect = {v for v in g.values()}
    rep.check(
        M,
        "m22_sample_counts_uniform",
        len(n_expect) <= 1,
        f"各组 measure 样本数不一致：{sorted(g.items())}",
        f"D2 共 {len(g)} 组，每组 {n_expect} 个 measure 样本",
    )
    n_warm = len([r for r in rows if r.get("phase") == "warmup"])
    rep.check(
        M,
        "m22_warmup_excluded",
        n_warm > 0 and len(meas) < len(rows),
        f"未见 warmup 样本（warmup={n_warm}, total={len(rows)}）",
        f"warmup {n_warm} 条已落盘且不进汇总",
    )

    # ---- 硬 invariant：per-rank 观测确实拿到了 ----
    d1 = [r for r in rows if r["experiment"] == "D1_memory_kv"]
    viol = [r for r in d1 if r["n_ranks_reported"] != r["tp"]]
    rep.check(
        M,
        "m22_d1_all_ranks_reported",
        not viol,
        f"{len(viol)} 档 TP 的 per-rank 报告数与 TP 不符："
        f"{[(r['tp'], r['n_ranks_reported'], r['observation_error']) for r in viol][:3]}",
        f"{len(d1)} 档 TP 的 per-rank 显存/KV 均已采到（collective_rpc）",
    )

    # ---- expected observation（不判 FAIL，只记录，要求解释） ----
    if d1:
        obs = []
        base = next((r for r in d1 if r["tp"] == 1), None)
        for r in sorted(d1, key=lambda x: x["tp"]):
            pr = [x for x in (r.get("per_rank") or []) if isinstance(x, dict)]
            wts = [
                x.get("weight_bytes")
                for x in pr
                if isinstance(x.get("weight_bytes"), int)
            ]
            kvs = [
                x.get("available_kv_bytes")
                for x in pr
                if isinstance(x.get("available_kv_bytes"), int)
            ]
            c = r["effective_cfg"]
            kv_rep = (
                c["num_kv_heads_per_rank"] * c["tensor_parallel_size"]
                > c["total_num_kv_heads"]
            )
            item = {
                "tp": r["tp"],
                "weight_MiB_per_rank": [round(w / 2**20, 1) for w in wts],
                "kv_MiB_per_rank": [round(k / 2**20, 1) for k in kvs],
                "kv_heads_replicated": kv_rep,
            }
            if base and wts:
                b = [
                    x.get("weight_bytes")
                    for x in (base.get("per_rank") or [])
                    if isinstance(x, dict) and isinstance(x.get("weight_bytes"), int)
                ]
                if b:
                    item["weight_ratio_vs_tp1"] = round(wts[0] / b[0], 4)
                    item["ideal_ratio"] = round(1 / r["tp"], 4)
            obs.append(item)
        rep.ok(
            M,
            "m22_memory_observation(expected, not invariant)",
            "每卡权重是否≈1/k、KV 容量是否随 TP 上升，属**预期观察**而非不变量："
            f"{obs}。若偏离 1/k，须用 embedding/lm_head 是否切分、"
            "KV head 复制（config/model.py:1259-1270）等结构事实解释，**不得判为实验失败**",
        )

    # ---- 输出一致性：如实记录，只有硬错误才 FAIL ----
    cons = load_csv(f"outputs/m22/{M22_TAG}_output_consistency.csv")
    if cons:
        bad_len = [r for r in cons if r["out_lens_a"] != r["out_lens_b"]]
        rep.check(
            M,
            "m22_consistency_same_length",
            not bad_len,
            f"{len(bad_len)} 组跨 TP 输出长度不同（**硬错误**，不是数值差异）",
            f"{len(cons)} 组跨 TP 输出长度一致",
        )
        bad_p = [r for r in cons if r["prompt_hash_match"] != "True"]
        rep.check(
            M,
            "m22_consistency_same_prompt",
            not bad_p,
            f"{len(bad_p)} 组跨 TP prompt 不同（对照不成立）",
        )
        n_match = sum(1 for r in cons if r["output_hash_match"] == "True")
        rep.ok(
            M,
            "m22_consistency_record(diagnostic, not pass/fail)",
            f"跨 TP 输出逐 token 完全一致 {n_match}/{len(cons)}。"
            "TP 改变 GEMM 切分与归约顺序，有限精度下贪心 argmax 可能翻转"
            "（M5/M19 已有同类先例），故**不一致本身不判 FAIL**；"
            "但若出现在第 0 个 token 或长度不符，需升级为「需要调查」",
        )
    rep.ok(
        M,
        "m22_level_note",
        f"当前已采集 TP 档：{tps}。**只有真实跑过的档位可标 E/H**；"
        "PP/DP/EP/CP/P-D 本轮未做，仍为 ⬜",
    )


def check_m21(rep):
    """M21 不变量：服务端指标是否可信 + 各实验的完整性。

    核心是 E2：服务端 counter 增量必须与客户端 bench 汇总守恒。这条不过，
    后面所有基于面板的结论都不该信，所以它是硬 FAIL 而不是「观察项」。

    刻意不把「容量 = 某个数」写成不变量：容量随会话漂移（见 results.md §4），
    把它当不变量只会逼出假结论。
    """
    M = "m21"

    # ---- artifact 可执行性守卫（2026-08-12 audit 新增）----
    # 教训：`m21_gate1.py` 曾因把逗号写进行尾注释而 SyntaxError，但当时 13 条 invariant
    # 全部 PASS —— 因为没有任何一条要求「模块脚本至少得是可解析的 Python」。
    # 这里真实编译（py_compile.compile + doraise），不是检查文件是否存在。
    m21_scripts = [
        "m21_common.py", "m21_host_exporter.py", "m21_gate1.py", "m21_capacity_sweep.py",
        "m21_validate.py", "m21_bottleneck.py", "m21_capacity_users.py",
        "m21_fault_drills.py", "m21_e6_consolidate.py", "m21_observability_cost.py",
        "m21_env_fingerprint.py", "m21_figures.py",
    ]
    missing, broken = [], []
    with tempfile.TemporaryDirectory(prefix="m21_pyc_") as tmpd:
        for name in m21_scripts:
            path = _p("scripts", name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            try:
                # 写到临时目录：py_compile 拒绝 /dev/null 这类非常规文件，
                # 且不能污染仓库里的 __pycache__
                py_compile.compile(path, cfile=os.path.join(tmpd, name + "c"), doraise=True)
            except py_compile.PyCompileError as e:
                broken.append(f"{name}: {str(e).strip().splitlines()[-1][:120]}")
    rep.check(M, "m21_scripts_compile", not missing and not broken,
              f"缺失 {missing}；编译失败 {broken}",
              f"{len(m21_scripts)} 个 M21 正式脚本全部 py_compile 通过")

    runs = load_jsonl("outputs/m21/raw/e3_shapeA_60s_runs.jsonl")
    if not runs:
        rep.pending(M, "m21_e3_ladder", "缺 outputs/m21/raw/e3_shapeA_60s_runs.jsonl")
        return

    rep.check(M, "m21_e3_sample_count", len(runs) == 21,
              f"期望 7 档 × 3 seed = 21 条，实得 {len(runs)}",
              f"{len(runs)} 条 measure 记录")

    bad = []
    for r in runs:
        d = r.get("counter_delta", {})
        if d.get("vllm:request_success_total") != r.get("client_completed"):
            bad.append((r["run_id"], "I1 完成数"))
        if d.get("vllm:prompt_tokens_total") != r.get("client_total_input_tokens"):
            bad.append((r["run_id"], "I2 prompt tokens"))
        if d.get("vllm:generation_tokens_total") != r.get("client_total_output_tokens"):
            bad.append((r["run_id"], "I3 generation tokens"))
    rep.check(M, "m21_e2_server_client_conservation", not bad,
              f"{len(bad)} 处服务端 counter 与客户端汇总不符：{bad[:3]}",
              f"{len(runs)} 个窗口的 I1/I2/I3 全部精确相等")

    over = [r["run_id"] for r in runs if (r.get("running_max") or 0) > 64]
    rep.check(M, "m21_i4_running_le_max_num_seqs", not over,
              f"{len(over)} 条 running 超过 max_num_seqs=64：{over[:3]}",
              "全程 running ≤ 64")

    badkv = [r["run_id"] for r in runs
             if r.get("kv_usage_max") is not None and not 0.0 <= r["kv_usage_max"] <= 1.0]
    rep.check(M, "m21_i5_kv_usage_in_range", not badkv,
              f"kv_cache_usage_perc 越界：{badkv[:3]}", "kv_cache_usage_perc ∈ [0,1]")

    neg = [r["run_id"] for r in runs
           if any(v is not None and v < 0 for v in (r.get("counter_delta") or {}).values())]
    rep.check(M, "m21_i6_counter_monotonic", not neg,
              f"counter 增量为负（回绕/重置）：{neg[:3]}", "所有 counter 增量非负")

    lowcov = [(r["run_id"], round((r.get("waiting_samples") or 0) / max(r.get("duration_s") or 1, 1), 2))
              for r in runs
              if (r.get("waiting_samples") or 0) / max(r.get("duration_s") or 1, 1) < 0.8]
    rep.check(M, "m21_i7_scrape_coverage", not lowcov,
              f"1s 抓取覆盖率 < 0.8：{lowcov[:3]}", "每个窗口的抓取覆盖率 ≥ 0.8")

    e4 = load_json("outputs/m21/e4_bottleneck.json")
    if e4:
        names = {r["scenario"] for r in e4["runs"]}
        preempt = [r for r in e4["runs"]
                   if (r.get("counter_delta") or {}).get("vllm:num_preemptions_total", 0) > 0]
        rep.check(M, "m21_e4_scenarios", {"B1_cpu_submit", "B2_gpu_compute", "B4_queue"} <= names,
                  f"E4 场景缺失，实得 {sorted(names)}", f"E4 场景齐全：{sorted(names)}")
        rep.check(M, "m21_e4_preemption_constructed", bool(preempt),
                  "没有任何场景真的触发抢占 —— 结论里不得声称验证了 KV 受限签名",
                  f"抢占已构造成功：{preempt[0]['run_id']}，"
                  f"{preempt[0]['counter_delta']['vllm:num_preemptions_total']:.0f} 次")
        supply_only = [r for r in e4["runs"] if r["scenario"] == "B3_kv_pressure"]
        if supply_only:
            rep.ok(M, "m21_e4_supply_side_negative(diagnostic)",
                   "只压显存预算（B3，Shape A）到 util=0.10 仍只到 "
                   f"{max(r.get('kv_usage_max') or 0 for r in supply_only):.2f} 且零抢占 —— "
                   "如实记为负结果，抢占是靠需求侧（B3b 长输出）才造出来的")
    else:
        rep.pending(M, "m21_e4", "缺 outputs/m21/e4_bottleneck.json")

    # ---- 2026-08-12 audit 新增：两条更严格的判据（不是放宽既有判据）----
    e5 = load_json("outputs/m21/e5_user_capacity.json")
    if e5:
        # E5 语义回归守卫：闭环 population 与「系统内并发请求数」必须分开报告，
        # 且不得再出现 estimated_concurrent_users 这种把 X·(R+Z) 说成并发请求的命名。
        blob = json.dumps(e5, ensure_ascii=False)
        has_split = ("average_in_system_requests" in e5
                     and "user_population_sensitivity" in e5
                     and all("estimated_closed_loop_users" in x
                             for x in e5["user_population_sensitivity"]))
        rep.check(M, "m21_e5_closed_loop_semantics",
                  has_split and "estimated_concurrent_users" not in blob.replace(
                      "首版把 N 命名为 estimated_concurrent_users，属误导，已修正", ""),
                  "E5 未把「闭环用户 population」与「系统内并发请求数」分开报告，"
                  "或仍在用 estimated_concurrent_users 这一误导命名",
                  f"E5 已分列：L=X·R={e5.get('average_in_system_requests')} 与 "
                  f"N=X·(R+Z)={[x['estimated_closed_loop_users'] for x in e5['user_population_sensitivity']]}")
    else:
        rep.pending(M, "m21_e5", "缺 outputs/m21/e5_user_capacity.json")

    e6 = load_json("outputs/m21/e6_drills.json")
    if e6:
        # 单位回归守卫：PromQL 的 *_seconds_* 直方图返回秒，存进 _ms 字段必须 ×1000。
        # 判据用「服务端 ms 与客户端 ms 同量级（0.1×~10×）」这一物理约束来兜，
        # 首版把 19.57s 写成 19.57ms 时该比值是 0.0011，会直接 FAIL。
        for d in e6["drills"]:
            srv = d.get("server_hist_ttft_p99_ms_during_probe")
            cli = d.get("background_load_p99_ttft_ms")
            if srv and cli:
                ratio = srv / cli
                rep.check(M, "m21_e6_ttft_unit_consistency", 0.1 <= ratio <= 10.0,
                          f"服务端/客户端 TTFT P99 比值 {ratio:.4f} 偏离量级 —— "
                          f"多半是秒/毫秒单位错（srv={srv}, cli={cli}）",
                          f"服务端 {srv / 1000:.2f}s vs 客户端 {cli / 1000:.2f}s，"
                          f"比值 {ratio:.2f}（同量级，单位一致）")

        fired = set()
        for d in e6["drills"]:
            fired |= set((d.get("alerts_firing") or {}).keys())
        expected = {"VLLMServerDown", "HostExporterTargetDown", "VLLMTTFTP99AboveSLO",
                    "VLLMQueueBacklog", "VLLMKVCacheNearFull", "VLLMPreemptionHappening",
                    "GPUMemoryNearFull"}
        rep.check(M, "m21_e6_all_alert_rules_fired", expected <= fired,
                  f"未真实触发过的规则：{sorted(expected - fired)}（必须在文档里标未验证）",
                  f"7 条规则全部真实 firing 过：{sorted(fired)}")
        failed = [d["drill"] for d in e6["drills"] if not d.get("PASS")]
        rep.check(M, "m21_e6_drills_pass", not failed, f"未通过的演练：{failed}",
                  f"{len(e6['drills'])} 个演练全部 PASS")
    else:
        rep.pending(M, "m21_e6", "缺 outputs/m21/e6_drills.json")


def check_m18(rep):
    """M18 不变量（设计冻结 `notes/m18/quantization_w4a16.md` §5）。

    核心是「除模型与量化外没有第二个变量」：dtype / attention backend / kv_cache_dtype
    三者必须跨配置一致，否则测出来的差异不能归因给量化（M10 的教训：TRITON_ATTN
    后端本身就比 FLASH 慢，混进来会把后端开销算成量化收益）。
    """
    M = "m18"
    models, configs = ["0.5b", "1.5b"], ["Q0", "Q1", "Q2"]

    confs, mems, quals = {}, {}, {}
    missing = []
    for m in models:
        for c in configs:
            tag = f"{m}_{c}"
            cf = load_json(f"outputs/m18/configs/{tag}.json")
            mm = load_json(f"outputs/m18/memory/{tag}.json")
            qq = load_json(f"outputs/m18/quality/{tag}.json")
            if not (cf and mm and qq):
                missing.append(tag)
                continue
            confs[tag], mems[tag], quals[tag] = cf, mm, qq
    if missing:
        rep.pending(M, "m18_artifacts", f"缺少配置产物：{missing}")
        return

    # I1 三配置的生效 dtype 必须都是 bfloat16（设计 §3 V1 写死的隐藏变量）
    dtypes = {t: c["effective"]["dtype"] for t, c in confs.items()}
    rep.check(M, "m18_i1_dtype_frozen_bfloat16",
              all(v == "torch.bfloat16" for v in dtypes.values()),
              f"存在非 bfloat16 的生效 dtype：{dtypes}",
              "6 个配置的生效 dtype 全部为 torch.bfloat16（量化 checkpoint 声明 float16，已被显式覆盖）")

    # I2 attention backend 跨配置一致
    backends = {t: c["effective"]["attention_backend"] for t, c in confs.items()}
    rep.check(M, "m18_i2_same_attention_backend", len(set(backends.values())) == 1,
              f"attention backend 不一致：{backends}",
              f"全部一致：{next(iter(set(backends.values())))}")

    # I3 kv_cache_dtype 一致（KV 不是本模块的变量）
    kvd = {t: c["effective"]["kv_cache_dtype"] for t, c in confs.items()}
    rep.check(M, "m18_i3_same_kv_cache_dtype", len(set(kvd.values())) == 1,
              f"kv_cache_dtype 不一致：{kvd}", f"全部一致：{next(iter(set(kvd.values())))}")

    # I5 W4A16 侧必须出现 int32 打包权重；且 embed/lm_head 未被量化（跨配置字节数相同）
    bad_int32, bad_embed = [], []
    for m in models:
        embeds = {c: mems[f"{m}_{c}"]["tensor_sum"]["embed_lm_head_MiB"] for c in configs}
        if len(set(round(v, 2) for v in embeds.values())) != 1:
            bad_embed.append((m, embeds))
        for c in ("Q1", "Q2"):
            by = mems[f"{m}_{c}"]["tensor_sum"]["weight_bytes_by_dtype"]
            if by.get("torch.int32", 0) <= 0:
                bad_int32.append(f"{m}_{c}")
    rep.check(M, "m18_i5a_w4a16_has_int32_packed_weights", not bad_int32,
              f"W4A16 配置缺少 int32 打包权重：{bad_int32}",
              "Q1/Q2 均存在 int32 打包权重（Marlin repack 后的布局）")
    rep.check(M, "m18_i5b_embed_lm_head_not_quantized", not bad_embed,
              f"embed/lm_head 字节数跨配置不一致（本应未被量化）：{bad_embed}",
              "embed/lm_head 在三配置中字节数完全相同 → 未被量化，这解释了整体降幅达不到理论 75%")

    # I6 权重变小 ⇒ KV token 不减（同 gpu_memory_utilization 下的显存预算守恒方向）
    viol = []
    for m in models:
        base_w = sum(mems[f"{m}_Q0"]["tensor_sum"]["weight_bytes_by_dtype"].values())
        base_kv = confs[f"{m}_Q0"]["effective"]["kv_cache_size_tokens"]
        for c in ("Q1", "Q2"):
            w = sum(mems[f"{m}_{c}"]["tensor_sum"]["weight_bytes_by_dtype"].values())
            kv = confs[f"{m}_{c}"]["effective"]["kv_cache_size_tokens"]
            if w < base_w and kv < base_kv:
                viol.append(f"{m}_{c}: 权重 {base_w:.0f}->{w:.0f} MiB 但 KV {base_kv}->{kv}")
    rep.check(M, "m18_i6_kv_capacity_not_decreasing", not viol,
              f"权重变小但 KV 容量反降：{viol}",
              "权重下降的配置，KV 容量均不减（显存预算守恒方向正确）")

    # I7/I8 性能样本：只用 valid=True，且样本量达冻结值
    grid_bad, invalid = [], []
    for m in models:
        for c in configs:
            rows = load_jsonl(f"outputs/m18/raw/{m}_{c}_perf.jsonl") or []
            n_invalid = sum(1 for r in rows if not r.get("valid", True))
            if n_invalid:
                invalid.append(f"{m}_{c}:{n_invalid}")
            buckets = {}
            for r in rows:
                key = (("prefill", r["prompt_tokens"], None) if r["experiment"] == "prefill"
                       else ("decode", r["context_len"], r["batch_size"]))
                buckets[key] = buckets.get(key, 0) + 1
            for key, n in buckets.items():
                if n != 15:
                    grid_bad.append(f"{m}_{c}{key}={n}")
    rep.check(M, "m18_i7_measure_sample_count", not grid_bad,
              f"measure 样本数不等于冻结的 15：{grid_bad[:5]}",
              "全部网格点均为 15 个 measure 样本（warmup 未写入 raw）")
    rep.check(M, "m18_i8_all_samples_valid", not invalid,
              f"存在 valid=False 的样本（调度形态不符）：{invalid}",
              "所有性能样本 valid=True（decode 每步 scheduled_tokens 与 running 数均符合预期）")

    # I9 PPL 评分 token 数跨配置相同，否则 ΔPPL 不可比
    scored = {t: q["overall"]["scored_tokens"] for t, q in quals.items()}
    rep.check(M, "m18_i9_same_scored_tokens", len(set(scored.values())) == 1,
              f"PPL 评分 token 数不一致：{scored}",
              f"六个配置的评分 token 数均为 {next(iter(set(scored.values())))}")

    # 诊断项（非判据）：把「实际降幅 vs 理论降幅」的差距如实记下来
    for m in models:
        base = mems[f"{m}_Q0"]["tensor_sum"]
        base_tot = sum(base["weight_bytes_by_dtype"].values())
        q1 = mems[f"{m}_Q1"]["tensor_sum"]
        q1_tot = sum(q1["weight_bytes_by_dtype"].values())
        embed = base["embed_lm_head_MiB"]
        whole = 100 * (q1_tot - base_tot) / base_tot
        part = 100 * ((q1_tot - embed) - (base_tot - embed)) / (base_tot - embed)
        rep.ok(M, f"m18_weight_saving_record[{m}](diagnostic)",
               f"整体权重 {base_tot:.0f}→{q1_tot:.0f} MiB（{whole:+.1f}%），"
               f"但**可量化部分** {base_tot - embed:.0f}→{q1_tot - embed:.0f} MiB（{part:+.1f}%，接近 4bit 理论值）；"
               f"差距来自未被量化的 embed/lm_head（{embed:.0f} MiB，占基线 {100 * embed / base_tot:.0f}%）")


def check_m23(rep):
    """M23 L1 不变量：2×2 性能对照 + 跨引擎 generation consistency。"""
    M = "m23"
    scripts = [
        "m23_gate.py", "m23_vctr_preflight.py", "m23_vo_correctness.py",
        "m23_bench.py", "m23_summarize.py", "m23_figures.py",
        "m23_cross_engine_consistency.py",
    ]
    missing, broken = [], []
    with tempfile.TemporaryDirectory(prefix="m23_pyc_") as tmpd:
        for name in scripts:
            path = _p("scripts", name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            try:
                py_compile.compile(path, cfile=os.path.join(tmpd, name + "c"), doraise=True)
            except py_compile.PyCompileError as e:
                broken.append(f"{name}: {str(e).strip().splitlines()[-1][:120]}")
    rep.check(M, "m23_scripts_compile", not missing and not broken,
              f"缺失 {missing}；编译失败 {broken}",
              f"{len(scripts)} 个 M23 脚本全部 py_compile 通过")

    cons = load_json("outputs/m23/cross_engine_consistency.json")
    if not cons:
        rep.pending(M, "m23_consistency_artifact",
                    "缺 outputs/m23/cross_engine_consistency.json")
        return

    for g in ("V-E", "V-O", "T-E", "T-O"):
        raw = load_json(f"outputs/m23/raw/consistency_{g}.json")
        gens = ((raw or {}).get("record") or {}).get("generations") or []
        rep.check(M, f"m23_consistency_raw[{g}]",
                  raw is not None and len(gens) == 6
                  and all((x.get("text") or "").strip() for x in gens),
                  f"{g} raw 缺失或不足 6 条非空生成",
                  f"{g} 6/6 非空")

    fs = cons.get("functional_sanity") or {}
    rep.check(M, "m23_functional_sanity_pass", fs.get("PASS") is True,
              f"functional_sanity.PASS={fs.get('PASS')} reasons={fs.get('hard_fail_reasons')}",
              "无崩溃/空输出/超长")
    disc = cons.get("discipline") or {}
    rep.check(M, "m23_mismatch_not_correctness_failure",
              "NOT correctness" in str(disc.get("mismatch_is", "")),
              f"discipline.mismatch_is 未标明非 correctness failure：{disc.get('mismatch_is')}",
              str(disc.get("mismatch_is")))

    ce = cons.get("cross_engine") or {}
    rep.check(M, "m23_cross_engine_pairs_only_same_lane",
              set(ce) == {"T-E vs V-E", "T-O vs V-O"},
              f"cross_engine keys 应为 T-E vs V-E / T-O vs V-O，实得 {list(ce)}",
              "仅 T-E vs V-E、T-O vs V-O")
    forbidden = disc.get("forbidden_for_engine_speedup") or []
    rep.check(M, "m23_no_cross_lane_engine_speedup_in_artifact",
              "T-E vs V-O" in forbidden and "T-O vs V-E" in forbidden
              and "T-E vs V-O" not in ce and "T-O vs V-E" not in ce,
              f"交叉 lane 纪律未落盘：forbidden={forbidden} keys={list(ce)}",
              "交叉 lane 未进入正式对照")

    off, on = ce.get("T-E vs V-E") or {}, ce.get("T-O vs V-O") or {}
    rep.check(M, "m23_off_lane_exact_2_of_6",
              (off.get("exact_match") or {}).get("matched") == 2
              and (off.get("exact_match") or {}).get("total") == 6,
              f"T-E vs V-E exact={off.get('exact_match')}",
              "2/6")
    rep.check(M, "m23_on_lane_exact_1_of_6",
              (on.get("exact_match") or {}).get("matched") == 1
              and (on.get("exact_match") or {}).get("total") == 6,
              f"T-O vs V-O exact={on.get('exact_match')}",
              "1/6")

    p4 = next((r for r in (off.get("per_prompt") or []) if r.get("prompt_index") == 4), None)
    rep.check(M, "m23_trt_eos_early_stop_is_prefix",
              p4 is not None and p4.get("T-E_n_tokens") == 45
              and p4.get("V-E_n_tokens") == 64
              and p4.get("shorter_is_prefix_of_longer") is True
              and p4.get("common_prefix_len") == 45,
              f"prompt4 不是「TRT EOS 前缀」形态：{p4}",
              "T-E prompt4 = 45 token，且是 V-E 的前缀")

    same = cons.get("same_engine_ablation") or {}
    vo_ve = (same.get("V-O vs V-E") or {}).get("exact_match") or {}
    pf = load_json("outputs/m23/vctr_preflight.json") or {}
    pf_cl = (pf.get("cross_lane_exact_match") or {})
    rep.check(M, "m23_vo_ve_matches_vctr_preflight",
              vo_ve.get("matched") == 1 and vo_ve.get("total") == 6
              and pf_cl.get("matched") == 1 and pf_cl.get("total") == 6,
              f"V-O vs V-E={vo_ve} preflight={pf_cl}",
              "两侧都是 1/6，与 V-ctr preflight 一致")

    summary = load_json("outputs/m23/cross_engine_summary.json")
    csv_rows = load_csv("outputs/m23/cross_engine.csv")
    groups = {r["group"] for r in (csv_rows or [])}
    rep.check(M, "m23_bench_four_groups",
              groups == {"V-E", "V-O", "T-E", "T-O"} and summary is not None,
              f"bench 组不全：{groups}",
              "四组性能汇总齐全")

    # Shape A 定长：四组 measure 输出全部 128 token（raw 字段 out_len_all_equal）
    len_bad = []
    for g in ("V-E", "V-O", "T-E", "T-O"):
        raw = load_json(f"outputs/m23/raw/bench_{g}.json")
        rec = (raw or {}).get("record") or {}
        if rec.get("out_len_all_equal") is not True:
            len_bad.append(f"{g}: out_len_all_equal={rec.get('out_len_all_equal')}")
        if rec.get("n_out_checked") != 330:
            len_bad.append(f"{g}: n_out_checked={rec.get('n_out_checked')} != 330")
    csv_len_bad = [
        r["group"] for r in (csv_rows or [])
        if str(r.get("out_len_all_equal")) != "True"
    ]
    rep.check(M, "m23_shapeA_out_len_128",
              not len_bad and not csv_len_bad,
              f"raw={len_bad} csv={csv_len_bad}",
              "四组 raw+csv 均为 out_len_all_equal=True（330 条 × 128 token）")

    doc = open(_p("notes/m23/results.md"), encoding="utf-8").read() if os.path.exists(_p("notes/m23/results.md")) else ""
    needles = [
        ("2/6", "T-E vs V-E exact 2/6"),
        ("1/6", "T-O vs V-O / V-O vs V-E exact 1/6"),
        ("禁止交叉 lane", "新比较纪律"),
        ("同引擎 execution-mode ablation", "允许 V-O vs V-E / T-O vs T-E"),
        ("numerical divergence", "mismatch 口径"),
        ("不得直接判为 correctness failure", "mismatch 非正确性失败"),
        ("eos_early_stop", "TRT EOS 记录"),
    ]
    missing_doc = [label for s, label in needles if s not in doc]
    rep.check(M, "m23_results_doc_numbers", not missing_doc,
              f"results.md 缺：{missing_doc}",
              "一致性数字与比较纪律已写入 results.md")


M24_TAG = "h20_formal"


def check_m24(rep):
    """M24 DCP 不变量。无 H20 数据时 PENDING，不得标 E/H。"""
    M = "m24"
    scripts = [
        "m24_dcp.py",
        "m24_run_all.py",
        "m24_pcp_negative.py",
        "m24_figures.py",
        "m24_preflight.sh",
    ]
    missing, broken = [], []
    with tempfile.TemporaryDirectory(prefix="m24_pyc_") as tmpd:
        for name in scripts:
            path = _p("scripts", name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            if name.endswith(".sh"):
                continue
            try:
                py_compile.compile(
                    path, cfile=os.path.join(tmpd, name + "c"), doraise=True
                )
            except py_compile.PyCompileError as e:
                broken.append(f"{name}: {str(e).strip().splitlines()[-1][:120]}")
    rep.check(
        M,
        "m24_scripts_present",
        not missing and not broken,
        f"缺失 {missing}；编译失败 {broken}",
        "M24 脚本齐全且 py_compile 通过",
    )
    plan = open(_p("notes/m24/M24_H20_PLAN.md"), encoding="utf-8").read() if os.path.exists(_p("notes/m24/M24_H20_PLAN.md")) else ""
    rep.check(
        M,
        "m24_plan_frozen",
        "decode_context_parallel_size" in plan and "1.5B" in plan,
        "缺 notes/m24/M24_H20_PLAN.md 或未钉死 1.5B/DCP",
        "H20 计划已冻结",
    )

    d = _p("outputs/m24/raw")
    rows = []
    if os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if n.startswith(f"{M24_TAG}_dcp") and n.endswith("_run.jsonl"):
                rows.extend(load_jsonl(f"outputs/m24/raw/{n}") or [])
    if not rows:
        rep.pending(
            M,
            "artifacts",
            f"outputs/m24/raw/{M24_TAG}_dcp*_run.jsonl 不存在 —— "
            "M24 多卡实验尚未执行。当前仅 T，不得标 E/H",
        )
        return

    meas = measure_rows(rows)
    dcps = sorted({r.get("dcp") for r in rows if r.get("dcp") is not None})
    cfgs = {}
    for r in rows:
        if r.get("effective_cfg"):
            cfgs[r.get("dcp")] = r["effective_cfg"]
    keys = [
        "async_scheduling",
        "tensor_parallel_size",
        "prefill_context_parallel_size",
        "max_num_batched_tokens",
        "max_num_seqs",
        "enable_prefix_caching",
        "enforce_eager",
        "max_model_len",
        "dtype",
    ]
    diff = []
    if len(cfgs) > 1:
        ref_d = min(cfgs)
        ref = cfgs[ref_d]
        for dcp, c in cfgs.items():
            for k in keys:
                if c.get(k) != ref.get(k):
                    diff.append((k, ref_d, ref.get(k), dcp, c.get(k)))
    rep.check(
        M,
        "m24_single_variable_dcp",
        not diff,
        f"DCP 档之间存在非 DCP 配置差异：{diff[:4]}",
        f"{len(cfgs)} 档除 DCP 外生效配置相同",
    )
    for dcp, c in cfgs.items():
        rep.check(
            M,
            f"m24_dcp_effective[{dcp}]",
            c.get("decode_context_parallel_size") == c.get("dcp_requested") == dcp,
            f"DCP={dcp} 回读不符：{c.get('dcp_requested')} vs {c.get('decode_context_parallel_size')}",
        )
        rep.check(
            M,
            f"m24_tp_frozen[{dcp}]",
            c.get("tensor_parallel_size") == 4,
            f"DCP={dcp} 的 TP={c.get('tensor_parallel_size')}，必须为 4",
        )
        rep.check(
            M,
            f"m24_pcp_off[{dcp}]",
            c.get("prefill_context_parallel_size") == 1,
            f"正式档不得开 PCP：{c.get('prefill_context_parallel_size')}",
        )
        qpk = None
        hq, hkv = c.get("total_num_attention_heads"), c.get("total_num_kv_heads")
        if hq and hkv:
            qpk = hq // hkv
        if dcp and dcp > 1 and qpk:
            rep.check(
                M,
                f"m24_gqa_divisible[{dcp}]",
                qpk % dcp == 0,
                f"H_q/H_kv={qpk} 不能被 dcp={dcp} 整除",
            )

    d2 = [r for r in meas if r.get("experiment") == "D2_decode"]
    viol = [
        r
        for r in d2
        if any(n != r["max_tokens_requested"] for n in (r.get("actual_output_tokens") or []))
    ]
    rep.check(
        M,
        "m24_d2_output_len_exact",
        not viol,
        f"{len(viol)} 个 D2 输出长度不符",
        f"{len(d2)} 个 D2 measure 输出长度精确",
    )
    viol = [r for r in d2 if r.get("steady_batch_stable") is not True]
    rep.check(
        M,
        "m24_d2_steady_batch_stable",
        not viol,
        f"{len(viol)} 个 D2 稳态 batch 不恒定",
        f"{len(d2)} 个 D2 稳态 batch 恒定",
    )
    csv_rows = load_csv(f"outputs/m24/{M24_TAG}_decode.csv") or []
    if csv_rows:
        n_csv = sum(int(r["n"]) for r in csv_rows if r.get("n"))
        rep.check(
            M,
            "m24_csv_n_matches_jsonl",
            n_csv == len(d2),
            f"CSV n 合计 {n_csv} != jsonl measure {len(d2)}",
            "decode CSV 与 jsonl measure n 一致",
        )


M27_TAG = "h20_formal"
M27_SCRIPTS = [
    "m27_common.py",
    "m27_launch.py",
    "m27_proxy.py",
    "m27_workload.py",
    "m27_run_all.py",
    "m27_summarize.py",
    "m27_trace_analyze.py",
    "m27_figures.py",
    "m27_final_evidence_audit.py",
    "m27_preflight.sh",
]
M27_PLAN = "notes/m27/M27_H20_PLAN.md"
M27_RAW_MANIFEST = "notes/m27/H20_RAW_ARCHIVE_MANIFEST.tsv"
M27_ARCHIVE_SHA256 = "fb1664ca9506efb75bb784ab19a8b65186c6eabedf69127805e8e72b29633ad6"
M27_FREEZE_RE = re.compile(
    r"<!-- M27_FROZEN_CONFIG:BEGIN -->\s*```json\s*(\{.*?\})\s*```\s*<!-- M27_FROZEN_CONFIG:END -->",
    re.S,
)
M27_EFFECTIVE_FREE_KEYS = {
    "max_num_batched_tokens",
    "kv_connector",
    "kv_role",
    "kv_load_failure_policy",
    "kv_engine_id",
}


def _m27_common():
    sd = _p("scripts")
    if sd not in sys.path:
        sys.path.insert(0, sd)
    import m27_common  # 仅标准库，不 import vllm

    return m27_common


def _m27_raw(tag):
    """{(exp, arm): rows}；不含 proxy / lifecycle 文件。"""
    d = _p("outputs/m27/raw")
    out = {}
    if not os.path.isdir(d):
        return out
    for n in sorted(os.listdir(d)):
        if not (n.startswith(tag + "_") and n.endswith(".jsonl")):
            continue
        if n.endswith("_proxy.jsonl") or n == f"{tag}_lifecycle.jsonl":
            continue
        exp, _, arm = n[len(tag) + 1 : -len(".jsonl")].partition("_")
        out[(exp, arm)] = load_jsonl(f"outputs/m27/raw/{n}") or []
    return out


def _m27_static(rep, C):
    M = "m27"
    missing, broken = [], []
    with tempfile.TemporaryDirectory(prefix="m27_pyc_") as tmpd:
        for name in M27_SCRIPTS:
            path = _p("scripts", name)
            if not os.path.exists(path):
                missing.append(name)
                continue
            if name.endswith(".sh"):
                continue
            try:
                py_compile.compile(path, cfile=os.path.join(tmpd, name + "c"), doraise=True)
            except py_compile.PyCompileError as e:
                broken.append(f"{name}: {str(e).strip().splitlines()[-1][:120]}")
    rep.check(M, "m27_scripts_present", not missing and not broken,
              f"缺失 {missing}；编译失败 {broken}", "M27 脚本齐全且 py_compile 通过")

    F = C.FROZEN
    plan = open(_p(M27_PLAN), encoding="utf-8").read() if os.path.exists(_p(M27_PLAN)) else ""
    m = M27_FREEZE_RE.search(plan)
    block = None
    if m:
        try:
            block = json.loads(m.group(1))
        except json.JSONDecodeError as e:
            block = f"json error: {e}"
    rep.check(M, "m27_config_freeze", block == json.loads(json.dumps(F)),
              f"{M27_PLAN} 的 M27_FROZEN_CONFIG 块缺失或 ≠ m27_common.FROZEN",
              "计划文档冻结块 == m27_common.FROZEN（逐键相等）")

    e = F["engine"]
    want = {"tensor_parallel_size": 1, "dtype": "bfloat16", "seed": 0, "enforce_eager": True,
            "enable_prefix_caching": False, "async_scheduling": False,
            "enable_chunked_prefill": True, "block_size": 16, "max_model_len": 32768,
            "max_num_batched_tokens": 8192, "max_num_seqs": 256,
            "gpu_memory_utilization": 0.85, "quantization": None, "speculative_config": None}
    bad = {k: e.get(k) for k, v in want.items() if e.get(k) != v}
    lf = F["legal_fallback"]
    if not (F["max_prompt_len"] == 16384 < e["max_model_len"]
            and lf["max_model_len"] == 16384 and lf["max_prompt_len"] == 8192):
        bad["max_prompt/fallback"] = (F["max_prompt_len"], lf)
    shapes = F["e3"]["shapes"]
    if any(i + o > e["max_model_len"] or i > F["max_prompt_len"] for i, o in shapes.values()):
        bad["e3_shapes"] = shapes
    if max(F["e1"]["contexts"]) > F["max_prompt_len"] or max(F["e2"]["p_lens"]) > F["max_prompt_len"]:
        bad["context_points"] = (F["e1"]["contexts"], F["e2"]["p_lens"])
    if abs(sum(F["e3"]["mixture"].values()) - 1.0) > 1e-9:
        bad["mixture_weights"] = F["e3"]["mixture"]
    if F["warmup"] < 5 or F["measure"] < 15:
        bad["samples"] = (F["warmup"], F["measure"])
    if F["kv_transfer"]["kv_load_failure_policy"] != "fail":
        bad["kv_load_failure_policy"] = F["kv_transfer"]["kv_load_failure_policy"]
    rep.check(M, "m27_frozen_constraints", not bad, f"冻结配置违反约束：{bad}",
              "TP1/eager/APC off/async off/chunked ON/bs16/mml32768/max_prompt16384/fail-policy；"
              "warmup≥5 measure≥15")

    per_tok = C.kv_bytes_per_token()
    mm = F["model"]
    rep.check(M, "m27_kv_bytes_per_token",
              per_tok == F["kv_bytes_per_token"] == 57344 ==
              2 * mm["num_hidden_layers"] * mm["num_key_value_heads"] * mm["head_dim"] * 2
              and C.kv_theory(16384)["theory_bytes_blocks"] == 16384 * 57344
              and C.kv_theory(17)["theory_blocks"] == 2,
              f"KV 理论值不符：{per_tok}", "2×28×4×128×2 = 57344 B/token；整 block 向上取整")
    delivered = C.calibration_delivered_rate(11, 30)
    rep.check(M, "m27_calibration_rate_window",
              delivered == 11 / 30 and delivered >= 0.95 * (11 / 30) and
              11 / 31.808343842625618 < 0.95 * (11 / 30),
              "校准用 30s offered 对排空后 wall achieved，首级会被机械判失败",
              "30s Poisson 到达窗口 offered 与 delivered 同分母；wall 吞吐另报")
    try:
        from m27_workload import build_schedule
        full = build_schedule("mixture", 2.0, 30, 9999)
        capped = build_schedule("mixture", 2.0, 1e9, 9999, max_requests=16)
        rep.check(M, "m27_warmup_schedule_bounded",
                  len(full) >= 16 and len(capped) == 16 and capped == full[:16] and
                  build_schedule("d_heavy", 2.0, 1e9, 9999, max_requests=16)[-1]["i"] == 15,
                  "warmup 截断改变了原首 16 个 Poisson 请求或仍无限生成",
                  "warmup 仅生成 16 个且保留 Poisson seed/key/shape，正式 schedule 不变")
    except ImportError:
        rep.pending(M, "m27_warmup_schedule_bounded", "缺 aiohttp，无法导入 workload 验证 E3 warmup")

    # 只核对仓库内的身份文件；模型目录是机器本地资源，由 H20 preflight 校验，不在这里要求存在
    sums_path = _p(mm["identity_manifest"])
    viol = []
    if not os.path.exists(sums_path):
        viol.append("identity manifest 缺失")
    else:
        with open(sums_path, "rb") as f:
            raw = f.read()
        if hashlib.sha256(raw).hexdigest() != mm["identity_manifest_sha256"]:
            viol.append("identity manifest sha256 ≠ FROZEN")
        entries = dict(reversed(line.split(None, 1)) for line in raw.decode().splitlines() if line.strip())
        names = set(entries)
        need = {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "vocab.json", "merges.txt", "model.safetensors.index.json"}
        shards = {n for n in names if re.fullmatch(r"model-\d{5}-of-\d{5}\.safetensors", n)}
        if len(names) != mm["file_count"] or not need <= names or len(shards) != 4:
            viol.append(f"manifest 条目不符：{len(names)} 个文件，{len(shards)} 个分片，缺 {sorted(need - names)}")
        if not all(re.fullmatch(r"[0-9a-f]{64}", h) for h in entries.values()):
            viol.append("manifest 中有非 sha256 值")
        hf = load_json("notes/m27/model_artifact/hf_revision_match.json") or {}
        if hf.get("hf_revision") != mm["hf_revision"] or hf.get("all_hf_files_matched") is not True:
            viol.append("hf_revision_match.json 缺失、revision 不符或未全部一致")
        elif any(r["sha256"] != entries.get(r["file"]) for r in hf.get("files", [])):
            viol.append("hf_revision_match.json 与 identity manifest 的 sha256 不一致")
    rep.check(M, "m27_model_identity", not viol, f"模型身份文件不一致：{viol}",
              f"identity manifest sha256 == FROZEN；{mm['file_count']} 个文件（4 分片 + index + config/tokenizer）；"
              f"内容与 HF {mm['hf_repo']}@{mm['hf_revision'][:8]} 一致")

    m22 = next((r for r in (load_jsonl("outputs/m22/raw/h20_formal_tp1_run.jsonl") or [])
                if r.get("sum_available_kv_bytes")), None)
    if m22 is None:
        rep.pending(M, "m27_kv_theory_cross_m22", "缺 M22 TP1 formal raw，无法做第二来源核对")
    else:
        avail = m22["sum_available_kv_bytes"]
        toks = (m22.get("effective_cfg") or {}).get("sched_kv_cache_size_tokens")
        bs = e["block_size"]
        rep.check(M, "m27_kv_theory_cross_m22", toks == (avail // (bs * per_tok)) * bs,
                  f"M22 引擎 KV 容量 {toks} tokens 与 {avail}//({bs}×{per_tok})×{bs} 不符",
                  f"M22 H20 TP1 引擎实测：{avail} B → {toks} tokens，与 57344 B/token 整 block 推算一致")

    all_arms = sorted(set(F["e2"]["arms"]) | set(F["e3"]["arms"]) | set(F["e4"]["arms"])
                      | set(F["e5_optional"]["arms"]) | {F["e3"]["calibration_arm"],
                                                         F["e4"]["calibration_arm"]})
    viol = []
    for arm in all_arms:
        t = C.parse_arm(arm)
        ins = C.build_instances(arm, list(range(t["n_gpus"])))
        for key in ("gpu", "http_port", "side_channel_port"):
            if len({i[key] for i in ins}) != len(ins):
                viol.append((arm, key))
        roles = [i["role"] for i in ins]
        if t["kind"] == "pd" and (roles.count("P"), roles.count("D")) != (t["n_p"], t["n_d"]):
            viol.append((arm, "roles"))
        if t["kind"] == "pd" and t["mbt"] != e["max_num_batched_tokens"]:
            viol.append((arm, "pd_mbt"))
        for i in ins:
            ktc = C.kv_transfer_config(i["role"])
            want_role = {"P": "kv_producer", "D": "kv_consumer", "colo": None}[i["role"]]
            if (ktc or {}).get("kv_role") != want_role:
                viol.append((arm, i["name"], "kv_role"))
            cmd = C.serve_cmd("/readonly/model", i)
            pos = cmd.index("--shutdown-timeout") if "--shutdown-timeout" in cmd else -1
            if pos < 0 or cmd[pos + 1] != str(C.SHUTDOWN_TIMEOUT_S):
                viol.append((arm, i["name"], "shutdown_timeout"))
    rep.check(M, "m27_arm_construction", not viol, f"arm 构造违规：{viol[:6]}",
              f"{len(all_arms)} 个 arm：GPU/HTTP/side-channel 端口唯一，角色与 kv_role 对应")

    m24_plan = _p("notes/m24/M24_H20_PLAN.md")
    rep.check(M, "m27_m24_plan_not_mutated",
              os.path.exists(m24_plan) and "M27" not in open(m24_plan, encoding="utf-8").read(),
              "M24_H20_PLAN.md 缺失或被写入 M27 内容（冻结计划不得改动）",
              "M24 冻结计划未被写入 M27 内容")

    try:
        import yaml

        man = yaml.safe_load(open(_p("config/repro_manifest.yaml"), encoding="utf-8"))
        mod = (man.get("modules") or {}).get("m27") or {}
        steps = mod.get("steps") or []
        leaks = [s.get("id") for s in steps
                 if s.get("requires_gpu") and (not s.get("formal_only")
                                               or s.get("status") in ("implemented", "in_progress"))]
        rep.check(M, "m27_manifest_not_runnable",
                  bool(steps) and mod.get("status") not in ("implemented", "in_progress") and not leaks,
                  f"manifest m27 缺失或 GPU 步骤可被 --run 启动：status={mod.get('status')} leaks={leaks}",
                  f"m27 status={mod.get('status')}；{sum(1 for s in steps if s.get('requires_gpu'))} 个 GPU "
                  "步骤均 formal_only 且非 runnable")
    except ImportError:
        rep.pending(M, "m27_manifest_not_runnable", "无 PyYAML，跳过 manifest 核对")


def _m27_external_raw(rep):
    """>100MB 的 formal raw 不在 Git 中：manifest 与 .gitignore 必须一致；本地存在时逐字节核对，缺失记 PENDING。"""
    M = "m27"
    path = _p(M27_RAW_MANIFEST)
    rows = []
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
    ignored = set()
    if os.path.exists(_p(".gitignore")):
        with open(_p(".gitignore"), encoding="utf-8") as f:
            ignored = {line.strip() for line in f}
    bad = [r.get("relative_path") for r in rows
           if r.get("archive_sha256") != M27_ARCHIVE_SHA256
           or r.get("repository_policy") != "external_evidence_not_git_tracked"
           or int(r.get("size_bytes") or 0) <= 100 * 2**20
           or not re.fullmatch(r"[0-9a-f]{64}", r.get("sha256") or "")
           or r.get("relative_path") not in ignored]
    rep.check(M, "m27_external_raw_manifest", len(rows) == 11 and not bad,
              f"{M27_RAW_MANIFEST} 缺失/条目数≠11/字段或 .gitignore 精确路径不符：{bad[:4]}",
              f"{len(rows)} 个 >100MB raw：archive sha256 一致、policy=external_evidence_not_git_tracked、"
              ".gitignore 逐条精确列出")
    absent, mismatch = [], []
    for r in rows:
        fp = _p(r["relative_path"])
        if not os.path.exists(fp):
            absent.append(r["relative_path"])
            continue
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        if os.path.getsize(fp) != int(r["size_bytes"]) or h.hexdigest() != r["sha256"]:
            mismatch.append(r["relative_path"])
    if mismatch or not absent:
        rep.check(M, "m27_external_raw_identity", bool(rows) and not mismatch,
                  f"本地外部 raw 与 manifest 的 size/sha256 不符：{mismatch[:4]}",
                  f"{len(rows)} 个外部 raw 本地存在且 size/sha256 与 manifest 逐个一致")
    else:
        rep.pending(M, "m27_external_raw_identity",
                    f"{len(absent)} 个外部 raw 不在本地（完整字节见 archive {M27_ARCHIVE_SHA256[:12]}…）；"
                    "依赖它们的 E3/E4 检查未运行")


def _m27_final_evidence_audit(rep):
    M = "m27"
    try:
        import m27_final_evidence_audit as G
        out = G.build()
    except ImportError as e:
        rep.pending(M, "m27_final_evidence_audit_regenerable", f"缺依赖，无法重算：{e}")
        return
    if out["status"] == "INCOMPLETE":
        rep.pending(M, "m27_final_evidence_audit_regenerable",
                    f"{out['n_unavailable']} 项依赖的外部 raw 不在本地，无法完整重算 final evidence audit")
        return
    on_disk = load_json(G.OUTPUT) or {}
    out = json.loads(json.dumps(out))
    rep.check(M, "m27_final_evidence_audit_regenerable",
              out["status"] == "PASS" and out["original"]["check_names_equal"]
              and out["original"]["all_original_true"] and on_disk == out,
              f"status={out['status']} {out['n_pass']}/{out['n_checks']}；"
              f"原 73 项同名={out['original']['check_names_equal']}；落盘一致={on_disk == out}",
              f"离线重算 {out['n_pass']}/{out['n_checks']} PASS，检查项与 H20 原文件同名同序，"
              f"与 {G.OUTPUT} 逐键一致")


def check_m27(rep):
    """M27 P-D 解耦 + NIXL 不变量。无 H20 raw 时 PENDING；「PD 更快」不是不变量。"""
    M = "m27"
    C = _m27_common()
    _m27_static(rep, C)
    _m27_external_raw(rep)
    tag = M27_TAG
    R = _m27_raw(tag)

    derived = sorted(
        n for n in (os.listdir(_p("outputs/m27")) if os.path.isdir(_p("outputs/m27")) else [])
        if n.startswith(tag + "_") and n.endswith((".csv", ".json"))
    ) + sorted(n for n in os.listdir(_p("figures"))
               if re.match(rf"fig1[678]_.*_{tag}\.png$", n))
    if not R:
        rep.check(M, "m27_no_derived_without_raw", not derived,
                  f"没有 raw 却存在派生产物（疑似伪造/残留）：{derived[:5]}",
                  "无 raw 时没有任何派生 CSV/JSON/图")
        rep.pending(M, "artifacts",
                    f"outputs/m27/raw/{tag}_*.jsonl 不存在 —— M27 H20 实验尚未执行。"
                    "当前仅 T/O + DESIGN，不得标 E/H")
        return

    F = C.FROZEN
    pf = load_json(f"outputs/m27/configs/preflight_{tag}.json") or {}
    gm = load_json(f"outputs/m27/configs/gpu_map_{tag}.json") or {}
    rep.check(M, "m27_preflight_pass", pf.get("result") == "PASS" and bool(gm.get("gpu_ids")),
              f"preflight={pf.get('result')} gpu_map={gm.get('gpu_ids')}",
              f"Gate 0 PASS @ {pf.get('hostname')}；gpu_ids={gm.get('gpu_ids')}")
    allowed_gpus = set(gm.get("gpu_ids") or [])
    mml = F["legal_fallback"]["max_model_len"] if gm.get("max_model_len_fallback") else F["engine"]["max_model_len"]

    bad_schema, not_done = [], []
    for (exp, arm), rows in R.items():
        for r in rows:
            if (r.get("schema") != C.SCHEMA or r.get("tag") != tag or r.get("experiment") != exp
                    or r.get("arm") != arm
                    or (r.get("phase") != "arm_done" and r.get("plan_version") != F["plan_version"])):
                bad_schema.append((exp, arm, r.get("phase")))
        if not any(r.get("phase") == "arm_done" for r in rows):
            not_done.append((exp, arm))
    rep.check(M, "m27_raw_schema", not bad_schema, f"{len(bad_schema)} 行 schema/身份不符：{bad_schema[:4]}",
              f"{sum(len(v) for v in R.values())} 行 schema={C.SCHEMA}、tag/experiment/arm 与文件名一致")
    proxy_bad = []
    for exp, arm in R:
        events = load_jsonl(f"outputs/m27/raw/{tag}_{exp}_{arm}_proxy.jsonl") or []
        configs = [r for r in events if r.get("event") == "upstream_transport_config"]
        if len(configs) != 1 or configs[0].get("force_close") is not True:
            proxy_bad.append((exp, arm))
    rep.check(M, "m27_upstream_connection_policy", not proxy_bad,
              f"PD/colo 的代理上游连接策略未双臂一致或旧 raw 未重跑：{proxy_bad[:5]}",
              "全部正式 arm 的 proxy 上游均 force_close（避免复用已断开的长连接）")
    rep.check(M, "m27_arm_complete", not not_done, f"未完成（无 arm_done）：{not_done}",
              f"{len(R)} 个 (exp, arm) 均有 arm_done")

    effs = {}
    for (exp, arm) in R:
        effs[(exp, arm)] = load_json(f"outputs/m27/configs/{tag}_{exp}_{arm}_effective.json")
    eff_bad = [k for k, v in effs.items() if not v or v.get("effective_matches_requested") is not True]
    rep.check(M, "m27_effective_config_readback", not eff_bad,
              f"缺回读或生效≠请求：{eff_bad}", "每个 (exp, arm) 启动回读：server_info + 日志双来源 == 请求")

    ref, diff, gpu_viol, role_viol = None, [], [], []
    for (exp, arm), ef in effs.items():
        if not ef:
            continue
        t = C.parse_arm(arm)
        insts = ef.get("instances") or {}
        gpus = [v["instance"]["gpu"] for v in insts.values()]
        ports = [v["instance"]["http_port"] for v in insts.values()]
        sides = [v["instance"]["side_channel_port"] for v in insts.values()]
        if (len(set(gpus)) != len(gpus) or len(set(ports)) != len(ports)
                or len(set(sides)) != len(sides) or not set(gpus) <= allowed_gpus
                or len(gpus) != t["n_gpus"]):
            gpu_viol.append((exp, arm, gpus))
        for name, v in insts.items():
            eff = v.get("effective") or {}
            common = {k: x for k, x in eff.items() if k not in M27_EFFECTIVE_FREE_KEYS}
            if ref is None:
                ref = (exp, arm, name, common)
            elif common != ref[3]:
                diff.append((exp, arm, name, sorted(k for k in common if common.get(k) != ref[3].get(k))))
            role = v["instance"]["role"]
            want_role = {"P": "kv_producer", "D": "kv_consumer", "colo": None}[role]
            if eff.get("kv_role") != want_role or eff.get("max_num_batched_tokens") != t["mbt"]:
                role_viol.append((exp, arm, name, eff.get("kv_role"), eff.get("max_num_batched_tokens")))
            if eff.get("max_model_len") != mml:
                role_viol.append((exp, arm, name, "max_model_len", eff.get("max_model_len")))
    rep.check(M, "m27_common_engine_args", not diff,
              f"除 mbt/kv_role 外存在生效差异：{diff[:4]}",
              "所有 arm / 实例除 max_num_batched_tokens 与 kv_transfer 角色外生效配置逐键相同")
    rep.check(M, "m27_only_variable", not role_viol, f"mbt/kv_role/max_model_len 不符：{role_viol[:4]}",
              "PD 的 mbt=8192；colo 的 mbt=arm 名；P=kv_producer / D=kv_consumer / colo=无 connector")
    rep.check(M, "m27_gpu_assignment", not gpu_viol, f"GPU/端口重复或越出 gpu_map：{gpu_viol[:4]}",
              f"每个 arm 的 GPU ⊆ gpu_map {sorted(allowed_gpus)} 且 GPU/端口唯一")

    def flat(rows):
        """展开到单请求记录（E2 的 d / inj 嵌套）。fault_probe 的 success 表示「拿到 502」，不是服务成功。"""
        for r in rows:
            if r.get("phase") == "fault_probe":
                continue
            if r.get("experiment") == "e2":
                for sub in ("d", "inj"):
                    if r.get(sub):
                        yield r, r[sub]
            elif "prompt_sha256" in r:
                yield r, r

    usage_viol, echo_viol, ident = [], [], {}
    for (exp, arm), rows in R.items():
        for _, q in flat(rows):
            u = q.get("usage") or {}
            if q.get("success") and (u.get("completion_tokens") != q.get("n_output_tokens")
                                     or u.get("prompt_tokens") != q.get("prompt_len")):
                usage_viol.append((exp, arm, q.get("label")))
            if q.get("prompt_sha256_returned") and q["prompt_sha256_returned"] != q["prompt_sha256"]:
                echo_viol.append((exp, arm, q.get("label")))
            if q.get("label"):
                ident.setdefault(q["label"], set()).add(q["prompt_sha256"])
    split = [k for k, v in ident.items() if len(v) > 1]
    rep.check(M, "m27_output_length", not usage_viol,
              f"{len(usage_viol)} 个成功请求的服务端 usage 与客户端 token 计数不符：{usage_viol[:4]}",
              "成功请求：客户端逐 chunk token 数 == 服务端 usage.completion_tokens；prompt_len == usage.prompt_tokens")
    rep.check(M, "m27_prompt_identity", not split and not echo_viol,
              f"同 key 不同 prompt：{split[:4]}；服务端回显不符：{echo_viol[:4]}",
              f"{len(ident)} 个 request key 在所有 arm 上 prompt sha256 相同，且与服务端回显一致")

    cnt_viol, warm_viol, xfer_viol, theory_viol = [], [], [], []
    for (exp, arm), rows in R.items():
        kind = C.parse_arm(arm)["kind"]
        groups = {}
        for r in rows:
            if r.get("phase") in ("measure", "warmup") and exp in ("e0", "e1", "e2"):
                g = r.get("context") if exp == "e1" else r.get("p_len") if exp == "e2" else 0
                groups.setdefault(g, {"measure": [], "warmup": []})[r["phase"]].append(r)
        for g, ph in groups.items():
            n_m = 5 if exp == "e0" else F["measure"]
            n_w = 3 if exp == "e0" else F["warmup"]
            ok = [r for r in ph["measure"] if r.get("success") is True]
            if len(ph["measure"]) != n_m or len(ok) != n_m:
                cnt_viol.append((exp, arm, g, len(ph["measure"]), len(ok)))
            if len(ph["warmup"]) != n_w:
                warm_viol.append((exp, arm, g, len(ph["warmup"])))
        if kind != "pd":
            continue
        d_names = [i["name"] for i in (rows[0].get("instances") or []) if i["role"] == "D"] if rows else []
        p_names = [i["name"] for i in (rows[0].get("instances") or []) if i["role"] == "P"] if rows else []
        for r in rows:
            if r.get("phase") != "measure":
                continue
            if exp in ("e0", "e1"):
                md, L = r.get("metrics_delta") or {}, r["prompt_len"]
                ok = (r.get("metrics_settled") is True
                      and sum((md.get(d) or {}).get("prompt_external_kv_transfer") or 0 for d in d_names) == L
                      and all((md.get(d) or {}).get("prompt_local_compute") == 0 for d in d_names)
                      and all((md.get(d) or {}).get("nixl_failed_transfers") == 0 for d in d_names)
                      and sum((md.get(p) or {}).get("prompt_local_compute") or 0 for p in p_names) == L)
                if not ok:
                    xfer_viol.append((exp, arm, r.get("request_key")))
                if exp == "e1":
                    th = C.kv_theory(r["context"])
                    nb = sum((md.get(d) or {}).get("nixl_bytes_sum") or 0 for d in d_names)
                    if any(r.get(k) != v for k, v in th.items()) or not nb > 0:
                        theory_viol.append((arm, r.get("request_key"), nb))
            elif exp == "e2" and r.get("p_len") and r.get("success"):
                md = r.get("metrics_delta_inj_phase") or {}
                if not (sum((md.get(d) or {}).get("prompt_external_kv_transfer") or 0 for d in d_names) == r["p_len"]
                        and all((md.get(d) or {}).get("nixl_failed_transfers") == 0 for d in d_names)):
                    xfer_viol.append((exp, arm, r.get("p_request_key")))
        for r in rows:
            if r.get("phase") != "measure_run_summary":
                continue
            md = r.get("metrics_delta") or {}
            if r.get("n_fail") == 0 and not all(
                    (md.get(d) or {}).get("prompt_local_compute") == 0
                    and (md.get(d) or {}).get("nixl_failed_transfers") == 0 for d in d_names):
                xfer_viol.append((exp, arm, r.get("workload_shape"), r.get("seed"), r.get("load_frac")))
    rep.check(M, "m27_sample_count", not cnt_viol, f"measure 数或成功数不符（期望全成功）：{cnt_viol[:4]}",
              "E0 每 arm 5 measure；E1/E2 每点 15 measure，全部 success（E2 含 overlap_valid）")
    rep.check(M, "m27_warmup_exclusion", not warm_viol, f"warmup 数不符：{warm_viol[:4]}",
              "warmup 单独标 phase（E0 3 / E1–E2 5），汇总只用 measure")
    rep.check(M, "m27_transfer_success", not xfer_viol,
              f"{len(xfer_viol)} 个 PD 样本未满足 D 外部 KV == prompt_len / D 本地计算 0 / 失败 0：{xfer_viol[:4]}",
              "PD：D external_kv_transfer == prompt_len、D local_compute == 0、P local_compute == prompt_len、"
              "nixl_failed == 0（metrics 稳定后读取）")
    if ("e1", "pd_1p1d") in R:
        rep.check(M, "m27_theoretical_kv_metadata", not theory_viol, f"理论值/实测 bytes 缺失：{theory_viol[:4]}",
                  "E1 每样本同时记录理论 bytes（token/block 口径）与 NIXL 实测 bytes（二者不要求相等）")

    life = load_jsonl(f"outputs/m27/raw/{tag}_lifecycle.jsonl") or []
    # A failed/replaced attempt stays in the immutable lifecycle/backup, but must
    # not be joined to the current raw after run_all has archived and rerun it.
    first_raw = {(exp, arm): min(r["unix_ts"] for r in rows) for (exp, arm), rows in R.items() if rows}
    def formal_stops(events, cutoffs):
        return [ev for ev in events if ev.get("event") == "stop" and
                (ev.get("exp"), ev.get("arm")) in cutoffs and
                ev["unix_ts"] >= cutoffs[(ev["exp"], ev["arm"])]]
    synthetic = [{"event": "stop", "exp": "e0", "arm": "pd_1p1d", "unix_ts": 1},
                 {"event": "stop", "exp": "e0", "arm": "pd_1p1d", "unix_ts": 3}]
    rep.check(M, "m27_lifecycle_attempt_boundary",
              formal_stops(synthetic, {("e0", "pd_1p1d"): 2}) == [synthetic[1]],
              "lifecycle retry 边界回归失败", "归档失败尝试不加入当前 raw 的日志判定")
    stops = formal_stops(life, first_raw)
    log_viol = []
    stop_keys = {(ev["exp"], ev["arm"]) for ev in stops if ev.get("reason") == "normal"}
    if set(R) != stop_keys:
        log_viol.append(("missing_normal_stop", sorted(set(R) - stop_keys)))
    for ev in stops:
        if ev.get("reason") != "normal":
            log_viol.append((ev["exp"], ev["arm"], ev.get("reason")))
        for name, le in (ev.get("log_evidence") or {}).items():
            if (le.get("kv_load_fail_events") or le.get("nixl_compat_mismatch") or le.get("tracebacks")
                    or le.get("log_missing")):
                log_viol.append((ev.get("exp"), ev.get("arm"), name))
            if name.startswith("D") and not le.get("nixl_compat_pass"):
                log_viol.append((ev.get("exp"), ev.get("arm"), name, "no_compat_pass"))
    rep.check(M, "m27_no_failures_log", bool(stops) and not log_viol,
              f"当前 raw 的生命周期缺失/异常或服务日志含 KV load 失败/compat mismatch/Traceback：{log_viol[:4]}",
              "当前正式 raw 对应生命周期的服务日志：无 KV load failure、compat mismatch、Traceback；D 有 compat pass")

    fp = [r for r in R.get(("e0", "pd_1p1d"), []) if r.get("phase") == "fault_probe"]
    if fp:
        f = fp[-1]
        d = next((i["name"] for i in f.get("instances", []) if i["role"] == "D"), None)
        dd = ((f.get("metrics_delta") or {}).get(d) or {}).get("prompt_tokens_total")
        rep.check(M, "m27_fail_closed_probe", f.get("http_status") == 502 and dd == 0,
                  f"故障注入未 fail-closed：http={f.get('http_status')} D prompt Δ={dd}",
                  "丢弃 kv_transfer_params → proxy 502，D 未收到请求（prompt_tokens Δ=0）")
    elif ("e0", "pd_1p1d") in R:
        rep.fail(M, "m27_fail_closed_probe", "E0 pd_1p1d 缺 fault_probe 行")

    e2_keys = {arm: {(r["p_len"], r["idx"]) for r in rows if r.get("phase") == "measure"}
               for (exp, arm), rows in R.items() if exp == "e2"}
    if e2_keys:
        ks = list(e2_keys.values())
        rep.check(M, "m27_paired_keys_e2", all(k == ks[0] for k in ks),
                  f"E2 各 arm 的 (p_len, idx) 不配对：{ {a: len(k) for a, k in e2_keys.items()} }",
                  f"E2 {len(e2_keys)} 个 arm 的 (p_len, idx) 完全配对")
    for exp in ("e3", "e4", "e5"):
        summ = {arm: rows for (x, arm), rows in R.items() if x == exp}
        if not summ:
            continue
        rates_path = _p(f"outputs/m27/configs/{tag}_{exp}cal_rates.json")
        rates_sha = None
        if os.path.exists(rates_path):
            rates_sha = C.sha256_file(rates_path)
        sched, viol, failed_runs = {}, [], []
        for arm, rows in summ.items():
            for r in rows:
                if r.get("phase") != "measure_run_summary":
                    continue
                key = (r["workload_shape"], r["seed"], r["load_frac"])
                if r.get("n_fail") != 0:
                    failed_runs.append((arm, key, r.get("n_fail")))
                sched.setdefault(key, {})[arm] = r["schedule_sha256"]
                if r.get("rates_file_sha256") != rates_sha:
                    viol.append((arm, key, "rates_sha"))
        for key, by_arm in sched.items():
            if len(set(by_arm.values())) != 1 or set(by_arm) != set(summ):
                viol.append((key, sorted(by_arm)))
        rep.check(M, f"m27_paired_schedules_{exp}", bool(sched) and not viol,
                  f"{exp} 到达序列/速率文件不配对：{viol[:4]}",
                  f"{exp}：{len(sched)} 个 (shape, seed, load) 在 {len(summ)} 个 arm 上 schedule sha256 相同，"
                  "速率文件 sha 一致")
        rep.check(M, f"m27_online_no_failed_requests_{exp}", not failed_runs,
                  f"{exp} 有请求失败的 run（不得静默剔除）：{failed_runs[:4]}",
                  f"{exp} 全部 run 的 n_fail == 0")

    e1csv = load_csv(f"outputs/m27/{tag}_e1_transfer.csv")
    if e1csv is not None and ("e1", "pd_1p1d") in R:
        want = {}
        for r in R[("e1", "pd_1p1d")]:
            if r.get("phase") == "measure" and r.get("success") is True:
                want[r["context"]] = want.get(r["context"], 0) + 1
        got = {int(r["context"]): int(r["n"]) for r in e1csv}
        rep.check(M, "m27_e1_csv_n_matches_jsonl", got == want, f"E1 CSV n {got} != jsonl {want}",
                  "E1 CSV n == jsonl measure∧success")
    e2csv = load_csv(f"outputs/m27/{tag}_e2_interference.csv")
    if e2csv is not None:
        want = {}
        for (exp, arm), rows in R.items():
            if exp != "e2":
                continue
            for r in rows:
                if r.get("phase") == "measure" and r.get("success") is True:
                    want[(arm, r["p_len"])] = want.get((arm, r["p_len"]), 0) + 1
        got = {(r["arm"], int(r["p_len"])): int(r["n"]) for r in e2csv}
        rep.check(M, "m27_e2_csv_n_matches_jsonl", got == want, "E2 CSV n != jsonl measure∧success",
                  "E2 CSV n == jsonl measure∧success")
    _m27_final_evidence_audit(rep)


CHECKS = {
    "m0": [check_env],
    "m17": [
        check_m17_e1,
        check_m17_e3_single,
        check_m17_e3_conc,
        check_m17_e3_correctness,
        check_m17_i5b,
        check_m17_csv_vs_jsonl,
        check_m17_doc_numbers,
    ],
    "m19": [
        check_m19_samples,
        check_m19_csv_vs_jsonl,
        check_m19_equivalence,
    ],
    "m18": [check_m18],
    "m21": [check_m21],
    "m22": [check_m22],
    "m23": [check_m23],
    "m24": [check_m24],
    "m27": [check_m27],
}


def run(modules, verbose=False):
    rep = Report()
    for m in modules:
        for fn in CHECKS.get(m, []):
            fn(rep)

    width = max((len(r[2]) for r in rep.rows), default=10)
    for status, module, rule, detail in rep.rows:
        if status == "PASS" and not verbose:
            continue
        mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "PENDING": "PENDING"}[status]
        line = f"[{mark}] {module:<16} {rule:<{width}}"
        print(f"{line}  {detail}" if detail else line)

    n_pass = len([r for r in rep.rows if r[0] == "PASS"])
    print(
        f"\n合计：{n_pass} PASS / {len(rep.failures)} FAIL / {len(rep.pendings)} PENDING"
        f"（模块：{', '.join(modules)}）"
    )
    if rep.pendings:
        print("\nPENDING（实验尚未执行，**不得记为通过**）：")
        for _, module, rule, detail in rep.pendings:
            print(f"  - {module} · {rule}: {detail}")
    if rep.failures:
        print("\nFAIL：")
        for _, module, rule, detail in rep.failures:
            print(f"  - {module} · {rule}: {detail}")
        return 1
    return 0


def main():
    p = argparse.ArgumentParser(
        description="正式不变量检查（不需 GPU、不 import vllm）"
    )
    p.add_argument("--module", action="append", choices=sorted(CHECKS), help="可重复")
    p.add_argument("--verbose", action="store_true", help="同时打印 PASS 明细")
    a = p.parse_args()
    return run(a.module or sorted(CHECKS), a.verbose)


if __name__ == "__main__":
    sys.exit(main())
