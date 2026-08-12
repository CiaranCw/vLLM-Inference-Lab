"""正式不变量 / 回归检查（M0 收口产物之一）——**不需 GPU、不 import vllm**。

把此前散落在各模块 notes 里的「跑完逐条核对」固化成可执行断言，覆盖三类核查：

  1. **逐样本不变量**：直接读 `outputs/mN/raw/*.jsonl`（权威证据），核对守恒律与字段语义；
  2. **CSV ↔ JSONL 一致性**：汇总 CSV 的 n 必须等于 JSONL 中 `phase=measure and success=True`
     的样本数——防止「改了脚本只重跑汇总、没重跑实验」；
  3. **文档数字可溯源**：把 `notes/m17/results.md` 主结果表里的每个数字**反向**核对回 CSV。
     M17 审计的教训是「自动核对只覆盖表格、没覆盖正文叙述句」，故这里同时做正向核对
     （由 CSV 生成的数字串必须出现在文档中，含正文关键句）。

三种结论，语义严格区分（诚实性红线）：
  PASS    —— 已执行且通过
  FAIL    —— 已执行但违反 → **非零退出**
  PENDING —— 实验尚未执行（产物不存在且 manifest 标注为 implemented）→ 不算通过，也不算失败

用法：
    python scripts/check_invariants.py                # 检查 manifest 里登记的全部模块
    python scripts/check_invariants.py --module m17
    python scripts/check_invariants.py --verbose      # 逐条打印 PASS 明细
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys

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
# Markdown 表格解析（用于文档数字的**反向**核对）
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
    """一个数在给定小数位下**可接受的书写形式**。

    浮点二进制表示会让 .5 边界两可（例如 58.45 的 f-string 给 '58.4'，而人手写
    通常是 '58.5'）——两者都是对同一 CSV 值的正确舍入，故都接受，避免把书写差异
    误报成数据对不上。
    """
    scale = 10**decimals
    half_up = (int(v * scale + (0.5 if v >= 0 else -0.5))) / scale
    return {f"{v:.{decimals}f}", f"{half_up:.{decimals}f}"}


def _num_matches(doc_cell, csv_value):
    """判据：文档数字必须等于 CSV 数字**在文档自身精度下**的正确舍入。

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
    """输出一致性：**如实记录**，不静默忽略差异。

    这里刻意不把「baseline 与 ngram 输出必须逐 token 相同」写成硬性 FAIL：
    `batch_shape_control.csv` 已证明**baseline 自己**换个 batch 组成就会在同一位置分歧，
    说明分歧来自批形状引起的浮点差异而非投机解码。因此硬判据只保留两条真正的功能红线
    ——**输出长度必须一致**、**分歧不得出现在第 0 位**；一致率本身只做记录与展示。
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

    **刻意区分两类**（这是本模块设计上的一个明确判断）：
      · correctness invariant —— 违反即实验无效，FAIL；
      · expected observation  —— 理论预期，实测不符时**记录并要求解释**，不自动判失败。
    例如「每卡权重 ≈ 1/k」就**不是** invariant：embedding/lm_head 是否切分随实现而变
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
    # 覆盖**全部**样本（含 warmup）：prompt_len >= max_model_len 属配置合法性问题，与 phase 无关
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

    核心是 E2：服务端 counter 增量必须与客户端 bench 汇总守恒。**这条不过，
    后面所有基于面板的结论都不该信**，所以它是硬 FAIL 而不是「观察项」。

    刻意**不**把「容量 = 某个数」写成不变量：容量随会话漂移（见 results.md §4），
    把它当不变量只会逼出假结论。
    """
    M = "m21"
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

    e6 = load_json("outputs/m21/e6_drills.json")
    if e6:
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
    "m21": [check_m21],
    "m22": [check_m22],
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
