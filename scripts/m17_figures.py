"""M17 · 从既有实验数据离线生成核心图（无需 GPU / 模型 / vllm）。

读取 M17 的 CSV 与逐样本 jsonl，产出 figures/ 下 2 张图，供 README 直观呈现：
  fig5_m17_apc.png       E1：APC 的 in-process 首 token 延迟 vs 前缀长度 + median 差的 Bootstrap 95% CI
  fig6_m17_chunked.png   E3：并发 decode 的 ITL（只改尾部不改中位）+ 单请求每多一个 prefill step 的成本

刻意只画两张：E1 的「收益随 P 浮出噪声」与 E3 的「尾部改善 / 中位不变」是本模块的两条主结论，
其余（不变量 I1–I4、上游配置约束）是布尔判定或源码事实，画图不增信息，故不画。

CI 与 results.md 一致：median difference 的 Bootstrap 95% CI，10000 次重采样，固定 seed=0，
仅用 stdlib random+statistics（与 scripts/m17_prefix_cache.py 内同一算法/同一种子，故数值可复现）。

复现：source .venv/bin/activate && python scripts/m17_figures.py
数据源固定（本机 sm_120 / eager / Qwen2.5-0.5B / in-process），图仅由已落盘数据生成，不重跑实验。
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

OFF_C, COLD_C, HOT_C = "#d62728", "#ff7f0e", "#1f77b4"
NOCHUNK_C, CHUNKED_C = "#d62728", "#1f77b4"


def _read(path):
    with open(os.path.join(ROOT, path)) as f:
        return list(csv.DictReader(f))


def _jsonl(path):
    rows = []
    with open(os.path.join(ROOT, path)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _measure(rows):
    """只取有效 measure 样本（与汇总脚本同口径：排除 warmup 与失败样本）。"""
    return [r for r in rows if r.get("phase") == "measure" and r.get("success", True)]


def _boot_ci_median_diff(a, b, iters=10000, seed=0):
    """median(a) - median(b) 的 Bootstrap 95% CI（与 m17_prefix_cache.py 同算法同种子）。"""
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
# fig5: E1 APC — latency vs prefix length + median difference with 95% CI
# --------------------------------------------------------------------------- #
def fig5():
    sweep = _read("outputs/m17/prefix_cache_sweep.csv")
    effect = _read("outputs/m17/prefix_cache_effect.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # 左：三条曲线（off / cold=APC on 但未命中 / hot=命中）
    for label, sel, color in (
        ("APC off", lambda r: r["prefix_caching"] == "off", OFF_C),
        ("APC on · cold (miss)", lambda r: r["kind"] == "cold", COLD_C),
        ("APC on · hot (prefix hit)", lambda r: r["kind"] == "hot", HOT_C),
    ):
        m = sorted((r for r in sweep if sel(r)), key=lambda r: int(r["prefix_len"]))
        x = [int(r["prefix_len"]) for r in m]
        y = [float(r["lat_median_ms"]) for r in m]
        e = [float(r["lat_stdev_ms"]) for r in m]
        ax1.errorbar(x, y, yerr=e, marker="o", capsize=3, label=label, color=color)
    ax1.set_xscale("log", base=2)
    ax1.set_xticks([256, 512, 1024, 2048])
    ax1.set_xticklabels(["256", "512", "1024", "2048"])
    ax1.axhline(7, ls="--", lw=0.8, color="grey")
    ax1.set_ylim(6.2, None)
    ax1.text(
        290, 6.45, "~7ms fixed CPU-submit floor (M7)", fontsize=8, color="grey"
    )
    ax1.set(
        xlabel="shared prefix length P (tokens)",
        ylabel="in-process first-token latency (ms)",
        title="off ≈ cold (both compute all P); hot is flat\n(only the 16 tail tokens are computed)",
    )
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8)

    # 右：median 差 (off − hot) + Bootstrap 95% CI
    m = sorted(effect, key=lambda r: int(r["prefix_len"]))
    x = range(len(m))
    diff = [float(r["median_diff_ms"]) for r in m]
    lo = [float(r["ci95_lo_ms"]) for r in m]
    hi = [float(r["ci95_hi_ms"]) for r in m]
    colors = ["#2ca02c" if r["ci_excludes_zero"] == "yes" else "grey" for r in m]
    ax2.bar(x, diff, color=colors, width=0.55)
    ax2.errorbar(
        x, diff,
        yerr=[[d - lo_ for d, lo_ in zip(diff, lo)], [hi_ - d for d, hi_ in zip(diff, hi)]],
        fmt="none", ecolor="black", capsize=4, lw=1.2,
    )
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([f"P={r['prefix_len']}" for r in m])
    ax2.set_ylim(min(lo) - 1.2, max(hi) * 1.16)
    for i, r in enumerate(m):
        ok = r["ci_excludes_zero"] == "yes"
        ax2.text(
            i, float(r["ci95_hi_ms"]) + 0.45,
            f"−{r['rel_reduction_pct']}%" if ok else "CI covers 0",
            ha="center", fontsize=8, fontweight="bold" if ok else "normal",
            color="#2ca02c" if ok else "grey",
        )
    ax2.set(
        ylabel="median(off) − median(hot)  (ms)",
        title="APC gain only significant at P≥1024 (Bootstrap 95% CI)",
    )
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "M17 · E1 Automatic Prefix Caching: gain must outgrow the fixed-overhead floor",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_m17_apc.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig6: E3 chunked prefill — concurrent-decode ITL (tail only) + per-step cost
# --------------------------------------------------------------------------- #
def fig6():
    conc = {
        m: _measure(_jsonl(f"outputs/m17/raw/e3_conc_{m}.jsonl"))
        for m in ("nochunk", "chunked")
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # 左：D 的 ITL median / p95 / max + 长 prefill 自身 TTFT（每 repeat 一个值，取 median）
    keys = [
        ("decode_itl_median_ms", "D's ITL\nmedian"),
        ("decode_itl_p95_ms", "D's ITL\np95"),
        ("decode_itl_max_ms", "D's ITL\nmax (worst stall)"),
        ("long_prefill_first_token_latency_ms", "long prefill's\nown TTFT"),
    ]
    width = 0.36
    per_mode = {}
    for mode in ("nochunk", "chunked"):
        per_mode[mode] = {k: [r[k] for r in conc[mode]] for k, _ in keys}
    for i, (mode, color) in enumerate((("nochunk", NOCHUNK_C), ("chunked", CHUNKED_C))):
        vals = [statistics.median(per_mode[mode][k]) for k, _ in keys]
        ax1.bar([j + i * width for j in range(len(keys))], vals, width,
                label=mode, color=color)
    # 显著性标注：nochunk − chunked 的 Bootstrap CI 是否排除 0
    top = 0
    for j, (k, _) in enumerate(keys):
        d, lo, hi = _boot_ci_median_diff(per_mode["nochunk"][k], per_mode["chunked"][k])
        y = max(statistics.median(per_mode[m][k]) for m in ("nochunk", "chunked"))
        top = max(top, y)
        sig = lo > 0 or hi < 0
        ax1.text(
            j + width / 2, y + 1.2,
            f"Δ={d:+.2f}\n{'significant' if sig else 'CI covers 0'}",
            ha="center", fontsize=7.5,
            color="#2ca02c" if sig else "grey",
            fontweight="bold" if sig else "normal",
        )
    ax1.set_ylim(0, top * 1.32)
    ax1.set_xticks([j + width / 2 for j in range(len(keys))])
    ax1.set_xticklabels([lab for _, lab in keys], fontsize=8)
    ax1.set(
        ylabel="latency (ms)",
        title="chunked halves the worst stall, leaves median/p95 alone\n(cost: the long prefill's own TTFT)",
    )
    ax1.grid(alpha=0.3, axis="y")
    ax1.legend(fontsize=8)

    # 右：单请求下 TTFT vs prefill step 数（同侧 chunked on，仅变 mbt）
    rows = [r for r in _read("outputs/m17/chunked_prefill.csv")
            if r["enable_chunked_prefill"] == "True"]
    pts = {}
    for r in rows:  # mbt=2048/4096 都是 1 步；取 mbt=2048 为同侧 1 步基准
        pts.setdefault(int(r["prefill_step_count"]), []).append(r)
    steps = sorted(pts)
    xs, ys, es, labels = [], [], [], []
    for s in steps:
        r = min(pts[s], key=lambda r: int(r["max_num_batched_tokens"]))
        xs.append(s)
        ys.append(float(r["ttft_median_ms"]))
        es.append(float(r["ttft_stdev_ms"]))
        labels.append(f"mbt={r['max_num_batched_tokens']}")
    ax2.errorbar(xs, ys, yerr=es, marker="o", capsize=3, color=CHUNKED_C)
    for x, y, lab in zip(xs, ys, labels):
        ax2.annotate(lab, (x, y), textcoords="offset points", xytext=(6, -12), fontsize=8)
    slope = (ys[-1] - ys[0]) / (xs[-1] - xs[0])
    ax2.text(
        1.05, max(ys) * 0.72,
        f"≈ +{slope:.1f} ms per extra prefill step\n(single request, prompt=2048, Σchunks≡2048)",
        fontsize=8.5,
    )
    ax2.set_xticks(steps)
    ax2.set_xlim(xs[0] - 0.35, xs[-1] + 0.55)
    ax2.set(
        xlabel="prefill step count (chunked on, only max_num_batched_tokens varies)",
        ylabel="in-process first-token latency (ms)",
        title="single request pays a fixed cost per extra step",
    )
    ax2.grid(alpha=0.3)

    fig.suptitle(
        "M17 · E3 Chunked prefill: trades the long request's own TTFT for others' decode smoothness",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig6_m17_chunked.png"), dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# fig7 (I5-b): APC 的相对收益 vs 输出长度
#   —— 只有 I5-b 数据落盘后才生成；无数据时跳过，不画伪图。
# --------------------------------------------------------------------------- #
def fig_apc_output_length():
    rel = "outputs/m17/apc_output_length_effect.csv"
    if not os.path.exists(os.path.join(ROOT, rel)):
        print(f"skip m17_apc_output_length.png：{rel} 尚未产生（I5-b 未执行）")
        return False
    rows = sorted(_read(rel), key=lambda r: int(r["max_tokens"]))
    x = [int(r["max_tokens"]) for r in rows]
    rel_red = [float(r["completion_rel_reduction_pct"]) for r in rows]
    diff = [float(r["completion_median_diff_ms"]) for r in rows]
    lo = [float(r["completion_ci95_lo_ms"]) for r in rows]
    hi = [float(r["completion_ci95_hi_ms"]) for r in rows]
    sig = [r["completion_ci_excludes_zero"] == "yes" for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 7.4))
    idx = range(len(x))

    # 上：相对收益（I5-b 的判据本身）
    ax1.bar(
        idx, rel_red,
        color=["#2ca02c" if s else "grey" for s in sig], width=0.55,
    )
    ax1.axhline(0, color="black", lw=1)
    for i, (v, s) in enumerate(zip(rel_red, sig)):
        ax1.text(
            i, v + (0.6 if v >= 0 else -1.4),
            f"{v:.1f}%" if s else f"{v:.1f}%\n(CI covers 0)",
            ha="center", fontsize=8,
            fontweight="bold" if s else "normal",
            color="#2ca02c" if s else "grey",
        )
    ax1.set_xticks(list(idx))
    ax1.set_xticklabels([f"max_tokens={v}" for v in x])
    ax1.set(
        ylabel="completion latency\nrelative reduction (%)",
        title="I5-b: APC only saves prefill — its relative gain should shrink\n"
              "as the output (decode) grows",
    )
    ax1.grid(alpha=0.3, axis="y")

    # 下：绝对差 + Bootstrap 95% CI。
    # 注意（实测修正）：恒定的是首 token 的绝对收益（≈15ms＝省掉的那次 prefill）；
    # 完成延迟的绝对差反而随输出变长而增大（14.6→35.7ms），来自 hot/off 之间约 2% 的
    # TPOT 差（8.15 vs 7.99 ms/token）在更多 token 上累积。原副标题「绝对收益恒定」与本
    # 面板自身数据矛盾，已按「数据推翻结论就改结论」修正，并把首 token 曲线一并画出。
    ft_diff = [float(r["ft_median_diff_ms"]) for r in rows]
    ft_lo = [float(r["ft_ci95_lo_ms"]) for r in rows]
    ft_hi = [float(r["ft_ci95_hi_ms"]) for r in rows]
    ax2.errorbar(
        idx, diff,
        yerr=[
            [d - lo_ for d, lo_ in zip(diff, lo)],
            [hi_ - d for d, hi_ in zip(diff, hi)],
        ],
        fmt="o", capsize=4, lw=1.2, color=HOT_C, label="completion latency",
    )
    ax2.errorbar(
        [i + 0.12 for i in idx], ft_diff,
        yerr=[
            [d - lo_ for d, lo_ in zip(ft_diff, ft_lo)],
            [hi_ - d for d, hi_ in zip(ft_diff, ft_hi)],
        ],
        fmt="s", capsize=4, lw=1.2, color="#d62728", label="first-token latency",
    )
    ax2.axhline(0, color="black", lw=1)
    ax2.set_xticks(list(idx))
    ax2.set_xticklabels([f"max_tokens={v}" for v in x])
    ax2.set(
        ylabel="median(off) − median(hot)  (ms)",
        xlabel="requested output length",
        title="first-token saving is flat (~15ms = the skipped prefill);\n"
              "completion-latency gap grows with output (Bootstrap 95% CI)",
    )
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "M17 · I5-b Automatic Prefix Caching vs output length",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "m17_apc_output_length.png"), dpi=130)
    plt.close(fig)
    return True


if __name__ == "__main__":
    fig5()
    fig6()
    fig_apc_output_length()
    print("figures written to", OUT)
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
