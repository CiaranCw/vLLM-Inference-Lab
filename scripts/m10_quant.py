"""M10 · 量化对比 单配置 driver（BF16 / 在线 FP8 W8A8 / FP8 KV）。

一次只处理一个配置（独立进程，见 m10_run_all.py）。同一进程内**分离** perf 与 quality 两阶段：
  - perf（prefill/decode 计时）：绝不开 prompt_logprobs。
  - quality（token 加权 PPL）：用 prompt_logprobs、不计时。
公共 Engine 参数由 COMMON 固定并落盘；各配置仅改 quantization / kv_cache_dtype / attention_backend(/calculate_kv_scales)。

指标为进程内 proxy：engine_first_step_latency_ms / steady_decode_step_latency_ms / decode_output_tokens_per_second（非在线 TTFT/TPOT）。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m6_benchmark as m6
from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.usage.usage_lib import UsageContext

# 公共 Engine 参数（所有配置一致，仅下方 CONFIG_MATRIX 三/四项变量不同）
COMMON = dict(model="Qwen/Qwen2.5-0.5B-Instruct", dtype="auto", max_model_len=2048,
              gpu_memory_utilization=0.6, tensor_parallel_size=1, enforce_eager=True,
              seed=0, max_num_seqs=64, max_num_batched_tokens=4096,
              enable_prefix_caching=False, async_scheduling=False, disable_log_stats=True)

CONFIG_MATRIX = {
    "C0":   dict(),  # BF16 / BF16 / FLASH_ATTN(默认)
    "C1":   dict(quantization="fp8"),
    "C0T":  dict(attention_backend="TRITON_ATTN"),
    "C2":   dict(kv_cache_dtype="fp8", attention_backend="TRITON_ATTN"),
    "C2dyn":dict(kv_cache_dtype="fp8", attention_backend="TRITON_ATTN", calculate_kv_scales=True),
    "C3":   dict(quantization="fp8", kv_cache_dtype="fp8", attention_backend="TRITON_ATTN"),
}

# 固定质量语料（内置、无下载）：中英文各一段较长文本，tokenize 后裁到目标长度。
# 注：为达 ≥4000 评分 token 会对源文本做 tile；PPL 绝对值因重复偏低，但**同一文本对所有配置一致，相对 ΔPPL 有效**（笔记注明）。
_ZH = ("大型语言模型的推理过程分为预填充和解码两个阶段。预填充阶段一次性处理整个提示词，"
       "计算所有输入 token 的键和值并写入缓存；解码阶段每一步只生成一个新 token，"
       "但需要读取此前全部历史的键值缓存。键值缓存的显存占用随序列长度线性增长，"
       "因此它往往是长上下文推理的显存瓶颈。量化通过降低权重或缓存的数值精度来节省显存，"
       "常见做法包括八比特浮点权重量化与八比特浮点键值缓存量化，代价是可能带来一定的精度损失。")
_EN = ("Large language model inference proceeds in two phases: prefill and decode. During prefill the "
       "entire prompt is processed at once, computing keys and values for every input token and writing "
       "them into the cache. During decode only a single new token is produced per step, yet the whole "
       "history of cached keys and values must be read back. Because the key value cache grows linearly "
       "with sequence length, it frequently becomes the memory bottleneck for long context inference. "
       "Quantization reduces memory by lowering the numerical precision of weights or of the cache, "
       "for example eight bit floating point weights or an eight bit floating point key value cache, "
       "at the possible cost of some accuracy degradation.")


def build(cfg_name):
    kw = dict(COMMON); kw.update(CONFIG_MATRIX[cfg_name])
    engine = LLMEngine.from_engine_args(
        EngineArgs(**kw), usage_context=UsageContext.ENGINE_CONTEXT, enable_multiprocessing=False)
    return engine


# ------------------------- dump: resolved config / scale / memory -------------------------
def dump_config(engine, cfg_name):
    vc = engine.vllm_config
    cc = vc.cache_config
    def nm(v): return getattr(v, "name", None) or (str(v) if v is not None else None)
    return {"config": cfg_name, "common": COMMON, "variant": CONFIG_MATRIX[cfg_name],
            "quantization": vc.model_config.quantization,
            "kv_cache_dtype": cc.cache_dtype,
            "calculate_kv_scales": cc.calculate_kv_scales,
            "num_gpu_blocks": cc.num_gpu_blocks, "block_size": cc.block_size,
            "kv_cache_size_tokens": cc.kv_cache_size_tokens,
            "kv_cache_max_concurrency": cc.kv_cache_max_concurrency}


def scale_audit(engine):
    def _audit(w):
        rows = []
        for n, mod in w.model_runner.model.named_modules():
            if hasattr(mod, "_k_scale") and hasattr(mod, "_v_scale"):
                rows.append({"layer": n, "k": float(mod._k_scale.item()), "v": float(mod._v_scale.item()),
                             "dtype": str(mod._k_scale.dtype),
                             "calc_kv_scales": bool(getattr(mod, "calculate_kv_scales", False))})
        ks = [r["k"] for r in rows]; vs = [r["v"] for r in rows]
        return {"n_attn_layers": len(rows),
                "k_all_one": all(abs(x - 1.0) < 1e-9 for x in ks) if ks else None,
                "v_all_one": all(abs(x - 1.0) < 1e-9 for x in vs) if vs else None,
                "k_range": [min(ks), max(ks)] if ks else None,
                "v_range": [min(vs), max(vs)] if vs else None,
                "example_layers": rows[:2]}
    try:
        return engine.model_executor.collective_rpc(_audit)[0]
    except Exception as e:  # noqa
        return {"error": repr(e)}


def mem_audit(engine):
    def _mem(w):
        import torch as _t
        from collections import defaultdict
        m = w.model_runner.model
        seen = set(); by_dtype = defaultdict(int); embed = 0; scale_bytes = 0
        for n, p in m.named_parameters():
            sid = p.untyped_storage().data_ptr()
            if sid in seen:  # 去重 tied/shared（如 embed<->lm_head）
                continue
            seen.add(sid)
            b = p.numel() * p.element_size()
            by_dtype[str(p.dtype)] += b
            if "embed" in n or "lm_head" in n:
                embed += b
            if "scale" in n:
                scale_bytes += b
        kv_bytes = int(sum(t.numel() * t.element_size() for t in getattr(w.model_runner, "kv_caches", []) if t is not None))
        return {"weight_bytes_by_dtype": {k: round(v / 2**20, 3) for k, v in by_dtype.items()},
                "embed_lm_head_MiB": round(embed / 2**20, 3),
                "weight_scale_MiB": round(scale_bytes / 2**20, 4),
                "kv_cache_tensor_MiB": round(kv_bytes / 2**20, 2),
                "kv_cache_tensor_bytes": kv_bytes}
    try:
        rpc = engine.model_executor.collective_rpc(_mem)[0]
    except Exception as e:  # noqa
        rpc = {"error": repr(e)}
    return {"tensor_sum": rpc, "snapshot": m6.mem_snapshot()}


# ------------------------- quality: token-weighted PPL (prompt_logprobs) -------------------------
def _tok_ids(tok, text, n):
    base = tok.encode(text, add_special_tokens=False)
    pool = base
    while len(pool) < n:
        pool = pool + base
    return pool[:n]


def quality_ppl(engine, tf, quick=False):
    lengths = [128, 256, 512] if quick else [128, 256, 512, 1024, 512, 256]
    samples = []
    sid = 0
    for lang, text in (("zh", _ZH), ("en", _EN)):
        for L in lengths:
            ids = _tok_ids(tf.tok, text, L)
            rid = f"q-{sid}"
            engine.add_request(rid, {"prompt_token_ids": ids},
                               SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1, seed=0))
            pls = None; ptids = None
            while engine.has_unfinished_requests():
                for o in engine.step():
                    if getattr(o, "finished", False):
                        pls = o.prompt_logprobs; ptids = list(o.prompt_token_ids)
            total_nll = 0.0; scored = 0; missing = 0; bad = 0
            for i in range(1, len(ptids)):  # 第 0 个无 logprob
                d = pls[i] if pls and i < len(pls) else None
                lp = d.get(ptids[i]) if d else None
                if lp is None:
                    missing += 1; continue
                v = lp.logprob
                if v is None or math.isnan(v) or math.isinf(v):
                    bad += 1; continue
                total_nll += -v; scored += 1
            samples.append({"sample_id": sid, "language": lang, "prompt_tokens": len(ptids),
                            "scored_tokens": scored, "missing_tokens": missing, "bad_tokens": bad,
                            "total_nll": round(total_nll, 4),
                            "mean_nll": round(total_nll / scored, 6) if scored else None,
                            "ppl": round(math.exp(total_nll / scored), 4) if scored else None})
            sid += 1
    # token 加权总 PPL（不先算每段再平均）
    tot_nll = sum(s["total_nll"] for s in samples); tot_scored = sum(s["scored_tokens"] for s in samples)
    def wppl(langf):
        n = sum(s["total_nll"] for s in samples if langf(s)); sc = sum(s["scored_tokens"] for s in samples if langf(s))
        return {"scored_tokens": sc, "mean_nll": round(n / sc, 6) if sc else None,
                "ppl": round(math.exp(n / sc), 4) if sc else None}
    return {"samples": samples,
            "overall": {"scored_tokens": tot_scored, "mean_nll": round(tot_nll / tot_scored, 6),
                        "ppl": round(math.exp(tot_nll / tot_scored), 4)},
            "zh": wppl(lambda s: s["language"] == "zh"),
            "en": wppl(lambda s: s["language"] == "en")}


# ------------------------- perf: prefill + decode (NO prompt_logprobs) -------------------------
def _stats(walls):
    ws = sorted(walls)
    return {"sample_count": len(ws), "median_ms": round(statistics.median(ws), 4),
            "mean_ms": round(statistics.mean(ws), 4), "stdev_ms": round(statistics.pstdev(ws), 4),
            "p95_ms": round(ws[min(len(ws) - 1, int(len(ws) * 0.95))], 4),
            "min_ms": round(ws[0], 4), "max_ms": round(ws[-1], 4)}


def perf_prefill(engine, tf, cfg_name, lengths, warmup, reps, raw):
    out = []
    for n in lengths:
        ids = tf.ids(n)
        def one():
            rid = f"pf-{n}-{time.perf_counter_ns()}"
            engine.add_request(rid, {"prompt_token_ids": ids}, SamplingParams(temperature=0.0, max_tokens=1, seed=0))
            dt, _ = m6.timed_step(engine); m6.drain(engine); return dt * 1e3
        for _ in range(warmup):
            one()
        walls = []
        for r in range(reps):
            w = one(); walls.append(w)
            raw.write(json.dumps({"config": cfg_name, "experiment": "prefill", "prompt_tokens": n,
                                  "repeat": r, "latency_ms": round(w, 4), "valid": True}, ensure_ascii=False) + "\n")
        st = _stats(walls); st["prompt_tokens"] = n; st["metric"] = "engine_first_step_latency_ms"; out.append(st)
    return out


def perf_decode(engine, spy, tf, cfg_name, matrix, warmup, reps, raw):
    sched = m6.get_scheduler(engine)
    out = []
    for ctx, batches in matrix:
        ids = tf.ids(ctx)
        for B in batches:
            rids = [f"d-{ctx}-{B}-{i}" for i in range(B)]
            for rid in rids:
                engine.add_request(rid, {"prompt_token_ids": ids},
                                   SamplingParams(temperature=0.0, max_tokens=100000, ignore_eos=True, seed=0))
            g = 0
            while g < 200:
                so = spy.last
                if so is not None and len(getattr(sched, "running", [])) == B and int(so.total_num_scheduled_tokens) == B:
                    break
                engine.step(); g += 1
            for _ in range(warmup):
                engine.step()
            seq_len = None
            try:
                seq_len = int(getattr(sched.running[0], "num_computed_tokens", 0))
            except Exception:  # noqa
                pass
            walls = []; valid = True
            for r in range(reps):
                dt, _ = m6.timed_step(engine)
                so = spy.last
                ok = so is not None and int(so.total_num_scheduled_tokens) == B and len(getattr(sched, "running", [])) == B
                valid = valid and ok
                walls.append(dt * 1e3)
                raw.write(json.dumps({"config": cfg_name, "experiment": "decode", "context_len": ctx,
                                      "batch_size": B, "repeat": r, "latency_ms": round(dt * 1e3, 4),
                                      "scheduled_tokens": int(so.total_num_scheduled_tokens) if so else -1,
                                      "running_requests": len(getattr(sched, "running", [])), "seq_len": seq_len,
                                      "valid": bool(ok)}, ensure_ascii=False) + "\n")
            st = _stats(walls); st.update({"context_len": ctx, "batch_size": B, "seq_len": seq_len, "all_valid": valid,
                                           "throughput_tok_s_median": round(B / (st["median_ms"] / 1e3), 2)})
            out.append(st)
            engine.abort_request(rids); m6.drain(engine)
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, choices=list(CONFIG_MATRIX))
    p.add_argument("--out-dir", default="outputs/m10")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = a.config
    for d in ("raw", "configs", "memory", "quality"):
        os.makedirs(os.path.join(a.out_dir, d), exist_ok=True)

    t0 = time.perf_counter()
    engine = build(cfg)
    init_s = round(time.perf_counter() - t0, 3)  # 单次观测
    spy = m6.ScheduleSpy(m6.get_scheduler(engine))
    tf = m6.TokenFactory(engine)

    conf = dump_config(engine, cfg); conf["engine_init_s_single_obs"] = init_s
    json.dump(conf, open(os.path.join(a.out_dir, "configs", f"{cfg}.json"), "w"), ensure_ascii=False, indent=2)
    json.dump(mem_audit(engine), open(os.path.join(a.out_dir, "memory", f"{cfg}.json"), "w"), ensure_ascii=False, indent=2)
    sc = scale_audit(engine)
    print(f"[m10 {cfg}] kv_dtype={conf['kv_cache_dtype']} calc_kv_scales={conf['calculate_kv_scales']} "
          f"kv_tokens={conf['kv_cache_size_tokens']} scale k_all_one={sc.get('k_all_one')} v_all_one={sc.get('v_all_one')}")

    # 阶段1：quality（prompt_logprobs，不计时）
    q = quality_ppl(engine, tf, a.quick)
    q["config"] = cfg
    json.dump(q, open(os.path.join(a.out_dir, "quality", f"{cfg}.json"), "w"), ensure_ascii=False, indent=2)
    print(f"[m10 {cfg}] PPL overall={q['overall']['ppl']} zh={q['zh']['ppl']} en={q['en']['ppl']} scored={q['overall']['scored_tokens']}")

    # 阶段2：perf（绝不开 prompt_logprobs）
    raw_path = os.path.join(a.out_dir, "raw", f"{cfg}_perf.jsonl")
    with open(raw_path, "w", encoding="utf-8") as raw:
        pf_lengths = [128, 512] if a.quick else [128, 512, 2000]
        dec_matrix = [(64, [1, 8])] if a.quick else [(64, [1, 8, 64]), (1024, [1, 8])]
        wu, rp = (2, 5) if a.quick else (8, 30)
        prefill = perf_prefill(engine, tf, cfg, pf_lengths, wu, rp, raw)
        decode = perf_decode(engine, spy, tf, cfg, dec_matrix, wu, rp, raw)
    spy.restore()

    perf = {"config": cfg, "scale_audit": sc, "prefill": prefill, "decode": decode}
    json.dump(perf, open(os.path.join(a.out_dir, "configs", f"{cfg}_perf.json"), "w"), ensure_ascii=False, indent=2)
    for d in decode:
        print(f"  decode ctx={d['context_len']} b={d['batch_size']}: median={d['median_ms']}ms "
              f"thru={d['throughput_tok_s_median']} seq={d['seq_len']} valid={d['all_valid']}")
    print(f"[m10 {cfg}] done -> {a.out_dir}")


if __name__ == "__main__":
    main()
