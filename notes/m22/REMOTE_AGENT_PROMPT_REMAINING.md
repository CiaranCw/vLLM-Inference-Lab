# M22 · 远端 Agent 续跑 Prompt（剩余约 2.5h · 可选补强）

> **粘贴给 H20 内部客户端 Agent。**  
> 本轮 **正式冻结矩阵（D1/D2/D3 TP=1/2/4 + P1 主路径）已完成并已回传本机校验通过**。  
> 以下任务是 **optional stretch**，不是重新开实验、不是改矩阵。

---

## 0. 先读结论（必须遵守）

### 已经完成（不要重跑正式 `h20_formal` run，除非操作者明确 `--force`）

- TP∈{1,2,4}：D1 + D2（batch 1/4/16/64）+ D3（256/1024/4096），每组 5 warmup + 15 measure  
- P1 profile JSONL：tp2/tp4 各 3 case success  
- NCCL smoke、environment、invariants 22 PASS、formal tar 已拉回本机  

### 明确禁止（剩余时间内）

1. **禁止** PP / DP / EP / CP / P-D（冻结文档写明：**未编写任何实现**；不得现场发明脚本当正式实验）  
2. **禁止** 改 D2/D3 网格、改 warmup/measure 数、改模型当「第二正式对照」  
3. **禁止** 源码编译 CUDA/C++；禁止 kill 他人 GPU 进程  
4. **禁止** 把 optional 结果写入与 `h20_formal` 相同 CSV 混肴；新产物用 **`h20_opt_*` tag** 或独立文件名  
5. **禁止** 宣称「又完成一套正式 M22」——正式线已结束  

### 时间盒

- 总预算约 **2.5h**；**T+2:00** 起只打包/校验，不再开新引擎  
- 任一步失败 → 记日志 + 跳过，**不阻塞打包**  
- 若 GPU 被占 / 环境坏了 → **BLOCKED**，释放机时，不要空转  

---

## 1. 环境（沿用既有）

```bash
# 在项目根 / tmux
export REPO_ROOT="$PWD"
export OFFLINE="$REPO_ROOT/m22_offline"
export MODEL_PATH="$OFFLINE/models/Qwen2.5-7B-Instruct"
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
unset VLLM_LAB_TRACE VLLM_LAB_M9_COALESCE
mkdir -p logs/m22 outputs/m22/raw outputs/m22/traces
```

先确认 4 卡仍可用且无他人进程：

```bash
nvidia-smi -L
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

---

## 2. 可选任务清单（按优先级；全部 optional）

### O1 · 补全 P1 chrome traces（高价值 · 建议先做）

**背景（本机审查发现）**：`h20_formal` 的 profile JSONL 显示 tp2/tp4 已跑  
`decode_small_batch` / `decode_large_batch` / `prefill_long`，但打包回来的  
`outputs/m22/traces/` **几乎只有 `decode_small_batch_*.pt.trace.json.gz`**，  
缺 large_batch / prefill_long 的 chrome trace → 离线 `m22_trace_analyze.py` 证据不完整。

**动作**：

```bash
# 1) 盘点现有 traces
ls -lah outputs/m22/traces/ | tee logs/m22/opt_trace_inventory_before.txt
# 统计是否已有 large_batch / prefill_long
ls outputs/m22/traces/ | grep -E 'large_batch|prefill_long' || echo 'MISSING_LARGE_OR_PREFILL_TRACES'

# 2) 若缺失：用新 tag 重跑 profile（避免误伤已有 formal JSONL 语义）
#    不要 --force 覆盖 h20_formal；改用 h20_opt_profile
python scripts/m22_run_all.py --tag h20_opt_profile --tp 2,4 --profile --model "$MODEL_PATH" \
  2>&1 | tee logs/m22/opt_profile_tp24.log

# 3) 盘点之后
ls -lah outputs/m22/traces/ | tee logs/m22/opt_trace_inventory_after.txt
ls outputs/m22/traces/ | grep -E 'h20_opt_profile.*(large_batch|prefill_long|decode_small)' | tee logs/m22/opt_trace_keys.txt
```

成功标准：`decode_large_batch` 与 `prefill_long` 的 per-rank `.pt.trace.json.gz` 落盘。  
失败：保留 log，标 `observation_gap: missing_chrome_traces_for_large_or_prefill`，继续 O2。

### O2 · TP=1 P1 profile 基线（中价值）

正式 P1 只跑了 tp2/tp4。补 TP=1 便于对照「无跨卡 all-reduce」时的 kernel 组成。

```bash
python scripts/m22_run_all.py --tag h20_opt_profile --tp 1 --profile --model "$MODEL_PATH" \
  2>&1 | tee logs/m22/opt_profile_tp1.log
```

### O3 · 远端快速 sanity（低成本）

```bash
# 不重跑 formal；只确认产物仍在
ls outputs/m22/raw/h20_formal_tp{1,2,4}_run.jsonl
ls outputs/m22/h20_formal_{decode,decode_effect,prefill,memory_kv,output_consistency}.csv
python scripts/check_invariants.py --module m22 2>&1 | tee logs/m22/opt_invariants_recheck.log
```

> 注：`m22_trace_analyze.py` **应在本机离线跑**（脚本注释写明不占 H20）。远端不必强跑；若你跑了，把 CSV 一并打包即可。

### O4 · 打包 optional 增量（必须做，哪怕 O1–O3 全跳过）

```bash
# 打一个「可选补强」包，文件名勿覆盖 formal full
TS=$(date +%Y%m%dT%H%M%S)
TAR="m22_h20_opt_${TS}.tar.gz"
tar -czf "$TAR" \
  outputs/m22/raw/h20_opt_profile_*.jsonl \
  outputs/m22/traces/h20_opt_profile_* \
  logs/m22/opt_*.log \
  logs/m22/opt_*.txt \
  2>/dev/null || true
# 若上面因无匹配而空：至少打包 traces 增量与日志
if [ ! -s "$TAR" ]; then
  tar -czf "$TAR" logs/m22/opt_* outputs/m22/traces 2>/dev/null || tar -czf "$TAR" logs/m22
fi
sha256sum "$TAR" | tee "$TAR.sha256"
ls -lh "$TAR" "$TAR.sha256"
echo "READY_FOR_OPERATOR_PULL $TAR"
```

通知操作者：**本机主动 pull** 该 tar + sha256（GitHub 仍可能不可用）。

---

## 3. 汇报模板（中文）

```text
## M22 optional stretch 报告
- 剩余机时估计 / 是否提前收工:
- O1 补 traces: DONE/SKIP/FAIL + 证据路径
- O2 TP=1 profile: DONE/SKIP/FAIL
- O3 invariants recheck: PASS/FAIL/SKIP
- 打包文件名 + sha256:
- 未做且不应做的事: PP/DP/...（确认未碰）
- NEED_HUMAN: pull 命令或路径
```

---

## 4. 一句话策略

**正式 M22 已闭环。** 剩余 2.5h 只做「补 chrome trace + TP1 profile 基线 + 打包」，  
做不完就收工放机时；**不要**为了填满时间去写 PP/DP 或重跑整网 `h20_formal`。
