# M13 断点快照（RESUME / 交接）

> **一句话状态**：M13 = 为 vLLM `BlockTable.compute_slot_mapping` 补单元测试。已基于**当前 upstream/main `cb81048`** 在**干净工作树**完成适配、实现与全套验证：**18 passed**（仓库根目录 + 真实 conftest）、pre-commit 影响本文件的 hook 全绿、**零产品代码改动**。
> **状态：READY FOR HUMAN REVIEW BEFORE PR —— 尚未 push、尚未创建 PR。**
>
> **最后更新**：2026-08-04 03:53 UTC（本轮全面重核，推翻多条旧结论）。

---

## 1. 最新上游状态（本轮实测，不再引用旧笔记）
| 项 | 值 |
|---|---|
| 核查时间 | 2026-08-04 03:21–03:53 UTC |
| upstream/main SHA | **`cb8104839c141609d99f1254459ef3a4f1bd4263`**（2026-08-04T03:05:45Z，`[XPU][CI]Adjust Samplers test ENV for Intel GPU (#50199)`） |
| 最新 release | **v0.26.0**（2026-07-27）——旧笔记的 v0.24.0 已落后 2 个 minor |
| 干净工作区 | `/home/ciaran/projects/vllm-m13-upstream`，分支 `m13/block-table-slot-mapping-tests`，base = `cb81048` |
| 环境 | Python 3.12.13 / torch 2.13.0+cu130 / triton 3.7.1 / RTX 5080 sm_120(12,0) / driver 610.62 |
| 安装方式 | `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`（遵 AGENTS.md，全程 `uv`，无裸 pip） |

## 2. 缺口与重复性（结论：**仍成立，值得做**）
- **`compute_slot_mapping` 在整个 `tests/` 中 0 命中**（`grep -rln "compute_slot_mapping\b" tests/` → 空）。
- `tests/v1/worker/test_block_table.py` **在 main 上仍不存在**（同目录已有 `test_gpu_block_table.py`，但它测 MRV2 `gpu/block_table.BlockTables` 的 staged writes / move_row，**不测 slot mapping**）。
- `test_gpu_model_runner.py` 有 `test_hybrid_block_table_initialization`，但**只验证初始化与表布局**，且自引用 `map_to_kernel_blocks` 作期望；`SlotMappingMode.NONE` 相关测试只断言 `max_num_blocks_per_req`。**均不计算 slot mapping。**
- **PR #30683**：`CLOSED`、未合并、labels `stale, v1`、作者 `pisceskkk`（closed 2026-04-09）。
- 无任何在途/已合入的等价测试 PR（`gh search prs` 查 slot mapping / block_table test / test_block_table）。
- 相关在途 PR 均**不删除**本路径：#41270/#41272 只从 **attention metadata** 移除 slot_mapping（文件为 backends/*），#46446/#45245/#43879 是 warmup 预编译，#47924 是 RFC。

## 3. 旧笔记中已被推翻的结论（本轮修正）
| 旧结论 | 实际（main `cb81048`） |
|---|---|
| “kernel 语义在当前 main 未变” | **错**。签名与语义均已变（见下三条） |
| slot mapping 用 `TOTAL_CP_WORLD_SIZE = pcp × dcp` | **错**。现在 `TOTAL_CP_WORLD_SIZE=self.dcp_world_size`、`TOTAL_CP_RANK=self.dcp_rank`——**PCP 已不参与 slot mapping**（`block_table.py:206-207`） |
| 只有单一 block_size | **错**。区分 `kv_cache_block_size`（分配）/ `block_size`（=kernel）/ `blocks_per_kv_block`；kernel 新增 `KV_CACHE_BLOCK_SIZE`、`BLOCKS_PER_KV_BLOCK` constexpr |
| 无 SlotMappingMode | **错**。新增 `SlotMappingMode.{TOKEN_TO_KV_SLOT,NONE}`；`NONE`（Mamba/GDN）直接 `return`、**不触碰 buffer**（实测确认） |
| PR 标题可用 `[Test]` | **错**。`docs/contributing/README.md:223+` 的合法前缀不含 `[Test]`；取 **`[CI/Build]`** |
| 旧 26/26 通过 | 该结论仅对 v0.24.0 成立，**已作废**；当前为 **18 passed** on `cb81048` |
| “/tmp standalone 跑测试” | 已不需要：装 `requirements/test/cuda.in` 后**真实 conftest 可导入**，正式从仓库根目录跑 |

## 4. 当前实现（18 tests, 1 new file, 0 product change）
`tests/v1/worker/test_block_table.py`（+322 行）：
- **手算 golden（3 组，独立于 oracle）**：普通映射 2 例（参数化）、DCP world2/rank1 1 例、Hybrid kv32/kernel16 1 例——注释里写出完整手算过程。
- **oracle**：`_reference_slot_mapping`，按 current main 语义逐步重建（virtual block → DCP 归属 → local offset → kv→kernel block 展开 → 物理块 → slot → PAD），**不镜像 kernel 索引式**。
- 覆盖：单 token / 块内首末 / 跨多块 / 非连续物理块 / ragged 多请求 / prefix offset / **跨 Triton tile（1030 tokens，非 1024 整数倍，padding 再跨一 tile）** / **DCP world{2,4} × interleave{1,4} 全 rank + “每 token 恰被一个 rank 拥有”** / **Hybrid (32,16) 与 (64,16) 端到端** / **SlotMappingMode.NONE buffer 不变（sentinel −12345 ≠ PAD −1）**。
- **删除**旧草稿的 PCP × DCP 参数矩阵（语义已不成立，保留会误导）。

## 5. 验证结果（全部在干净树、仓库根目录）
| 检查 | 结果 |
|---|---|
| `pytest tests/v1/worker/test_block_table.py -v` | **18 passed** in 5.34s |
| `pytest ... --collect-only` | 18 collected（真实 conftest） |
| **变异测试**（临时给 kernel `slot_offsets` 注入 +1） | **17 failed / 1 passed** → 测试非空转；随后已还原，`git diff vllm/` 为空 |
| `pre-commit run --files <本文件>` | 全绿（ruff check / ruff format / typos / **mypy** / SPDX / forbidden imports / …） |
| `pre-commit run --all-files` | 除 **`shellcheck`**（基线问题：`docs/pre_run_check.sh` SC2317，与本改动无关）外全绿 |
| `git diff --check` | 无空白错误；无行 > 88 字符 |
| 邻近回归 `test_gpu_block_table.py` | 3 failed —— **基线问题**：移除本文件后在纯净 main 上同样 3 failed，根因 `RuntimeError: UVA is not available`（WSL2 无 UVA，MRV2 pinned-host 表）。本文件用 `pin_memory=False` 故不受影响 |

日志：`notes/m13/validation/{environment.txt,pytest_block_table.txt,precommit_changed_files.txt,precommit_all_files.txt}`。

## 6. 未解决 / 待人工决策
1. **DCO 身份**：git 配置为 `CiaranCw <CiaranCw@users.noreply.github.com>`。**尚未创建本地 commit**——请确认该 sign-off 身份是否即为你要用的（AGENTS.md 要求 `Signed-off-by`）。
2. **AI 披露与署名**：AGENTS.md 强制要求 PR 描述声明 AI 辅助、并用 `Co-authored-by:`/`Assisted-by:` trailer；**且明确“纯 code-agent PR 不允许，提交人须能端到端理解并为改动负责”**。→ 你需要**通读本测试文件**再提。
3. **是否 credit #30683**：建议在 PR 描述致谢 `@pisceskkk` 的方向（但**不夸大继承**：本实现是按当前 API 重写，其旧测试无法在 main 运行）。
4. **PR 标题前缀**：建议 `[CI/Build]`（合法列表内，测试基础设施）；若你倾向 `[Core]` 也在合法列表。

## 7. 人工下一步（命令备好，我不会自行执行）
```bash
cd /home/ciaran/projects/vllm-m13-upstream
git status --short                 # 应只有 A tests/v1/worker/test_block_table.py
git diff HEAD                      # 人工逐行审查
# 确认身份后（示例）：
git add tests/v1/worker/test_block_table.py
git commit -s -m "[CI/Build] Add unit tests for BlockTable.compute_slot_mapping"
# 之后才是（需你明确授权）：
# git remote add fork <your-fork-url> && git push fork m13/block-table-slot-mapping-tests
# gh pr create --repo vllm-project/vllm --title ... --body-file <PR_DESCRIPTION.md 的英文正文>
```

## 8. 环境注意
- 干净工作区 `/home/ciaran/projects/vllm-m13-upstream`（**独立 clone + 独立 `.venv`**），与带中文学习注释的 `/home/ciaran/projects/vllm-src`（22 个脏文件、shallow/grafted、pinned v0.24.0）**完全隔离**；后者不可用于 PR。
- GPU 使用前后均查 `nvidia-smi`，**未 kill 任何进程**、未 reset GPU（测试前 2537MiB/1% 占用，来自 WSL 显示，无其他计算进程）。
