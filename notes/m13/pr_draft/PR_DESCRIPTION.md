# PR 草稿：Add unit tests for `BlockTable.compute_slot_mapping`

> 这是 M13 首次上游 PR 的**草稿描述**，尚未提交。GPU 验证挂起（本地 GPU 占用中）。
> 提交前须完成：① 本地 GPU 跑通测试、填入真实 Test Result；② 干净工作树 + DCO sign-off + pre-commit 全绿；③ 决定署名口径（见文末）。

---

## 建议标题
`[Test][V1] Add unit tests for BlockTable.compute_slot_mapping (slot mapping kernel)`

## Purpose
`BlockTable.compute_slot_mapping` / `_compute_slot_mapping_kernel`（`vllm/v1/worker/block_table.py`）把逻辑 token 位置映射到物理 KV-cache 槽位，是 KV 写入正确性的关键路径，且含 Context-Parallel（DCP/PCP）interleave 与 CUDA-graph padding 等易错分支——但当前 `tests/` 无任何直接覆盖（`compute_slot_mapping` 在 `tests/` 下 0 命中）。本 PR 补充针对该入口的单元测试。

本 PR 取代已 stale 关闭的 #30683（`[UT][PCP&DCP] UT for block_table.py`）：那版测试写在旧 API `compute_slot_mapping(req_indices, positions)` 上，现签名已改为 `compute_slot_mapping(num_reqs, query_start_loc, positions)`；本 PR 对齐当前 API，并在其 CP 覆盖之外补充非 CP 正确性边界与 padding 覆盖。

## What's covered
- **非 CP 边界**：块内首/末位置、跨块序列、非连续/乱序物理块号、ragged 多请求（变长首尾相接）、非零起始位置（prefix 已缓存）。
- **CUDA-graph padding**：`[num_tokens, max_num_batched_tokens)` 区间恒为 `PAD_SLOT_ID`。
- **Context Parallel**：DCP=4 × PCP=2 全 8 rank，interleave ∈ {1,16}；以及 world_size=1 退化到普通 slot 公式的一致性。
- 断言基准是一份独立的 plain-numpy oracle（`_reference_slot_mapping`），与 kernel 实现相互独立、交叉验证。

## Test Plan
```bash
pytest tests/v1/worker/test_block_table.py -v
```
（CP interleave 逻辑仅存在于 GPU Triton kernel——CPU fallback 断言 world_size==1——故测试需 CUDA，`@pytest.mark.skipif(not cuda)`。）

## Test Result
```
# 本机 RTX 5080 (sm_120) / WSL2 / lab venv，standalone 运行（绕开需额外依赖的 tests/conftest）
$ python -m pytest test_block_table.py -q
........................                                                 [100%]
24 passed in 3.86s
```
- 24 例＝7 非 CP 边界 + 16 DCP4/PCP2(interleave{1,16}×8 rank) + 1 退化一致性。
- 规范：在 **vLLM `pyproject.toml` 配置下** `ruff check` 与 `ruff format --check` 均通过。
- 待在干净环境补跑：完整 `pre-commit run --all-files`（mypy/codespell 等）与 CI 下 `pytest tests/v1/worker/test_block_table.py`（需 `requirements/test.txt` 依赖如 tblib）。

## Notes / 署名
- 思路承接 #30683（作者 @pisceskkk）；如采「复活」路径应 credit 原作者。
- vLLM 要求披露 AI 辅助并加 `Co-authored-by:` trailer——本仓库 no-coauthor 约定不延伸到上游，署名口径待定。
- 提交须 `git commit -s`（DCO）、`pre-commit run --all-files` 全绿（#30683 正死于 pre-commit 未过）。

---

## 与本仓库草稿的对应
- 草稿测试文件：`notes/m13/pr_draft/test_block_table.py`（**未经 GPU 验证**）。
- 目标落点：vLLM `tests/v1/worker/test_block_table.py`。
- 设计依据与研究留痕：`notes/m13/pr_slot_mapping_test.md`。
