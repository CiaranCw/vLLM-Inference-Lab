# M27 · 模型 artifact 身份清单（Qwen/Qwen2.5-7B-Instruct）

> 2026-10-03 pre-H20 closure 冻结。模型权重**不入 Git**；仓库里只放身份文件。
> 本文件区分三件事：**artifact 身份**（与机器无关，入库）、**本地当前可用性**（只是本机事实）、**H20 formal 时的要求**。

## 1. 身份（与机器无关）

| 项 | 值 |
|---|---|
| 模型 | `Qwen/Qwen2.5-7B-Instruct` |
| 内容对应的 HF revision | `a09a35458c702b33eeacc393d103063234e8bc28`（HF 仓库当前 head，最后一次提交 2025-01-12） |
| 原始下载来源 | ModelScope（M22，2026-08-08；`qwen2.5-7b_files.json` 记为 “modelscope (primary); huggingface metadata first”）。ModelScope 当时的 revision 只记为 `master`，没有 commit id |
| 文件数 / 总字节 | 15 / 15,242,807,272 B（与已入库的 `notes/m22/M22_EXECUTION_FREEZE_MANIFEST.json` 中 `file_count` / `total_size_bytes` 相同） |
| safetensors | 4 个分片 `model-0000{1..4}-of-00004.safetensors` + `model.safetensors.index.json` |
| 身份 manifest | [`model_artifact/qwen2.5-7b-instruct.sha256`](model_artifact/qwen2.5-7b-instruct.sha256)（15 行，相对文件名；该文件本身的 sha256 = `157f08c69340e16b5d0358a31a4d20716f73a3b90ce6538ed557cff5fbe49638`，登记在 `m27_common.FROZEN["model"]`） |
| HF 比对证据 | [`model_artifact/hf_revision_match.json`](model_artifact/hf_revision_match.json) |

逐文件 sha256（即身份 manifest 的内容）：

| 文件 | 字节 | sha256 |
|---|---:|---|
| `.gitattributes` | 1519 | `11ad7efa24975ee4b0c3c3a38ed18737f0658a5f75a0a96787b576a78a023361` |
| `LICENSE` | 11343 | `832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e` |
| `README.md` | 6240 | `f366f33bbf6bcadbb7d87f0a21a7b65584a56b8d58b0743c77c88bee625b93a6` |
| `config.json` | 663 | `7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c` |
| `configuration.json` | 2 | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| `generation_config.json` | 243 | `3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f` |
| `merges.txt` | 1671839 | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |
| `model-00001-of-00004.safetensors` | 3945441440 | `a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7` |
| `model-00002-of-00004.safetensors` | 3864726352 | `f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185` |
| `model-00003-of-00004.safetensors` | 3864726424 | `8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5` |
| `model-00004-of-00004.safetensors` | 3556377672 | `1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd` |
| `model.safetensors.index.json` | 27752 | `624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028` |
| `tokenizer.json` | 7031645 | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` | 7305 | `5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583` |
| `vocab.json` | 2776833 | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |

## 2. 来源与身份等级（不夸大）

找回过程：在 `/home/ciaran`、`../`、`/mnt/{c/Users,d,e,f}`（depth ≤ 4–7）中按名称搜索，只找到**一个**候选：
`/mnt/f/wsl_backup/m22_offline/models/Qwen2.5-7B-Instruct`（同目录还有 `manifests/`、`bundles/`、`wheelhouse/`）。
HF cache 中的 `models--Qwen--Qwen2.5-7B-Instruct` 只有 `refs/main`（12K），ModelScope cache 只剩 `.lock` 文件，都不是候选。

证据链：

1. **备份目录就是 M22 offline pack**：其中 `bundles/m22_runtime_bundle.tar.gz` 的 sha256
   `ebc3c7d5…86b1` 与已入库的 `notes/m22/frozen_runtime_snapshot/m22_runtime_bundle.tar.gz.sha256` 相同；
   文件数与总字节与已入库的 M22 freeze manifest 相同。
2. **当前字节 == M22 下载时生成的 manifest**：`manifests/qwen2.5-7b.sha256`（mtime 2026-08-08 12:54，即下载完成时）
   被逐字节复制为本目录的身份 manifest；2026-10-03 对恢复副本重新计算 `sha256sum -c`，15/15 OK，见
   [`logs/m27/model_artifact_sha256_check_20261003.log`](../../logs/m27/model_artifact_sha256_check_20261003.log)。
3. **内容 == HF 官方 revision**：用 `HfApi.model_info(files_metadata=True)` 比对（未下载任何文件）：
   4 个 safetensors 的 LFS sha256 与身份 manifest 相同；其余 10 个文件的 git blob id 与本地文件
   `git hash-object` 相同；HF 仓库的 14 个文件全部一致，多出的 `configuration.json`（`{}`）是 ModelScope 下载器附带的。

**身份等级**：recovered artifact believed to be the M22 artifact based on provenance / path / package evidence;
current identity is now frozen by SHA256。另外，文件内容与 HF `Qwen/Qwen2.5-7B-Instruct@a09a3545` 逐文件一致。

**不能声称的**：与 M22 在 H20 上实际加载的那份**逐位相同**。M22 在 H20 上的 `logs/m22/model_7b_checksum.log` 只有
“OK”，没有保存 hash 数值，也没有保存当时所用 manifest 的 hash，所以无法用密码学证据把 H20 副本与这份 manifest 绑定；
另外也没有原始的模型 tar 包。因此不升级为 “exact recovered M22 package”。

M27 uses the same model family/name as M22, but this frozen artifact is not claimed to be bitwise-identical to the
historical M22 H20 copy。这不影响 M27：M27 的正式结论都是模块内部同一 artifact 下的 A/B 对照。
M22 absolute latency numbers and M27 absolute latency numbers must not be directly subtracted and attributed solely
to parallel strategy（另见计划 §6 的 HND / 口径边界）。

## 3. 模型 Gate（实读 `config.json`，2026-10-03）

| 字段 | 实测 | 冻结值 |
|---|---|---|
| architectures / model_type | `Qwen2ForCausalLM` / `qwen2` | 同 |
| num_hidden_layers | 28 | 28 |
| num_attention_heads（Q） | 28 | 28 |
| num_key_value_heads | 4 | 4 |
| hidden_size → head_dim | 3584 → 128 | 128 |
| torch_dtype | bfloat16 | bfloat16 |
| max_position_embeddings | 32768（`use_sliding_window=false`，无 `rope_scaling`） | ≥ 32768 才用 mml 32768 |
| quantization_config | 无 | 无 |

结论：与冻结设计一致，**不触发回退**，max_model_len = 32768、最长 prompt = 16384（< 32768，满足 vLLM 的严格小于约束）。
KV 理论值 = 2 × 28 × 4 × 128 × 2 B = **57344 B/token**，与冻结值相同（公式值，不是实测传输量）。

## 4. 本地当前可用性（只是本机事实，不是不变量）

- 路径：`/mnt/f/wsl_backup/m22_offline/models/Qwen2.5-7B-Instruct`（Windows F: 盘的 WSL 挂载；只读使用，没有改动任何文件）。
- 同一备份中的 `wheelhouse/vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl`：279209310 B，sha256 `2d2831ae…22cf`，与 PyPI digest 相同。
- `check_invariants.py` 只检查仓库内的身份文件，**不要求**任何开发机存在上述绝对路径。

## 5. H20 formal 时的要求

- H20 上 `M27_MODEL` 指向的目录必须通过 preflight §5：`sha256sum -c notes/m27/model_artifact/qwen2.5-7b-instruct.sha256`
  15/15 OK，且 config 字段与冻结值一致。preflight 默认使用仓库内的身份 manifest；如果另行指定 `M27_MODEL_SHA256`，
  其内容必须与身份 manifest 逐字节相同，否则 FAIL。
- 首选 H20 上 M22 遗留的 `/root/m22_offline/models/Qwen2.5-7B-Instruct`；不存在时，把本地恢复副本经 scp/rsync 传过去
  （≈15.2 GB，见 [`offline_package.md`](offline_package.md) §3），**不在 H20 上重新下载**。

## 6. 附记

An accidental config-only Qwen3-0.6B HuggingFace cache created during local self-test was removed after review.
No model weights were downloaded and no formal M27 artifact or result was affected.
（删除的是 `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/`，36K。同一次意外还留下了 0 字节的
`~/.cache/huggingface/hub/.locks/models--Qwen--Qwen3-0.6B/f5c3703b….lock`，不在本轮授权删除的范围内，保留未动。）
