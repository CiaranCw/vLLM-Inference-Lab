# M27 · 离线依赖清单 / H20 传输包

> 原则：用 `pip download` + checksum 清单，**不安装到本地 lab 环境**，不使用第三方中转站、网盘或匿名分享服务。
> 本地 wheelhouse 在仓库外：`~/m27_offline/wheelhouse/`（`*.whl` 已被 .gitignore 排除，不入库）。

## 1. 需要新增到 H20 的依赖（只有 NIXL）

| 文件 | 大小（B） | sha256 | 来源 / 核对 |
|---|---:|---|---|
| `nixl-1.2.0-py3-none-any.whl` | 8842 | `eb666a3691e3d98be35164128175d7763eea3217125b5ead9fc54cd9754e0f16` | PyPI；本地 `sha256sum -c` OK，与 PyPI JSON 的 digest 一致 |
| `nixl_cu13-1.2.0-cp311-cp311-manylinux_2_28_x86_64.whl` | 42499749 | `dae1330fd5eeecd23b252b556fb1a528c22ea02ee2019cae92a5a3fad7e8f401` | 同上 |
| `nixl_cu12-1.2.0-cp311-cp311-manylinux_2_28_x86_64.whl` | 47934649 | `2dca720380d516762109541fd71ee8d75482ffc625751cac2d2506ac97c49c09` | **不需要、未下载**，仅记录 PyPI digest |

包结构事实（直接读 wheel，未安装）：

- `nixl` 是 meta 包：`Requires-Dist: nixl-cu12==1.2.0, nixl-cu13==1.2.0`。导入时按 `torch.version.cuda` 的主版本
  选择 `nixl_cu{major}`（`nixl/__init__.py` `_load_cuda_backend`）。torch 2.11.0+cu130 → 只需要 `nixl-cu13`。
- `nixl-cu13` 声明 `Requires-Dist: torch, numpy`。**必须 `--no-deps`**，否则 pip 可能改动 H20 上的 torch。
- `nixl-cu13` 自带 UCX（`nixl_cu13.libs/ucx/libuct_{cuda,cma,ib,ib_mlx5,ib_mlx5_gda,ib_efa,rdmacm}.so`）以及 NIXL 插件
  （`.nixl_cu13.mesonpy.libs/plugins/libplugin_{UCX,LIBFABRIC,GDS,GDS_MT,POSIX,OBJ,GPUNETIO,...}.so`）；
  wheel 中没有 `ucx_info` 可执行文件，所以传输层的确认改用 E0 时的 `UCX_PROTO_INFO=y` 日志。
- `libuct_cuda` 运行时需要 driver 提供的 `libcuda.so.1` 与 `libnvidia-ml.so.1`。
- wheel 标签为 `manylinux_2_28`。M22 的环境记录显示 H20 上 llguidance 因为需要 glibc 2.31 而无法安装、
  只有 glibc 2.28 可用（`logs/m22/environment.txt:5,120`），vLLM 0.24.0 的 `manylinux_2_28` wheel 也能正常运行，
  所以预期满足要求；preflight §0 会打印 `ldd --version` 作为实测值。

本地安装演练（只演练目录布局，不导入、不改任何 venv）：

```bash
python -m pip install --no-deps --no-index --target /tmp/m27_target_rehearsal \
  --python-version 3.11 --only-binary=:all: --platform manylinux_2_28_x86_64 --implementation cp \
  nixl-1.2.0-py3-none-any.whl nixl_cu13-1.2.0-cp311-cp311-manylinux_2_28_x86_64.whl
# → Successfully installed nixl-1.2.0 nixl-cu13-1.2.0；.nixl_cu13.mesonpy.libs/plugins 与 nixl_cu13.libs/ucx（10 个 .so）完整
```

本地是 Python 3.12 且没有 cu130 torch，所以**导入与 UCX agent 创建只能在 H20 的 preflight §3 中验证**。

## 2. H20 上应已存在的依赖（M22 留下，不重新安装）

| 依赖 | 期望 | 证据 / 如果缺失 |
|---|---|---|
| Python | 3.11.13（`/root/.venv`） | `logs/m22/environment.txt` |
| torch | 2.11.0+cu130 | 同上 |
| vLLM | 0.24.0，wheel `vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl`（279209310 B，sha256 `2d2831aeba311292250df0132dbc4d8e9f42c654536eaec48e6fe58acb1822cf`，PyPI digest） | `logs/m22/pip_install.log`；缺失 → STOP 并询问，不自行重装 |
| aiohttp / numpy | vLLM 的依赖 | proxy / workload / summarize 需要；preflight §2–3 间接验证 |
| CUDA userspace | cuda-compat 13.0 + driver 535.247.01 | 不修改 |
| 模型 | `/root/m22_offline/models/Qwen2.5-7B-Instruct` | 身份 = 仓库内 `notes/m27/model_artifact/qwen2.5-7b-instruct.sha256`（见 [`model_artifact_manifest.md`](model_artifact_manifest.md)）；缺失 → 用本地恢复副本（下表） |

本地 `/home/ciaran/m22_offline` 已不存在，但 M22 offline pack 的完整备份在 **`/mnt/f/wsl_backup/m22_offline/`**
（2026-10-03 找回；其中 runtime bundle 的 sha256 与仓库内 M22 记录相同）。H20 上缺资源时可以从这里直接传：

| 资源 | 本地路径 | 身份 / 校验 |
|---|---|---|
| 7B 模型（15 个文件，15,242,807,272 B） | `/mnt/f/wsl_backup/m22_offline/models/Qwen2.5-7B-Instruct/` | 身份 manifest `sha256sum -c` 15/15 OK（`logs/m27/model_artifact_sha256_check_20261003.log`）；内容 == HF `Qwen/Qwen2.5-7B-Instruct@a09a3545` |
| vLLM 0.24.0 wheel | `/mnt/f/wsl_backup/m22_offline/wheelhouse/vllm-0.24.0-cp38-abi3-manylinux_2_28_x86_64.whl` | 279209310 B，sha256 `2d2831ae…22cf`，2026-10-03 重新计算，与 PyPI digest 相同 |

## 3. 交接包分三部分（不是“一包全有”）

| 部分 | 内容 | 身份 / manifest |
|---|---|---|
| code handoff | lab 仓库的 pre-H20 anchor commit（`git clone` 或 `git archive`） | commit SHA（见 `HANDOFF.md` §5）；archive 方式另记 tar 的 sha256 |
| dependency wheelhouse | `nixl-1.2.0-py3-none-any.whl` + `nixl_cu13-1.2.0-cp311-…whl`（`~/m27_offline/wheelhouse/SHA256SUMS`）；vLLM 0.24.0 wheel 只在 H20 环境缺失时才需要 | §1 / §2 的 sha256（均与 PyPI digest 相同） |
| model artifact | 7B 目录（≈15.2 GB，权重不入 Git） | `notes/m27/model_artifact/qwen2.5-7b-instruct.sha256`（manifest 自身 sha256 登记在 `m27_common.FROZEN`） |

传输一律用 scp/rsync 直接点对点，传完在 H20 上按各自 manifest 校验。

## 4. NIXL 下载计划

| 方式 | 命令 | 条件 |
|---|---|---|
| A（推荐） | 在 H20 上：`pip download --no-deps nixl==1.2.0 nixl-cu13==1.2.0 --python-version 3.11 --only-binary=:all: -d /root/m27_offline/wheelhouse`，然后对照 §1 的 sha256 | M22 时 H20 可以访问 PyPI |
| B | 本地 `~/m27_offline/wheelhouse/`（两个 wheel + `SHA256SUMS`）经 scp/rsync 直接传到 H20 | H20 无法访问 PyPI 时 |

安装命令见 [`M27_H20_RUNBOOK.md`](M27_H20_RUNBOOK.md) §2（`--no-deps --no-index --target /root/m27_offline/site`，再设置 `PYTHONPATH`）。

## 5. 未闭环的风险（不伪造，只记录）

| 依赖 | 期望 | 方法 | 未解决的问题 |
|---|---|---|---|
| UCX 同机 GPU 传输 | cuda_ipc（NVLink） | NIXL 自带 UCX 1.21.x（wheel 内字符串证据） | driver 535 + cuda-compat 13 组合下 `libuct_cuda` 是否可用，尚未验证；UCX 可能选择 mlx5 网卡而不是 cuda_ipc（M22 的 `topo -m` 中有 42 个 NIC，即 NIC0–41）→ Gate 1 用 `UCX_PROTO_INFO=y` 确认，不符时 STOP |
| NIXL 与 vLLM 0.24.0 的兼容性 | vLLM 0.24.0 的 NIXL 路径 | preflight 导入 `vllm.distributed.nixl_utils.NixlWrapper` 与 `NixlConnector` | 只有在 H20 上才能端到端验证 |
| 8 卡拓扑 | 未知 | Gate 0 `topo -m` | M22 时容器只看到 4 卡；E5 可能无法执行 |

> 上表是上机前的风险登记（历史）。H20 结果回填（2026-10-06）：Gate 0 实测 GPU0–7 两两 NV18，E0 证实 GPU KV 走 `cuda_ipc/cuda`；正式实验最多用到 4 GPU（E4），可选的 8-GPU E5 未执行、不属于 M27 最终范围。见 [`results.md`](results.md)。
