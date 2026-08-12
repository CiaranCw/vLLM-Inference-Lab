# M21 监控栈（Prometheus + Grafana + DCGM + 自写 host exporter）

配置全部以文件入库、由 provisioning 加载 —— **面板不是 UI 手点出来的**，换台机器也能复现。

## 拓扑

| 组件 | 端口 | 采什么 | 备注 |
|---|---|---|---|
| `vllm serve` | 8014 | vLLM 官方 `/metrics`（122 条 series） | 端口沿用 M12 冻结配置，便于交叉验证 |
| `dcgm-exporter`（容器） | 9400 | GPU util / 显存 / 功耗 / 温度 / 时钟 | **本环境 `DCGM_FI_PROF_*`（SM active 等）全为 N/A** |
| `scripts/m21_host_exporter.py` | 9836 | 服务进程 CPU%/RSS/线程数 + NVML 交叉校验读数 | dcgm-exporter 不提供进程级指标 |
| Prometheus（容器） | 9090 | 1s 抓取 + 7 条告警规则 | 仅监听 127.0.0.1 |
| Grafana（容器） | 3000 | 面板 `m21-vllm-capacity` | 匿名只读，便于截图取证 |

三个容器都用 `network_mode: host`，因为 Prometheus 要抓 host 上的 vLLM 与自写 exporter。

## 启动

```bash
# 本机 WSL session 常缺 docker 补充组，故走 wrapper
~/bin/docker compose -f monitoring/docker-compose.yml up -d

# host exporter（长驻；靠 PID 文件跟踪被测 server，server 重启也不用重启它）
source .venv/bin/activate
nohup python scripts/m21_host_exporter.py --port 9836 \
  --pid-file outputs/m21/raw/server.pid > logs/m21/host_exporter.log 2>&1 &

# 自证：四个 target 全绿 + 规则加载 + 关键 PromQL 有数 + 面板已 provision
python scripts/m21_gate1.py
```

停止：`~/bin/docker compose -f monitoring/docker-compose.yml down`

## 踩过的坑（都已修，写在这里免得重犯）

1. **`dcgm-exporter` 默认 30s 采集一次**（`-c/--collect-interval` 默认 30000ms）。
   秒级压测窗口里 Prometheus 每秒抓到的是同一份最多 30 秒前的旧值 ——
   实测 DCGM 报 `GPU_UTIL=2%` 而同期 NVML 是 33%。**靠 NVML 这个正交第二证据源才发现**。
   现已在 compose 里固定 `-c 1000`。
2. **改了 `prometheus.yml` 必须 reload**：`curl -X POST http://127.0.0.1:9090/-/reload`
   （compose 已开 `--web.enable-lifecycle`），否则还在抓旧 target。
3. **进程 CPU 必须复用 `psutil.Process` 对象**：每次 scrape 重新 `children()` 构造新对象的话，
   每个对象的 `cpu_percent()` 都是首次调用、恒返回 0 —— 会把 vLLM V1 跑在**子进程**里的
   EngineCore 的 CPU 全漏掉（正是最该看的那个）。
4. **告警在「冷启动」时会真的响**：warmup 首请求含 JIT，TTFT ~0.42s > 300ms SLO，
   30s rate 窗口里会触发 `VLLMTTFTP99AboveSLO`。这不是误报，是判据要写清「空载」得等样本滑出窗口。
5. **脚本必须互斥**：并发跑两个实验会轮流抢 8014 端口，写出交错且标签错位的记录
   （M21 的 B3 阶梯就这么废过一次）。现由 `m21_common.acquire_lock()` 兜住。

## 告警规则（`alerts.yml`）

7 条，全部在 E6 演练中**真实 firing 过**（证据见 `outputs/m21/e6_drills.json`）：
`VLLMServerDown` / `HostExporterTargetDown` / `VLLMTTFTP99AboveSLO` / `VLLMQueueBacklog` /
`VLLMKVCacheNearFull` / `VLLMPreemptionHappening` / `GPUMemoryNearFull`。

TTFT 阈值直接沿用 M12 冻结 SLO（P99 ≤ 300ms），不另造标准。
