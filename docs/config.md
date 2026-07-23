# Qwen3.6-27B-FP8 部署配置文档

> 记录所有部署参数、技术栈版本、以及各阶段配置变更对照。

## 环境信息

| 项目 | 值 |
|------|-----|
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition（97,887 MiB VRAM；历史 baseline 为 L40S 46GB） |
| **系统内存** | 57GB |
| **模型** | Qwen/Qwen3.6-27B-FP8（29GB） |
| **模型路径** | `/data/models/Qwen3.6-27B-FP8` |
| **vLLM 版本** | 0.25.1（容器 `vllm/vllm-openai:v0.25.1`） |
| **Docker 运行时** | nvidia-container-runtime |

---

## 当前配置（Baseline: MTP k=3）

```yaml
# docker-compose.yml — 当前运行版本
command: >
  /models/Qwen3.6-27B-FP8
  --served-model-name qwen3.6-27b
  --host 0.0.0.0
  --port 18001
  --trust-remote-code
  --dtype auto
  --kv-cache-dtype fp8_e4m3
  --max-model-len 32768
  --max-num-seqs 10
  --gpu-memory-utilization 0.90
  --max-num-batched-tokens 8192
  --enable-prefix-caching
  --enable-chunked-prefill
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  --reasoning-parser qwen3
  --language-model-only

environment:
  - NVIDIA_VISIBLE_DEVICES=0
  - HF_HOME=/models/.cache
  - TZ=Asia/Shanghai
```

### 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--max-model-len` | 32768 | 最大上下文长度 |
| `--max-num-seqs` | 10 | 最大并发序列数 |
| `--gpu-memory-utilization` | 0.90 | 显存使用比例 |
| `--max-num-batched-tokens` | 8192 | 单 batch 最大 prefill tokens |
| `--kv-cache-dtype` | fp8_e4m3 | KV Cache FP8 量化 |
| `--enable-prefix-caching` | enabled | 前缀 KV 缓存复用 |
| `--enable-chunked-prefill` | enabled | 长 prompt 分块预填 |
| `--speculative-config` | MTP k=3 | 3-token 投机解码 |
| `--reasoning-parser` | qwen3 | 思维链解析 |
| `--language-model-only` | enabled | 跳过视觉编码器 |
| DeepGEMM | 默认决策 | vLLM 0.25.1 可启动，但 `qwen3_5_text` 在 Blackwell 上仍自动回落 CUTLASS；target 日志为 `Selected CutlassFp8BlockScaledMMKernel` |

---

## GPU 显存分配

| 项目 | 大小 |
|------|------|
| FP8 模型权重 | 28.08 GiB（vLLM 日志 `Model loading took`） |
| 运行时总显存 | ~86.2-86.7 / 97.9 GB（RTX PRO 6000，MTP k=3 baseline 参数） |
| KV Cache 可用 | 54.05 GiB |
| GPU KV cache size | 947,541 tokens |

> L40S 历史 baseline 稳态约 ~39-40 / 46 GB；RTX PRO 6000 使用同一 `gpu-memory-utilization=0.90` 后会分配更大的 KV cache，因此旧 `<46GB` 验收口径不再适用于当前硬件。

---

## 各阶段配置对照表

| 参数 | Baseline (=Stage 0) | Stage A (DFlash) | Stage B (DSparkLite) | Stage C (DeepSpec) |
|------|:---:|:---:|:---:|:---:|
| 投机方法 | MTP k=3 ✅ 当前部署 | DFlash k=3（不适用，已回退） | DSparkLite k=7（不适用） | 训练 DSpark drafter |
| `--gpu-memory-utilization` | 0.90 | 0.90 | 0.90 | — |
| `--max-num-batched-tokens` | 8192 | 8192 | 8192 | — |
| `--enable-flashinfer-autotune` | 默认已开 | 默认已开 | 默认已开 | — |
| `--prefix-caching-hash-algo` | sha256 | sha256 | sha256 | — |
| DeepGEMM | 默认决策（v0.25.1 可启动，target 回落 CUTLASS） | 默认决策 | 默认决策 | — |
| Drafter 模型 | — | z-lab/Qwen3.6-27B-DFlash (3.3GB) | DSparkLiteProposer | DeepSpec 训练产出 |

> **Stage 状态**：Stage 0 ✅ 默认参数最优。Stage A ✅ 已验证 DFlash 不适用（全面劣于 MTP，最佳 k=3 仅 0.71×），回退 MTP k=3。Stage B ❌ 依赖 DFlash drafter + `disable_padded_drafter_batch` 触发 `NotImplementedError`，不适用。Stage C ❌ DeepSpec 不支持 qwen3_5 架构 + 27B BF16 显存超限 + target cache ~76TB 超存储 + Python 3.9 不兼容，当前硬件不可行。
> Stage 0 验证结论：可调参数（gpu-util 0.92、batched-tokens 16384、partial-prefills、xxhash）经测试均无收益或不支持，默认值即为最优。详见 [`performance.md#stage-0`](performance.md) 与 [`performance.md#stage-a`](performance.md)。
> RTX PRO 6000 + vLLM 0.25.1 复测结论：MTP k=4/k=5 均会在并发 5 烟测后触发 `cudaErrorIllegalAddress` 并重启服务，仍不可用；当前部署继续固定 MTP k=3。

---

## Benchmark 工具

- **工具**: `benchmark.py` 原生 streaming client（httpx；计时口径与 `measure_latency.py` 同源）
- **模式**: streaming (OpenAI 兼容 API)
- **命令**: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
- **报告路径**: `docs/benchmark-<label>.md` + `docs/benchmark-<label>.json`

---

## 当前部署决策（vLLM 0.25.1）

- 当前运行 `vllm/vllm-openai:v0.25.1`，不设置 `VLLM_USE_DEEP_GEMM=0`。
- v0.25.1 默认决策可在 RTX PRO 6000 Blackwell 上正常启动；但 `qwen3_5_text` target FP8 linear 仍自动回落 CUTLASS，不是 DeepGEMM target 推理路径。
- MTP `num_speculative_tokens` 固定为 3。k=4/k=5 在 RTX PRO 6000 + v0.25.1 上短请求可通过，但并发 5 烟测后 EngineCore 触发 `cudaErrorIllegalAddress`，不可作为服务配置。
- 若回退 v0.24.0，RTX PRO 6000 上需要恢复 `VLLM_USE_DEEP_GEMM=0`，否则默认启动会在 DeepGEMM warmup 阶段触发 `Unknown recipe`。
- 两版本性能差异记录在 [`performance.md#vllm-0251--deepgemm-默认决策复测`](performance.md#vllm-0251--deepgemm-默认决策复测)。

---

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-07-08 | baseline | MTP k=3 基线配置文档 |
| 2026-07-08 | stage0 | 参数验证：默认最优，Stage 0 = Baseline（详见 performance.md） |
| 2026-07-08 | stageA | DFlash 验证：k=7/4/3 全面劣于 MTP（最佳 k=3 仅 0.71×），不适用，回退 MTP k=3 |
| 2026-07-23 | rtx-pro-6000 | 当前 GPU 更新为 RTX PRO 6000 Blackwell；新增 `VLLM_USE_DEEP_GEMM=0` 规避 DeepGEMM warmup 启动失败 |
| 2026-07-23 | vllm-0.25.1 | 升级到 `vllm/vllm-openai:v0.25.1`；移除 `VLLM_USE_DEEP_GEMM=0`，默认 DeepGEMM 决策可启动但 target 自动回落 CUTLASS |
| 2026-07-23 | mtp-k4-k5-rtx-pro | RTX PRO 6000 + vLLM 0.25.1 复测 MTP k=4/k=5：短请求可用但并发 5 后 `cudaErrorIllegalAddress`，恢复 k=3 |
