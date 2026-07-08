# Qwen3.6-27B-FP8 部署配置文档

> 记录所有部署参数、技术栈版本、以及各阶段配置变更对照。

## 环境信息

| 项目 | 值 |
|------|-----|
| **GPU** | NVIDIA L40S 46GB VRAM |
| **系统内存** | 57GB |
| **模型** | Qwen/Qwen3.6-27B-FP8（29GB） |
| **模型路径** | `/data/models/Qwen3.6-27B-FP8` |
| **vLLM 版本** | 0.24.0（容器 `vllm/vllm-openai:latest`） |
| **Docker 运行时** | nvidia-container-runtime |

---

## 当前配置（Baseline: MTP k=3）

```yaml
# docker-compose.yml — 当前运行版本
command: >
  /models/Qwen3.6-27B-FP8
  --served-model-name qwen3.6-27b
  --host 0.0.0.0
  --port 8000
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

---

## GPU 显存分配

| 项目 | 大小 |
|------|------|
| FP8 模型权重 | ~27 GB |
| 框架开销 + MTP | ~3 GB |
| 运行时总显存 | ~39-40 GB |
| KV Cache 可用 | ~7 GB |

---

## 各阶段配置对照表

| 参数 | Baseline (当前) | Stage 0 (优化后) | Stage A (DFlash) | Stage B (DSparkLite) |
|------|:---:|:---:|:---:|:---:|
| 投机方法 | MTP k=3 | MTP k=3 | DFlash k=7 | DSparkLite k=7 |
| `--gpu-memory-utilization` | 0.90 | **0.92** | 0.92 | 0.92 |
| `--max-num-batched-tokens` | 8192 | **16384** | 16384 | 16384 |
| `--max-num-partial-prefills` | — (1) | **2** | 2 | 2 |
| `--max-long-partial-prefills` | — (1) | **2** | 2 | 2 |
| `--enable-flashinfer-autotune` | disabled | **enabled** | enabled | enabled |
| `--prefix-caching-hash-algo` | sha256 | **xxhash** | xxhash | xxhash |
| Drafter 模型 | — | — | z-lab/DFlash (~1.5GB) | DSparkLiteProposer |

---

## Benchmark 工具

- **工具**: [awslabs/llmeter](https://github.com/awslabs/llmeter) v0.1.12
- **模式**: streaming (OpenAI 兼容 API)
- **命令**: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
- **报告路径**: `docs/benchmark-<label>.md` + `docs/benchmark-<label>.json`

---

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-07-08 | baseline | MTP k=3 基线配置文档 |
