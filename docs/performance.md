# 性能追踪报告

> 所有 benchmark 均在 NVIDIA L40S 46GB + Qwen3.6-27B-FP8 + vLLM 0.24.0 上运行。
> 测试命令: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
> 每并发级别 10 requests, max_tokens=2048, streaming mode.

## 性能总览

| 阶段 | 日期 | 投机方法 | 参数 | output_tps (c=1) | output_tps (c=5) | output_tps (c=10) | TPOT mean (c=10) | vs 基线 |
|------|------|---------|------|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 2026-07-08 | MTP k=3 | 默认参数 | 36.3 | 170.0 | 309.8 | 6.4ms | — |
| Stage 0 | — | MTP k=3 | 优化参数 | — | — | — | — | 失败(见下方) |
| Stage A | — | DFlash k=7 | 优化参数 | — | — | — | — | — |
| Stage B | — | DSparkLite k=7 | 优化参数 | — | — | — | — | — |
| Stage C | — | DSpark Trained k=7 | 优化参数 | — | — | — | — | — |

---

## Baseline: MTP k=3 (当前配置)

**配置**: `gpu-memory-utilization=0.90`, `max-num-batched-tokens=8192`, MTP `num_speculative_tokens=3`
**GPU 显存**: ~40/46 GB
**报告**: [`docs/benchmark-mtp_k3_baseline.md`](benchmark-mtp_k3_baseline.md)
**日期**: 2026-07-08

### 关键指标

| 并发 | output_tps | TTLT p50 | TTLT p90 | TTFT p50 | TPOT mean | 失败 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 36.3 | 56.3s | 59.2s | N/A | N/A | 0 |
| 5 | 170.0 | 59.6s | 63.6s | 54.8s | 1.1ms | 0 |
| 10 | 309.8 | 65.1s | 69.1s | 52.4s | 6.4ms | 0 |

### 分析

- **单流吞吐**: ~36.3 tok/s — 受限于 2048 token 长输出
- **5 并发**: ~170.0 tok/s — 聚合吞吐约单流的 4.7x
- **10 并发**: ~309.8 tok/s — 聚合吞吐约单流的 8.5x，TPOT 6.4ms 表明 decode 高效
- **TTFT**: 52.4s (c=10) — 长 prompt prefill 排队，瓶颈在 `max-num-batched-tokens=8192`
- **所有并发 0 失败**

---

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-07-08 | v1.0 | 初始基线报告，MTP k=3 |

---

## Stage 0 尝试记录

**目标**: 在 MTP k=3 上应用 vLLM 参数优化（`max-num-batched-tokens=16384`, `enable-flashinfer-autotune`, `xxhash`, `gpu-memory-utilization=0.92`）

### 尝试的参数组合（全部失败）

| 尝试 | `batched-tokens` | `gpu-util` | `flashinfer-autotune` | `xxhash` | 结果 |
|:---:|:---:|:---:|:---:|:---:|------|
| 1 | 16384 | 0.92 | ✓ | ✓ | Server 崩溃（c=10 时） |
| 2 | 12288 | 0.92 | ✓ | ✓ | Server 崩溃（c=10 时） |
| 3 | 8192 | 0.92 | ✓ | ✓ | Server 崩溃（c=10 时） |
| 4 | 8192 | 0.92 | ✗ | ✓ | Server 崩溃（c=10 时） |

### 不支持的参数
- `--max-num-partial-prefills 2` → vLLM 0.24 报错: `Concurrent Partial Prefill is not supported`
- `--max-long-partial-prefills 2` → 同上

### 现象
- 服务启动正常，health check 返回 200，单次 `curl` 请求成功
- benchmark 运行时所有请求失败（包括 c=1 的 10 个请求）
- Server 在 benchmark 期间自动重启

### 下一步建议
- 逐个测试参数变化，先只改 `gpu-memory-utilization 0.90→0.92`，跑 benchmark 确认稳定
- 再逐个加 `xxhash`、`max-num-batched-tokens`、`flashinfer-autotune`
- 或者直接跳到 Stage A（DFlash），保持当前稳定的 MTP 参数不变
