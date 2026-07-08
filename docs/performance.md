# 性能追踪报告

> 所有 benchmark 均在 NVIDIA L40S 46GB + Qwen3.6-27B-FP8 + vLLM 0.24.0 上运行。
> 测试命令: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
> 每并发级别 10 requests, max_tokens=2048, streaming mode.

## 性能总览

| 阶段 | 日期 | 投机方法 | 参数 | output_tps (c=1) | output_tps (c=5) | output_tps (c=10) | TPOT mean (c=10) | vs 基线 |
|------|------|---------|------|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 2026-07-08 | MTP k=3 | 默认参数 | 36.3 | 170.0 | 309.8 | 6.4ms | — |
| Stage 0 | — | MTP k=3 | 优化参数 | — | — | — | — | — |
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
