# 性能追踪报告

> 所有 benchmark 均在 NVIDIA L40S 46GB + Qwen3.6-27B-FP8 + vLLM 0.24.0 上运行。
> 测试命令: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
> 每并发级别 10 requests, max_tokens=2048, streaming mode.

## 性能总览

| 阶段 | 日期 | 投机方法 | 参数 | output_tps (c=1) | output_tps (c=5) | output_tps (c=10) | TPOT mean (c=10) | vs 基线 |
|------|------|---------|------|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 待测试 | MTP k=3 | 默认参数 | — | — | — | — | — |
| Stage 0 | — | MTP k=3 | 优化参数 | — | — | — | — | — |
| Stage A | — | DFlash k=7 | 优化参数 | — | — | — | — | — |
| Stage B | — | DSparkLite k=7 | 优化参数 | — | — | — | — | — |
| Stage C | — | DSpark Trained k=7 | 优化参数 | — | — | — | — | — |

---

## Baseline: MTP k=3 (当前配置)

**配置**: `gpu-memory-utilization=0.90`, `max-num-batched-tokens=8192`, MTP `num_speculative_tokens=3`
**GPU 显存**: ~40/46 GB
**报告**: 待测试

---

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-07-08 | v1.0 | 初始基线报告，MTP k=3 |
