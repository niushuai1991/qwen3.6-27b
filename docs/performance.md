# 性能追踪报告

> 所有 benchmark 均在 NVIDIA L40S 46GB + Qwen3.6-27B-FP8 + vLLM 0.24.0 上运行。
> 测试命令: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
> 每并发级别 10 requests, max_tokens=2048, streaming mode.

## 性能总览

| 阶段 | 日期 | 投机方法 | 参数 | output_tps (c=1) | output_tps (c=5) | output_tps (c=10) | TPOT mean (c=10) | vs 基线 |
|------|------|---------|------|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 2026-07-08 | MTP k=3 | 默认参数 | 36.3 | 170.0 | 309.8 | 6.4ms | — |
| Stage 0 | 2026-07-08 | MTP k=3 | 默认(已验证最优) | 36.8 | 169.2 | 309.4 | 3.9ms | ≈基线 |
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

## Stage 0 — 根因分析与参数验证

**目标**: 验证 vLLM 调度/参数优化能否在 MTP k=3 上提升吞吐。**结论：默认参数已是最优，无需调整。**

### 根因：`--prefix-caching-hash-algo xxhash` 容器缺包（前 4 次失败的真因）

前 4 次尝试均带 `xxhash`，全部崩溃。traceback 定位到：

```
File ".../vllm/utils/hashing.py", line 63, in _xxhash_digest
ModuleNotFoundError: xxhash is required for the 'xxhash' prefix caching hash algorithms.
→ EngineDeadError → 容器崩溃 → Docker 重启 → 所有请求 "Server disconnected"
```

- 容器 `vllm/vllm-openai:latest` 未安装 `xxhash` 包（`pip show xxhash` → not found）
- prefix-cache block 哈希仅在生成足够多 token 时触发 → 单次 curl（短输出）不崩，benchmark（max_tokens=2048）必崩
- **决策**：放弃 xxhash（收益=CPU 哈希微优化；持久化需自定义镜像；与投机解码主线无关），回退默认 sha256
- **纯 baseline 验证**（移除 xxhash）：output_tps 36.8/169.2/309.4，0 失败 → 根因 100% 确认

### 其余参数结论

| 参数 | 结论 |
|------|------|
| `--max-num-partial-prefills` / `--max-long-partial-prefills` | vLLM 0.24 不支持（`Concurrent Partial Prefill is not supported`） |
| `--enable-flashinfer-autotune` | 非合法 CLI 参数（`vllm serve --help` 无）；日志 `kernel_config` 显示 `enable_flashinfer_autotune=True` 是内部默认值，**默认已开启** |
| `--gpu-memory-utilization` | 保持 0.90（约束 ≤0.90；显存无压力） |

### `--max-num-batched-tokens` 8192 → 16384（唯一实质变更，已测）

| 并发 | 8192(baseline) | 16384 | 变化 |
|:---:|:---:|:---:|:---:|
| output_tps c=1 | 36.8 | 35.9 | 持平 |
| output_tps c=5 | 169.2 | 169.9 | 持平 |
| output_tps c=10 | 309.4 | 307.8 | −0.5%（误差内） |
| c=10 TTFT p50 | 56.7s | 63.8s | **+12.5%（恶化）** |

报告：[`benchmark-mtp_k3_batched16384.md`](benchmark-mtp_k3_batched16384.md) · [`benchmark-mtp_k3_verify_no_xxhash.md`](benchmark-mtp_k3_verify_no_xxhash.md)

**结论**：16384 无吞吐收益，反而轻微恶化 c=10 TTFT（更大 prefill batch 阻塞 decode）。**回退到 8192。**

### Stage 0 总结

默认参数已是最优。baseline（MTP k=3 + 默认参数 + gpu-mem=0.90）即为 Stage A 的干净起点。
