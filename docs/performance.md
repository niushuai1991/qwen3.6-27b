# 性能追踪报告

> 所有 benchmark 均在 NVIDIA L40S 46GB + Qwen3.6-27B-FP8 + vLLM 0.24.0 上运行。
> 测试命令: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
> 每并发级别 10 requests, max_tokens=2048, streaming mode.

## 性能总览

| 阶段 | 日期 | 投机方法 | 参数 | output_tps (c=1) | output_tps (c=5) | output_tps (c=10) | TPOT mean (c=10) | vs 基线 |
|------|------|---------|------|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 2026-07-08 | MTP k=3 | 默认参数 | 36.3 | 170.0 | 309.8 | 6.4ms | — |
| Stage 0 | 2026-07-08 | MTP k=3 | 默认(已验证最优) | 36.8 | 169.2 | 309.4 | 3.9ms | ≈基线 |
| Stage A | 2026-07-08 | DFlash k=3（最佳） | baseline 参数 | 35.2 | 162.3 | 219.9 | 3.7ms | 0.71× ✗ 不适用→回退 |
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

---

## Stage A — DFlash 验证（结论：不适用，回退 MTP）

**目标**：MTP k=3 → DFlash，目标 output_tps ≥ MTP × 1.30。
**结论：DFlash 在当前硬件/配置下全面劣于 MTP，未达标，回退 MTP k=3 baseline。**

### 配置变更

`--speculative-config` 从 MTP k=3 改为 DFlash（drafter `/models/Qwen3.6-27B-DFlash` = `z-lab/Qwen3.6-27B-DFlash`，3.3GB，5 层 qwen3，`block_size=16`，vLLM 0.24 原生 `method:"dflash"`，FlashInfer 支持 non-causal）。其余参数沿用 baseline（8192/0.90/无 xxhash）。

### 测试数据

| 配置 | c=1 | c=5 | c=10 | TPOT(c=10) | acceptance | vs MTP c=10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| MTP k=3（baseline） | 36.8 | 169.2 | 309.4 | 3.9ms | — | — |
| DFlash k=7 | 40.5 | 129.3 | 127.5 | — | ~2.7 | 0.41× |
| DFlash k=4 | — | — | 174.6* | — | ~2.4 | 0.56× |
| DFlash k=3（最佳） | 35.2 | 162.3 | 219.9 | 3.7ms | ~2.4 | 0.71× |

\* 快速验证（5 reqs）。报告：[`dflash_k7`](benchmark-dflash_k7.md) · [`dflash_k3`](benchmark-dflash_k3.md) · [`dflash_k3_quick`](benchmark-dflash_k3_quick.md) · [`dflash_k4_quick`](benchmark-dflash_k4_quick.md)

### 根因分析（结构性，非抢占）

1. **独立 drafter 开销（主因）**：DFlash 需 3.3GB 独立 drafter（5 层 Qwen3），每个 decode step 额外一次 forward；MTP 用 target 内置 head，零 drafter 开销。高并发时 drafter forward 被 batch 放大，成为串行瓶颈。
2. **acceptance 无优势**：DFlash 实测 acceptance ~2.4-2.7（c=1 smoke 的 5.42 为单次峰值异常，中位数 2.66），与 MTP k=3 相当。无 acceptance 收益却有 drafter 开销 → 净负。
3. **KV cache 压力**：DFlash drafter + spec tokens 占用 KV cache，c=10 usage 92-96%，限制实际并发到 7-8（`max-num-seqs=10` 跑不满）。k=7 KV cache 61,224 tokens vs k=3 79,488。
4. **k 值 trade-off**：c=1 大 k 略优（acceptance 高），c=10 小 k 优（verify batch 省）。无单一 k 全场景达标。
5. **非 preemption**：所有配置 0 抢占重算。
6. `disable_padded_drafter_batch`：`NotImplementedError`（draft models only support padded），不可用。

### 为何 DFlash 理论强但此处失效

DFlash 论文（2-3× over Eagle-3）优势在大模型 + 充足算力，drafter forward 能与 target overlap。本配置：27B FP8 在单 L40S，decode 时 GPU 已被 target 占满，drafter forward 无法 overlap 变串行开销；且 qwen3_5 混合架构下 drafter acceptance 仅 ~2.5。

### 决策

**回退 MTP k=3 baseline**（Stage 0 已验证最优）。DFlash drafter 模型保留在 `/data/models/Qwen3.6-27B-DFlash`（3.3GB），供未来换硬件（多卡/更强 GPU）复用。Stage B（DSparkLite custom_class）依赖 DFlash drafter，同样不适用；Stage C（DeepSpec 训练）受架构/显存/存储多重阻塞，当前硬件不可行。DSpark 路线在单 L40S + 27B 配置下暂止于 MTP baseline。
