# MTP k≥4 崩溃修复验证：FLASH_ATTN 后端（2026-07-23）

**目标**：AGENTS.md「下一步」#1——解决 MTP k≥4 在并发 ≥5 必触发 `cudaErrorIllegalAddress` 的崩溃。待验证假设：根因在 FlashInfer + spec-decode 强制 **PIECEWISE cudagraph**（k≥4 越界），切 `--attention-backend FLASH_ATTN`（支持 **FULL cudagraph**）绕过。

**结论**：假设**机制成立、崩溃已修复**（k=4/k=5 × c=1/5/10 全程 0 失败、容器零重启）。但附带两个决定性发现：① FLASH_ATTN **拒绝 fp8 KV cache**，k≥4 必须牺牲 fp8 KV（改 bf16）；② k≥4 仍**不赢 k=3**（acceptance 随深度递减）。**生产配置维持 FlashInfer + fp8 KV + k=3 不变。**

## 1. 关键发现（按重要性）

### 1.1 崩溃根因机制——确认

FlashInfer 在 spec-decode 下只支持 `UNIFORM_SINGLE_TOKEN_DECODE` → 自动降级 **PIECEWISE** cudagraph；k≥4（每序列 ≥5 候选 token）在并发 batch 下 PIECEWISE graph shape 与实际 spec-decode batch 不匹配，replay 越界 → `cudaErrorIllegalAddress`。

切 FLASH_ATTN 后，启动日志明确捕获了 decode 的 **FULL cudagraph**（之前 FlashInfer 只有 PIECEWISE）：

```
Profiling CUDA graph memory: PIECEWISE=12, FULL=7   (k=4)
Profiling CUDA graph memory: PIECEWISE=16, FULL=8   (k=5)
Capturing CUDA graphs (decode, FULL): 7/7 [done]    ← 关键：decode 拿到 FULL
```

→ FULL cudagraph 覆盖完整 spec-decode batch，无 PIECEWISE shape 越界。k=4、k=5 均稳定。验证：完整 benchmark c=1/5/10 **0 失败**，`RestartCount=0`（容器从未重启），全程无 `cudaErrorIllegalAddress`。

### 1.2 硬约束——FLASH_ATTN 拒 fp8 KV cache（新发现）

直接在现有配置（`--kv-cache-dtype fp8_e4m3`）上加 `--attention-backend FLASH_ATTN`，启动即崩：

```
ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid
for this configuration. Reason: ['kv_cache_dtype not supported']
  (vllm/platforms/cuda.py:411 get_attn_backend_cls)
```

→ **要用 FLASH_ATTN（即要用 k≥4）必须放弃 fp8 KV cache**，改默认 bf16。代价：KV cache 占用带宽翻倍、可用 KV block 数减半（本项目 96GB 下容量仍充足，崩溃时 KV usage 仅 ~8%）。

### 1.3 k≥4 仍不赢 k=3——acceptance 随深度递减

同配置（FLASH_ATTN+bf16）下横向比 k，acceptance 单调下降：**k=3: 0.54 → k=4: 0.45 → k=5: 0.39**。每加深一档，marginal 候选位置接受的 token 越来越少，却要多跑一次 MTP draft forward，得不偿失。印证旧结论：**该模型 MTP 的 acceptance 上限就在 k=3**，加深 k 无收益。

## 2. 完整结果矩阵

| 配置 | backend | KV | k | c=1 | c=5 | c=10 | accept(c=10) | 失败 | 状态 |
|---|---|---|---|---|---|---|---|---|---|
| **A（prod baseline）** | FlashInfer | fp8 | **3** | 81.5 | 357.5 | 653.7 | 0.55 | 0 | ✅ 稳定 |
| C（新配置） | FLASH_ATTN | bf16 | **3** | 68.6 | 367.7 | 672.9 | 0.54 | 0 | ✅ 稳定 |
| B（新配置） | FLASH_ATTN | bf16 | **4** | 69.3 | 337.6 | 665.7 | 0.46 | 0 | ✅ **修复** |
| D（新配置） | FLASH_ATTN | bf16 | **5** | 70.1 | 348.8 | 620.0 | 0.39 | 0 | ✅ **修复** |
| （历史） | FlashInfer | fp8 | 4 | — | — | — | — | 全失败 | ❌ cudaErrorIllegalAddress |
| （历史） | FlashInfer | fp8 | 5 | — | — | — | — | 全失败 | ❌ cudaErrorIllegalAddress |

> output_tps 单位 tok/s；A 为既有 baseline（v0.25.1/RTX PRO 6000，见 `performance.md`）。B/C/D 为本次实测（v0.25.1/RTX PRO 6000）。

**两个干净归因**（控制变量）：

- **C vs A**（同 k=3，只换 backend + KV dtype）→ 配置本身的影响，与 k 无关：
  - c=1：68.6 vs 81.5 = **0.84×**（单流退步 16%；fp8 KV + FlashInfer 单流更快）
  - c=5/10：367.7/672.9 vs 357.5/653.7 = **+3%**（高并发 FLASH_ATTN+bf16 略优）
  - acceptance 完全一致（0.54 vs 0.54）→ backend/KV 不影响投机命中率
- **C/B/D**（同新配置，k=3/4/5）→ k 本身的影响：
  - c=10：C(672.9) > B(665.7) > D(620.0)；c=5：C(367.7) > D(348.8) > B(337.6)
  - **k=3 在新配置下仍最优**；k=4/k=5 不赢 k=3

## 3. per-position acceptance（c=10，FLASH_ATTN+bf16）

| 位置 | k=3 | k=4 | k=5 |
|---|---|---|---|
| pos0 | 5751 | 5292 | 4980 |
| pos1 | 4072 | 3714 | 3427 |
| pos2 | 2794 | 2530 | 2346 |
| pos3 | — | 1758 | 1623 |
| pos4 | — | — | （未解析） |

每个后续位置接受的 token 约为前一位置的 ~68%，几何衰减。k=4 的 pos3、k=5 的 pos3/4 贡献的 accepted token 很少，却各要一次 draft forward——这是「加深 k 无收益」的直接证据。

> 注：`benchmark.py` 的 per-pos 解析硬编码 `range(4)`（k=3 口径），故 k=5 的 pos4 未采集；总体 `acceptance_rate` 由 accepted/draft 总量计算，仍准确。原始数据见 `benchmark-flashattn_k4_full.json` / `_k3.json` / `_k5.json`。

## 4. 决策与建议

- **生产配置不变**：维持 **FlashInfer + fp8 KV + k=3**。它是单流最优（c=1: 81.5，领先所有新配置 16%+），高并发也与新配置持平；且保留 fp8 KV 的容量/带宽优势。
- **崩溃修复的价值**：k≥4 现在**可运行**了。这是能力解锁，非当前吞吐收益。若未来 acceptance 因更强 drafter（如自训 DSpark）量级跃迁，k≥4 才有发挥空间——前提仍是多卡/disaggregation 给 drafter overlap 空隙（见 AGENTS.md「下一步」#3）。
- **若必须跑 k≥4**（实验性）：用 `--attention-backend FLASH_ATTN`（去掉 `--kv-cache-dtype fp8_e4m3`），k=4 是新配置下的次优、k=5 最差，均不如 k=3。

## 5. 复现步骤

```yaml
# docker-compose.yml 片段（实验性 k=4）
--attention-backend FLASH_ATTN          # 关键：换后端
# （移除 --kv-cache-dtype fp8_e4m3）    # 关键：FLASH_ATTN 不支持 fp8 KV
--speculative-config '{"method":"mtp","num_speculative_tokens":4}'
```

```bash
docker compose down && docker compose up -d
# 等 ~3min（torch.compile + cudagraph capture）
uv run python benchmark.py --label flashattn_k4_full --concurrency "1,5,10"
```

诊断要点：启动日志应见 `Capturing CUDA graphs (decode, FULL): N/N`（FULL=可跑 k≥4）；若只见 PIECEWISE 则仍会崩。

---
*本次实验：RTX PRO 6000 Blackwell 96GB / vLLM 0.25.1 / Qwen3.6-27B-FP8。原始报告：[`benchmark-flashattn_k4_full.md`](benchmark-flashattn_k4_full.md) / [`benchmark-flashattn_k3.md`](benchmark-flashattn_k3.md) / [`benchmark-flashattn_k5.md`](benchmark-flashattn_k5.md)。*
