# 性能追踪报告

> 除另行标注外，历史 benchmark 均在 NVIDIA L40S 46GB + Qwen3.6-27B-FP8 + vLLM 0.24.0 上运行；2026-07-23 起新增 RTX PRO 6000 Blackwell 复测结果。
> 测试命令: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
> 每并发级别 10 requests, max_tokens=2048, streaming mode.
> **2026-07-20 起 `benchmark.py` 改用原生 streaming 计时**（移除 llmeter 依赖）：主指标改为 decode 阶段吞吐 `decode tok/s = (content_tokens−1)/(TTLT−TTFT)`（不含 prefill），与 `measure_latency.py` 同源；保留 `Output TPS`（含 TTFT 的整体聚合）对齐历史 llmeter 口径。历史 llmeter 报告保留作对比。

## 性能总览

| 阶段 | 日期 | 投机方法 | 参数 | output_tps (c=1) | output_tps (c=5) | output_tps (c=10) | TTFT p50 (c=10) | TPOT mean (c=10) | vs 基线 |
|------|------|---------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline** | 2026-07-09 | MTP k=3 | 默认参数 | 37.8 | 172.1 | 317.0 | 0.32s | 31.0ms | — |
| **RTX PRO 6000 复测** | 2026-07-23 | MTP k=3 | Baseline + `VLLM_USE_DEEP_GEMM=0` | 81.5 | 357.5 | 653.7 | 0.347s | 13.9ms | 2.06× vs L40S c=10 |
| **RTX PRO 6000 / vLLM 0.25.1** | 2026-07-23 | MTP k=3 | DeepGEMM 默认决策（target 自动回落 CUTLASS） | 80.1 | 345.5 | 644.7 | 0.336s | 14.1ms | 0.99× vs v0.24 RTX c=10 |
| **RTX PRO 6000 / vLLM 0.25.1 k=4 复测** | 2026-07-23 | MTP k=4 | 默认 cudagraph | 短请求 OK | 烟测后崩溃 | — | — | — | ✗ `cudaErrorIllegalAddress`→回退 k=3 |
| **RTX PRO 6000 / vLLM 0.25.1 k=5 复测** | 2026-07-23 | MTP k=5 | 默认 cudagraph | 短请求 OK | 烟测后崩溃 | — | — | — | ✗ `cudaErrorIllegalAddress`→回退 k=3 |
| **RTX PRO 6000 / FLASH_ATTN k=3** | 2026-07-23 | MTP k=3 | `--attention-backend FLASH_ATTN`（bf16 KV，去 fp8） | 68.6 | 367.7 | 672.9 | — | 13.9ms | 配置归因：c=1 0.84× / c=10 +3%（vs fp8 baseline）accept 0.54 |
| **RTX PRO 6000 / FLASH_ATTN k=4** | 2026-07-23 | MTP k=4 | `--attention-backend FLASH_ATTN`（bf16 KV） | 69.3 | 337.6 | 665.7 | — | 13.9ms | ✅ **崩溃已修复**（0 失败）；accept 0.46；仍不赢 k=3 |
| **RTX PRO 6000 / FLASH_ATTN k=5** | 2026-07-23 | MTP k=5 | `--attention-backend FLASH_ATTN`（bf16 KV） | 70.1 | 348.8 | 620.0 | — | 14.6ms | ✅ **崩溃已修复**（0 失败）；accept 0.39；仍不赢 k=3 |
| **RTX PRO 6000 / vLLM 0.24.0 DFlash k=7** | 2026-07-23 | DFlash k=7 | 0.24.0 + `VLLM_USE_DEEP_GEMM=0`（绕过 0.25.1 NotImplementedError） | 76.0 | 295.9 | 499.2 | 0.328s | 17.8ms | 0.76× ✗ acceptance 0.20→回退 MTP |
| **RTX PRO 6000 / vLLM 0.24.0 DFlash k=15** | 2026-07-23 | DFlash k=15 | README 推荐满配（block_size=16） | 72.0 | 256.5 | 400.7 | 0.344s | 22.0ms | 0.61× ✗ acceptance 0.10→回退 MTP |
| ~~Baseline（thinking 开启）~~ | 2026-07-08 | MTP k=3 | 默认+thinking | 36.3 | 170.0 | 309.8 | 52.4s* | 6.4ms* | 参考 |
| Stage 0 | 2026-07-08 | MTP k=3 | 默认(已验证最优) | 36.8 | 169.2 | 309.4 | 56.7s* | 3.9ms* | ≈基线 |
| Stage A | 2026-07-08 | DFlash k=3（最佳） | baseline 参数 | 35.2 | 162.3 | 219.9 | N/A* | 3.7ms* | 0.71× ✗ 不适用→回退 |
| k=4 验证 | 2026-07-09 | MTP k=4 | 默认参数 | 36.8 | 崩溃 | 崩溃 | — | — | ✗ cudagraph 越界→回退 k=3 |
| k=5 验证 | 2026-07-09 | MTP k=5 | max-num-seqs=5 | 可跑* | 崩溃 | — | — | — | ✗ 同 k=4 根因→回退 k=3 |
| no_mtp | 2026-07-09 | 无投机（纯 target） | 移除 speculative-config | 16.9 | 97.1 | 182.7 | N/A* | 3.1ms* | 0.59× ✗ 全面劣于 MTP→回退 k=3 |
| 容器优化 | 2026-07-09 | MTP k=3 | host 网络 + ipc host | 36.6 | 171.8 | 310.1 | N/A* | N/A* | ≈基线（+0.1~1.1% 噪声内）✗ 裸机无意义 |
| Stage B | — | DSparkLite k=7 | 优化参数 | — | — | — | — | — | — |
| Stage C | — | DSpark Trained k=7 | 优化参数 | — | — | — | — | — | — |

> \* thinking 开启时的数据，TTFT/TPOT 被 thinking tokens 污染（首个 `content` token 需等 thinking 结束，详见 [`disable-thinking.md`](disable-thinking.md)）。**2026-07-09 起 benchmark 默认关闭 thinking，TTFT/TPOT 已修正。**

---

## RTX PRO 6000 Blackwell 复测（2026-07-23）

**配置**: MTP k=3 baseline 参数不变；RTX PRO 6000 Blackwell 上显式设置 `VLLM_USE_DEEP_GEMM=0`，规避 vLLM 0.24.0 DeepGEMM warmup `Unknown recipe` 启动失败。
**GPU 显存**: 稳态 ~86.2/97.9 GB；benchmark 采样峰值 ~86.7/97.9 GB（`gpu-memory-utilization=0.90` 会分配大 KV cache，旧 L40S `<46GB` 验收口径不再适用）。
**报告**: [`docs/benchmark-rtx_pro_6000_mtp_k3.md`](benchmark-rtx_pro_6000_mtp_k3.md)

| 并发 | output_tps | TTFT p50 | TTFT p90 | TPOT mean | decode tok/s | Acceptance | 失败 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 81.5 | 0.103s | 0.125s | 12.2ms | 82.2 | 0.55 | 0 |
| 5 | 357.5 | 0.887s | 2.090s | 13.1ms | 76.6 | 0.54 | 0 |
| 10 | 653.7 | 0.347s | 0.348s | 13.9ms | 72.3 | 0.55 | 0 |

### 与 L40S baseline 对比

| 并发 | L40S baseline | RTX PRO 6000 | 提升 |
|:---:|:---:|:---:|:---:|
| 1 | 37.8 | 81.5 | 2.16× |
| 5 | 172.1 | 357.5 | 2.08× |
| 10 | 317.0 | 653.7 | 2.06× |

结论：在同一 MTP k=3 / max_tokens=2048 / streaming 口径下，RTX PRO 6000 将聚合 output_tps 提升到约 2.1×，acceptance 仍维持 ~0.54-0.55，说明收益主要来自 Blackwell GPU decode 算力与更大 KV cache 余量，而非 acceptance 改变。

### vLLM 0.25.1 + DeepGEMM 默认决策复测

**结论**: vLLM 0.25.1 不再需要 `VLLM_USE_DEEP_GEMM=0` 才能启动，v0.24.0 的 DeepGEMM warmup `Unknown recipe` 启动失败未复现；但日志仍显示 `Auto-disabled DeepGemm for model_type=qwen3_5_text on Blackwell`，target 线性层实际选择 `CutlassFp8BlockScaledMMKernel`，因此这不是 DeepGEMM target 推理性能。

报告：[`docs/benchmark-rtx_pro_6000_vllm0251_deepgemm_default.md`](benchmark-rtx_pro_6000_vllm0251_deepgemm_default.md)

| 并发 | v0.24.0 + DeepGEMM off | v0.25.1 默认决策 | 变化 |
|:---:|:---:|:---:|:---:|
| 1 | 81.5 | 80.1 | -1.7% |
| 5 | 357.5 | 345.5 | -3.4% |
| 10 | 653.7 | 644.7 | -1.4% |

v0.25.1 结果：output_tps 80.1 / 345.5 / 644.7，TPOT(c=10)=14.1ms，acceptance=0.54，0 失败。整体与 v0.24.0 RTX baseline 接近，略慢 1-3%。

#### v0.24.0 vs v0.25.1 差异明细

| 项目 | v0.24.0 on RTX PRO 6000 | v0.25.1 on RTX PRO 6000 |
|------|------|------|
| 镜像 | `vllm/vllm-openai:v0.24.0` | `vllm/vllm-openai:v0.25.1` |
| DeepGEMM 配置 | 需要显式 `VLLM_USE_DEEP_GEMM=0` 才能稳定启动 | 移除 `VLLM_USE_DEEP_GEMM=0`，使用默认决策即可启动 |
| 默认启动行为 | 默认路径会进入 `deep_gemm_warmup` 并触发 `Unknown recipe`，EngineCore 启动失败 | 未复现 `Unknown recipe`，服务正常启动并完成 benchmark |
| target FP8 kernel | 强制关闭 DeepGEMM 后使用 CUTLASS | 自动禁用 `qwen3_5_text` DeepGEMM 后使用 CUTLASS |
| 关键日志 | `VLLM_USE_DEEP_GEMM=0` 后无 DeepGEMM warmup crash | `Auto-disabled DeepGemm for model_type=qwen3_5_text on Blackwell` + `Selected CutlassFp8BlockScaledMMKernel` |
| DeepGEMM 实际状态 | 全局禁用；不是 DeepGEMM 性能 | DeepGEMM PDL 库可用，但 target 推理未走 DeepGEMM |
| output_tps c=1/5/10 | 81.5 / 357.5 / 653.7 | 80.1 / 345.5 / 644.7 |
| TPOT mean c=10 | 13.9ms | 14.1ms |
| Acceptance | 0.54-0.55 | 0.53-0.54 |
| 失败请求 | 0 | 0 |
| 结论 | 稳定但必须关 DeepGEMM | 当前推荐部署版本；启动更干净，但没有 DeepGEMM target 性能收益 |

当前决策：保留 v0.25.1 作为当前运行版本，保持 DeepGEMM 默认决策，不再手动设置 `VLLM_USE_DEEP_GEMM=0`。若未来 vLLM 对 `qwen3_5_text` + Blackwell 放开 DeepGEMM，需要重新确认日志中的 target kernel，再跑同口径 benchmark。

### MTP k=4/k=5 复测（RTX PRO 6000 + vLLM 0.25.1）

**目标**：新 GPU 与 vLLM 0.25.1 后，先验证 MTP k=4/k=5 是否可用；若可用再跑完整 benchmark。
**结论**：k=4/k=5 都不可用。二者短请求可通过，并发 5 烟测表面 0 失败，但随后 EngineCore 触发 `cudaErrorIllegalAddress` 并重启服务；因此不进入完整性能测试，不继续 DSpark 前置验证。

报告：
- k=4 烟测：[`docs/benchmark-rtx_pro_6000_vllm0251_mtp_k4_smoke.md`](benchmark-rtx_pro_6000_vllm0251_mtp_k4_smoke.md)
- k=5 烟测：[`docs/benchmark-rtx_pro_6000_vllm0251_mtp_k5_smoke.md`](benchmark-rtx_pro_6000_vllm0251_mtp_k5_smoke.md)

| 配置 | 验证路径 | 烟测结果 | 随后健康检查 | 根因日志 | 结论 |
|------|------|------|------|------|------|
| MTP k=4 | 短请求 + c=5 / 5 requests / 512 max_tokens | output_tps=98.8，acceptance=0.44，0 失败 | 失败；容器重启中 | `torch.AcceleratorError: CUDA error: an illegal memory access was encountered`；`num_running_reqs=5`，`total_num_scheduled_tokens=25` | 不可用 |
| MTP k=5 | 短请求 + c=5 / 5 requests / 512 max_tokens | output_tps=56.1，acceptance=0.39，0 失败 | 失败；容器重启中 | 同样 `cudaErrorIllegalAddress`；`num_running_reqs=5`，`total_num_scheduled_tokens=30` | 不可用 |

关键判断：

- 不是显存不足：崩溃时 KV cache usage 约 7.8%（k=4）/ 9.2%（k=5），RTX PRO 6000 稳态显存仍有余量。
- 不是 DeepGEMM target 路径：v0.25.1 日志仍显示 `qwen3_5_text` 在 Blackwell 上自动回落 CUTLASS。
- 失败路径与 L40S k≥4 现象一致：FlashInfer + spec decode + PIECEWISE cudagraph 下，attention metadata 构建阶段异步暴露 `cudaErrorIllegalAddress`。
- 当前部署继续使用 MTP k=3；k≥4 已列入「主动解决」（见 AGENTS.md「下一步」#1）：根因在 FlashInfer 强制 PIECEWISE cudagraph，待验证切 `--attention-backend FLASH_ATTN`（支持 FULL cudagraph，AEON 同款卡用此跑通）是否绕过 k≥4 越界 bug；`enforce-eager` / PR #46324 已证无效。

### MTP k≥4 崩溃修复：FLASH_ATTN 后端（2026-07-23）

**目标**：执行 AGENTS.md「下一步」#1——验证「切 `--attention-backend FLASH_ATTN`（FULL cudagraph）绕过 FlashInfer PIECEWISE 越界」假设。**结论：假设机制成立、崩溃已修复；但 k≥4 仍不赢 k=3，生产维持 baseline。** 综合分析见 [`benchmark-flashattn-k4-k5-analysis.md`](benchmark-flashattn-k4-k5-analysis.md)。

**修复机制（已确认）**：FLASH_ATTN 在 spec-decode 下捕获了 decode 的 **FULL cudagraph**（k=4: FULL=7/7，k=5: FULL=8/8），而 FlashInfer 只给 PIECEWISE（`UNIFORM_SINGLE_TOKEN_DECODE`）。FULL 覆盖完整 spec-decode batch，无 PIECEWISE shape 越界。k=4、k=5 × c=1/5/10 全程 **0 失败、容器零重启**。

**附带发现（硬约束）**：FLASH_ATTN **拒绝 fp8 KV cache**——直接加 `--attention-backend FLASH_ATTN` 到现有配置启动即 `ValueError: ... kv_cache_dtype not supported`（`vllm/platforms/cuda.py:411`）。→ **要用 k≥4 必须放弃 fp8 KV cache**（改 bf16，KV 容量/带宽成本翻倍）。

**完整结果**（output_tps tok/s，accept 取 c=10）：

| 配置 | backend | KV | k | c=1 | c=5 | c=10 | accept | 失败 |
|---|---|---|---|---|---|---|---|---|
| A（prod baseline） | FlashInfer | fp8 | 3 | 81.5 | 357.5 | 653.7 | 0.55 | 0 |
| C | FLASH_ATTN | bf16 | 3 | 68.6 | 367.7 | 672.9 | 0.54 | 0 |
| B | FLASH_ATTN | bf16 | 4 | 69.3 | 337.6 | 665.7 | 0.46 | 0 ✅修复 |
| D | FLASH_ATTN | bf16 | 5 | 70.1 | 348.8 | 620.0 | 0.39 | 0 ✅修复 |

**两个干净归因**：
- **C vs A**（同 k=3，只换 backend+KV）：c=1 退步 16%（fp8 KV+FlashInfer 单流更快），c=5/10 反而 +3%；acceptance 一致 0.54 → backend/KV 不影响投机命中率。
- **C/B/D**（同新配置，k=3/4/5）：acceptance 随深度递减 0.54→0.46→0.39，**k=3 在新配置下仍最优**。加深 k 无收益（marginal 位置接受的 token 几何衰减，各要一次 draft forward）。

**决策**：生产维持 **FlashInfer + fp8 KV + k=3**（单流最优、高并发持平、保留 fp8 KV 优势）。崩溃修复的价值是「k≥4 可运行」的能力解锁（待未来 acceptance 跃迁/多卡 overlap 才有吞吐收益），非当前收益。原始报告：[`benchmark-flashattn_k4_full.md`](benchmark-flashattn_k4_full.md) / [`benchmark-flashattn_k3.md`](benchmark-flashattn_k3.md) / [`benchmark-flashattn_k5.md`](benchmark-flashattn_k5.md)。

### DFlash 复测（RTX PRO 6000 + vLLM 0.24.0，2026-07-23）

**目标**：换 96GB Blackwell 新卡后复测 DFlash——L40S 上因独立 drafter 开销 + acceptance 无优势而「不适用」（见 Stage A），验证新卡的算力与显存余量是否翻转该结论。
**结论**：仍不适用。DFlash k=7 / k=15 在所有并发下均慢于 MTP k=3（c=10 分别 0.76× / 0.61×），且 k 越大、并发越高越劣。换硬件未改变动态。

报告：[`dflash_k7`](benchmark-rtx_pro_6000_vllm0240_dflash_k7.md) · [`dflash_k15`](benchmark-rtx_pro_6000_vllm0240_dflash_k15.md)

#### 版本阻塞与解法（关键）

| 版本 | DFlash 启动结果 |
|------|------|
| vLLM 0.25.1（当前生产版本） | ✗ `NotImplementedError: DFlash does not yet support mixed sliding/full attention via layer_types`（`qwen3_dflash.py:_resolve_layer_attention`）。drafter 的 `layer_types=[sliding×4, full×1]` 是混合注意力，0.25.1 重写了逐层解析但把该路径留作 `NotImplementedError`，修复卡在未合并的 [PR #40898](https://github.com/vllm-project/vllm/issues/40898) |
| vLLM 0.24.0 | ✓ 无此检查（旧实现更粗放），可启动；但需 `VLLM_USE_DEEP_GEMM=0` 规避 Blackwell DeepGEMM warmup `Unknown recipe`（0.25.1 已自动禁用 DeepGEMM，0.24.0 需手动） |

→ 本次复测用 **v0.24.0 + `VLLM_USE_DEEP_GEMM=0`**，与 MTP k=3 RTX baseline（81.5/357.5/653.7）同版本同 env，唯一变量是 speculative-config，口径干净。

#### 测试数据（同口径对比 MTP k=3 @ v0.24.0）

| 配置 | c=1 | c=5 | c=10 | TPOT(c=10) | decode tok/s(c=1) | acceptance | vs MTP c=10 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| MTP k=3（baseline） | 81.5 | 357.5 | 653.7 | 13.9ms | 82.2 | 0.54-0.55 | — |
| DFlash k=7 | 76.0 | 295.9 | 499.2 | 17.8ms | 77.1 | 0.20 | 0.76× |
| DFlash k=15 | 72.0 | 256.5 | 400.7 | 22.0ms | 74.1 | 0.10 | 0.61× |

#### 根因（与 Stage A 一致，新卡未翻转）

1. **acceptance 太低、且 k 越大越低**：k=7 acceptance 0.20（≈1.4 accepted/step）、k=15 仅 0.10（≈1.5 accepted/step）。每 step 有效 token ≈ 2.4-2.5，与 MTP k=3（≈2.6）相当——**无 acceptance 收益**。
2. **drafter 开销 dominant**：DFlash 每 step 额外跑 3.3GB 独立 drafter + 验证 k 个 token；MTP 零开销复用 target 自身 MTP 层。acceptance 无优势 + 有额外开销 → 净负；k 越大 verify batch 越大、并发越高 drafter forward 串行放大 → 越劣（c=10：k=7 0.76× → k=15 0.61×）。
3. **KV cache 压力本次不成立**：96GB 下 k=7 KV cache 560K tokens、k=15 376K tokens，远超 L40S 时代（k=7 仅 61K），不再是并发瓶颈——证明根因是 drafter 开销 + acceptance，非显存。
4. **k=15（README 推荐满配）最差**：block_size=16 verify 开销最大、acceptance 却最低 → 全场景垫底。

#### caveat：mixed-SWA 未正确处理可能压低 acceptance

v0.24.0 对 drafter 的混合 sliding/full 注意力处理粗放（无 PR #40898 的逐层 causal metadata），可能损害 draft 质量导致 acceptance 偏低。严谨复测需 PR #40898（正确 non-causal SWA）；但即便 acceptance 提升，DFlash 仍需显著高于 MTP 的 ~2.6 token/step 才能抵消 drafter 开销，而当前 acceptance 离该门槛甚远。Stage A（L40S）同版本同 FlashInfer non-causal 已得一致结论（0.41-0.71×）。

#### 决策

**回退 MTP k=3 @ v0.25.1**（生产配置不变）。DFlash 在单 RTX PRO 6000 + 27B 配置下确认不适用——硬件升级解决了显存/算力，但 drafter 开销与 acceptance 天花板这两个结构性问题不变。DSpark 路线见下「DSpark 调研」：#40898 已关闭但可不依赖，单卡实测仍输 MTP。

### DSpark 调研（2026-07-23，单卡仍输 MTP，但两条阻塞已解除）

**背景**：DSpark（半自回归 + 置信度调度）直击 DFlash 的 acceptance 短板，vLLM main 已原生支持 `qwen3_dspark`。核查「单卡能否跑通 DSpark」。

**两条旧阻塞均已解除**：

1. **[PR #40898](https://github.com/vllm-project/vllm/issues/40898)（DFlash SWA 支持）已关闭、不会合并——但不依赖它**。`Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft` 在 **vLLM 0.23.0 + 自带两个 patch**（`qwen3_dflash.py` + `llm_base_proposer.py`，独立实现 Markov 半自回归路径）上把 DSpark 跑通；本项目早先「降 0.24.0 能跑 DFlash」亦印证 SWA 报错是 0.25.1 版本问题、非死墙。
2. **「无 EN/ZH 27B drafter」不成立**。`Qwen3.6-27B-DSpark-FR` 是法语专用、不可用；但 AEON 证明可在 `z-lab/Qwen3.6-27B-DFlash`（本项目已下好）基础上自训：加 rank-256 Markov 头 + on-policy 蒸馏，语料仅 15,936 序列、6000 步收敛，**单卡可训**，recipe 开源（`github.com/hikarioyama/dspark-aeon-27b`）。

**决定性同卡实测**：AEON 在**同款单 RTX PRO 6000** 上（vLLM 0.23.0, K=8, T=1.0, NVFP4 target）测得 DSpark-style 头端到端吞吐仅比 z-lab DFlash **+11.0%**（194.8 vs 175.5 tok/s，accept 0.420 vs 0.342）；按域 chat +8.1% / math +7.1%。结合本项目 DFlash=0.76× MTP → **DSpark≈0.84× MTP，仍输 MTP k=3**。

**根因不变**：单卡 decode 占满 GPU → drafter 无 overlap 空隙 → acceptance 提升只能换回 +11%，远不够把 0.76× 翻到 >1×。多卡 / disaggregation 才是翻转前提。

**决策**：DSpark 在单卡配置下确认仍输 MTP，不自训 drafter 优先；但「能跑通」已无技术阻塞。下一步转向先攻 **MTP k≥4 崩溃**（更便宜的杠杆），自训 drafter 列为次选。详见 AGENTS.md「下一步」。

---

## Baseline: MTP k=3（当前配置）

**配置**: `gpu-memory-utilization=0.90`, `max-num-batched-tokens=8192`, MTP `num_speculative_tokens=3`, thinking 关闭（`"reasoning_effort": "none"`）
**GPU 显存**: ~40/46 GB
**报告**: [`docs/benchmark-mtp_k3_no_think.md`](benchmark-mtp_k3_no_think.md)
**日期**: 2026-07-09

### 关键指标

| 并发 | output_tps | TTLT p50 | TTLT p90 | TTFT p50 | TTFT p90 | TPOT mean | Acceptance | 失败 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 37.8 | 55.7s | 57.2s | 0.21s | 0.49s | 26.3ms | 0.54 (≈2.6 tok/step) | 0 |
| 5 | 172.1 | 59.1s | 62.4s | 0.31s | 0.40s | 28.4ms | 0.53 (≈2.6 tok/step) | 0 |
| 10 | 317.0 | 65.0s | 68.7s | 0.32s | 0.49s | 31.0ms | 0.54 (≈2.6 tok/step) | 0 |

### 分析

- **单流吞吐**: ~37.8 tok/s — 受限于 2048 token 长输出
- **5 并发**: ~172.1 tok/s — 聚合吞吐约单流的 4.6×
- **10 并发**: ~317.0 tok/s — 聚合吞吐约单流的 8.4×，TPOT 31.0ms 表明 decode 高效
- **TTFT**: 0.21-0.32s（并发无关），说明 prefill 排队未构成瓶颈（`max-num-batched-tokens=8192` 足够覆盖 ~144 token prompt）
- **Acceptance**: 0.54 ≈ 2.62 tokens/step
- **所有并发 0 失败**
- **Thinking 说明**: benchmark 默认关闭 thinking（`"reasoning_effort": "none"`）。若开启 thinking，TTFT 会飙升至 ~52s（thinking tokens 出现在 `reasoning` 字段，首个 `content` token 需等 thinking 结束），TPOT 也会被快速 thinking tokens 拉低至 ~6ms。详见 [`docs/disable-thinking.md`](disable-thinking.md)

### Acceptance 分解

MTP k=3 每个 decode step 产出 = 1（target 保证 token）+ P₀ + P₁ + P₂。vLLM `spec_decode_num_accepted_tokens_per_pos_total` 计数器给出各 position 的独立 acceptance 概率：

| 并发 | P₀ (第1个draft) | P₁ (第2个draft) | P₂ (第3个draft) | 合计 tokens/step |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 73.5% | 52.9% | 35.6% | 2.62 |
| 5 | 72.5% | 51.1% | 35.3% | 2.59 |
| 10 | 73.3% | 52.4% | 36.3% | 2.62 |

- **并发无关**：三个并发级 acceptance 一致（差异 < 0.2%），说明 acceptance 是模型内在属性
- **递减规律**：P₀ ≈ 73%，P₁ ≈ 52%，P₂ ≈ 36%——每个位置被接受的独立概率随位置递增而下降，符合预期（越远的 draft 越难猜中）
- 这解释了 MTP 吞吐是无投机（纯 target）的 1.69–2.18×

---

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-07-08 | v1.0 | 初始基线报告，MTP k=3 |
| 2026-07-09 | v1.1 | 将 thinking 关闭设为默认，同时保留 thinking 开启数据作为参考 |
| 2026-07-20 | v1.2 | 新增网宿 edgecloud API vs 自部署横向对比（c=1, TTFT/decode），见末章 |
| 2026-07-20 | v1.3 | MTP k≥4 补丁验证：PR #46324 实测生效但仍崩（同症不同因），见 `benchmark-mtp-k4-pr46324.md` |
| 2026-07-23 | v1.4 | RTX PRO 6000 Blackwell 复测：MTP k=3 c=10 output_tps=653.7，0 失败 |
| 2026-07-23 | v1.5 | 升级 vLLM 0.25.1 并验证 DeepGEMM 默认决策：可启动但 target 自动回落 CUTLASS，c=10 output_tps=644.7 |
| 2026-07-23 | v1.6 | RTX PRO 6000 + vLLM 0.25.1 复测 MTP k=4/k=5：短请求可用但并发烟测后 `cudaErrorIllegalAddress`，回退 k=3 |
| 2026-07-23 | v1.7 | RTX PRO 6000 复测 DFlash：0.25.1 受 `NotImplementedError`(PR #40898) 阻塞，降 0.24.0+`VLLM_USE_DEEP_GEMM=0` 跑通；k=7/k=15 分别 0.76×/0.61× MTP，acceptance 仅 0.20/0.10，回退 MTP |
| 2026-07-23 | v1.8 | **MTP k≥4 崩溃修复**：切 `--attention-backend FLASH_ATTN`（decode 拿到 FULL cudagraph）后 k=4/k=5 全程 0 失败。硬约束：FLASH_ATTN 拒 fp8 KV（须 bf16）。k≥4 仍不赢 k=3（accept 0.54→0.46→0.39 递减），生产维持 FlashInfer+fp8+k3。详见 [`benchmark-flashattn-k4-k5-analysis.md`](benchmark-flashattn-k4-k5-analysis.md) |

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
| output_tps c=1 | 37.8 | 37.6 | −0.5% |
| output_tps c=5 | 172.1 | 174.2 | +1.2% |
| output_tps c=10 | 317.0 | 317.4 | +0.1% |
| c=10 TTFT p50 | 0.32s | 0.32s | **无差异** |

报告：[`benchmark-mtp_k3_batched16384.md`](benchmark-mtp_k3_batched16384.md) · [`benchmark-mtp_k3_verify_no_xxhash.md`](benchmark-mtp_k3_verify_no_xxhash.md)

**结论**：16384 无吞吐收益，TTFT 也无差异（thinking 开启时的数据显示 +12.5% 是噪声）。**保持 8192。**

### Stage 0 总结

默认参数已是最优。baseline（MTP k=3 + 默认参数 + gpu-mem=0.90）即为 Stage A 的干净起点。

---

## Stage A — DFlash 验证（结论：不适用，回退 MTP）

**目标**：MTP k=3 → DFlash，目标 output_tps ≥ MTP × 1.30。
**结论：DFlash 在当前硬件/配置下全面劣于 MTP，未达标，回退 MTP k=3 baseline。**（2026-07-23 在 RTX PRO 6000 96GB 新卡上复测确认同一结论：k=7/k=15 分别 0.76×/0.61× MTP，详见上文「DFlash 复测」章节。）

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

---

## MTP k≥4 验证（PIECEWISE cudagraph 崩溃，不适用，回退 k=3）

**目标**：增大 MTP 投机深度 k=3→4/5，看能否提升吞吐。**结论：k≥4 在并发 ≥5 时必触发 `cudaErrorIllegalAddress` 崩溃；根因是 vLLM 0.24 PIECEWISE cudagraph 在 spec-decode 下的越界 bug，与显存、max-num-seqs 均无关。唯一规避（enforce_eager）比 k=3 慢且无收益。回退 k=3。**

### 测试数据

| 配置 | max-num-seqs | c=1 | c=5 | c=10 | 说明 |
|------|:---:|:---:|:---:|:---:|------|
| MTP k=3（baseline） | 10 | 36.8 | 169.2 | 309.4 | 0 失败 |
| MTP k=4（默认 cudagraph） | 10 | 36.8 | **崩溃** | **崩溃** | c≥5 全 500 |
| MTP k=4（enforce_eager） | 10 | ~27.4 | ~91 | — | 稳定但慢 25-46% |
| MTP k=5（默认 cudagraph） | 5 | 可跑* | **崩溃** | — | c=5 全 500（10.6s 同时返回） |

\* k=5 c=1 warmup http=200（单序列不崩）；完整 benchmark 未跑（c≥5 必崩）。

c≥5 错误：`EngineDeadError` → `torch.AcceleratorError: CUDA error: an illegal memory access was encountered (cudaErrorIllegalAddress)`。报告：[`benchmark-mtp_k4.md`](benchmark-mtp_k4.md)（k=4 仅 c=1 完整，c=5/10 全失败）。

### 根因分析（cudagraph replay 越界，非显存、非 max-num-seqs）

1. **错误类型**：`cudaErrorIllegalAddress`（非法内存访问），不是 OOM（OOM 会报 `out of memory`）。崩溃点 `gpu_model_runner.py:3767 synchronize_input_prep`（异步报告，真实越界 kernel 更早）。
2. **排除显存**：崩溃瞬间显存峰值仅 **40.2 / 46 GB（k=4）/ 38.6 / 46 GB（k=5）**，余量 5.8–7.4 GB，远未触顶。
3. **enforce_eager 验证（决定性，k=4）**：关闭 cudagraph + torch.compile 后，k=4 c=5 全部 http=200 稳定 → 根因 = **cudagraph replay 越界**，非 draft 逻辑 bug。
4. **机制**：vLLM 0.24 的 spec-decode + FlashInfer 不支持 FULL cudagraph（日志 `AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE`），自动降级 PIECEWISE；k≥4（每序列 ≥5 候选 token）在并发 batch 下 PIECEWISE graph shape 与实际 spec-decode batch 不匹配，replay 时越界。k=3（每序列 4 候选）落在稳定 capture size，不触发。
5. **max-num-seqs=5 不能规避（k=5 实验证伪）**：缩小 max-num-seqs 只缩小 cudagraph capture range（k=5 时 `[1,2,4,8,16,24,32,40,48,56]`，max 56），但 **PIECEWISE 降级依然触发**（启动警告同 k=4），c=5 仍崩溃。根因在 PIECEWISE + k≥4 的 shape 越界，与并发上限无关。
6. **c=1 零收益**：k=4 c=1 output_tps=36.8 = k=3，说明增大 k 未提升 accepted length / TPOT（MTP 单流吞吐由 acceptance 决定，k=3 已接近该模型 acceptance 上限）。

### 补丁验证（PR #46324，2026-07-20，无效）

社区 PR [#46324](https://github.com/vllm-project/vllm/pull/46324)（PIECEWISE spec-decode 捕获尺寸对齐）描述的根因与本现象高度吻合。bind-mount 打补丁实测：**补丁确认生效**（运行时 `cudagraph_capture_sizes` 对齐成 `1+k=5` 的倍数 `[5,10,20,25,35,40,50,60,65,75,80,90]`）**但 c=5 仍 `cudaErrorIllegalAddress`**（failed 50/50）——#46324 修的那类越界已被排除，本崩溃是另一个 bug（同症不同因；PR 针对 DFlash+Blackwell，我们是 MTP+L40S/Ada）。详见 [`benchmark-mtp-k4-pr46324.md`](benchmark-mtp-k4-pr46324.md)。

### 规避方法与代价

- **唯一规避**：`--enforce-eager`（关 cudagraph + torch.compile）。代价：k=4 下 c=1 ~27.4 / c=5 ~91 tok/s，比 k=3 baseline 慢 **25–46%**，且 c=1 零收益。得不偿失。
- **不可行**：降 max-num-seqs、换 k 值（k=4/5 均崩）均无效。

### 决策

**回退 MTP k=3 baseline**。k≥4 在单 L40S + vLLM 0.24 配置下：默认崩溃不可用，唯一规避（enforce_eager）稳定但慢且无收益。根因是 vLLM PIECEWISE cudagraph 的 k≥4 边界 bug；待 vLLM 修复 spec-decode FULL cudagraph 支持（或换硬件/版本）后可重试。

---

## 无 MTP 对比验证（结论：MTP 全面占优，确认 k=3 最优）

**目标**：禁用投机解码（移除 `--speculative-config`，纯 target model decode），量化 MTP k=3 的实际增益。**结论：MTP 在所有并发档全面优于无投机（单流 2.18×，高并发 1.69×），确认 MTP k=3 为最优配置。**

### 测试数据

| 配置 | c=1 | c=5 | c=10 | TPOT(c=10) | KV cache tokens |
|------|:---:|:---:|:---:|:---:|:---:|
| MTP k=3（baseline） | 36.8 | 169.2 | 309.4 | 3.9ms | 136,953 |
| 无 MTP（纯 target） | 16.9 | 97.1 | 182.7 | 3.1ms | 228,162 |
| **MTP 提升** | **2.18×** | **1.74×** | **1.69×** | — | — |

报告：[`benchmark-no_mtp.md`](benchmark-no_mtp.md)（0 失败）。

### 分析

1. **MTP 全面占优**：所有并发档 MTP 吞吐显著高于无投机。MTP 的 acceptance（~2.4）让每 decode step 产出多个 token，直接转化为吞吐收益。
2. **提升随并发递减（2.18× → 1.69×）**：c=1 时纯串行 decode 最慢，MTP acceptance 收益最大；c=10 时无投机也能 batch 并行共享算力，且无投机 KV cache 更大（228k vs 137k，+66%，可容纳更多并发序列），缩小差距。
3. **TPOT 反直觉但不影响结论**：无 MTP c=10 TPOT 3.1ms < MTP 3.9ms（单步只算 1 token，无 draft verify 开销），但总吞吐仍输给 MTP——MTP 每步产出更多 token 抵消了 verify 开销。
4. **结论**：MTP k=3 是单 L40S + 27B 的正确选择，带来 1.69-2.18× 实测吞吐增益。回退 k=3 baseline。

---

## 容器优化验证（host 网络 + ipc host，结论：无实质提升，裸机无意义）

**目标**：验证「直接在宿主机装 vLLM（裸机）是否比容器快」的假设。**结论：host 网络 + ipc host 优化对吞吐无实质影响（+0.1~1.1%，benchmark 噪声内），证明容器已等价于裸机，无需裸机部署。**

### 背景

容器理论上对 GPU 推理零开销（`runtime: nvidia` 直通 GPU + bind mount 模型权重走宿主机磁盘）。容器仅有的微小开销点是：① 网络桥接 NAT（`ports` 映射 vs host 网络）；② 默认 `/dev/shm` 64MB 限制（vs `ipc: host` 共享宿主机 shm）。

为验证这两点是否影响吞吐，给容器加 `network_mode: host` + `ipc: host`（vLLM 官方推荐高性能配置，使容器的 CPU/网络/shm 层面等价于裸机），重跑 benchmark 对比。若此优化无提升，则裸机（再省掉容器层）必然也无意义——因为 GPU 本就直通。

### 配置变更

`docker-compose.yml`：
- 移除 `ports: ["0.0.0.0:8000:8000"]`（host 网络下端口映射冲突）
- 新增 `network_mode: host`（容器直接用宿主机网络栈，benchmark localhost 走内核 loopback，不经 docker 桥接 NAT/docker-proxy）
- 新增 `ipc: host`（共享宿主机 `/dev/shm` + IPC namespace，消除 torch worker 间共享内存的 64MB 容器限制；与 `shm_size` 互斥）

vLLM 启动参数与 baseline 完全相同。

### 测试数据

| 配置 | c=1 | c=5 | c=10 | 显存 | 失败 |
|------|:---:|:---:|:---:|:---:|:---:|
| MTP k=3 baseline（端口映射） | 36.3 | 170.0 | 309.8 | ~38.8/46 | 0 |
| host_net（host 网络 + ipc host） | 36.6 | 171.8 | 310.1 | 38.8/46 | 0 |
| **变化** | +0.8% | +1.1% | +0.1% | 一致 | — |

报告：[`benchmark-host_net.md`](benchmark-host_net.md)（0 失败）。

### 根因分析

1. **GPU 本就直通**：`runtime: nvidia` 下 CUDA kernel、显存、算力全部直接走宿主机 GPU，推理瓶颈（GPU 算力 + 显存带宽）在容器内外物理相同——这是吞吐无差异的根本原因。
2. **模型走 bind mount**：`/data/models:/models` 直接读宿主机磁盘，无 overlay2 存储驱动开销，模型加载 I/O 无差异。
3. **网络/shm 非瓶颈**：vLLM 是 GPU-bound，单次生成耗时数十秒由 GPU decode 主导，网络桥接（μs 级）与 shm 在总耗时占比可忽略。
4. **波动属噪声**：+0.1~1.1% 落在 benchmark 运行间正常波动内（baseline 自身两次跑即为 36.3 vs 36.8），无统计意义。

### 决策

**host_net 配置保留**（vLLM 官方推荐、无副作用、略优）。更重要的是：**裸机部署（直接在服务器装 vLLM）已被证明无意义**——`network_mode: host` + `ipc: host` 让容器在 CPU/网络/shm 层面等价于裸机，而 GPU 本就直通，故容器 = 裸机（性能上）。无需花 30-60min 在宿主机装 vLLM 环境。

## 外部对比：网宿 edgecloud API vs 自部署（2026-07-20）

横向对比自部署 vLLM (L40S) 与网宿托管 API。相同 prompt（LC 字段抽取，prompt_tokens=3423）、两端 thinking 均关闭（`reasoning_effort=none`）、c=1、streaming 逐 chunk 测 TTFT/decode。各端 warmup 后 3 轮取中位数。

| 指标 | 自部署 (vLLM L40S) | 网宿 (edgecloud) |
|:---|---:|---:|
| TTFT（首 token）≈ Prefill | **0.50s** | 2.65s |
| decode 间隔 | 17.3 ms/tok | **12.6 ms/tok** |
| decode 吞吐 | 57.7 tok/s | **79.2 tok/s** |
| 端到端 wall（~1261 tok） | 22.4s | 18.4s |
| 稳定性 | 零抖动（独占） | ±30%（共享集群） |

**结论**：网宿 decode 更快（后端算力强于单张 L40S），但 TTFT 慢（公网往返）且波动大（共享集群）；自部署首字延迟与稳定性更优。长输出场景网宿端到端快 ~18%，短输出/交互式场景自部署更优。数据合规/成本/可控性上自部署占优。

> ⚠️ 本对比为 c=1；自部署在 c=5/10 下 output_tps 达 169/309，网宿并发扩展性未测。详细数据、原始各轮与测量局限见 [`benchmark-wangsu-vs-selfhosted.md`](benchmark-wangsu-vs-selfhosted.md)，复现脚本 `measure_latency.py`（项目根）。
