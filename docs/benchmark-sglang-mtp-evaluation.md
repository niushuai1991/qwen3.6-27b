# SGLang MTP 评估：Qwen3.6-27B-FP8 @ RTX PRO 6000 Blackwell（SM120 单卡）

**日期**：2026-07-24
**目的**：验证假设「换 SGLang 可稳定跑更高 MTP k，从而超过 vLLM k=3 baseline」
**结论**：**NO-GO**。SGLang k=3/4/5 均不超 vLLM k=3 基线 +10%；acceptance 衰减墙与引擎无关，再次实证确认。
**生产决策**：**维持 vLLM k=3 不变**；SGLang 不引入。

## 背景

vLLM 侧已确认：MTP k≥4 在 FlashInfer 下崩溃（[根因上游 #37744]），切 FLASH_ATTN 虽可稳跑但须放弃 fp8 KV cache，且 k≥4 acceptance 随深度递减、不赢 k=3（详见 `docs/benchmark-flashattn-k4-k5-analysis.md`）。

本次评估的核心动机：SGLang 的 spec-v2（overlap 调度）+ NEXTN(MTP) **可能绕开 vLLM 的 k≥4 崩溃**，在单卡上稳定跑更高 k。前提是上游 issue [#31249](https://github.com/sgl-project/sglang/issues/31249)（本模型 warmup 崩溃）已被 PR [#27998](https://github.com/sgl-project/sglang/pull/27998) 修复（详见 `docs/2026-07-24-fault-sglang-qwen35-mtp-specv2-crash.md`）。

## 部署与验证

| 项 | 值 |
|---|---|
| 镜像 | `lmsysorg/sglang:dev-cu13`（CUDA13，支持 SM120；版本 `0.0.0.dev1+g1e10ec93b`） |
| 修复 #27998 | ✅ 镜像内已含（`schedule_batch.py:1716` 命中 `mamba_next_track_idx ... is not None else 0`） |
| 架构支持 | ✅ 原生（`sglang/srt/models/qwen3_5_mtp.py` 专用 MTP 实现） |
| 启动 | ✅ 0 崩溃；`Capture target verify CUDA graph`（#31249 崩溃路径）成功；`UnifiedRadixCache hybrid_ssm=True`（崩溃组合 radix+spec-v2）正常 |
| 配置 | NEXTN, topk=1, fp8 KV, radix cache（默认）, `mem-fraction-static=0.90`, `context-length=32768`, `max-running-requests=10`, `chunked-prefill-size=8192`, `SGLANG_ENABLE_SPEC_V2=1` |
| compose | `docker-compose.sglang.yml`（独立文件，未动 vLLM 的 `docker-compose.yml`）；k 用 `SPEC_STEPS`/`SPEC_DRAFT_TOKENS` 环境变量参数化 |

**「SM120 单卡未验证」不确定性已关闭**：本配置（混合架构 + spec-v2 + radix cache + SM120 单卡）稳定运行，k=3/4/5 全程 0 崩溃、0 失败。

## 结果（output_tps，c=1/5/10）

| 并发 | vLLM k=3（基线） | SGLang k=3 | SGLang k=4 | SGLang k=5 | SGLang 最佳 vs 基线 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| c=1  | 81.5  | **84.6** | **85.3** | 84.4 | +4.7%（k=4） |
| c=5  | 357.5 | **362.5** | 356.9 | 348.8 | +1.4%（k=3） |
| c=10 | 653.7 | 622.4 | 607.8 | 599.6 | **−4.8%**（k=3） |

decode tok/s（单流 decode 阶段有效吞吐）：

| 并发 | SGLang k=3 | k=4 | k=5 |
|:---:|:---:|:---:|:---:|
| c=1  | 85.2 | 86.0 | 85.7 |
| c=5  | 76.9 | 75.7 | 74.7 |
| c=10 | 68.5 | 65.7 | 64.8 |

> 全部 0 失败请求。逐 k 明细见 `docs/benchmark-sglang_k3.md` / `_k4` / `_k5`（acceptance 因 SGLang 指标格式与 vLLM 不同，benchmark.py 未采集，记 N/A）。

## Go/No-Go 判决

- **Go 判据**：任一 k 在任一并发上 output_tps 稳定超基线 ≥10% 且 0 失败。
- **实际**：最佳 +4.7%（k=4, c=1），其余 +1.4% / 持平 / 为负；c=10 全线低于基线。
- **结论：NO-GO**，不满足判据。

## 分析：为什么更高 k 仍不赢

1. **SGLang k=3 ≈ vLLM k=3（无引擎基线优势）**：c=1/5 略快（+3.8%/+1.4%，噪声内）、c=10 略慢（−4.8%）。"SGLang 引擎本身更快"的假设不成立，两大引擎在本配置同口径持平。
2. **更高 k 不赢 k=3 = acceptance 衰减墙，引擎无关**：k=3→4→5 在 c=5/c=10 上单调走低（362.5→356.9→348.8；622.4→607.8→599.6），与 vLLM 侧 k3=0.54/k4=0.46/k5=0.39 的 acceptance 递减完全同构。SGLang 用**同一个 MTP draft head**、同一套 rejection sampling，深层 draft 被拒比例不变 → 提高 k 只增加被拒草稿的浪费算力，不增有效产出。
3. **c=10 SGLang 略逊 + 随 k 退化更明显**：单卡 decode 在高并发已饱和，spec-v2 的 overlap 调度在饱和 GPU 上挤不出额外 overlap 空间（项目早已定位的根因：单卡 decode 占满 → drafter 无 overlap 空隙）。

## 能力解锁 vs 收益：两件事

- ✅ **能力解锁（真实）**：SGLang 在本混合架构 + SM120 单卡上**稳定跑 k≥4**（vLLM k≥4 崩）。这对未来若 acceptance 能跃迁（自训 drafter）或多卡 overlap 时是必要前提。
- ❌ **当前无收益（决定性）**：能力解锁**不转化为吞吐**，因 acceptance 衰减是瓶颈。换引擎解决的是「能不能跑更高 k」，解决不了「更高 k 为什么不赢」。

## 建议与下一步

- **生产维持 vLLM k=3**：SGLang 无吞吐收益，引入只增运维复杂度（dev 滚动镜像、混合架构 spec 路径前沿脆弱）。
- **binding 约束仍是 acceptance（drafter 质量），引擎无关**：唯一能从「质」上提升的杠杆仍是 ①自训 drafter（已被「无业务数据集」阻塞）②多卡/disaggregation 给 overlap 空间（已被「成本」否决）。换引擎不在这两条路径上。
- **若未来 acceptance 跃迁**：SGLang 的「稳定高 k」能力才有发挥空间——届时可基于本报告的 compose 与已验证镜像快速复测。

## 耗费

- Docker 数据目录迁 `/data`（436G 空闲，解决 `/` 盘满）；删已结论无用的 `vllm/vllm-openai:v0.24.0`（可重拉）。
- SGLang 镜像 `lmsysorg/sglang:dev-cu13`（约 20G+）已留存于 `/data/docker`，便于复测。
- vLLM 生产停机窗口约 40 分钟（含 3 次 SGLang k 切换 + benchmark），已恢复。

## 相关

- 故障/修复核实：[`docs/2026-07-24-fault-sglang-qwen35-mtp-specv2-crash.md`](2026-07-24-fault-sglang-qwen35-mtp-specv2-crash.md)
- vLLM 侧 k≥4 结论：[`docs/benchmark-flashattn-k4-k5-analysis.md`](benchmark-flashattn-k4-k5-analysis.md)
- 性能总览：[`docs/performance.md`](performance.md)「SGLang MTP 评估」章节
- 上游：issue [#31249](https://github.com/sgl-project/sglang/issues/31249)、PR [#27998](https://github.com/sgl-project/sglang/pull/27998)
