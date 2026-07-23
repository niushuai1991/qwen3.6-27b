# Qwen3.6-27B-FP8 部署项目

## 项目概述

Qwen3.6-27B-FP8 vLLM 容器化部署，目标：DSpark 投机解码优化，三阶段推进（MTP → DFlash → DSparkLite → 完整 DSpark）。

## 技术栈

- **模型**: Qwen/Qwen3.6-27B-FP8（29GB）
- **推理引擎**: vLLM 0.25.1（`vllm/vllm-openai:v0.25.1`；DFlash 复测时降级 0.24.0）
- **投机解码**: MTP k=3（当前）→ DFlash k=7/15（已测，不适用）→ DSpark
- **Benchmark**: 原生 streaming 计时（`benchmark.py`，measure_latency 同源口径）
- **训练** (Stage C): DeepSpec (PyTorch + FSDP)

## 硬件约束

- GPU: NVIDIA RTX PRO 6000 Blackwell 96GB VRAM（2026-07-23 起；原 L40S 46GB）
- 系统内存: 62GB
- 显存硬上限: 96GB（`gpu-memory-utilization=0.90` 稳态 ~85GB；旧 `<46GB` 验收口径不再适用）
- 模型路径: `/data/models/Qwen3.6-27B-FP8`
- DFlash drafter: `/data/models/Qwen3.6-27B-DFlash`（3.3GB，`z-lab/Qwen3.6-27B-DFlash`，5 层 qwen3，block_size=16）

## 关键文件

| 文件 | 用途 |
|------|------|
| `docker-compose.yml` | 服务编排 + vLLM 参数 |
| `benchmark.py` | llmeter 基准测试脚本 |
| `DEPLOY.md` | 部署指南（面向用户） |
| `docs/config.md` | 配置参数文档（技术参考） |
| `docs/design.md` | 架构设计概览 |
| `docs/performance.md` | 性能追踪总览（跨阶段对比） |
| `docs/benchmark-*.md` | 各阶段详细性能报告 |
| `docs/superpowers/plans/2026-07-08-dspark-deploy.md` | 实施计划 |

## 工作约定

- **不要使用 git worktree** — 所有工作直接在 `main` 分支上进行，无需创建 worktree 隔离。

## 当前进度

- **阶段**: RTX PRO 6000 复测 MTP baseline（2.06× L40S）；DFlash 新卡复测仍不适用；DSpark 经 AEON 同卡实测确认单卡仍输 MTP（+11% over DFlash → ~0.84× MTP）；**MTP k≥4 崩溃已修复**（切 `--attention-backend FLASH_ATTN`，FULL cudagraph，k=4/5 全 0 失败），但 k≥4 仍不赢 k=3（acceptance 随深度递减）→ 生产维持 FlashInfer+fp8+k3；下一步转向评估自训 drafter（次选）
- **GPU 显存**: ~85/96 GB（MTP baseline 稳态，RTX PRO 6000 Blackwell）
- **当前配置**: vLLM 0.25.1 + MTP k=3 + baseline 参数 + 容器 host 网络/ ipc host 优化（Stage 0 验证最优）
- **基线性能（RTX PRO 6000）**: output_tps = 81.5(c=1) / 357.5(c=5) / 653.7(c=10)，acceptance 0.54-0.55，0 失败（v0.24.0 同配置与 0.25.1 几乎等价 80.1/345.5/644.7）
- **MTP k≥4 崩溃修复（2026-07-23，已解决→结论「无吞吐收益」）**: 切 `--attention-backend FLASH_ATTN` 后，decode 拿到 **FULL cudagraph**（vs FlashInfer 只给 PIECEWISE），k=4/k=5 × c=1/5/10 **全程 0 失败、容器零重启**——假设机制成立。**硬约束**：FLASH_ATTN 拒 fp8 KV cache（启动即 `ValueError: ... kv_cache_dtype not supported`），k≥4 必须放弃 fp8 KV（改 bf16）。**但 k≥4 仍不赢 k=3**：acceptance 随深度递减（k3=0.54 / k4=0.46 / k5=0.39），同新配置下 k=3 仍最优；配置归因（FLASH_ATTN+bf16 vs fp8 baseline）单流退步 16%、高并发 +3%。**决策：生产维持 FlashInfer+fp8+k3**；修复价值是「k≥4 可运行」能力解锁（待 acceptance 跃迁/多卡 overlap 才有收益）。详见 `docs/benchmark-flashattn-k4-k5-analysis.md` / `docs/performance.md`「MTP k≥4 崩溃修复」章节
- **Stage A 结论（DFlash 不适用）**: DFlash k=7/4/3 全面劣于 MTP（最佳 k=3: c=10=219.9 vs MTP 309.4 = 0.71×），回退 MTP
  - 根因：独立 drafter（3.3GB）每 step 额外 forward（MTP 零开销）+ KV cache 压力限并发到 7-8 + acceptance 仅 ~2.4-2.7（与 MTP 相当，无收益）
  - DFlash 优势需充足算力 overlap drafter；单 L40S 跑 27B decode 时 GPU 已占满，drafter 变串行开销
  - 详细记录见 `docs/performance.md#stage-a--dflash-验证结论不适用回退-mtp`
- **DFlash RTX PRO 6000 复测（2026-07-23，仍不适用→回退 MTP）**: 新卡上 DFlash k=7/k=15 分别 0.76×/0.61× MTP（c=10: 499.2/400.7 vs 653.7），acceptance 仅 0.20/0.10
  - 版本阻塞：vLLM 0.25.1 对 drafter 混合 sliding/full attention 报 `NotImplementedError`（[PR #40898](https://github.com/vllm-project/vllm/issues/40898) **已关闭、不会合并**，但**不依赖它**——见下「DSpark 调研」：vLLM 0.23.0 + 独立 patch 可跑通）；降 0.24.0 + `VLLM_USE_DEEP_GEMM=0` 亦可跑通
  - 根因不变：独立 drafter 每步 forward 开销 + acceptance 无优势（≈2.4-2.5 token/step ≈ MTP）；96GB 下 KV cache 已不再是瓶颈，证明根因是 drafter 开销而非显存
  - caveat：0.24.0 未含正确 non-causal SWA，可能压低 acceptance；AEON 已用独立 patch 提供 SWA 路径，可严谨复测（见下「DSpark 调研」）
  - 详细记录见 `docs/performance.md`「DFlash 复测（RTX PRO 6000）」章节
- **DSpark 调研结论（2026-07-23，单卡输 MTP，但两条阻塞已解除）**: DSpark 半自回归头（DFlash backbone + Markov/置信度头）直击 DFlash 的 acceptance 短板，vLLM main 已原生支持 `qwen3_dspark`。两条旧阻塞已解：① #40898 已关闭但**不依赖它**——`Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft` 在 vLLM 0.23.0 + 自带两个 patch（独立实现 Markov 半自回归路径）跑通；②「无 EN/ZH 27B drafter」不成立——`Qwen3.6-27B-DSpark-FR` 是法语专用不可用，但 AEON 证明可在 z-lab DFlash 基础上自训（rank-256 Markov 头，on-policy 蒸馏，16k 序列 6k 步，单卡可训，recipe 开源）。**但同卡实测决定性**：AEON 在同款单 RTX PRO 6000 测得 DSpark-style 头端到端仅比 DFlash **+11%**（194.8 vs 175.5 tok/s，K=8/T=1.0）；结合本项目 DFlash=0.76× MTP → **DSpark≈0.84× MTP，仍输 MTP k=3**。根因不变：单卡 decode 占满 GPU → drafter 无 overlap 空隙。详见 `docs/performance.md`「DSpark 调研」章节
- **Stage B/C 不可行**:
  - Stage B（DSparkLite custom_class）依赖 DFlash drafter，同样不适用；且 `disable_padded_drafter_batch` 触发 `NotImplementedError`
  - Stage C（DeepSpec 训练）多重硬阻塞：DeepSpec 只支持标准 Qwen3 不支持 qwen3_5 混合架构 + 27B BF16 54GB>46GB 显存 + target cache ~76TB>>465GB + Python 3.9 不兼容。repo 已 clone 到 `~/code/DeepSpec`
- **Stage 0 结论**: 默认参数即为最优（详见 `docs/performance.md#stage-0`）
- **容器优化验证（2026-07-09，裸机无意义）**: 容器加 `network_mode: host` + `ipc: host` vs baseline 仅 +0.1~1.1%（噪声内）。证明容器已等价裸机（`runtime: nvidia` GPU 直通 + bind mount 模型），无需宿主机裸机部署。详见 `docs/performance.md`「容器优化验证」
- **下一步（2026-07-23 修订，按性价比排序）**:
  1. ✅ **解决 MTP k≥4 崩溃**（已完成；结论：无吞吐收益）: 切 `--attention-backend FLASH_ATTN`（FULL cudagraph）已让 k=4/k=5 稳跑（0 失败），但硬约束是须放弃 fp8 KV cache，且 k≥4 acceptance 随深度递减、仍不赢 k=3。能力解锁但非当前收益，生产维持 k=3。详见 [`docs/benchmark-flashattn-k4-k5-analysis.md`](docs/benchmark-flashattn-k4-k5-analysis.md)。
  2. **自训 DSpark drafter**（当前首选；针对 stock Qwen3.6-27B-FP8 / EN+ZH）: 用 AEON 开源 recipe（z-lab DFlash 基础 + rank-256 Markov 头，on-policy 蒸馏，单卡可训）。**可行性已调研**（见 [`docs/dspark-selftrain-feasibility.md`](docs/dspark-selftrain-feasibility.md)）：vLLM 0.25.1 原生 `qwen3_dspark.py` schema 与 recipe 同构（很可能免 patch 原生 serve），target/drafter 完全兼容；**但** ①FP8 target 与训练 loader 不匹配（需改 loader 或换 BF16 target，首日烟测）②数据全自采+占 GPU ③训练独占整卡需停在线服务 ~1 天 ④预期仍输 MTP（≈0.84×）。工作量 ~20–50 工程小时 + ~1 天 GPU。下一步先做 /tmp 烟测阻塞 ①+集成（不碰项目文件），再决定是否投入全量。
  3. **翻转 DSpark 结论的真正前提**（非近期）: 多卡 / disaggregation 给 drafter overlap 空间，或 acceptance 量级跃迁——而非更强的 drafter 算法或更高 k。注意：k≥4 已可跑（见 #1），若自训 drafter 把 acceptance 量级提上去，更高 k 才有发挥空间。

## 文档约定

- **性能追踪** (`docs/performance.md`) — 各阶段性能总览，跨阶段对比，优化方向分析
- **性能报告** (`docs/benchmark-*.md`) — 每次 benchmark 自动生成
- **配置文档** (`docs/config.md`) — 记录所有参数变更，各阶段对照
- **部署指南** (`DEPLOY.md`) — 面向用户的操作手册
- **架构设计** (`docs/design.md`) — 高层设计，不超过一屏

## 常用命令

```bash
# 服务管理
docker compose up -d                         # 启动
docker compose logs -f                       # 查看日志
docker compose down                          # 停止

# 健康检查
curl -s http://localhost:18001/health         # 服务状态
nvidia-smi --query-gpu=memory.used --format=csv  # 显存使用

# 关闭 thinking（reasoning_effort；详见 docs/disable-thinking.md）
curl -s http://localhost:18001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
    "max_tokens": 100,
    "reasoning_effort": "none"
  }'

# 基准测试
uv run python benchmark.py --label <label> --concurrency "1,5,10"

# 查看性能结果
grep -E "output_tps|TPOT_mean" docs/benchmark-<label>.md
```

## 验收标准

- 所有 benchmark: 0 失败请求
- GPU 显存: < 46GB at all times
- Stage A: output_tps ≥ MTP × 1.30
- Stage B: output_tps ≥ Stage A × 1.05
- Stage C: accepted length 达论文水平的 70%+
