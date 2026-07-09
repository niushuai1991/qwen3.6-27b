# Qwen3.6-27B-FP8 部署项目

## 项目概述

Qwen3.6-27B-FP8 vLLM 容器化部署，目标：DSpark 投机解码优化，三阶段推进（MTP → DFlash → DSparkLite → 完整 DSpark）。

## 技术栈

- **模型**: Qwen/Qwen3.6-27B-FP8（29GB）
- **推理引擎**: vLLM 0.24.0（`vllm/vllm-openai:latest`）
- **投机解码**: MTP k=3 → DFlash k=7 → DSpark
- **Benchmark**: llmeter v0.1.12（streaming mode）
- **训练** (Stage C): DeepSpec (PyTorch + FSDP)

## 硬件约束

- GPU: NVIDIA L40S 46GB VRAM
- 系统内存: 57GB
- 显存硬上限: 46GB（必须留余量）
- 模型路径: `/data/models/Qwen3.6-27B-FP8`

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

- **阶段**: Stage A 完成（DFlash 不适用）→ DSpark 路线暂止于 MTP baseline
- **GPU 显存**: ~38.8/46 GB（MTP baseline 稳态）
- **当前配置**: MTP k=3 + baseline 参数（Stage 0 验证最优，Stage A 后已回退）
- **基线性能**: output_tps = 36.8(c=1) / 169.2(c=5) / 309.4(c=10)，0 失败
- **MTP k=4 验证（2026-07-09，不适用→回退 k=3）**: k=4 默认在并发 ≥5 触发 `cudaErrorIllegalAddress`（vLLM PIECEWISE cudagraph 越界，**非显存**——峰值仅 40.2/46GB）；`--enforce-eager` 关 cudagraph 可稳定但比 k=3 慢 25-46%，且 c=1 零收益（36.8=36.8）。详见 `docs/performance.md#mtp-k4-验证`
- **Stage A 结论（DFlash 不适用）**: DFlash k=7/4/3 全面劣于 MTP（最佳 k=3: c=10=219.9 vs MTP 309.4 = 0.71×），回退 MTP
  - 根因：独立 drafter（3.3GB）每 step 额外 forward（MTP 零开销）+ KV cache 压力限并发到 7-8 + acceptance 仅 ~2.4-2.7（与 MTP 相当，无收益）
  - DFlash 优势需充足算力 overlap drafter；单 L40S 跑 27B decode 时 GPU 已占满，drafter 变串行开销
  - 详细记录见 `docs/performance.md#stage-a--dflash-验证结论不适用回退-mtp`
- **Stage B/C 不可行**:
  - Stage B（DSparkLite custom_class）依赖 DFlash drafter，同样不适用；且 `disable_padded_drafter_batch` 触发 `NotImplementedError`
  - Stage C（DeepSpec 训练）多重硬阻塞：DeepSpec 只支持标准 Qwen3 不支持 qwen3_5 混合架构 + 27B BF16 54GB>46GB 显存 + target cache ~76TB>>465GB + Python 3.9 不兼容。repo 已 clone 到 `~/code/DeepSpec`
- **Stage 0 结论**: 默认参数即为最优（详见 `docs/performance.md#stage-0`）
- **下一步**: DSpark 路线在单 L40S + 27B 配置下已穷尽，待硬件升级（多卡 ≥2×46GB 或 80GB）+ qwen3_5 draft model 实现后重启

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
curl -s http://localhost:8000/health         # 服务状态
nvidia-smi --query-gpu=memory.used --format=csv  # 显存使用

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
