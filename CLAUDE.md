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

- **阶段**: Stage 0 完成 → 准备 Stage A (DFlash)
- **GPU 显存**: ~39.5/46 GB
- **基线性能**: output_tps = 36.8(c=1) / 169.2(c=5) / 309.4(c=10)，0 失败（默认参数已验证最优）
- **Stage 0 结论**: 默认参数即为最优，无需调整
  - xxhash 崩溃根因：容器缺 `xxhash` 包（`ModuleNotFoundError` → `EngineDeadError`）；单次 curl 不触发、benchmark 必崩。放弃该参数
  - `enable-flashinfer-autotune`：非合法 CLI 参数，功能默认已开
  - `max-num-batched-tokens 16384`：无收益且轻微恶化 c=10 TTFT，回退 8192
  - `partial-prefills`：vLLM 0.24 不支持
  - 详细记录见 `docs/performance.md#stage-0--根因分析与参数验证`
- **下一步**: Stage A — MTP → DFlash k=7（保持当前 baseline 参数）

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
