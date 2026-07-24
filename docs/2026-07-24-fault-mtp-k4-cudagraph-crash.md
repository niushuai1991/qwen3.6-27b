# [故障事件] MTP k≥4 spec-decode `cudaErrorIllegalAddress` 崩溃——根因定位与修复验证

- **日期**：2026-07-24（崩溃复现首见 2026-07-23）
- **时间**：2026-07-24 01:41 - 02:21（Asia/Shanghai，UTC+8；容器日志时区）
- **服务器/系统**：vLLM 0.25.1（`vllm/vllm-openai:v0.25.1`）容器，服务 Qwen3.6-27B-FP8，单卡 NVIDIA RTX PRO 6000 Blackwell 96GB；docker compose（`network_mode: host`/`ipc: host`），项目 `/data/qwen3.6-27b`，端点 `http://localhost:18001`
- **严重程度**：P3（k≥4/k=5 能力不可用；生产 k=3 baseline 全程未受影响，无终端用户感知）
- **影响范围**：仅 MTP 投机解码 k≥4/k=5 不可用；k=3 生产路径不受影响
- **处理人员**：Claude Code（按 AGENTS.md「下一步」#1 自主执行）

## 事件概述

在 RTX PRO 6000 Blackwell + vLLM 0.25.1 上，将 MTP 投机深度从 k=3 提升到 k=4/k=5 时，并发 ≥5 必触发 `cudaErrorIllegalAddress`（非法内存访问）并导致 EngineCore 崩溃、容器自动重启；该问题在 2026-07-23 已复现并记录。本事件为 2026-07-24 的**根因定位与修复验证**：确认根因是 **FlashInfer 后端在 spec-decode 下只支持 PIECEWISE cudagraph，k≥4 导致 graph shape 越界**；切换 `--attention-backend FLASH_ATTN`（decode 可捕获 FULL cudagraph）后 k=4/k=5 全程 0 失败、容器零重启，**崩溃已修复**。但附带两个决定性约束：① FLASH_ATTN 不支持 fp8 KV cache，k≥4 必须改 bf16 KV；② k≥4 仍不赢 k=3（acceptance 随深度递减）。**决策：生产维持 FlashInfer + fp8 KV + k=3，配置已还原并验证。** 修复价值为「k≥4 可运行」能力解锁，非当前吞吐收益。

## 时间线

| 时间（Asia/Shanghai） | 事件 | 操作/结果 |
|------|------|-----------|
| 2026-07-23（前序会话） | k=4/k=5 在 c≥5 烟测后崩溃 | `cudaErrorIllegalAddress`，容器重启；记录于 `performance.md` |
| 2026-07-23（前序会话） | `enforce-eager` / PR #46324 验证 | enforce-eager 稳定但慢 25-46%；#46324 补丁生效但仍崩（同症不同因）|
| 01:41 | 本会话：加 `--attention-backend FLASH_ATTN`（保留 fp8 KV）+ k=4 启动 | **启动失败**：`ValueError: ... kv_cache_dtype not supported`（`vllm/platforms/cuda.py:411`）|
| 01:45 | 移除 `--kv-cache-dtype fp8_e4m3`（改默认 bf16）+ FLASH_ATTN + k=4 重启 | 容器启动中（capture）|
| 01:48 | capture 完成 | 日志 `Capturing CUDA graphs (decode, FULL): 7/7`；服务 healthy。机制确认 |
| ~01:50–02:00 | FLASH_ATTN+bf16+k=4 完整 benchmark c=1/5/10 | **0 失败，RestartCount=0**。崩溃修复 |
| 01:57 | 同配置 k=3 归因 benchmark | 0 失败（控制变量：区分「k=4 本身」vs「bf16 KV 退步」）|
| 02:08 | 同配置 k=5 benchmark | 0 失败。修复对 k=5 亦成立 |
| 02:21 | 还原生产配置 FlashInfer+fp8+k=3 并重启 | healthy；日志 `decode_backend=flashinfer-native, kv_cache_dtype=torch.float8_e4m3fn`；烟测返回正常 |

## 根本原因

### 直接原因
vLLM 0.25.1 的 spec-decode + **FlashInfer 后端**在 Blackwell 上只支持 `UNIFORM_SINGLE_TOKEN_DECODE` → 自动降级 **PIEWISE cudagraph**；k≥4（每序列 ≥5 候选 token）在并发 batch 下 PIECEWISE graph 的 capture shape 与实际 spec-decode batch 不匹配，cudagraph replay 时越界 → `cudaErrorIllegalAddress`。k=3（每序列 4 候选）落在稳定 capture size，不触发。

### 根本原因
1. **后端能力错配**：FlashInfer（Blackwell 默认后端，支持 fp8 KV cache）在 spec-decode 下无法提供 FULL cudagraph；能提供 FULL cudagraph 的 FLASH_ATTN 后端又**不支持 fp8 KV cache**。二者不可兼得 → 单卡 Blackwell 上「fp8 KV + k≥4」无解。
2. **模型 acceptance 上限**：该 MTP drafter 的 acceptance 在 k=3 已接近上限，加深 k 收益递减（acceptance 0.54→0.46→0.39），即使崩溃修复也无吞吐意义。

### 触发条件
MTP `num_speculative_tokens` ≥ 4 且并发 ≥5，于 FlashInfer（默认）后端 + 默认 cudagraph 下。

## 影响分析

### 服务影响
- [ ] 服务中断：**否**（生产 k=3 baseline 全程在线；本会话期间的数次重启为**计划内测试操作**，每次 ~3min，非故障停机）
- [ ] 性能下降：**否**（k=3 生产路径未变）
- [ ] 数据丢失：**否**

### 用户影响
- 终端用户：**无感知**（k=3 服务持续）。确切的在线用户数 `待确认`（无访问日志在本事件中采集）

### 业务影响
- 业务流程：**不适用**（k≥4 非生产配置，仅为优化探索）
- 经济损失：**不适用**（无证据）

## 排查过程与数据来源

### 数据收集方法

| 数据类型 | 命令/工具 | 输出位置 | 说明 |
|----------|-----------|----------|------|
| 容器启动日志 | `docker compose logs vllm` | 终端 | attention backend / cudagraph mode / 报错栈 |
| 容器重启检测 | `docker inspect ... StartedAt/RestartCount` | 终端 | 判断是否崩溃重启 |
| 崩溃关键词 | `docker compose logs \| grep IllegalAddress/ValueError` | 终端 | 复现/消除证据 |
| benchmark 结果 | `uv run python benchmark.py --label ...` | `docs/benchmark-*.md`/`.json` | 吞吐/acceptance/失败数 |
| GPU 显存 | `nvidia-smi --query-gpu=memory.used` | 终端 | 排除 OOM |
| 健康检查 | `curl http://localhost:18001/health` | 终端 | 服务状态 |
| 历史记录 | `docs/performance.md`、前序 benchmark 报告 | 仓库 | 崩溃历史、enforce-eager/#46324 结论 |

### 关键发现

**发现1：FLASH_ATTN 拒绝 fp8 KV cache（新发现，阻塞）**
- 数据来源：`docker compose logs`（`vllm/platforms/cuda.py:411 get_attn_backend_cls`）
- 数据内容：`ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this configuration. Reason: ['kv_cache_dtype not supported']`
- 分析结论：原假设（切 FLASH_ATTN 绕过 PIECEWISE）在现有 fp8 KV 配置下**连加载都过不了**；必须先放弃 fp8 KV（改 bf16）才能验证。

**发现2：FLASH_ATTN 使 decode 获得 FULL cudagraph（机制确认）**
- 数据来源：`docker compose logs`（capture 阶段）
- 数据内容：`Profiling CUDA graph memory: PIECEWISE=12 (largest=90), FULL=7 (largest=50)`；`Capturing CUDA graphs (decode, FULL): 7/7`（k=4）、`8/8`（k=5）
- 分析结论：FLASH_ATTN 对 decode 捕获 FULL cudagraph（FlashInfer 仅 PIECEWISE）。FULL 覆盖完整 spec-decode batch，无 PIECEWISE shape 越界 → 崩溃根因被消除。

**发现3：k≥4 崩溃已消除（运行时验证）**
- 数据来源：`docker inspect` RestartCount + benchmark 失败计数
- 数据内容：FLASH_ATTN+bf16 下 k=4/k=5 × c=1/5/10 全程 `RestartCount=0`、`failed=0`；无 `cudaErrorIllegalAddress`
- 分析结论：修复对 k=4、k=5 均成立，且非「烟测假阴性」（完整 benchmark 全绿）。

**发现4：k≥4 仍不赢 k=3（收益判定）**
- 数据来源：`docs/benchmark-flashattn_k4_full.md` / `_k3.md` / `_k5.md` + 既有 baseline
- 数据内容：acceptance k3=0.54 / k4=0.46 / k5=0.39；同配置下 k=3 吞吐最优（c=10: 672.9 > 665.7 > 620.0）
- 分析结论：加深 k 无吞吐收益，与 k=3 acceptance 上限一致。

**发现5：崩溃非显存问题**
- 数据来源：`performance.md`（2026-07-23 复测）+ 本会话 `nvidia-smi`
- 数据内容：崩溃时 KV cache usage ~7.8%(k=4)/9.2%(k=5)，稳态显存 85.4/97.9GB 有余量
- 分析结论：排除 OOM，确认为 cudagraph replay 越界。

## 解决方案

### 紧急处理
1. 不适用——生产 k=3 全程未中断；本事件为**计划内优化探索**，无故障停机需紧急处理。

### 长期修复
1. **k≥4 崩溃修复（已验证）**：`--attention-backend FLASH_ATTN` + 放弃 `--kv-cache-dtype fp8_e4m3`（bf16 KV）。代价：KV cache 带宽翻倍、可用 KV block 数减半（96GB 下容量仍充足）。复现命令见 `docs/benchmark-flashattn-k4-k5-analysis.md` §5。
2. **生产决策（已执行）**：因 k≥4 无吞吐收益且须牺牲 fp8 KV，**生产维持 FlashInfer + fp8 KV + k=3**；`docker-compose.yml` 已还原 baseline 并验证 healthy。

## 建议措施

### 短期（1-3天）
- [x] 生产配置还原并验证（已完成，`docker-compose.yml` = baseline）
- [ ] （可选）若临时实验需 k≥4：用 FLASH_ATTN + bf16 KV + k=4；启动日志须见 `Capturing CUDA graphs (decode, FULL)` 方算就绪

### 中期（1-2周）
- [ ] 仅当 drafter acceptance 量级跃迁（如自训 DSpark）或多卡/disaggregation 提供 overlap 空隙时，再评估 k≥4 的吞吐价值
- [ ] 关注 vLLM 上游 FlashInfer spec-decode FULL cudagraph 支持（若实现，可同时保留 fp8 KV + k≥4）

### 长期（1-3月）
- [ ] 跟踪 vLLM PIECEWISE cudagraph 在 k≥4 边界的修复（本崩溃与 PR #46324 同症不同因，待上游覆盖）
- [ ] 若引入多卡，重测 DSpark/更高 k（单卡 GPU 占满是 drafter 无 overlap、k≥4 无收益的结构性原因）

## 经验教训

### 做得好的
- 根因定位严谨：`enforce-eager`（前序）已证「cudagraph replay 越界」非 draft 逻辑 bug；本会话进一步定位到「FlashInfer PIECEWISE」并捕获 FULL cudagraph 实证闭环。
- 控制变量归因：在同新配置下补测 k=3，干净区分「k=4 本身」与「bf16 KV 配置退步」，避免误判。
- 决策务实：修复成功但无收益时，明确维持 baseline 并记录「能力解锁」而非强行上线。

### 需要改进的
- FLASH_ATTN ↔ fp8 KV cache 的不兼容是 Blackwell 上 k≥4 的隐性前提，此前未在文档中显式标注，易让后续操作者重蹈「直接加 backend 即崩」的弯路（已补记于 `docs/benchmark-flashattn-k4-k5-analysis.md` §1.2 与 memory）。
- `benchmark.py` 的 per-position acceptance 解析硬编码 `range(4)`（k=3 口径），k=5 的 pos4 未采集；总体 acceptance_rate 仍准确，但 per-pos 对 k≥5 不完整。

### 后续行动
- k≥4 可运行路径已记录，待「acceptance 跃迁 / 多卡」前提满足时复用。
- DSpark 自训 drafter 可行性已调研（`docs/dspark-selftrain-feasibility.md`），其推进需停在线服务 ~1 天，待决策。

## 参考资料

- 综合分析：[`docs/benchmark-flashattn-k4-k5-analysis.md`](benchmark-flashattn-k4-k5-analysis.md)
- 原始 benchmark：[`docs/benchmark-flashattn_k4_full.md`](benchmark-flashattn_k4_full.md) / [`benchmark-flashattn_k3.md`](benchmark-flashattn_k3.md) / [`benchmark-flashattn_k5.md`](benchmark-flashattn_k5.md)
- 性能追踪：`docs/performance.md`「MTP k≥4 崩溃修复」章节 / 「MTP k≥4 验证」章节
- 历史崩溃记录：`docs/benchmark-rtx_pro_6000_vllm0251_mtp_k4_smoke.md` / `_k5_smoke.md`；`docs/benchmark-mtp_k4.md`；`docs/benchmark-mtp-k4-pr46324.md`
- vLLM：`vllm/platforms/cuda.py:411`（backend 校验）、`vllm/model_executor/models/qwen3_dspark.py`（原生 DSpark 后端）
