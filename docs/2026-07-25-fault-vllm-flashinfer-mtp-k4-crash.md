# [故障事件] vLLM FlashInfer + MTP k≥4 在 SM120 崩溃（混合架构大页越 spec，不可实际修复）

- **事件日期**：2026-07-24 ~ 2026-07-25（复现 + 根因核证）
- **报告日期**：2026-07-25
- **服务器/系统**：本项目 vLLM 部署容器（`vllm/vllm-openai:v0.25.1`，compose `docker-compose.yml`）
- **相关模型 / 硬件**：`Qwen/Qwen3.6-27B-FP8`（**混合架构** `model_type=qwen3_5`：64 层 = 48 `linear_attention`(GDN/SSM) + 16 `full_attention`，`full_attention_interval=4`，`head_dim=256`，`num_attention_heads=24`，`num_key_value_heads=4`，自带 `mtp_num_hidden_layers=1`）；NVIDIA RTX PRO 6000 Blackwell（**SM120**，96GB）。
- **严重程度**：P3（**非生产故障**——生产为 k=3 + FlashInfer + fp8，稳定无影响；本事件=k≥4 吞吐路径在 FlashInfer 下不可用的能力/根因核证）
- **影响范围**：任何对本模型启用 `--attention-backend flashinfer`（默认）+ `num_speculative_tokens≥4` + 并发 ≥5 的部署。k=3 不受影响。
- **处理人员**：本项目（复现 + 源码核证 + 双向对比 sglang）

## 事件概述

在评估「能否不走 `triton_attn`、直接修好 vLLM 的 FlashInfer 让 MTP k≥4 稳跑」时，在 **v0.25.1 / SM120** 上稳定复现了崩溃：`--attention-backend flashinfer`（默认）+ `--kv-cache-dtype fp8_e4m3` + `num_speculative_tokens=4`，并发 `c=1` 稳定、**`c=10` 全崩**（10/10 请求失败、`cudaErrorIllegalAddress`、容器 Exited）。

经复现 traceback + 双 agent 源码核证 + 排除实验，**根因不是 cudagraph、也不是 FlashInfer workspace**，而是**混合架构的 GDN/Mamba 状态页（~3.3MB）强制 attention `block_size=1616`**（`vllm/platforms/interface.py:_align_hybrid_block_size`，attention 页必须 ≥ mamba 页、共享 block table、不可独立配置）。这个 1616-token 大页喂给 FlashInfer **native fa2 `BatchPrefillWithPagedKVCache`** 内核**越 spec**——vLLM 自家 gate（`flashinfer.py:748`）只允许 `page_size>64` 走 **SM100 trtllm-gen** 路径，而 **SM120 没有 trtllm-gen cubin**，只能 native fa2 硬扛 1616 页，高并发/k≥4 下越界。

参考 sglang（用**同一个** `BatchPrefillWithPagedKVCache` kernel）确认：sglang 不崩是因为其 `HybridLinearAttnBackend` 把 full-attention 分页设成 **`page_size=1`**（in-spec）并**与 GDN 状态分页解耦**；vLLM 把二者**耦合**（共享 block_size=1616）。**定论：本 hybrid 模型 + SM120 下，FlashInfer native fa2 在 k≥4 无实际可修路径**（除非 ① NVIDIA 出 SM120 trtllm-gen cubin，或 ② vLLM 改成 sglang 式按层组解耦分页——架构级改动，非 patch）。务实正确解仍是 `triton_attn`（已实测保 fp8 + k≥4 稳）。

## 时间线

| 时间 | 事件 | 操作/结果 |
|------|------|-----------|
| 2026-07-23 | 项目首次发现 k≥4 崩溃 | 切 `--attention-backend FLASH_ATTN` 绕过（弃 fp8 KV）；结论记为「k≥4 不赢 k=3」。**后续核证：该修复同时改了 backend+kv dtype 两变量，属 confound。** |
| 2026-07-24 | triton_attn 修复验证 | `triton_attn`(target+draft) + fp8 + k=4：c=1/5/10 全程 0 失败 0 崩溃（保住 fp8）。见 [`docs/performance.md`](performance.md)「MTP k≥4 崩溃修复（真因 + triton_attn）」。 |
| 2026-07-25 00:00 | 复现 v0.25.1 FlashInfer k=4 崩溃 | 起 `docker-compose.k4-flashinfer.yml`（FlashInfer + fp8 + k=4，默认 workspace）。 |
| 2026-07-25 00:09:47 | 崩溃触发（容器日志时间戳） | benchmark c=10 → `EngineDeadError: cudaErrorIllegalAddress`，容器 Exited。c=1=0 失败 / c=5=降级 / **c=10=10 失败**。 |
| 2026-07-25 00:17 | workspace 排除实验 | 加 `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824`（1GiB，默认 394MiB）重测：**c=10 k=4 仍 10/10 崩溃** → 排除 #48428（workspace 放大）为修复。 |
| 2026-07-25 | 双 agent 源码核证 | vLLM 侧（verify kernel/调用点/seq_lens 重构定位）+ sglang 侧（同 kernel + page_size=1 + 解耦分页）+ metadata 对比 → 定位 `block_size=1616` 强制逻辑。 |
| 2026-07-25 | 恢复生产基线 | 容器切回 `docker-compose.yml`（k=3 + FlashInfer + fp8）。 |

## 根本原因

### 直接原因

FlashInfer **native fa2 `BatchPrefillWithPagedKVCache`** 内核在 spec-decode verify 阶段越界（`cudaErrorIllegalAddress`）。v0.25.1 实测 traceback（容器 `/usr/local/lib/python3.12/dist-packages/vllm/`）：

```
gpu_model_runner.py:4282 execute_model
→ gpu_model_runner.py:2553 _build_attention_metadata
→ gpu_model_runner.py:2487 _build_attn_group_metadata
→ flashinfer.py:1176 build
→ backend.py:493 seq_lens_cpu        ← GPU→CPU sync，在此抛 cudaErrorIllegalAddress
→ torch.AcceleratorError: an illegal memory access was encountered
```

**注意**：`seq_lens_cpu`（`vllm/v1/attention/backend.py:493`，已 deprecated 的 `seq_lens.cpu()` 同步）是**异步检测点**，不是发生点——真实 OOB 在**上一步**的 `BatchPrefillWithPagedKVCache` kernel 里异步发生，到此处 sync 才暴露（与上游 [issue #37754](https://github.com/vllm-project/vllm/issues/37754) 描述一致）。

### 根本原因

**① hybrid 模型强制 attention `block_size=1616`。** 模型有 48 层 GDN/Mamba（需逐 token 追踪状态，状态页 ~3.3MB）。vLLM 的 hybrid paging 要求 attention 页 ≥ mamba 页且**共享同一个 block table**，由 `vllm/platforms/interface.py:_align_hybrid_block_size` 强制对齐：

```python
attn_block_size = kernel_block_alignment_size * cdiv(
    mamba_page_size, kernel_block_alignment_size * attn_page_size_1_token)
# → 1616 tokens（完全由 mamba_page_size 决定，不可独立配置，非 bug）
```

容器启动日志确认：`[interface.py:891] Setting attention block size to 1616 tokens to ensure that attention page size is >= mamba page size.`

**② 1616 大页对 FlashInfer native fa2 越 spec。** vLLM 自家 gate（`vllm/v1/attention/backends/flashinfer.py:748`）只允许 `page_size>64` 走 **SM100 trtllm-gen GQA** 路径。但 **SM120 不在 trtllm-gen 支持范围**（`vllm/utils/flashinfer.py:394` `is_device_capability_family(100)`：SM120→`to_int()//10=12≠10` → False）→ trtllm-gen 不可用 → 只能 native fa2 扛 1616 页 → 越界。每 block 元素 stride ≈ `2 × 1616 × num_kv_heads(4) × head_dim(256)`，block_id × stride 随并发/k 增大逼近/超出 int32 表示范围，解释了并发与 k 的依赖性。

### 触发条件

四者同时成立即触发：

1. **混合架构模型**（含 GDN/Mamba 层，触发 attention 大页对齐）——纯 Transformer 不会强制 1616 页；
2. **FlashInfer attention backend**（默认；native fa2 `BatchPrefillWithPagedKVCache` 路径）；
3. **`num_speculative_tokens ≥ 4`**（k=3 不触发、k≥4 触发——更长 derived KV → 更高 block_id / 更多页）；
4. **并发 ≥5**（c=1 稳、c=10 崩——更多请求 → 更高 pool block-id 量级）。

> 注：fp8 KV cache 会令 `attn_page_size_1_token` 更小（fp8=1 byte vs bf16=2 byte），从而令强制 block_size 更大——但非根因（bf16 下 block_size 仍 >64、仍越 spec）。

## 影响分析

### 服务影响

- [ ] 服务中断：**否**。生产配置为 k=3 + FlashInfer + fp8，全程稳定，未受影响。
- [ ] 性能降级：**否**（生产未启用 k≥4）。
- [ ] 数据丢失：**否**（崩溃发生在推理期，无持久化数据）。

### 用户影响

- 生产用户：**无影响**（k=3 稳定服务）。
- 评估/开发：k≥4 吞吐路径在 FlashInfer 下不可用；需切 `triton_attn` 才能跑 k≥4。

### 业务影响

- 经济损失：**不适用**（无生产故障）。
- 路线影响：确认「单卡 + 现有 drafter + FlashInfer」下 k≥4 不可行；本事件与 SGLang 评估（[`docs/2026-07-24-fault-sglang-qwen35-mtp-specv2-crash.md`](2026-07-24-fault-sglang-qwen35-mtp-specv2-crash.md)）共同构成「混合架构 spec-decode 是两大引擎前沿高风险区」的本地实证。

## 排查过程与数据来源

### 数据收集方法

| 数据类型 | 命令/工具 | 来源 | 说明 |
|----------|-----------|------|------|
| 模型架构 | 读 `/data/models/Qwen3.6-27B-FP8/config.json` | 本地文件 | 确认 `model_type=qwen3_5`、48 linear + 16 full、head_dim=256 |
| 崩溃复现 | `docker compose -f docker-compose.k4-flashinfer.yml up` + `benchmark.py --concurrency 1,5,10` | 容器日志 + `docs/benchmark-mtp_k4_flashinfer_CRASH.{md,json}` | c=10 全崩、traceback |
| 排除实验 | `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=1073741824` 重测 | `docs/benchmark-mtp_k4_flashinfer_ws1g.{md,json}` | workspace 放大不修复 |
| triton 验证 | `docker-compose.k4-triton.yml` / `k3-triton.yml` + benchmark | `docs/benchmark-mtp_k4_triton.{md,json}` / `benchmark-mtp_k3_triton.*` | triton_attn 保 fp8 + 0 崩溃 |
| 源码（vLLM） | 读 `/data/vllm`（main，a454a1dd2）+ 容器 v0.25.1 镜像 | `flashinfer.py`、`platforms/interface.py`、`attention/backend.py`、`utils/flashinfer.py`、`envs.py` | 定位 verify 调用点、gate、强制对齐逻辑、trtllm-family 判定 |
| 源码（sglang） | 读 `/data/sglang` | `flashinfer_backend.py`、`hybrid_linear_attn_backend.py`、`kernels/ops/kvcache/kv_indices.py`、`server_args.py`、`attention_registry.py` | sglang 用同 kernel + page_size=1 + 解耦分页 |
| 上游对照 | `gh issue/pr view` vllm-project/vllm | GitHub API | #37754 / #36613 / #48428 |

### 关键发现

**发现 1：崩溃真实可复现，并发 + k 双依赖**
- 数据来源：`docs/benchmark-mtp_k4_flashinfer_CRASH.json` + 容器日志（`00:09:47 cudaErrorIllegalAddress`）。
- 数据内容：k=4 FlashInfer fp8：c=1 = 0 失败（82.2 tok/s）；c=5 = 0 失败但降级（57.4）；**c=10 = 10/10 失败（0.0 tok/s）**，容器 Exited。
- 结论：c=1 稳 / c=10 崩 / k=4 触发——与上游 #37754（同款 SM121）现象同构。

**发现 2：排除 workspace（#48428 不适用）**
- 数据来源：`docs/benchmark-mtp_k4_flashinfer_ws1g.json`；默认值 `vllm/envs.py:202`（`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE = 394*1024*1024`）。
- 数据内容：workspace 设 1GiB（> #48428 公式 `8192×24×256×16≈768MiB`），c=10 k=4 **仍 10/10 崩溃**。
- 结论：上游 #48428（按头足迹放大 workspace）**不是**本崩溃修复——「一个 env 变量搞定」的捷径堵死。

**发现 3：真因 = hybrid 强制 attention `block_size=1616`**
- 数据来源：容器日志 `[interface.py:891] Setting attention block size to 1616`；源码 `vllm/platforms/interface.py:_align_hybrid_block_size`（`attn_block_size = kernel_align * cdiv(mamba_page_size, kernel_align*attn_page_size_1_token)`）。
- 数据内容：block_size=1616 完全由 GDN/Mamba 状态页决定，attention 页与 mamba 页共享 block table、不可独立配置。
- 结论：1616-token 大页是混合架构的硬约束产物，喂给 FlashInfer native fa2 越出其设计 spec。

**发现 4：SM120 无 trtllm-gen，无法走大页安全路径**
- 数据来源：`vllm/utils/flashinfer.py:394`（`is_device_capability_family(100)` 对 SM120 返回 False）；`vllm/v1/attention/backends/flashinfer.py:748`（gate：page>64 仅 SM100 trtllm-gen）；`get_cudagraph_support` 返回 `UNIFORM_SINGLE_TOKEN_DECODE`（启动日志确认 `setting cudagraph_mode=PIECEWISE`）。
- 结论：SM120 既无 trtllm-gen cubin 走大页安全路径，又被迫 PIECEWISE cudagraph，是 FlashInfer 在本卡的脆弱性来源。

**发现 5：sglang 用同一 kernel 但 page_size=1（解耦分页）**
- 数据来源：sglang `/data/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py:1312`（`prefill_wrapper_paged.forward()` = `BatchPrefillWithPagedKVCache`）；`hybrid_linear_attn_backend.py:952-968`（full-attn 分发到 FlashInfer）；`arg_groups/overrides.py:1944`（`page_size=1` 默认）；`server_args.py:5447-5451`（SM120 默认 flashinfer）。
- 数据内容：sglang 的 full-attention verify 走**同一个** `BatchPrefillWithPagedKVCache`，但分页 `page_size=1`，与 GDN/linear 状态分页**解耦**（`HybridLinearAttnBackend`）。
- 结论：sglang 不崩不是靠换 kernel，而是靠**小页 + 解耦分页 + 边界 mask 的 metadata**（`kv_indices.py:36-44` mask、`cache_locs.py:171-173` 零填充）。

## 解决方案

### 紧急处理（已做）

1. 生产维持 `FlashInfer + fp8 + k=3`（`docker-compose.yml`，k=3 不触发，稳定）。
2. k≥4 / 稳定性裕度场景：切 `triton_attn`（`docker-compose.k4-triton.yml`），保 fp8 KV、给 FULL cudagraph、实测 c=1/5/10 全程 0 失败。

### 长期修复（本 hybrid 模型 + SM120 下，FlashInfer 路径）

**定论：无实际可修路径**（非 patch 级别）。两条理论上可行的路均不在本项目可控范围：

1. **等待 NVIDIA 发布 SM120 trtllm-gen cubin**——届时 vLLM 可走 trtllm-gen 大页安全路径（gate `flashinfer.py:748` 已就绪）。外部依赖，时间不定。
2. **vLLM 采纳 sglang 式按层组解耦分页**——让 full-attention 用小页（如 page_size=1）、与 GDN 状态分页解耦。属 vLLM 架构级改动（重设计 hybrid block table / paging），非短期可行。

> 备选（已评估，不推荐）：仅做 metadata null-pad（把 `paged_kv_indices` 尾部填 0，对齐 sglang 的 `zeros_like` 尾处理）。但因根因是 page_size 越 spec（非 metadata 形状），此 patch 属「必要但不充分」，agent 评估全修概率 ~30%，未采用。

## 建议措施

### 短期（1–3 天）

- [ ] 生产继续 `FlashInfer + fp8 + k=3`，**不引入 k≥4**（acceptance 随深度递减本就不赢 k=3）。
- [ ] 若未来需要 k≥4（如自训 drafter 把 acceptance 量级提上去），用 `docker-compose.k4-triton.yml`（triton_attn），勿用 FlashInfer。

### 中期（1–2 周）

- [ ] 跟踪 vLLM 上游 SM120 trtllm-gen 支持进展（关注 `vllm/utils/flashinfer.py` 的 family 判定与 NVIDIA cubin 发布）；一旦可用，重测 FlashInfer k=4 是否随之稳定。
- [ ] 跟踪 vLLM hybrid 解耦分页的相关 PR（若出现）。

### 长期（1–3 月）

- [ ] 把「混合架构模型在 Blackwell 单卡的 spec-decode 稳定性」列为持续风险项（vLLM FlashInfer k≥4 + SGLang spec-v2 均命中），定期复核两大引擎进展。
- [ ] 翻转 k≥4 收益的前提仍是「acceptance 量级跃迁（自训 drafter）」或「多卡/disaggregation 给 drafter overlap 空间」——与引擎/backend 选择无关。

## 经验教训

### 做得好的

- **不靠推断，靠实验排除**：对 workspace 假设（上游 #48428 + vLLM agent 一致指向）做了实测（1GiB），用数据否证——避免采纳一个看似合理实则无效的「修复」。
- **双向对比 sglang 源码**：确认 sglang 用的是**同一个 kernel**，从而把「为何 sglang 不崩」锁定在**分页架构差异**（page_size=1 + 解耦），而非 kernel 选择——这直接指向「vLLM 侧需解耦分页」的定论。
- **拿到的 traceback 区分了「检测点」与「发生点」**：明确 `seq_lens_cpu` 是异步 sync 检测点，真实 OOB 在前一步 kernel，避免误把同步逻辑当根因。

### 需要改进的

- 项目 2026-07-23 首次「修复」（切 FLASH_ATTN）**同时改了 backend 与 kv dtype 两个变量**，属 confound，归因「FULL cudagraph」不严谨。本次已更正，但应在此类多变量对比中**单变量隔离**。
- 混合架构模型（qwen3_5）的 hybrid paging 约束（attention 大页对齐）此前未被纳入 k≥4 崩溃的归因——应在引入新混合架构模型时预先评估其分页对 attention backend 选型的影响。

### 后续行动

- 关闭本事件的「能否修好 FlashInfer」问题：**结论=本模型不可实际修复，triton_attn 是等价正确解**。
- 持续跟踪 SM120 trtllm-gen cubin 与 vLLM 解耦分页上游进展；出现后重测。

## 参考资料

- 上游 issue（本事件同类）：[#37754](https://github.com/vllm-project/vllm/issues/37754) FlashInfer + MTP crashes on SM121；[#36613](https://github.com/vllm-project/vllm/issues/36613) MTP illegal memory access；[#40756](https://github.com/vllm-project/vllm/issues/40756)。
- 上游 PR（排除项）：[#48428](https://github.com/vllm-project/vllm/pull/48428) FlashInfer prefill workspace sizing（实测不修复本事件）。
- vLLM 源码：`vllm/platforms/interface.py:_align_hybrid_block_size`（强制对齐）、`vllm/v1/attention/backends/flashinfer.py:748`（大页 gate）、`vllm/utils/flashinfer.py:394`（trtllm-family 判定）、`vllm/v1/attention/backend.py:493`（seq_lens_cpu 检测点）。
- sglang 源码：`flashinfer_backend.py:1312`（同 kernel）、`hybrid_linear_attn_backend.py`（解耦分页）、`kv_indices.py`（边界 mask）。
- 本项目相关：[`docs/performance.md`](performance.md)「深层根因」小节、[`docs/2026-07-24-fault-sglang-qwen35-mtp-specv2-crash.md`](2026-07-24-fault-sglang-qwen35-mtp-specv2-crash.md)（跨引擎对照）、`docker-compose.k4-flashinfer.yml`（复现）/ `docker-compose.k4-triton.yml`（triton 修复）。
