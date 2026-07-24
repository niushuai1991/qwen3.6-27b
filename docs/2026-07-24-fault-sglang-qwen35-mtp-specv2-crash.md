# [故障事件] SGLang Qwen3.6-27B-FP8 + MTP spec-v2 warmup 崩溃（issue #31249 / 修复 PR #27998）

- **事件日期**：2026-07-15（上报） ~ 2026-07-16（修复合并）
- **报告日期**：2026-07-24
- **服务器/系统**：**上游 SGLang**（`sgl-project/sglang`）。⚠️ 本项目尚未部署 SGLang（当前生产为 vLLM 0.25.1 + MTP k=3）；本事件为评估 SGLang 替代方案时核实的**上游已知故障 + 修复**，记录其对本项目部署计划的交叉影响。
- **相关模型 / 硬件**（上游复现环境）：`Qwen/Qwen3.6-27B-FP8`（与本项目同款）；8× NVIDIA B200（Compute Capability 10.0 / SM100）。本项目目标硬件为 **RTX PRO 6000 Blackwell（SM120）单卡**——与上游环境不同，详见「影响分析」。
- **严重程度**：P2（上游事件；对本项目=部署评估阻塞，非生产故障）
- **影响范围**：任何在混合架构模型（qwen3_5 / NemotronH 等）上启用 **spec-v2（overlap 调度）+ radix cache（前缀缓存）** 的 SGLang 部署，warmup 即必崩。
- **处理人员**：上游——修复作者 Serge Panev（`Kh4L`）；上报人 Daya Khudia（`dskhudia`）；指引者 `violet73`。本项目——核实与记录。

## 事件概述

SGLang 在用 `Qwen/Qwen3.6-27B-FP8` + NEXTN（MTP）投机解码启动时，于 warmup 阶段崩溃：`TypeError: 'NoneType' object cannot be interpreted as an integer`，栈底落在 `schedule_batch.py:1593 set_mamba_track_indices_from_reqs`。根因是该函数假设每个 request 都有有效的 `mamba_next_track_idx`，但在 **Spec-v2 verify 路径**上，部分 request 未经过 `_alloc_ping_pong_buffer`，该字段为 `None`，传给 `torch.tensor(...)` 即抛异常。崩溃由 **spec-v2（overlap scheduler）+ radix cache（extra_buffer）+ cache-eligible 请求**三者组合触发，warmup 的并发请求即可命中。

上游在同日给出修复 PR #27998（`guard None → 0`），上报人确认有效，次日（2026-07-16）合并、issue 关闭。**但该修复合并于 2026-07-16，晚于最新稳定版 `v0.5.15.post1`（2026-07-14）——所有稳定 release 均不含此修复**，目前仅存在于 `main`/dev 构建。对本项目而言：若用稳定版 SGLang 评估，目标配置（spec-v2 + radix cache + 本模型）必崩；必须使用含合并 commit `1c4892d7` 的 dev 镜像或源码构建。

## 时间线

| 时间（UTC） | 事件 | 操作/结果 |
|------|------|-----------|
| 2026-07-15 01:18:28 | 崩溃发生（traceback 内时间戳） | warmup 序列触发 `TypeError`，scheduler 异常退出 |
| 2026-07-15 01:28:20 | 上报 issue #31249 | `dskhudia` 提交复现命令与环境；checklist 中「最新版仍存在」未勾选 |
| 2026-07-15 06:14:45 | 指引修复 | `violet73` 建议「You can try PR #27998」 |
| 2026-07-15 06:55:10 | 上报人确认修复 | `dskhudia`：「Just confirmed that PR #27998 fixes the issue」 |
| 2026-07-16 01:23:15 | 修复合并 | PR #27998 MERGED（merge commit `1c4892d7`） |
| 2026-07-16 01:23:16 | issue 关闭 | #31249 自动关闭 |
| 2026-07-24 | 本项目核实 | 评估 SGLang 时核实事件 + 发现稳定版版本 gap |

## 根本原因

### 直接原因

`set_mamba_track_indices_from_reqs`（`sglang/srt/managers/schedule_batch.py:1593`）将每个 request 的 `mamba_next_track_idx` 传入 `torch.tensor(...)`，但该字段对部分 request 为 `None` → `TypeError: 'NoneType' object cannot be interpreted as an integer`。完整调用链：

```
scheduler.run_event_loop → event_loop_overlap（spec-v2 / overlap 调度）
→ run_batch → eagle_worker_v2.forward_batch_generation → verify
→ eagle_prepare_for_verify → prepare_mamba_track_for_verify
→ set_mamba_track_indices_from_reqs → torch.tensor(None)  # 崩
```

### 根本原因

`Qwen3.6-27B` 为**混合架构**（本项目读 config.json 确认：`model_type=qwen3_5`，`layer_types` = `3× linear_attention + 1× full_attention` 循环，共 64 层 = 48 linear + 16 full，`full_attention_interval=4`，含 `mtp_num_hidden_layers` 自带 MTP 头）。其中 **48 层线性注意力（SSM/Mamba 族）需逐 token 追踪状态**（`mamba_track` / ping-pong buffer）。投机解码 verify 阶段需为 draft token 分配这些 SSM 状态 buffer，但在 **spec-v2 + radix cache 的 cache-eligible 快速路径**上，部分 request 走了未调用 `_alloc_ping_pong_buffer` 的分支，导致状态索引缺失（`None`），且调用点未做 None guard。

修复（PR #27998）即在此处将 `None` guard 为 `0`。

### 触发条件

四者同时成立即触发：
1. **混合架构模型**（含 SSM/线性注意力层，如 qwen3_5、NemotronH）——纯 Transformer 不会走 `mamba_track` 路径；
2. **spec-v2（overlap scheduler）**——SGLang 默认、也是其性能招牌的投机调度路径；
3. **radix cache（extra_buffer）开启**——SGLang 默认开（等价 vLLM 的 prefix caching）；
4. 出现 **cache-eligible 请求**——warmup 的并发/重复请求即可命中，无需真实业务流量。

> 注：上游同源 fix 还修复了 #27325（生产环境 Qwen3.5 命中同类问题）。

## 影响分析

### 对上游 SGLang 用户

- [x] 服务中断：是。目标配置（混合模型 + spec-v2 + radix cache）启动 warmup 阶段必崩，服务不可用。
- [ ] 性能下降：不适用（直接崩溃，非降级）。
- [ ] 数据丢失：不适用（启动期，无业务数据）。

### 对本项目（交叉影响）

- [x] **部署评估阻塞**：是。本项目评估 SGLang 的目标正是「用 NEXTN/MTP 跑更高 k、且保留 radix cache（前缀缓存）+ spec-v2（overlap）」——恰好是会崩的配置。稳定版 SGLang 无法承载此评估。
- [ ] 生产服务影响：否（当前生产为 vLLM k=3，与本事件无关，未受影响）。
- [ ] 用户/业务影响：不适用（评估阶段，无在线 SGLang 服务）。

### 关键不确定性（未验证项）

| 维度 | 上游已验证 | 本项目待验证 |
|------|-----------|------------|
| 模型 | `Qwen3.6-27B-FP8`（reporter 确认修复有效） | 同款模型 ✅ |
| GPU 架构 | B200（SM100），PR 另在 GB10（SM121）验证 | **RTX PRO 6000（SM120）——未验证** ⚠️ |
| 卡数 | `--tp 2`（多卡） | **单卡**——未验证 ⚠️ |
| 缓存路径 | radix cache + spec-v2 | 同 ✅ |

> 即：修复对「同模型 + radix cache + spec-v2」有效已确认；但对「SM120 单卡」这一本项目实际环境，无任何上游验证记录。仅 smoke test 可关闭此项。

## 排查过程与数据来源

### 数据收集方法

| 数据类型 | 命令/工具 | 来源 | 说明 |
|----------|-----------|------|------|
| Issue 详情 | `gh issue view 31249 --repo sgl-project/sglang --json ...` | GitHub API | 标题/状态/时间/正文/评论/环境段，全文核实 |
| PR 详情 | `gh pr view 27998 --repo sgl-project/sglang --json ...` | GitHub API | 标题/状态/合并时间/merge commit/body |
| 版本时间线 | `gh release list --repo sgl-project/sglang --limit 15` | GitHub API | 判断修复是否进入稳定版 |
| 架构支持 | `gh search code --repo sgl-project/sglang "qwen3_5"` | GitHub API | 确认 SGLang 原生支持本架构 |
| 本模型架构 | 读 `/data/models/Qwen3.6-27B-FP8/config.json` | 本地文件 | 确认混合架构（解释为何走 mamba_track） |

### 关键发现

**发现 1：崩溃真实、已修、reporter 确认**
- 数据来源：issue #31249 评论（2026-07-15 06:55:10Z）
- 数据内容：「@violet73 : Thanks. Just confirmed that PR #27998 fixes the issue.」
- 结论：本模型 + NEXTN + spec-v2 + radix cache 的 warmup 崩溃，经上报人确认由 PR #27998 修复。

**发现 2：修复不在任何稳定版（版本 gap）**
- 数据来源：PR #27998 `mergedAt=2026-07-16T01:23:15Z`；release 列表最新稳定版 `v0.5.15.post1=2026-07-14`、`v0.5.15=2026-07-10`。
- 结论：合并时间晚于最新稳定版 2 天 → 稳定 release 均不含修复；修复仅在 `main`/dev（merge commit `1c4892d7` 之后）。

**发现 3：本模型是混合架构，解释崩溃路径**
- 数据来源：`config.json` → `layer_types` = `3× linear_attention + 1× full_attention`（interleave 4），含 `mamba_ssm_dtype` / `linear_conv_kernel_dim` / `mtp_num_hidden_layers`。
- 结论：48 层 SSM 族线性注意力需逐 token 状态追踪，故 spec-decode verify 走 `mamba_track` 路径——纯 Transformer 不会命中。这也解释了 vLLM 侧 k≥4 的脆弱性同属「混合架构 spec-decode」前沿问题。

**发现 4：SGLang 原生支持本架构 + MTP**
- 数据来源：`gh search code` 命中 `python/sglang/srt/models/qwen3_5_mtp.py`（"Inference-only Qwen3_5 MTP model"）、`configs/qwen3_5.py`（`model_type=qwen3_5`，`Qwen3_5TextConfig(Qwen3NextConfig)`）。
- 结论：SGLang 对本模型 + 自带 MTP 头是一等公民支持，非 hack；崩溃是 spec-v2 缓冲逻辑 bug，非架构缺失。

## 解决方案

### 上游修复

PR #27998 `[Mamba] Fix spec-v2 + extra_buffer crash (guard None mamba_next_track_idx)`：在 `set_mamba_track_indices_from_reqs` 调用点将 `mamba_next_track_idx` 为 `None` 时 guard 为 `0`（+7 / −1）。已 MERGED，merge commit `1c4892d7bb5f0967aa54cd67b4923493f850a36e`。

### 本项目可执行对策（择一）

1. **（推荐）使用含 `1c4892d7` 的构建**：官方 `lmsysorg/sglang:dev-cu13`（dev 滚动标签，现拉取应含修复；启动需核验容器内 sglang git commit ≥ `1c4892d7`），或源码在 `1c4892d7` 上用 CUDA 13 base 自建。SM120 需 CUDA 13 构建。
2. **（保底绕过）关闭 radix cache**：`--disable-radix-cache` 可规避 cache-eligible 路径，理论上避免触发；但① warmup 是否仍崩未确认 ② 丢失前缀缓存（本项目在用）③ 为「阉割配置」，评估不公平。

## 建议措施

### 短期（本次 smoke test）

- [ ] SGLang 评估镜像选含 `1c4892d7` 的构建（dev-cu13 或源码 @ commit），启动后**显式核验容器内 sglang commit** ≥ `1c4892d7` 再开始 benchmark。
- [ ] smoke test 启动后，先跑最小 warmup（复现 issue 中的并发 warmup 脚本思路）确认本卡（SM120 单卡）不崩，再进入 k 扫描。
- [ ] 明确「SM120 单卡」为本项目独有未验证项，结果需单独记录。

### 中期（1–2 周）

- [ ] 跟踪 #27998 进入哪个稳定 release（关注 `sgl-project/sglang` release）；进入后切换到固定稳定版镜像，避免长期依赖 dev 滚动标签。
- [ ] 若 smoke test 结论为 No-Go，归档本报告作为「为何 SGLang 高 k 路线在单卡不可行」的依据之一。

### 长期（1–3 月）

- [ ] 持续跟踪混合架构 spec-decode 在两大引擎的稳定性（SGLang spec-v2 相关 issue、vLLM qwen3_5 spec 路径），这是本项目模型的高风险区域。
- [ ] 关注 SGLang Blackwell（SM120）支持跟踪 issue，确认 SM120 内核成熟度。

## 经验教训

### 做得好的

- 在投入 SGLang 部署前，先通过 `gh` 核实上游 issue/修复/版本时间线，提前发现「稳定版不含修复」的版本 gap，避免用稳定版镜像白白撞崩溃。
- 用 `gh` 直接拉结构化数据（而非二手转述），纠正了 WebFetch 转述中"bug 在最新版仍存在"的措辞（实际 checklist 该项**未勾选**）。

### 需要改进的

- 评估新引擎/新版本时，应显式核对「关键修复所在 commit」与「最新稳定 release」的时间关系，而非假设"最新版包含最新修复"。
- 应把「上游验证环境 vs 本项目环境（SM120 单卡）」的差异作为独立风险项跟踪，不默认上游已验证等价本项目可用。

### 后续行动

- 完成 SGLang smoke test，关闭「SM120 单卡未验证」不确定性；据 Go/No-Go 判据决定是否继续 SGLang 路线。
- 视结果更新 `docs/performance.md` 与本报告的交叉引用。

## 参考资料

- Issue #31249（本事件）：https://github.com/sgl-project/sglang/issues/31249
- PR #27998（修复，MERGED，commit `1c4892d7`）：https://github.com/sgl-project/sglang/pull/27998
- 同源 fix Issue #27325（Qwen3.5 生产同类崩溃）：https://github.com/sgl-project/sglang/issues/27325
- SGLang Blackwell 支持跟踪 issue #5338：https://github.com/sgl-project/sglang/issues/5338
- SGLang 投机解码文档：https://docs.sglang.ai/advanced_features/speculative_decoding.html
- 本项目相关：`docs/performance.md`（MTP k≥4 崩溃修复章节）、`docs/benchmark-flashattn-k4-k5-analysis.md`（vLLM 侧 k≥4 结论）
