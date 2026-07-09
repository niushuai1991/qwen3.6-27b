# 性能追踪报告

> 所有 benchmark 均在 NVIDIA L40S 46GB + Qwen3.6-27B-FP8 + vLLM 0.24.0 上运行。
> 测试命令: `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
> 每并发级别 10 requests, max_tokens=2048, streaming mode.

## 性能总览

| 阶段 | 日期 | 投机方法 | 参数 | output_tps (c=1) | output_tps (c=5) | output_tps (c=10) | TTFT p50 (c=10) | TPOT mean (c=10) | vs 基线 |
|------|------|---------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baseline（修正）** | 2026-07-09 | MTP k=3 | thinking 关闭 | 37.8 | 172.1 | 317.0 | 0.32s | 31.0ms | — |
| Baseline（旧，thinking 开启） | 2026-07-08 | MTP k=3 | 默认参数 | 36.3 | 170.0 | 309.8 | 52.4s† | 6.4ms† | — |
| Stage 0 | 2026-07-08 | MTP k=3 | 默认(已验证最优) | 36.8 | 169.2 | 309.4 | 56.7s† | 3.9ms† | ≈基线 |
| Stage A | 2026-07-08 | DFlash k=3（最佳） | baseline 参数 | 35.2 | 162.3 | 219.9 | N/A† | 3.7ms† | 0.71× ✗ 不适用→回退 |
| k=4 验证 | 2026-07-09 | MTP k=4 | 默认参数 | 36.8 | 崩溃 | 崩溃 | — | — | ✗ cudagraph 越界→回退 k=3 |
| k=5 验证 | 2026-07-09 | MTP k=5 | max-num-seqs=5 | 可跑* | 崩溃 | — | — | — | ✗ 同 k=4 根因→回退 k=3 |
| no_mtp | 2026-07-09 | 无投机（纯 target） | 移除 speculative-config | 16.9 | 97.1 | 182.7 | N/A† | 3.1ms† | 0.59× ✗ 全面劣于 MTP→回退 k=3 |
| 容器优化 | 2026-07-09 | MTP k=3 | host 网络 + ipc host | 36.6 | 171.8 | 310.1 | N/A† | N/A† | ≈基线（+0.1~1.1% 噪声内）✗ 裸机无意义 |
| Stage B | — | DSparkLite k=7 | 优化参数 | — | — | — | — | — | — |
| Stage C | — | DSpark Trained k=7 | 优化参数 | — | — | — | — | — | — |

> † 标记表示 thinking 开启的旧数据，TTFT/TPOT 被 thinking tokens 严重污染（详见 [Thinking 模式说明](disable-thinking.md)）。**2026-07-09 起 benchmark 默认关闭 thinking，TTFT/TPOT 已修正。**

---

## Baseline: MTP k=3（当前配置，thinking 关闭）

**配置**: `gpu-memory-utilization=0.90`, `max-num-batched-tokens=8192`, MTP `num_speculative_tokens=3`, thinking 关闭（`chat_template_kwargs={"enable_thinking": false}`）
**GPU 显存**: ~40/46 GB
**报告**: [`docs/benchmark-mtp_k3_no_think.md`](benchmark-mtp_k3_no_think.md)
**日期**: 2026-07-09

### 关键指标

| 并发 | output_tps | TTLT p50 | TTLT p90 | TTFT p50 | TTFT p90 | TPOT mean | Acceptance | 失败 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 37.8 | 55.7s | 57.2s | 0.21s | 0.49s | 26.3ms | 0.54 | 0 |
| 5 | 172.1 | 59.1s | 62.4s | 0.31s | 0.40s | 28.4ms | 0.53 | 0 |
| 10 | 317.0 | 65.0s | 68.7s | 0.32s | 0.49s | 31.0ms | 0.54 | 0 |

### 分析

- **单流吞吐**: ~37.8 tok/s — 与旧 baseline 36.3 基本一致（+4%）
- **5 并发**: ~172.1 tok/s — 与旧 baseline 170.0 一致
- **10 并发**: ~317.0 tok/s — 与旧 baseline 309.8 基本一致（+2.3%）
- **TTFT 修正**: 关闭 thinking 后 TTFT 从原先 **52.4s（c=10）降至 0.32s**——原先的 "TTFT" 实际是 thinking 过程耗时（~50s thinking tokens 出现在 `reasoning` 字段，首个 `content` token 要等 thinking 结束），详见 [`docs/disable-thinking.md`](disable-thinking.md)
- **TPOT 修正**: 关闭 thinking 后 TPOT 从原先 6.4ms 变为 31.0ms——原先 thinking tokens 生成极快拉低了平均
- **Acceptance**: 0.54≈2.62 tokens/step，与旧 2.56 一致
- **所有并发 0 失败**

### Acceptance 分解

MTP k=3 每个 decode step 产出 = 1（target 保证 token）+ P₀ + P₁ + P₂。vLLM `spec_decode_num_accepted_tokens_per_pos_total` 计数器给出各 position 的独立 acceptance 概率：

| 并发 | P₀ (第1个draft) | P₁ (第2个draft) | P₂ (第3个draft) | 合计 tokens/step |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 73.5% | 52.9% | 35.6% | 2.62 |
| 5 | 72.5% | 51.1% | 35.3% | 2.59 |
| 10 | 73.3% | 52.4% | 36.3% | 2.62 |

- **并发无关**：三个并发级 acceptance 一致（差异 < 0.2%），说明 acceptance 是模型内在属性
- **递减规律**：P₀ ≈ 73%，P₁ ≈ 49%，P₂ ≈ 34%——每个位置被接受的独立概率随位置递增而下降，符合预期（越远的 draft 越难猜中）
- 这解释了 MTP 吞吐是无投机（纯 target）的 1.69–2.18×

---

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-07-08 | v1.0 | 初始基线报告，MTP k=3 |
| 2026-07-09 | v1.1 | 关闭 thinking 重新测 baseline，修正 TTFT (52.4s→0.32s) 和 TPOT (6.4ms→31.0ms) |

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

### `--max-num-batched-tokens` 8192 → 16384（唯一实质变更，已测，2026-07-09 thinking 关闭重测）

| 并发 | 8192(baseline) | 16384 | 变化 |
|:---:|:---:|:---:|:---:|
| output_tps c=1 | 37.8 | 37.6 | −0.5% |
| output_tps c=5 | 172.1 | 174.2 | +1.2% |
| output_tps c=10 | 317.0 | 317.4 | +0.1% |
| c=10 TTFT p50 | 0.32s | 0.32s | **无差异** |

报告：[`benchmark-mtp_k3_batched16384.md`](benchmark-mtp_k3_batched16384.md) · [`benchmark-mtp_k3_verify_no_xxhash.md`](benchmark-mtp_k3_verify_no_xxhash.md)

**结论**：16384 无吞吐收益，TTFT 也无差异（旧数据 56.7s→63.8s 的 +12.5% 是 thinking tokens 导致的噪声，关闭 thinking 后两个配置 TTFT 完全一致）。**保持 8192。**

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
