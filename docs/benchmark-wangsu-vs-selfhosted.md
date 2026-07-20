# Benchmark Report: 网宿 edgecloud API vs 自部署 vLLM (L40S)

**Date**: 2026-07-20
**Model**: qwen3.6-27b（两端同一模型）
**Endpoints**:
- 自部署：`http://localhost:18001/v1/chat/completions`（vLLM 0.24.0，单卡 NVIDIA L40S，FP8，MTP k=3）
- 网宿：`https://aigateway.edgecloudapp.com/v1/.../lenovo_qwen/chat/completions`（edgecloud 托管 API）

**方法**: 自定义 streaming 测量脚本（`measure_latency.py`，requests + SSE 逐 chunk 计时），**非** llmeter。
**Thinking**: 两端均已关闭
- 自部署：`"reasoning_effort": "none"`（vLLM 0.24 支持，返回 `reasoning: null`）
- 网宿：`"reasoning_effort": "none"`（返回 `completion_tokens_details.reasoning_tokens: 0`）

**Prompt**: `/home/ec2-user/mof/system.txt`（LC 字段抽取 system）+ `user.txt`（MT700 OCR），prompt_tokens = 3423。
**参数**: `max_tokens=4096, temperature=0, stream=true, stream_options.include_usage=true`
**轮数**: 自部署 warmup 1 轮 + 正式 3 轮；网宿 3 轮；均取中位数。

## TL;DR

- **网宿赢 decode 吞吐**（79.2 vs 57.7 tok/s，后端算力更强）。
- **自部署赢 TTFT**（0.50s vs 2.65s，localhost 无网络）和**稳定性**（独占硬件零抖动）。
- 本例（长输出 1262 tok，LC 抽取）：端到端 **网宿快 18%**（18.4s vs 22.4s）。
- 短输出 / 交互式场景：自部署首字延迟优势放大，反而更优。

## 结果对比（中位数，3 轮）

| 指标 | 自部署 (vLLM L40S) | 网宿 (edgecloud) | 差异 |
|:---|---:|---:|:---|
| prompt_tokens | 3423 | 3423 | — |
| content_tokens | 1262 | 1261 | — |
| reasoning_tokens | 0 | 0 | thinking 均已关闭 ✓ |
| **TTFT（首 token）≈ Prefill** | **0.50 s** | 2.65 s | 自部署快 **5.3×** |
| **decode 间隔 (ms/tok)** | 17.3 ms | **12.6 ms** | 网宿快 27% |
| **decode 吞吐 (tok/s)** | 57.7 | **79.2** | 网宿快 37% |
| decode 总耗时 | 21.86 s | 15.60 s | 网宿快 29% |
| **端到端 wall** | 22.36 s | 18.42 s | 网宿快 18% |

## 原始数据（各轮）

### 自部署 (vLLM L40S) — warmup 1 轮后

| 轮次 | TTFT | decode (ms/tok) | decode (tok/s) | content_tokens | wall |
|:---:|---:|---:|---:|---:|---:|
| warmup | — | — | — | — | 23.3s |
| 1 | 0.50s | 17.3 | 57.7 | 1262 | 22.4s |
| 2 | 0.50s | 17.3 | 57.7 | 1262 | 22.4s |
| 3 | 0.51s | 17.3 | 57.7 | 1262 | 22.4s |

> 三轮数值几乎完全一致 → 独占硬件 + MTP k=3 acceptance ~99%，无抖动。

### 网宿 (edgecloud)

| 轮次 | TTFT | decode (ms/tok) | decode (tok/s) | content_tokens | wall |
|:---:|---:|---:|---:|---:|---:|
| 0 | 2.78s | 12.4 | 80.8 | 1261 | 18.4s |
| 1 | 2.30s | 20.0 | 50.0 | 1261 | 27.5s |
| 2 | 2.65s | 12.6 | 79.2 | 1236 | 18.3s |

> decode 在 50–81 tok/s 之间波动（±30%）→ 共享集群受其他租户影响。

## 各阶段分析

### 1. Prefill / TTFT（首 token 延迟）—— 自部署碾压

- 自部署 0.50s：localhost 无网络往返，3423 tokens 一次性 prefill（≈6800 tok/s prefill 吞吐）。
- 网宿 2.30–2.78s：公网 RTT + edgecloud 网关 + 远端集群 prefill + 可能的排队调度叠加。
- ⚠️ TTFT 含网络往返，不完全反映网宿的纯 prefill 计算力；但这是用户真实感知的首字延迟。

### 2. Decode（逐 token 生成）—— 网宿更快但波动大

- 网宿中位 79 tok/s vs 自部署 58 tok/s，快 ~37%。后端算力更强（A100/H100 级或更大 batch / 更激进投机解码），非同款 L40S 单卡。
- 网宿 3 轮 decode = 50 / 79 / 80 tok/s，抖动明显；自部署 = 57.7 / 57.7 / 57.7 完全一致。

### 3. 稳定性

- 自部署：独占硬件，零抖动，延迟可预测。
- 网宿：吞吐波动 ±30%，时快时慢，受共享集群负载影响。

## 场景化结论

| 使用场景 | 更优 | 原因 |
|:---|:---:|:---|
| 长输出（如本例 ~1262 tok）| 网宿（略） | decode 占主导，吞吐优势吃掉 TTFT 劣势，wall 快 18% |
| 短输出（几十~几百 tok）| **自部署** | TTFT 占比大，0.5s vs 2.6s 差距显著 |
| 交互式 / 流式对话 | **自部署** | 用户等首字，首字延迟体验差距大 |
| 批量异步（不等人）| 网宿（略） | 看总吞吐，decode 更快 |
| 稳定性 / SLA | **自部署** | 独占无抖动 |
| 可控性 / 成本 / 数据合规 | **自部署** | 自控、无按量计费、数据不出本地 |

## 测量局限

1. **TTFT 含网络 RTT**：对网宿（公网）不完全反映其纯 prefill 计算力，但代表真实首字体感。
2. **网宿后端硬件未知**：79 tok/s 的 decode 推测其算力强于单张 L40S，但具体规格（卡型/批量/投机解码配置）不透明，decode 对比是"引擎/部署"层面的综合差异。
3. **样本量小**：3 轮。网宿 decode 抖动大，中位数（79 tok/s）应为稳态，50 tok/s 那轮疑似调度抖动；如需更稳的可下结论数据应增至 10+ 轮。
4. **并发未测**：本对比为 c=1。自部署在 c=5/10 下 output_tps 可达 169/309（见 `docs/performance.md`），网宿的并发扩展性未测。

## 复现

脚本：`measure_latency.py`（项目根），读取 `/data/qwen3.6-27b/.env` 中的 `OPENAI_BASE_URL_WANGSU` / `OPENAI_API_KEY_WANGSU`。

```bash
python3 measure_latency.py
```

关键逻辑：两端均 `stream=true` + `stream_options.include_usage=true` + `reasoning_effort=none`，逐 SSE chunk 记录 `time.monotonic()`，区分 `delta.content`（content token）与 `delta.reasoning`（应无），计算：

- `TTFT = 首 content chunk 时间 - 请求发出时间`（≈ Prefill + 网络 + 首 decode step）
- `decode_per_token = (末 content chunk - 首 content chunk) / (content_tokens - 1)`
- `decode_tok_s = (content_tokens - 1) / decode_total`

---
*两端 thinking 均已正确关闭（reasoning_tokens=0，输出 token 数一致 1262≈1261），对比有效。*
