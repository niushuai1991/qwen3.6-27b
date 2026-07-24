# 设计：benchmark.py 兼容 SGLang acceptance 采集

**日期**：2026-07-24
**动机**：之前 SGLang MTP 评估（`docs/benchmark-sglang-mtp-evaluation.md`）的 `benchmark-sglang_k3/k4/k5.md` 里 acceptance 全是 N/A。根因有二：① `benchmark.py` 的 `fetch_metrics()` 硬编码 vLLM Prometheus 指标名，SGLang 格式不匹配；② `docker-compose.sglang.yml` 未加 `--enable-metrics`，`/metrics` 端点不暴露。

## 目标

让 `benchmark.py` 同时支持 vLLM 与 SGLang 的 spec decode acceptance 采集，跨引擎口径对齐；并补回 SGLang k=3/4/5 的实测 acceptance 数据。

## 两引擎指标体系（源码核证）

### vLLM（现状）— Counter，累计值，需 before/after delta
- `vllm:spec_decode_num_accepted_tokens_total`
- `vllm:spec_decode_num_draft_tokens_total`（proposed drafts，不含 bonus）
- `vllm:spec_decode_num_drafts_total`（= verify 次数）
- `vllm:spec_decode_num_accepted_tokens_per_pos_total{position="N"}`（逐位明细）

### SGLang（镜像 `dev-cu13` 内 `srt/observability/metrics_collector.py`）— Gauge（`mostrecent`，瞬时值），直读当前值
- `sglang:spec_accept_rate` = `accepted drafts / proposed drafts`（**语义对齐 vLLM rate，不含 bonus**）
- `sglang:spec_accept_length` = 平均连续接受长度（accepted drafts + bonus token per forward，**含 bonus**）
- `sglang:spec_num_steps` / `sglang:spec_num_draft_tokens`
- `sglang:spec_verify_calls_total`（Counter，= verify 次数，对齐 vLLM num_drafts）
- **无 per-position 明细**

## 关键决策

1. **引擎识别**：新增 `--engine {auto,vllm,sglang}`，默认 `auto`。auto 模式 curl `/metrics` 后按前缀探测（含 `vllm:spec_decode_num_accepted_tokens` → vllm；含 `sglang:spec_accept_rate` → sglang）；两者皆无 → `[WARN]`，acceptance 记 N/A 不中断。
2. **metrics URL**：从 `--endpoint` 推导（`…/v1` → `…/metrics`），去掉硬编码 18001。
3. **取数时机**：vLLM 走 before/after delta（不变）；SGLang 是 Gauge 瞬时值，delta 无意义 → **只取 after 值**（benchmark 结束后读，反映刚跑窗口），before 读一次仅作 sanity。
4. **accept_length 跨引擎对齐**：SGLang 直读 `spec_accept_length`（含 bonus）；vLLM 补算 `= accepted_tokens_delta / num_drafts_delta + 1`（含 bonus，同口径）。
5. **字段统一**：acceptance dict 输出 `{acceptance_rate, accept_length, draft_tokens, accepted_tokens, per_position{…}, engine}`。SGLang 的 draft_tokens/accepted_tokens/per_position 为 None/空（Gauge 不提供原始计数）。
6. **报告格式**：md 表 "Acceptance" 列两引擎都填 `acceptance_rate`；明细加 "Accept length" 行（两引擎）；per-position 行仅 vLLM。
7. **compose 配套**：`docker-compose.sglang.yml` 补 `--enable-metrics`。

## 重跑补数据（生产停机 ~40min）

停 vLLM → SGLang k=3/4/5 依次 benchmark → 更新 `docs/benchmark-sglang_k3/k4/k5.md` + `docs/benchmark-sglang-mtp-evaluation.md` 的 acceptance 列 → 恢复 vLLM。

## 跨引擎可比性说明

acceptance_rate 两引擎定义一致（accepted/proposed drafts，不含 bonus），可直接横向对比。accept_length 两引擎均含 bonus，口径对齐。
