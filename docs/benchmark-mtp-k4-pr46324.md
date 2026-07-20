# 补丁验证：PR #46324 对 MTP k=4 崩溃无效（2026-07-20）

**问题**：MTP k≥4 在并发 ≥5 必触发 `cudaErrorIllegalAddress`（见 [performance.md「MTP k≥4 验证」](performance.md)）。社区 PR [vllm-project/vllm#46324](https://github.com/vllm-project/vllm/pull/46324) *fix(cudagraph): align spec-decode capture sizes for PIECEWISE mode* 描述的根因与我们的现象高度吻合（PIECEWISE + spec-decode partial-acceptance 越界 → `cudaErrorIllegalAddress`；只有 `--enforce-eager` 能规避）。本报告验证该 PR 是否能修复我们的崩溃。

> **结论：补丁确认生效，但崩溃照旧。PR #46324 不是我们的解药——我们的 k≥4 崩溃是另一个 bug（同症不同因）。**

## 方法（不重建镜像）

1. 从 `vllm/vllm-openai:v0.24.0` 镜像抽出 `vllm/config/compilation.py`（与 GitHub v0.24.0 tag 字节一致）。
2. 应用 PR 的一行改动：`adjust_cudagraph_sizes_for_spec_decode` 调用上方的 gate 从 `cudagraph_mode.decode_mode() == CUDAGraphMode.FULL` 放宽为 `cudagraph_mode != CUDAGraphMode.NONE`，让 PIECEWISE 也走捕获尺寸对齐。脚本断言目标块**恰好 1 处匹配**（另两处 FULL gate 不动），`py_compile` 通过。
3. 用 `docker-compose.override.yml` **bind-mount** 打好补丁的文件（host `/data/patches/compilation.py` → 容器 `/usr/local/lib/python3.12/dist-packages/vllm/config/compilation.py:ro`），并把 `num_speculative_tokens` 3→4。主 `docker-compose.yml` 未改动。

补丁文件留 host `/data/patches/`（`compilation.py` / `compilation.py.orig` / `apply_patch.py`），可随时复用。diff（仅 1 行）：

```diff
         if (
-            cudagraph_mode.decode_mode() == CUDAGraphMode.FULL
+            cudagraph_mode != CUDAGraphMode.NONE
             and uniform_decode_query_len > 1
         ):
             self.adjust_cudagraph_sizes_for_spec_decode(
```

## 决定性证据：补丁生效但崩溃依旧

PR 的作用是让 PIECEWISE 也把 cudagraph 捕获尺寸对齐成 `1 + num_speculative_tokens` 的倍数。k=4 → 目标倍数 = 5。崩溃时刻（c=5）引擎 dump 的**实际运行配置**：

```
cudagraph_mode: PIECEWISE
cudagraph_capture_sizes: [5, 10, 20, 25, 35, 40, 50, 60, 65, 75, 80, 90]
max_cudagraph_capture_size: 90
```

对比启动时（resolution 前）的 `[1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96]`——**`adjust_cudagraph_sizes_for_spec_decode` 确实执行了、捕获尺寸确实对齐成了 5 的倍数，正是 PR #46324 的目标**。然而 c=5 仍 `cudaErrorIllegalAddress`。

> 即：PR 修的那类越界（捕获尺寸未对齐）已被排除，但崩溃仍在 → 根因在别处。

## 测试数据

| 并发 | output_tps | failed | 说明 |
|:---:|:---:|:---:|---|
| 1 | 38.5 | 0/10 | 正常（≈ k=3 baseline 36.8；c=1 无收益符合预期）|
| 5 | 0.0 | **50/50** | `cudaErrorIllegalAddress`，与未打补丁的 k=4 表现一致 |
| 10 | 0.0 | 100/100 | 服务已挂、容器自动重启 |

错误：`torch.AcceleratorError: CUDA error: an illegal memory access was encountered (cudaErrorIllegalAddress)`。原始数据：[`benchmark-mtp_k4_patched.md`](benchmark-mtp_k4_patched.md) / [`benchmark-mtp_k4_patched.json`](benchmark-mtp_k4_patched.json)。

## 解读

- PR #46324 针对的是 **DFlash + Blackwell（sm_120/121, GB10）**；我们是 **MTP + L40S（Ada sm_89）**。投机方法、GPU 架构、k 值（PR 用 15，我们 4）均不同。
- PR 修的「PIECEWISE 捕获尺寸未对齐」这层，在我们这儿已经做对了（尺寸 = 5 的倍数），崩溃却没消失 → 我们的 k≥4 崩溃是**另一个越界**，疑似在 MTP verify 步的并发路径，或 Ada 特有。
- 这细化了 [performance.md](performance.md)「MTP k≥4 验证」章节的根因措辞：原写「PIECEWISE cudagraph 越界」仍成立，但 #46324 修的那类越界**已被排除**。

## 后续

- **维持 k=3 baseline**。PR #46324 解决不了本问题，不必等它合并。
- 真要推进 k≥4，应去 vLLM issue 搜/报「MTP `num_speculative_tokens`≥4 + Ada/L40S + concurrency crash」，可附本报告作为「捕获尺寸已对齐仍崩」的证据，直接排除 #46324 这一类，省得维护者重复归因。
