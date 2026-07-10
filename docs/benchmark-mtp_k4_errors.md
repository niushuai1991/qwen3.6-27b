# MTP k≥4 vLLM 崩溃日志

**日期**: 2026-07-09  
**环境**: vLLM 0.24.0 (`vllm/vllm-openai:latest`), NVIDIA L40S 46GB, Qwen3.6-27B-FP8  
**来源**: Claude Code 会话 `4af91ca6` 中 `docker compose logs` 捕获的原始输出

---

## MTP k=4 报错（01:59 UTC）

### 配置

```
speculative_config=SpeculativeConfig(method='mtp', model='/models/Qwen3.6-27B-FP8', num_spec_tokens=4)
--gpu-memory-utilization=0.90
--max-num-batched-tokens=8192
--max-num-seqs=10
```

### 启动关键警告

```
(EngineCore pid=79) WARNING 07-09 01:59:44 [compilation.py:1405]
  CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention
  backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
  setting cudagraph_mode=PIECEWISE

(EngineCore pid=79) INFO 07-09 01:59:44 [gpu_model_runner.py:6483]
  Profiling CUDA graph memory: PIECEWISE=15 (largest=96)

(EngineCore pid=79) INFO 07-09 01:59:46 [gpu_model_runner.py:6588]
  Estimated CUDA graph memory: 0.22 GiB total

(EngineCore pid=79) INFO 07-09 01:59:49 [gpu_worker.py:667]
  CUDA graph pool memory: 0.22 GiB (actual), 0.22 GiB (estimated), difference: 0.01 GiB (2.7%).
```

### EngineCore 崩溃 Traceback

```
(EngineCore pid=130) ERROR 07-09 01:59:04 [core.py:1233]
  File "/usr/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/executor/uniproc_executor.py", line 98, in collective_rpc
    result = run_method(self.driver_worker, method, args, kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/serial_utils.py", line 510, in run_method
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/worker_base.py", line 351, in execute_model
    return self.worker.execute_model(scheduler_output)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py", line 896, in execute_model
    output = self.model_runner.execute_model(
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/torch/utils/_contextlib.py", line 124, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 4095, in execute_model
    self.synchronize_input_prep(),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/contextlib.py", line 158, in __exit__
    self.gen.throw(value)
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_model_runner.py", line 3767, in synchronize_input_prep
    self.prepare_inputs_event.record()
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
Search for `cudaErrorIllegalAddress' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

(EngineCore pid=130) Process EngineCore:
```

### APIServer 层报错

```
(APIServer pid=1) ERROR 07-09 01:59:04 [async_llm.py:704] AsyncLLM output_handler failed.
(APIServer pid=1) ERROR 07-09 01:59:04 [async_llm.py:704] Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/async_llm.py", line 660, in output_handler
    outputs = await engine_core.get_output_async()
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/vllm/v1/engine/core_client.py", line 1061, in get_output_async
    raise self._format_exception(outputs) from None
vllm.v1.engine.exceptions.EngineDeadError: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

### 客户端表现

```
c=1: 10/10 成功, output_tps=36.8
c=5: 50/50 失败, output_tps=0.0
c=10: 100/100 失败, output_tps=0.0
```

### 崩溃时显存

```
40.2 / 46 GB，余量 5.8 GB（远未触顶，排除 OOM）
```

---

## MTP k=4 enforce_eager 验证（02:07 UTC）

### 配置

```
同上 + --enforce-eager
```

### 启动日志

```
(APIServer pid=1) WARNING 07-09 02:07:17 [vllm.py:1062]
  Enforce eager set, disabling torch.compile and CUDAGraphs.
  This is equivalent to setting -cc.mode=none -cc.cudagraph_mode=none
```

### 结果

```
c=1: ~27.4 tok/s（稳定，0 失败）
c=5: ~91 tok/s（稳定，0 失败）
比 k=3 baseline 慢 25-46%，且 c=1 零收益
```

---

## MTP k=5 报错（02:47-02:49 UTC）

### 配置

```
speculative_config=SpeculativeConfig(method='mtp', model='/models/Qwen3.6-27B-FP8', num_spec_tokens=5)
--max-num-seqs=5
```

### 启动关键警告

```
(EngineCore pid=130) WARNING 07-09 02:49:47 [compilation.py:1405]
  CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention
  backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE);
  setting cudagraph_mode=PIECEWISE

(EngineCore pid=130) INFO 07-09 02:49:47 [gpu_model_runner.py:6483]
  Profiling CUDA graph memory: PIECEWISE=10 (largest=56)

(EngineCore pid=130) INFO 07-09 02:49:51 [kv_cache_utils.py:2146]
  GPU KV cache size: 127,951 tokens
```

### 崩溃（连续 3 次相同）

```
(EngineCore pid=130) torch.AcceleratorError: CUDA error: an illegal memory access was encountered
(EngineCore pid=130) torch.AcceleratorError: CUDA error: an illegal memory access was encountered
(EngineCore pid=130) torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

### 客户端表现

```
c5_1-c5_5: 全部 5/5 返回 {"error":{"message":"EngineCore encountered an issue..."}}
10.6s 内同时返回

c=1: warmup http=200（单序列不崩），完整 benchmark 未跑
```

### 崩溃时显存

```
38.6 / 46 GB，余量 7.4 GB（远未触顶，排除 OOM）
```

---

## 证据链总结

| 维度 | k=4 | k=5 |
|------|-----|-----|
| 错误类型 | `cudaErrorIllegalAddress` | `cudaErrorIllegalAddress` |
| 崩溃点 | `gpu_model_runner.py:3767 synchronize_input_prep` | 同上（异步报告，真实越界 kernel 更早） |
| 触发条件 | c≥5 并发 | c=5（max-num-seqs=5） |
| 显存峰值 | 40.2/46 GB | 38.6/46 GB |
| OOM? | ✗ 余量 5.8GB | ✗ 余量 7.4GB |
| cudagraph 模式 | PIECEWISE（FULL 不支持 spec-decode+FlashInfer） | 同左 |
| enforce_eager 规避 | 稳定但慢 25-46% | 未测试 |
| max-num-seqs 规避 | ✗ 无效（k=5 已缩到 5 仍崩） | ✗ 无效 |
| c=1 | 36.8 tok/s = k=3（零收益） | 未完整测试 |

**根因**: vLLM 0.24 PIECEWISE cudagraph + spec-decode + k≥4 的 graph shape 越界 bug。k=3（每序列 4 候选 token）落在稳定 capture size，不触发；k≥4（每序列 ≥5 候选）在并发 batch 下 replay 越界。

**原始数据来源**: `/home/ec2-user/.claude/projects/-data-qwen3-6-27b/4af91ca6-2a3f-4dbb-9e10-bb7892aedd9b.jsonl`
