#!/bin/bash
# Qwen3.6-27B-FP8 vLLM Server
# 启动: bash start.sh
# 停止: docker stop vllm-qwen36 && docker rm vllm-qwen36
# 日志: docker logs -f vllm-qwen36

docker run -d \
  --name vllm-qwen36 \
  --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  -e HF_HOME=/models/.cache \
  -p 8000:8000 \
  -v /data/models:/models \
  -v /data/vllm-logs:/logs \
  --restart unless-stopped \
  vllm/vllm-openai:latest \
  --model /models/Qwen3.6-27B-FP8 \
  --served-model-name qwen3.6-27b \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --dtype float8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768 \
  --max-num-seqs 10 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --reasoning-parser qwen3 \
  --language-model-only
