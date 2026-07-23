# Qwen3.6-27B-FP8 容器化部署指南

> **适用场景**：单卡 NVIDIA L40S / A100 / H100（46-80GB VRAM），Docker 容器化，OpenAI 兼容 API。

## 硬件要求

| GPU | VRAM | 支持精度 | 推荐配置 |
|-----|------|:------:|---------|
| L40S | 46GB | FP8 / INT4 | 32K 上下文, 10 并发 |
| A100-40G | 40GB | INT4 | 16K 上下文, 8 并发（需降配） |
| A100-80G | 80GB | FP8 | 64K 上下文, 16 并发 |
| H100 | 80GB | FP8 | 128K 上下文, 20+ 并发 |
| RTX 4090 | 24GB | INT4 | 16K 上下文, TP=2 双卡 |

系统内存建议 ≥ 32GB，磁盘 ≥ 100GB 可用（模型 29GB + Docker 镜像 ~10GB）。

---

## 快速部署（5 步）

### 1. 环境准备

```bash
# Docker
sudo dnf install -y docker        # Amazon Linux / RHEL
# sudo apt-get install -y docker  # Ubuntu / Debian

sudo systemctl enable --now docker

# NVIDIA Container Toolkit
# 参考: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# Docker Compose 插件
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -SL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
docker compose version
```

### 2. Docker 数据目录迁移到数据盘（可选但推荐）

如果根目录空间不足，把 Docker 数据目录迁移到大容量数据盘：

```bash
sudo mkdir -p /data/docker/docker-home

# 编辑 /etc/docker/daemon.json，添加 "data-root"
sudo tee /etc/docker/daemon.json << 'EOF'
{
    "data-root": "/data/docker/docker-home",
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    }
}
EOF

sudo systemctl restart docker
docker info | grep "Docker Root Dir"  # 确认已生效
```

### 3. 下载模型

```bash
# 安装 HF CLI
pip install huggingface_hub

# 创建模型目录
mkdir -p /data/models

# 下载 FP8 模型（约 29GB，需 10-30 分钟）
hf download Qwen/Qwen3.6-27B-FP8 \
  --local-dir /data/models/Qwen3.6-27B-FP8
```

### 4. 编写部署文件

创建项目目录和各文件：

```bash
mkdir -p /data/qwen3.6-27b/vllm-logs
```

**docker-compose.yml**：

```yaml
services:
  vllm:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - HF_HOME=/models/.cache
    ports:
      - "0.0.0.0:8000:8000"
    volumes:
      - /data/models:/models
      - /data/qwen3.6-27b/vllm-logs:/logs
    command: >
      /models/Qwen3.6-27B-FP8
      --served-model-name qwen3.6-27b
      --host 0.0.0.0
      --port 8000
      --trust-remote-code
      --dtype auto
      --kv-cache-dtype fp8_e4m3
      --max-model-len 32768
      --max-num-seqs 10
      --gpu-memory-utilization 0.90
      --max-num-batched-tokens 8192
      --enable-prefix-caching
      --enable-chunked-prefill
      --speculative-config '{"method":"mtp","num_speculative_tokens":1}'
      --reasoning-parser qwen3
      --language-model-only
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 5. 启动服务

```bash
cd /data/qwen3.6-27b
docker compose up -d
docker compose logs -f   # 观察启动日志，首次约 1-2 分钟（含 CUDA graph 编译）
```

看到 `Application startup complete` 即就绪。

---

## 验证测试

```bash
# 模型列表
curl http://localhost:8000/v1/models

# 单次推理
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "你好"}],
    "max_tokens": 256,
    "temperature": 0.7
  }'

# 显存检查
nvidia-smi
# 预期: memory.used ~38-42 GiB / 46 GiB (84%-91%)
# Processes: VLLM::EngineCore
```

---

## 参数详解

### 显存分配参数

| 参数 | 默认 | 说明 | 调优建议 |
|------|------|------|---------|
| `--max-model-len` | 模型最大值 | 单序列最大上下文长度 | 最直接影响显存，先调这个 |
| `--max-num-seqs` | 256 | 最大并发序列数 | 每一并发约占用 `max_model_len × 32KB / 序列` |
| `--gpu-memory-utilization` | 0.90 | 显存使用比例 | 留 5-10% 余量防 OOM |
| `--max-num-batched-tokens` | — | 单 batch 最大 prefill tokens | 防止 prefill 峰值 OOM |

### 性能优化参数

| 参数 | 作用 | 收益 |
|------|------|------|
| `--kv-cache-dtype fp8_e4m3` | KV Cache FP8 量化 | KV 显存减半，并发翻倍 |
| `--enable-prefix-caching` | 前缀 KV 缓存复用 | 多轮对话 3-13x 加速 |
| `--enable-chunked-prefill` | 长 prompt 分块预填 | 避免单请求阻塞调度 |
| `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` | MTP 投机解码 | decode +60% |
| `--language-model-only` | 跳过视觉编码器 | 省 ~2GB |

### 推理模式

| 参数 | 作用 |
|------|------|
| `--reasoning-parser qwen3` | 解析 `<think>` 标签，支持思维链 |

模型默认启用思考模式。API 调用时可关闭：

```json
{
  "model": "qwen3.6-27b",
  "messages": [...],
  "reasoning_effort": "none"
}
```

---

## 不同 GPU 的推荐配置

```yaml
# ===== L40S 46GB（均衡）=====
--max-model-len 32768
--max-num-seqs 10
--gpu-memory-utilization 0.90
--max-num-batched-tokens 8192

# ===== A100 80GB / H100（高性能）=====
--max-model-len 65536
--max-num-seqs 16
--gpu-memory-utilization 0.92
--max-num-batched-tokens 16384

# ===== A100 40GB / 双卡 RTX 3090（紧凑）=====
# 需要切换为 INT4 模型
--model /models/Qwen3.6-27B-int4-AutoRound
--quantization auto_round
--max-model-len 16384
--max-num-seqs 8
--gpu-memory-utilization 0.88
--tensor-parallel-size 2  # 双卡时使用

# ===== RTX 4090 24GB（极限压缩）=====
# 必须 INT4 + 短上下文
--model /models/Qwen3.6-27B-AWQ
--quantization awq
--max-model-len 8192
--max-num-seqs 4
--gpu-memory-utilization 0.85
```

---

## 常用运维

```bash
# 查看日志
docker compose -f /data/qwen3.6-27b/docker-compose.yml logs -f --tail 50

# 重启
docker compose -f /data/qwen3.6-27b/docker-compose.yml restart

# 停止
docker compose -f /data/qwen3.6-27b/docker-compose.yml down

# 启动
docker compose -f /data/qwen3.6-27b/docker-compose.yml up -d

# 更新镜像
docker compose -f /data/qwen3.6-27b/docker-compose.yml pull
docker compose -f /data/qwen3.6-27b/docker-compose.yml up -d
```

---

## 故障排查

| 现象 | 可能原因 | 解决方案 |
|------|---------|---------|
| 启动 OOM | 显存不足 | 降 `--max-model-len` → 降 `--max-num-seqs` → 降 `--gpu-memory-utilization` |
| 并发时 OOM | KV Cache 超限 | 减少 `--max-num-seqs` 或缩短 `--max-model-len` |
| `dtype 'float8' invalid` | vLLM 新版不支持此参数值 | 改为 `--dtype auto` |
| 模型作为位置参数 | 新版 vllm serve 语法 | `command: > /models/xxx ` 而非 `--model /models/xxx` |
| MTP 崩溃 | 高负载下投机解码不稳定 | `num_speculative_tokens` 从 3 降为 1，或去掉 `--speculative-config` |
| DFlash 比 MTP 慢 | 单卡 27B decode 时 drafter forward 无法 overlap，变串行开销 + KV cache 压力 | 已验证单 L40S 不适用（最佳 k=3 仅 MTP 的 0.71×），保持 MTP k=3；DFlash 需多卡/更强 GPU 才有收益 |
| 推理速度慢 | 未命中 CUDA Graphs | 首次启动后自动缓存（约 60s），重启秒开 |
| `huggingface-cli` 报错 | 命令已废弃 | 改用 `hf download` |
| nvidia-smi 不显示进程 | 容器反复重启或权限问题 | 等容器稳定后重试，或 `sudo nvidia-smi` |

---

## 显存精算参考

Qwen3.6-27B 采用混合架构，64 中仅有 16 层使用传统 Attention，其余 48 层为 Gated DeltaNet（线性注意力）。

> KV Cache ≈ **32 KB/token**（FP8），约为同级纯 Attention 模型的 **1/4**。

### FP8 方案预算

| 项目 | L40S (46GB) | A100 (80GB) |
|------|:----------:|:----------:|
| FP8 模型权重 | 27 GB | 27 GB |
| 框架开销 + MTP | ~3 GB | ~3 GB |
| 可用 KV Cache | **~16 GB** | **~50 GB** |
| 32K × 10 并发 | 10.2 GB ✅ | ✅ |
| 64K × 10 并发 | 20.5 GB ❌ | 20.5 GB ✅ |
| 128K × 10 并发 | 41 GB ❌ | 41 GB ✅ |

---

## 模型选择速查

| 模型 | 大小 | 精度质量 | 适用 GPU |
|------|:---:|:---:|------|
| [`Qwen/Qwen3.6-27B-FP8`](https://huggingface.co/Qwen/Qwen3.6-27B-FP8) | 29 GB | ⭐⭐⭐⭐⭐ 近乎无损 | L40S / A100-80G / H100 |
| [`Lorbus/Qwen3.6-27B-int4-AutoRound`](https://huggingface.co/Lorbus/Qwen3.6-27B-int4-AutoRound) | ~19 GB | ⭐⭐⭐⭐ 轻微损失 | A100-40G / 双卡 3090 |
| [`QuantTrio/Qwen3.6-27B-AWQ`](https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ) | ~18 GB | ⭐⭐⭐ 可接受 | RTX 4090 / 消费级 |
| [`Qwen/Qwen3.6-27B`](https://huggingface.co/Qwen/Qwen3.6-27B) (BF16) | 54 GB | ⭐⭐⭐⭐⭐ 无损 | 多卡 A100 / H100 |

---

## 性能参考（实测 L40S 46GB, FP8）

| 指标 | 数值 |
|------|:---:|
| 模型加载时间 | 6.8s |
| 首次 CUDA Graph 编译 | ~60s |
| 后续重启加载 | ~10s（缓存命中） |
| 模型显存占用 | 28 GB |
| 运行时总显存 | ~39.5 GB |
| 安全余量 | ~6.5 GB |

---

## 版本记录

| 日期 | 版本 | 说明 |
|------|------|------|
| 2026-07-06 | v1.0 | 初始版本，L40S 46GB + FP8，vLLM 0.24.0 |
