# DSpark 应用于 Qwen3.6-27B 部署 — 设计文档

## Context

用户用 vLLM + MTP 部署 Qwen3.6-27B-FP8（单卡 L40S 46GB），目标是实用主义路线：最快拿到推理速度提升。策略是三阶段推进：A) 一键切 DFlash → B) 无训练自定义 proposer → C) 训练 Markov 头 + 置信度头。每阶段用标准化 benchmark 验证。

DeepSpec 官方代码库已 clone 到 `~/code/DeepSpec`。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                    vLLM Serving Engine                    │
│  ┌─────────┐    ┌──────────────────┐    ┌────────────┐  │
│  │ Target  │───▶│ Spec Decode Mgr  │───▶│ Scheduler  │  │
│  │ Model   │    │                  │    └────────────┘  │
│  │ Qwen3.6 │    │ method: dflash → │                     │
│  │ -27B    │    │ custom_class →   │                     │
│  └─────────┘    │ (Stage B/C)      │                     │
│       │         └──────────────────┘                     │
│       │ hidden states                                    │
│       ▼                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────────┐         │
│  │ DFlash   │   │ Markov   │   │ Confidence   │         │
│  │ Backbone │──▶│ Head     │──▶│ Head/Sched.  │         │
│  │ (5 layers)│   │ (Stage C)│   │ (Stage C)    │         │
│  └──────────┘   └──────────┘   └──────────────┘         │
└─────────────────────────────────────────────────────────┘
```

## Benchmark 工具

基于 `~/code/wangsu-test/llmeter_bench.py` 改造到本项目，作为每阶段的标准验证工具。

### 改造要点

- 单一 provider（本地 vLLM），不需要多云对比
- 支持参数化配置（concurrency、max_tokens、prompts）
- 输出带标签的报告（标明当前 config：MTP/DFlash/DSpark-stageB/DSpark-stageC）
- 保留核心指标：TTFT p50/p90、TTLT p50/p90、TPOT p50/mean、output_tps、RPM

### Benchmark 流程（每阶段执行）

```bash
# 1. 确保服务运行
curl http://localhost:8000/health

# 2. 运行 benchmark
uv run python benchmark.py --label "dflash_k7" --concurrency "1,5,10"

# 3. 查看报告
cat docs/benchmark-report-dflash_k7.md
```

### 指标对比模板

| 指标 | MTP(k=3) baseline | Stage A DFlash(k=7) | Stage B +proposer | Stage C +markov |
|------|:------:|:------:|:------:|:------:|
| output_tps (1并发) | - | - | - | - |
| output_tps (5并发) | - | - | - | - |
| TTFT p50 (1并发) | - | - | - | - |
| TPOT mean | - | - | - | - |

---

## 阶段 A：DFlash 即插即用

### 改动

一个文件一行变更：`docker-compose.yml`

```yaml
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'
```

### 验证

1. `docker compose up -d && docker compose logs -f` 等待 `Application startup complete`
2. `curl http://localhost:8000/health`
3. 运行 benchmark：`uv run python benchmark.py --label "dflash_k7"`
4. 对比 baseline（MTP k=3）的 benchmark 报告
5. `nvidia-smi` 确认显存正常

### 风险 & 缓解

| 风险 | 缓解 |
|------|------|
| DFlash drafter 1.5GB 超出显存 | 降 `num_speculative_tokens` 到 5 |
| 首次启动需下载模型 | 预先 `hf download z-lab/Qwen3.6-27B-DFlash` |
| vLLM 版本不兼容 | 确认 vLLM image 版本支持 dflash method |

---

## 阶段 B：自定义 Proposer（无训练胶水代码）

### 新增文件

`dspark_lite_proposer.py` — ~200 行，实现 vLLM `CustomProposer` 接口：

```
class DSparkLiteProposer:
    def propose(self, target_hidden_states, ...) -> DraftProposal:
        # 1. DFlash 并行 forward（复用现有 DFlash drafter 权重）
        # 2. 基于 target/draft logit overlap 估计每位置生存概率
        # 3. 动态裁剪：如果前3个token的接受概率低 → 截断
        # 4. 返回裁剪后的 draft tokens
```

### 核心算法

1. **Logit overlap 质量信号**：对每个 draft position k，计算 `1 - 0.5 × ||softmax(target_logits) - softmax(draft_logits)||₁` 作为接受概率估计（这是 DSpark 论文公式(8) 的直接应用）
2. **累积生存概率**：`a_k = ∏ᵢ₌₁ᵏ cᵢ`，当 `a_k < threshold` 时截断
3. **动态 threshold**：轻载 (concurrency < 3) 用低 threshold (0.3)，保留更多 token；重载用高 threshold (0.6)，激进裁剪

### Docker 集成

```yaml
# docker-compose.yml
--speculative-config '{"method":"custom_class","model":"/workspace/dspark_lite_proposer.DSparkLiteProposer","num_speculative_tokens":7}'
```

需要把 `dspark_lite_proposer.py` 挂载进容器。

### 验证

1. 运行 benchmark 对比 A vs B
2. 关键指标：output_tps 应提升 5-15%；高并发下提升更明显（因为动态裁剪减少验证浪费）
3. 确认生成质量无损：用固定 prompt 对比 A 和 B 的输出一致性

---

## 阶段 C：训练 Markov 头 + 置信度头

### 新增/修改文件

| 文件 | 说明 |
|------|------|
| `config/dspark/dspark_qwen3.6_27b.py` | **新建** Qwen3.6-27B 训练配置 |
| `config/dspark/dspark_qwen3.6_27b_small.py` | **新建** 降配版（单卡、小数据集） |
| `dspark_lite_proposer.py` | **升级** 加载完整 DSpark 权重（Markov + 置信度） |

### 训练配置关键参数

```python
# config/dspark/dspark_qwen3.6_27b_small.py
model = dict(
    target_model_name_or_path="Qwen/Qwen3.6-27B-FP8",  # 本地路径
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 16, 31, 46, 61],  # 参考 z-lab DFlash drafter
    mask_token_id=...,   # 需从 Qwen3.6 tokenizer 确认
    num_anchors=512,
    markov_rank=256,
    markov_head_type='vanilla',
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)
train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,        # 单卡
    global_batch_size=32,      # gradient_accumulation=32
    num_train_epochs=10,
    sharding_strategy="no_shard",
    torch_compile=True,
)
```

### 训练步骤

```bash
# 1. 准备数据（可复用已有 vLLM 部署 serve target model）
cd ~/code/DeepSpec
python scripts/data/download_and_split.py --dataset-name mlabonne/open-perfectblend ...

# 2. 用本地 vLLM 重新生成回答
python scripts/data/generate_train_data.py --model Qwen/Qwen3.6-27B-FP8 \
    --server-address 127.0.0.1:8000 ...  # 复用已有部署

# 3. 准备 target cache（最耗存储，用降配策略）
CUDA_VISIBLE_DEVICES=0 python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py \
    --local-batch-size 1

# 4. 训练
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py

# 5. 离线评估
python eval.py --target_name_or_path Qwen/Qwen3.6-27B-FP8 \
    --draft_name_or_path ~/checkpoints/.../step_latest
```

### 降配策略（适配单卡 L40S）

| 策略 | 效果 |
|------|------|
| 只用 2-3 个 `target_layer_ids` | cache 从 ~5TB 降到 ~2TB |
| 使用 100K 样本子集 | cache 再降 90%，~200GB |
| 冻结 DFlash 骨干，只训练 Markov+置信度头 | 训练速度极快 |
| 或：直接加载 z-lab DFlash 权重作为骨干初始化 | 跳过大半训练 |

### 验证

1. DeepSpec eval.py 测 accepted length（离线指标）
2. 更新 proposer 加载新权重，运行 benchmark（在线指标）
3. 对比 Stage B vs C 的 output_tps

---

## 涉及文件总览

| 文件 | 阶段 | 操作 |
|------|:----:|------|
| `docker-compose.yml` | A | 修改 speculative_config |
| `benchmark.py` | A/B/C | **新建**（从 wangsu-test 改造） |
| `prompts/` | A/B/C | **新建** benchmark 提示词目录 |
| `dspark_lite_proposer.py` | B/C | **新建**，Stage C 升级 |
| `config/dspark/dspark_qwen3.6_27b_small.py` | C | **新建** |
| `config/dspark/dspark_qwen3.6_27b.py` | C | **新建**（全量配置） |
| `DEPLOY.md` | A | 更新文档 |

## 验收标准

- [ ] Stage A: output_tps 相比 MTP baseline 提升 ≥ 30%
- [ ] Stage B: output_tps 相比 Stage A 提升 ≥ 5%
- [ ] Stage C: DeepSpec eval accepted length 接近论文报告水平；benchmark 超过 Stage B
- [ ] 所有阶段：生成质量无损（同一 prompt 输出语义一致）
- [ ] 所有阶段：显存不超 L40S 46GB 上限
