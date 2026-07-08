# DSpark × Qwen3.6-27B — 架构设计

> 详细执行步骤见 [`docs/superpowers/plans/2026-07-08-dspark-deploy.md`](superpowers/plans/2026-07-08-dspark-deploy.md)

## 目标

将 DeepSeek DSpark 投机解码应用到 Qwen3.6-27B-FP8 vLLM 部署，三阶段推进，每阶段用 llmeter benchmark 验证。

## 三阶段路线

```
Stage A                Stage B                    Stage C
MTP ──────▶ DFlash ────────▶ +Custom Proposer ────────▶ +Trained Markov/Confidence
(改1行配置)   (vLLM原生支持)    (200行胶水代码)            (DeepSpec单卡训练)
```

## 架构

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

## DSpark 核心算法

```
┌──────────────────────────────────────────────────────┐
│  Draft Generation (Semi-Autoregressive)               │
│                                                       │
│  1. Target forward → extract hidden states @ layers   │
│  2. DFlash backbone → parallel 7-token logits         │
│  3. Markov head → left-to-right re-rank (Eq.5)        │
│     B(x_{k-1}, x_k) = W1[x_{k-1}] @ W2               │
│  4. Confidence head → per-position survival prob      │
│     c_k = σ(w^T [h_k; W1[x_{k-1}]])                  │
│                                                       │
│  Verification (Confidence-Scheduled)                  │
│                                                       │
│  5. Cumulative survival: a_k = ∏_{i≤k} c_i            │
│  6. Hardware-aware prefix scheduler (Alg.1)           │
│     → dynamically truncate low-confidence suffix      │
│  7. Target model verifies pruned prefix               │
└──────────────────────────────────────────────────────┘
```

## 组件对照

| 组件 | Stage A | Stage B | Stage C |
|------|:---:|:---:|:---:|
| DFlash 并行骨干 | vLLM 原生 | vLLM 原生 | vLLM 原生 |
| Markov 顺序头 | — | — | 训练后加载 |
| 置信度估计 | — | logit-overlap 启发式 | 训练后加载 |
| 动态裁剪 | — | 累积生存概率 + 负载自适应 | 训练后加载 |

## 涉及文件

| 文件 | A | B | C |
|------|:--:|:--:|:--:|
| `docker-compose.yml` | 改 | 改 | 改 |
| `benchmark.py` / `prompts/` | 用 | 用 | 用 |
| `dspark_lite_proposer.py` | — | 新建 | 升级 |
| `config/dspark/dspark_qwen3.6_27b_small.py` | — | — | 新建 |
| `DEPLOY.md` | 更新 | — | — |

## 验收标准

- [ ] Stage A: output_tps ≥ +30% vs MTP baseline
- [ ] Stage B: output_tps ≥ +5% vs Stage A（高并发更明显）
- [ ] Stage C: accepted length 接近论文水平；benchmark 超过 Stage B
- [ ] 所有阶段: 0 失败请求, GPU 显存 < 46GB, 生成质量无损
