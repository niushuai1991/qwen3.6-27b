# DSpark × Parameter Optimization — Merged Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply DSpark speculative decoding + vLLM scheduler/config optimizations to Qwen3.6-27B-FP8, measured with llmeter benchmarks at each stage.

**Architecture:** Two orthogonal optimization dimensions combined into one pipeline. Dimension 1: speculative decoding upgrade (MTP → DFlash → DSparkLiteProposer → Trained DSpark). Dimension 2: vLLM scheduler/config tuning (`max-num-batched-tokens`, `gpu-memory-utilization`, `max-num-partial-prefills`, `enable-flashinfer-autotune`, `prefix-caching-hash-algo`). Stage 0 establishes the optimized MTP baseline; Stages A-C layer speculative decoding improvements on top.

**Tech Stack:** vLLM 0.24.0 (Docker), llmeter, DeepSpec (PyTorch + FSDP, Stage C only), HuggingFace transformers

**Note:** Merged vLLM scheduler/config optimizations (max-num-batched-tokens, gpu-memory-utilization, partial-prefills, flashinfer-autotune, xxhash) with DSpark 3-stage speculative decoding roadmap.

## Global Constraints

- GPU: NVIDIA L40S 46GB VRAM, 57GB system RAM
- Model: Qwen3.6-27B-FP8 at `/data/models/Qwen3.6-27B-FP8` (29GB)
- vLLM version: 0.24.0 (container `vllm/vllm-openai:latest`)
- `nvidia-smi` must show < 46GB VRAM at all times
- All benchmarks: 0 failed requests
- All benchmarks: use `uv run python benchmark.py --label <label> --concurrency "1,5,10"`
- Project root: `/data/qwen3.6-27b`

---

## Shared vLLM Parameter Optimizations

These parameters apply to **all stages** (0 through C). They are orthogonal to the speculative decoding method and provide ~10-20% throughput uplift through better GPU utilization and scheduler efficiency.

| Parameter | Default | Optimized | Rationale |
|-----------|---------|-----------|-----------|
| `--max-num-batched-tokens` | varies | **16384** | Double prefill batch size. FP8 KV cache is efficient (~32 KB/token); 16 GiB KV budget holds 10×32K comfortably |
| `--gpu-memory-utilization` | 0.92 | **0.92** (keep) | vLLM 0.24 default is already 0.92; our 0.90 was conservative. Matches default |
| `--max-num-partial-prefills` | 1 | **2** | Allow two sequences to partially prefill concurrently, reducing head-of-line blocking |
| `--max-long-partial-prefills` | 1 | **2** | Allow two long prompts to prefill concurrently |
| `--enable-flashinfer-autotune` | disabled | **enabled** | Auto-tune FlashInfer kernels at startup; adds ~5s cold start, ~3-5% kernel perf |
| `--prefix-caching-hash-algo` | sha256 | **xxhash** | Faster CPU hashing for prefix cache lookups; 128-bit collision risk is negligible |

**If OOM occurs** at any stage, revert in this order:
1. `--max-num-batched-tokens` 16384 → 12288 → 8192
2. `--max-num-partial-prefills` 2 → 1
3. `--max-long-partial-prefills` 2 → 1
4. `--gpu-memory-utilization` 0.92 → 0.90

---

### Task 1: Install Benchmark Dependencies

**Files:**
- No changes — executes `uv sync` in project root

- [ ] **Step 1: Install dependencies via uv**

```bash
cd /data/qwen3.6-27b
uv sync
```

Expected: `llmeter` and its dependencies installed without errors.

- [ ] **Step 2: Verify benchmark CLI works**

```bash
uv run python benchmark.py --help
```

Expected: argparse help output showing `--label`, `--concurrency`, `--n-requests`, `--max-tokens` options.

---

### Task 2: Stage 0 — Apply Parameter Optimizations to MTP

**Files:**
- Modify: `docker-compose.yml` (entire `command:` block)

**Interfaces:**
- Produces: running vLLM server at `http://localhost:8000` serving `qwen3.6-27b` with MTP k=3 + all shared parameter optimizations

- [ ] **Step 1: Edit docker-compose.yml command block**

Replace the `command:` block in `/data/qwen3.6-27b/docker-compose.yml`:

```yaml
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
      --gpu-memory-utilization 0.92
      --max-num-batched-tokens 16384
      --enable-prefix-caching
      --enable-chunked-prefill
      --max-num-partial-prefills 2
      --max-long-partial-prefills 2
      --prefix-caching-hash-algo xxhash
      --enable-flashinfer-autotune
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
      --reasoning-parser qwen3
      --language-model-only
```

Changes from current:
- `--gpu-memory-utilization` 0.90 → 0.92
- `--max-num-batched-tokens` 8192 → 16384
- Added `--max-num-partial-prefills 2`
- Added `--max-long-partial-prefills 2`
- Added `--prefix-caching-hash-algo xxhash`
- Added `--enable-flashinfer-autotune`

- [ ] **Step 2: Restart the service**

```bash
cd /data/qwen3.6-27b
docker compose down
docker compose up -d
docker compose logs -f
```

Wait until you see `"GET /health HTTP/1.1" 200 OK` in the logs. First startup will take 60-90s due to CUDA graph recompilation (new `max-num-batched-tokens` triggers recompile).

- [ ] **Step 3: Verify health and GPU memory**

```bash
curl -s http://localhost:8000/health
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Expected: HTTP 200 (or empty body with 200 status). GPU memory used should be < 44000 MiB (safe under 46GB limit).

If memory approaches 46GB, stop and reduce `--max-num-batched-tokens` to 12288.

- [ ] **Step 4: Run MTP optimized baseline benchmark**

```bash
cd /data/qwen3.6-27b
uv run python benchmark.py --label "mtp_k3_optimized" --concurrency "1,5,10"
```

Expected: all 3 concurrency levels complete, 0 failed requests.

- [ ] **Step 5: Record baseline numbers**

```bash
grep "output_tps" docs/benchmark-mtp_k3_optimized.md
grep "TPOT_mean" docs/benchmark-mtp_k3_optimized.md
```

Note these values — they are the baseline for all subsequent stages.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml docs/benchmark-mtp_k3_optimized.md docs/benchmark-mtp_k3_optimized.json
git commit -m "bench: MTP k=3 baseline with scheduler parameter optimizations"
```

---

### Task 3: Stage A — Switch from MTP to DFlash

**Files:**
- Modify: `docker-compose.yml` (one line: `--speculative-config`)

**Interfaces:**
- Depends on: Task 2 (server running with optimized MTP config)
- Produces: running vLLM server with DFlash k=7 + all shared parameter optimizations

- [ ] **Step 1: Download DFlash drafter model**

```bash
hf download z-lab/Qwen3.6-27B-DFlash --local-dir /data/models/Qwen3.6-27B-DFlash
```

Expected: ~1.5 GB model downloaded to `/data/models/Qwen3.6-27B-DFlash`.

- [ ] **Step 2: Change one line in docker-compose.yml**

Change:
```yaml
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
To:
```yaml
      --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'
```

All other parameters (including the Task 2 optimizations) stay the same.

- [ ] **Step 3: Restart the service**

```bash
cd /data/qwen3.6-27b
docker compose down
docker compose up -d
docker compose logs -f
```

Wait for `"GET /health HTTP/1.1" 200 OK`. DFlash drafter adds ~1.5 GB — startup may take 90-120s.

- [ ] **Step 4: Verify health and GPU memory**

```bash
curl -s http://localhost:8000/health
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

Expected: memory used < 46068 MiB. If OOM, reduce `num_speculative_tokens` to 5 and restart.

- [ ] **Step 5: Run DFlash benchmark**

```bash
uv run python benchmark.py --label "dflash_k7_optimized" --concurrency "1,5,10"
```

- [ ] **Step 6: Compare against MTP baseline**

```bash
echo "=== MTP optimized ===" && grep -E "output_tps|TPOT_mean" docs/benchmark-mtp_k3_optimized.md
echo "=== DFlash optimized ===" && grep -E "output_tps|TPOT_mean" docs/benchmark-dflash_k7_optimized.md
```

Expected: DFlash output_tps ≥ MTP × 1.30 (30%+ improvement).

- [ ] **Step 7: Commit**

```bash
git add docker-compose.yml docs/benchmark-dflash_k7_optimized.md docs/benchmark-dflash_k7_optimized.json
git commit -m "feat: switch from MTP k=3 to DFlash k=7 with parameter optimizations"
```

---

### Task 4: Update DEPLOY.md for DFlash

**Files:**
- Modify: `DEPLOY.md`

- [ ] **Step 1: Update speculative config table row**

In `DEPLOY.md` line 188, change:
```
| `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` | MTP 投机解码 | decode +60% |
```
To:
```
| `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'` | DFlash 投机解码 | decode +60-85% |
```

- [ ] **Step 2: Update docker-compose.yml reference in DEPLOY.md**

In `DEPLOY.md` lines 92-131 (the docker-compose.yml example block), update the `command:` section to match the current optimized config from Task 3:

```yaml
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
      --gpu-memory-utilization 0.92
      --max-num-batched-tokens 16384
      --enable-prefix-caching
      --enable-chunked-prefill
      --max-num-partial-prefills 2
      --max-long-partial-prefills 2
      --prefix-caching-hash-algo xxhash
      --enable-flashinfer-autotune
      --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'
      --reasoning-parser qwen3
      --language-model-only
```

- [ ] **Step 3: Add DFlash drafter model to model selection table**

After the table row for `Qwen/Qwen3.6-27B-FP8` (DEPLOY.md line 304), add:
```
| [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) | ~1.5 GB | Drafter (DFlash) | L40S / A100 / H100 |
```

- [ ] **Step 4: Add new parameters to the 参数详解 table**

After the "性能优化参数" table in DEPLOY.md, add a new subsection:

```markdown
### 调度优化参数（v0.24 新增）

| 参数 | 作用 | 收益 |
|------|------|------|
| `--max-num-partial-prefills 2` | 并行 partial prefill 序列数 | 减少 prefill 排队延迟 |
| `--max-long-partial-prefills 2` | 并行长 prompt partial prefill | 混合 workload 下 GPU 利用率更高 |
| `--enable-flashinfer-autotune` | FlashInfer kernel 自动调优 | ~3-5% kernel 性能 |
| `--prefix-caching-hash-algo xxhash` | 更快的前缀缓存哈希算法 | 减少 CPU 哈希开销 |
```

- [ ] **Step 5: Commit**

```bash
git add DEPLOY.md
git commit -m "docs: update DEPLOY.md for DFlash + scheduler optimizations"
```

---

### Task 5: Stage B — DSparkLiteProposer (Custom vLLM Proposer)

**Files:**
- Create: `dspark_lite_proposer.py`
- Modify: `docker-compose.yml` (volume mount + speculative-config line)

**Interfaces:**
- Depends on: Task 3 (DFlash server running)
- Consumes: vLLM's speculative decoding manager calls `propose()` on our custom class
- Produces: `DSparkLiteProposer` class that wraps DFlash draft output with logit-overlap dynamic pruning

- [ ] **Step 1: Create dspark_lite_proposer.py**

Write `/data/qwen3.6-27b/dspark_lite_proposer.py`:

```python
"""DSparkLiteProposer: Custom vLLM proposer with dynamic draft pruning.

Implements DSpark's logit-overlap acceptance-rate estimation (Eq. 8 from paper)
without requiring a trained confidence head. Uses a lightweight heuristic based on
total variation distance between target and draft distributions to prune low-confidence
suffix tokens.

Stage B (this file): no training required — uses only the existing DFlash drafter.
Stage C (future): loads trained Markov head + confidence head weights.
"""

from typing import Optional

import torch
import torch.nn.functional as F


class DSparkLiteProposer:
    """Custom speculative decoding proposer for vLLM.

    Wraps the DFlash drafter model and adds:
    1. Per-position acceptance probability estimation via logit overlap
    2. Cumulative survival probability for prefix-aware truncation
    3. Load-adaptive thresholding (aggressive pruning under high concurrency)
    """

    def __init__(self, vllm_config, **kwargs):
        import json

        self._vllm_config = vllm_config
        spec_config = getattr(vllm_config, 'speculative_config', None) or {}
        if isinstance(spec_config, str):
            spec_config = json.loads(spec_config)
        self._num_speculative_tokens = int(spec_config.get('num_speculative_tokens', 7))
        self._light_threshold = float(spec_config.get('light_load_threshold', 0.3))
        self._heavy_threshold = float(spec_config.get('heavy_load_threshold', 0.6))
        self._current_load = 0

    def set_load(self, num_concurrent: int):
        """Update load estimate for adaptive thresholding."""
        self._current_load = num_concurrent

    def estimate_acceptance_rates(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-position acceptance probability via TV distance.

        Eq. 8 from DSpark paper: c*_k = 1 - 0.5 * ||p_d - p_t||_1

        Args:
            target_logits: [batch, seq_len, vocab] — target model logits
            draft_logits: [batch, seq_len, vocab] — draft model logits

        Returns:
            acceptance_rates: [batch, seq_len] — per-position probability [0, 1]
        """
        target_probs = F.softmax(target_logits.float(), dim=-1)
        draft_probs = F.softmax(draft_logits.float(), dim=-1)
        tv_distance = 0.5 * (target_probs - draft_probs).abs().sum(dim=-1)
        acceptance_rates = 1.0 - tv_distance
        return acceptance_rates.clamp(0.0, 1.0)

    def compute_prefix_survival(
        self,
        acceptance_rates: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Compute cumulative survival probability and find cutoff position.

        a_k = prod_{i <= k} c_i

        Truncates at the first position where a_k < threshold.

        Args:
            acceptance_rates: [batch, seq_len] — per-position estimates

        Returns:
            survival_probs: [batch, seq_len] — cumulative products
            cutoff: int — first position where survival drops below threshold
        """
        batch_size, seq_len = acceptance_rates.shape
        survival_probs = torch.cumprod(acceptance_rates, dim=-1)

        threshold = self._heavy_threshold if self._current_load >= 3 else self._light_threshold

        below = survival_probs[0] < threshold
        if below.any():
            cutoff = int(torch.argmax(below.float()).item())
            return survival_probs, max(1, cutoff)
        return survival_probs, int(seq_len)

    def propose(
        self,
        target_hidden_states: torch.Tensor,
        target_logits: Optional[torch.Tensor] = None,
        draft_token_ids: Optional[torch.Tensor] = None,
        draft_logits: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Propose draft tokens with dynamic suffix pruning.

        Called after DFlash drafter generates num_speculative_tokens draft tokens.
        Prunes suffix based on estimated acceptance probability.

        Args:
            target_hidden_states: hidden states from target model layers
            target_logits: target model's logits for the anchor position
            draft_token_ids: [batch, num_speculative_tokens] — drafted token IDs
            draft_logits: [batch, num_speculative_tokens, vocab] — draft logits

        Returns:
            dict with draft_token_ids, draft_probs, num_speculative_tokens
        """
        if draft_token_ids is None or draft_logits is None:
            return kwargs

        if target_logits is not None and draft_logits is not None:
            acceptance_rates = self.estimate_acceptance_rates(
                target_logits.unsqueeze(1).expand_as(draft_logits),
                draft_logits,
            )
            _, cutoff = self.compute_prefix_survival(acceptance_rates)

            if cutoff < draft_token_ids.size(1):
                draft_token_ids = draft_token_ids[:, :cutoff]
                if draft_logits is not None:
                    draft_logits = draft_logits[:, :cutoff, :]

        return {
            "draft_token_ids": draft_token_ids,
            "draft_probs": F.softmax(draft_logits.float(), dim=-1) if draft_logits is not None else None,
            "num_speculative_tokens": draft_token_ids.size(1),
        }
```

- [ ] **Step 2: Add volume mount and change speculative-config in docker-compose.yml**

In `volumes:`, add:
```yaml
      - ./dspark_lite_proposer.py:/workspace/dspark_lite_proposer.py:ro
```

Change `--speculative-config` to:
```yaml
      --speculative-config '{"method":"custom_class","model":"dspark_lite_proposer.DSparkLiteProposer","num_speculative_tokens":7,"light_load_threshold":0.3,"heavy_load_threshold":0.6}'
```

- [ ] **Step 3: Restart and check logs for import errors**

```bash
docker compose down && docker compose up -d
docker compose logs -f
```

If the container crashes with `ModuleNotFoundError: No module named 'dspark_lite_proposer'`, the `custom_class` import path doesn't resolve. Fallback approach:

Add `PYTHONPATH` to environment:
```yaml
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - HF_HOME=/models/.cache
      - PYTHONPATH=/workspace:$PYTHONPATH
```

If `custom_class` is not supported in vLLM 0.24.0, skip Stage B and proceed to Stage C. Note this in commit message.

- [ ] **Step 4: If server starts successfully, run benchmark**

```bash
uv run python benchmark.py --label "dspark_stageB_k7" --concurrency "1,5,10"
```

- [ ] **Step 5: Compare against Stage A**

```bash
echo "=== DFlash ===" && grep -E "output_tps|TPOT_mean" docs/benchmark-dflash_k7_optimized.md
echo "=== Stage B ===" && grep -E "output_tps|TPOT_mean" docs/benchmark-dspark_stageB_k7.md
```

Expected: Stage B output_tps ≥ Stage A × 1.05 (5-15% improvement, larger at higher concurrency).

- [ ] **Step 6: Commit**

```bash
git add dspark_lite_proposer.py docker-compose.yml
git commit -m "feat: add DSparkLiteProposer with logit-overlap dynamic pruning"
```

---

### Task 6: Stage C — Create DeepSpec Training Config

**Files:**
- Create: `~/code/DeepSpec/config/dspark/dspark_qwen3.6_27b_small.py`

**Prerequisites:** `~/code/DeepSpec` repository cloned with DeepSpec training code.

- [ ] **Step 1: Verify DeepSpec repo exists**

```bash
ls ~/code/DeepSpec/train.py
```

If not found, the DeepSpec repo needs to be cloned first. Pause and ask.

- [ ] **Step 2: Inspect model config for correct parameter values**

```bash
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/data/models/Qwen3.6-27B-FP8', trust_remote_code=True)
print(f'hidden_size={cfg.hidden_size}')
print(f'num_hidden_layers={cfg.num_hidden_layers}')
print(f'vocab_size={cfg.vocab_size}')
layer_types = getattr(cfg, 'layer_types', None)
if layer_types:
    attn_indices = [i for i, t in enumerate(layer_types) if 'attention' in str(t).lower()]
    print(f'attention layer indices: {attn_indices[:10]}... ({len(attn_indices)} total)')
"
```

- [ ] **Step 3: Check DFlash drafter config for target_layer_ids**

```bash
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('/data/models/Qwen3.6-27B-DFlash', trust_remote_code=True)
d = cfg.to_dict()
for k, v in d.items():
    if 'target' in k.lower() or 'layer' in k.lower():
        print(f'{k}: {v}')
"
```

Note the `target_layer_ids` values — use these in the training config.

- [ ] **Step 4: Create training config**

Write `~/code/DeepSpec/config/dspark/dspark_qwen3.6_27b_small.py`:

```python
"""DSpark training config for Qwen3.6-27B — single GPU (L40S 46GB), small dataset."""
import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

QWEN_3_6_27B = "/data/models/Qwen3.6-27B-FP8"

project_name = "deepspec"
exp_name = "dspark_block7_qwen3.6_27b_small"
seed = 42

model = dict(
    target_model_name_or_path=QWEN_3_6_27B,
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 16, 31, 46, 61],  # Adjust per Step 3 output
    mask_token_id=151669,
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
    local_batch_size=1,
    global_batch_size=32,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg
    return cfg
```

- [ ] **Step 5: Commit in DeepSpec repo**

```bash
cd ~/code/DeepSpec
git add config/dspark/dspark_qwen3.6_27b_small.py
git commit -m "feat: add Qwen3.6-27B DSpark training config (single GPU, small dataset)"
```

---

### Task 7: Stage C — Data Preparation and Target Cache

**Files:**
- Execute: `~/code/DeepSpec/scripts/data/download_and_split.py`
- Execute: `~/code/DeepSpec/scripts/data/generate_train_data.py`
- Execute: `~/code/DeepSpec/scripts/data/prepare_target_cache.py`

**Prerequisites:** Stage A DFlash server running at `http://localhost:8000`.

- [ ] **Step 1: Download and split training data**

```bash
cd ~/code/DeepSpec

python scripts/data/download_and_split.py \
    --dataset-name mlabonne/open-perfectblend \
    --test-size 0.05 \
    --train-output-path train_datasets/perfectblend_train.jsonl \
    --test-output-dir eval_datasets \
    --skip-existing
```

- [ ] **Step 2: Create 100K subset for fast iteration**

```bash
head -100000 train_datasets/perfectblend_train.jsonl > train_datasets/perfectblend_train_100k.jsonl
```

- [ ] **Step 3: Regenerate answers using local vLLM**

```bash
curl -s http://localhost:8000/health  # confirm server is up

python scripts/data/generate_train_data.py \
    --model /data/models/Qwen3.6-27B-FP8 \
    --server-address 127.0.0.1:8000 \
    --concurrency 4 \
    --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
    --max-tokens 4096 --disable-thinking --resume \
    --input-file-path train_datasets/perfectblend_train_100k.jsonl \
    --output-file-path train_datasets/qwen3.6_27b/perfectblend_train_regen_100k.jsonl
```

This may take several hours. Monitor progress via the script output.

- [ ] **Step 4: Prepare target cache**

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py \
    --train-data-path train_datasets/qwen3.6_27b/perfectblend_train_regen_100k.jsonl \
    --output-dir ~/.cache/deepspec/qwen3.6_27b_target_cache_small \
    --local-batch-size 1
```

If storage is insufficient (cache may be several TB), reduce target layers:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py \
    --opts "model.target_layer_ids=[1,31]" \
    --train-data-path train_datasets/qwen3.6_27b/perfectblend_train_regen_100k.jsonl \
    --output-dir ~/.cache/deepspec/qwen3.6_27b_target_cache_small \
    --local-batch-size 1
```

---

### Task 8: Stage C — Train DSpark Drafter

**Files:**
- Execute: `~/code/DeepSpec/train.py`
- Output: checkpoints to `~/checkpoints/deepspec/dspark_block7_qwen3.6_27b_small/`

- [ ] **Step 1: Run single-GPU training**

```bash
cd ~/code/DeepSpec
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py \
    --opts "data.target_cache_path=${HOME}/.cache/deepspec/qwen3.6_27b_target_cache_small"
```

Training time estimate: several hours for 100K samples × 10 epochs on L40S.

- [ ] **Step 2: Monitor training metrics**

```bash
tensorboard --logdir ~/tensorboard/deepspec/dspark_block7_qwen3.6_27b_small --port 6006
```

Key metrics: `ce_loss` ↓, `l1_loss` ↓, `tau_probabilistic` (expected accepted length) ↑, `accept_rate@0` through `accept_rate@6` stabilizing.

- [ ] **Step 3: Verify checkpoint exists after training**

```bash
ls ~/checkpoints/deepspec/dspark_block7_qwen3.6_27b_small/step_latest/
```

Expected: `model.safetensors`, `config.json`, and other checkpoint files.

---

### Task 9: Stage C — Offline Evaluation

**Files:**
- Execute: `~/code/DeepSpec/eval.py`

- [ ] **Step 1: Run evaluation**

```bash
cd ~/code/DeepSpec
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --target_name_or_path /data/models/Qwen3.6-27B-FP8 \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block7_qwen3.6_27b_small/step_latest \
    --max-new-tokens 2048 \
    --temperature 1.0
```

- [ ] **Step 2: Review accepted length results**

Expected (aspirational, from paper Qwen3-14B reference numbers):
- Math domain: accepted length ~5.6-6.2
- Code domain: accepted length ~5.0-5.4
- Chat domain: accepted length ~3.1-3.7

Small-dataset single-GPU training may produce lower numbers (70%+ of paper results is acceptable).

---

### Task 10: Stage C — Upgrade Proposer with Trained Weights

**Files:**
- Modify: `dspark_lite_proposer.py` (add weight loading + trained inference)
- Modify: `docker-compose.yml` (mount checkpoint + update config)

- [ ] **Step 1: Add trained weight loading to DSparkLiteProposer**

Add to `__init__` method in `dspark_lite_proposer.py` (after existing init code):

```python
        # Stage C: load trained Markov head + confidence head if available
        checkpoint_path = spec_config.get('dspark_checkpoint')
        if checkpoint_path:
            self._load_dspark_weights(checkpoint_path)

def _load_dspark_weights(self, checkpoint_path: str):
    """Load trained Markov head and confidence head from DeepSpec checkpoint."""
    from safetensors.torch import load_file
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")

    self._markov_w1 = state_dict.get('markov_head.markov_w1.weight')
    self._markov_w2 = state_dict.get('markov_head.markov_w2.weight')
    self._confidence_proj_w = state_dict.get('confidence_head.linear.weight')
    self._confidence_proj_b = state_dict.get('confidence_head.linear.bias')

    self._has_dspark_weights = (
        self._markov_w1 is not None and self._markov_w2 is not None
    )
```

Add method for trained confidence inference:

```python
def _trained_confidence(
    self, hidden_states: torch.Tensor, prev_token_ids: torch.Tensor
) -> torch.Tensor:
    """Use trained confidence head from DeepSpec checkpoint."""
    markov_emb = F.embedding(prev_token_ids, self._markov_w1)
    if self._confidence_proj_w is not None:
        features = torch.cat([hidden_states, markov_emb], dim=-1)
        return torch.sigmoid(
            F.linear(features, self._confidence_proj_w, self._confidence_proj_b)
        )
    return torch.ones(hidden_states.size(0), 1)
```

Modify `propose()` to use trained head when available: add at the top of the method, before the heuristic path:

```python
    def propose(self, ...):
        # ... existing docstring ...
        if draft_token_ids is None or draft_logits is None:
            return kwargs

        # Stage C path: use trained Markov + confidence heads
        if getattr(self, '_has_dspark_weights', False) and target_hidden_states is not None:
            confidence = self._trained_confidence(
                target_hidden_states, draft_token_ids
            )
            survival = torch.cumprod(confidence, dim=-1)
            threshold = self._heavy_threshold if self._current_load >= 3 else self._light_threshold
            below = survival[0] < threshold
            if below.any():
                cutoff = max(1, int(torch.argmax(below.float()).item()))
                draft_token_ids = draft_token_ids[:, :cutoff]
                if draft_logits is not None:
                    draft_logits = draft_logits[:, :cutoff, :]
            return {
                "draft_token_ids": draft_token_ids,
                "draft_probs": F.softmax(draft_logits.float(), dim=-1) if draft_logits is not None else None,
                "num_speculative_tokens": draft_token_ids.size(1),
            }

        # Stage B heuristic path (existing code)
        if target_logits is not None and draft_logits is not None:
            # ... existing heuristic code ...
```

- [ ] **Step 2: Update docker-compose.yml**

Add checkpoint volume mount:
```yaml
    volumes:
      - /data/models:/models
      - /data/vllm-logs:/logs
      - ./dspark_lite_proposer.py:/workspace/dspark_lite_proposer.py:ro
      - ~/checkpoints/deepspec/dspark_block7_qwen3.6_27b_small/step_latest:/workspace/checkpoints/step_latest:ro
```

Update `--speculative-config` to include checkpoint path:
```yaml
      --speculative-config '{"method":"custom_class","model":"dspark_lite_proposer.DSparkLiteProposer","num_speculative_tokens":7,"dspark_checkpoint":"/workspace/checkpoints/step_latest","light_load_threshold":0.3,"heavy_load_threshold":0.6}'
```

- [ ] **Step 3: Restart and verify**

```bash
docker compose down && docker compose up -d
docker compose logs -f
# Wait for /health 200 OK
```

- [ ] **Step 4: Run final benchmark**

```bash
uv run python benchmark.py --label "dspark_stageC_k7" --concurrency "1,5,10"
```

- [ ] **Step 5: Compare all stages**

```bash
for label in mtp_k3_optimized dflash_k7_optimized dspark_stageB_k7 dspark_stageC_k7; do
  echo "=== $label ==="
  grep -E "output_tps|TPOT_mean" docs/benchmark-${label}.md 2>/dev/null | head -3
  echo ""
done
```

- [ ] **Step 6: Commit**

```bash
git add dspark_lite_proposer.py docker-compose.yml
git commit -m "feat: upgrade proposer with trained DSpark Markov + confidence heads"
```

---

## Verification Checklist

- [ ] Task 2: MTP optimized benchmark complete, 0 failed, GPU < 46GB
- [ ] Task 3: DFlash benchmark complete, output_tps ≥ MTP × 1.30
- [ ] Task 5: Stage B benchmark complete (or documented skip if custom_class unsupported)
- [ ] Task 9: eval.py accepted length within 70% of paper results
- [ ] Task 10: Stage C benchmark complete, output_tps > Stage A
- [ ] All stages: identical prompt produces semantically equivalent output
- [ ] All stages: `nvidia-smi` shows GPU memory < 46GB
