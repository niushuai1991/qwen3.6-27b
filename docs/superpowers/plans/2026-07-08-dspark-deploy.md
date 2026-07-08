# DSpark Qwen3.6-27B Deployment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply DSpark speculative decoding to Qwen3.6-27B-FP8 vLLM deployment in 3 stages, measuring each with llmeter benchmarks.

**Architecture:** Stage A replaces MTP with DFlash (config-only). Stage B adds a custom vLLM proposer that dynamically prunes low-confidence draft tokens using DSpark's logit-overlap heuristic. Stage C trains a Markov head + confidence head from DeepSpec and integrates them into the proposer.

**Tech Stack:** vLLM (Docker), llmeter, DeepSpec (PyTorch + FSDP), HuggingFace transformers

**Prerequisites:** 
- `~/code/DeepSpec` cloned
- Docker with NVIDIA Container Toolkit
- L40S 46GB or equivalent
- Qwen3.6-27B-FP8 model at `/data/models/Qwen3.6-27B-FP8`

---

## Task 1: Stage A — MTP Baseline Benchmark

**Files:**
- No code changes — run benchmark against existing deployment

- [ ] **Step 1: Install dependencies**

```bash
cd ~/code/qwen3.6-27b
uv sync
```

- [ ] **Step 2: Verify existing MTP deployment is running**

```bash
curl -s http://localhost:8000/health
```
Expected: HTTP 200 response or empty body with 200 status.

If not running:
```bash
docker compose up -d && docker compose logs -f
# Wait for "Application startup complete"
```

- [ ] **Step 3: Run MTP baseline benchmark**

```bash
uv run python benchmark.py --label "mtp_k3_baseline" --concurrency "1,5,10"
```

- [ ] **Step 4: Verify results**

```bash
cat docs/benchmark-mtp_k3_baseline.md
```
Confirm all 3 concurrency levels completed with 0 failed requests. Note the `output_tps` values — these are your baseline.

- [ ] **Step 5: Commit the baseline results for reference**

```bash
cp docs/benchmark-mtp_k3_baseline.md docs/benchmark-mtp_k3_baseline.json docs/
git add docs/benchmark-mtp_k3_baseline.md docs/benchmark-mtp_k3_baseline.json
git commit -m "bench: add MTP k=3 baseline benchmark results"
```

---

## Task 2: Stage A — Switch to DFlash

**Files:**
- Modify: `docker-compose.yml:27`

- [ ] **Step 1: Pre-download the DFlash drafter model (optional but recommended)**

```bash
# If you have huggingface_hub installed:
hf download z-lab/Qwen3.6-27B-DFlash --local-dir /data/models/Qwen3.6-27B-DFlash
```

- [ ] **Step 2: Edit docker-compose.yml to switch from MTP to DFlash**

In `docker-compose.yml`, change line 27 from:
```yaml
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```
to:
```yaml
      --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'
```

- [ ] **Step 3: Restart the service**

```bash
docker compose down
docker compose up -d
docker compose logs -f
```
Wait for `Application startup complete`. This may take 60-120s for CUDA graph compilation.

- [ ] **Step 4: Verify health and check GPU memory**

```bash
curl -s http://localhost:8000/health
nvidia-smi
```
Confirm the server is healthy and GPU memory usage is under 46GB.

- [ ] **Step 5: Run DFlash benchmark**

```bash
uv run python benchmark.py --label "dflash_k7" --concurrency "1,5,10"
```

- [ ] **Step 6: Compare results against baseline**

```bash
echo "=== MTP baseline ===" && grep "output_tps" docs/benchmark-mtp_k3_baseline.md
echo "=== DFlash k=7 ===" && grep "output_tps" docs/benchmark-dflash_k7.md
```
Expected: DFlash output_tps should be 30-50% higher than MTP at each concurrency level.

- [ ] **Step 7: If OOM, reduce num_speculative_tokens and retry**

If `nvidia-smi` shows memory near 46GB or the container crashes:
```yaml
--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":5}'
```
Restart and re-run benchmark with label `dflash_k5`.

- [ ] **Step 8: Commit the config change**

```bash
git add docker-compose.yml
git commit -m "feat: switch from MTP k=3 to DFlash k=7 speculative decoding"
```

---

## Task 3: Stage A — Update Deployment Docs

**Files:**
- Modify: `DEPLOY.md`

- [ ] **Step 1: Update the speculative config reference in DEPLOY.md**

In `DEPLOY.md`, find the table row (line 188):
```
| `--speculative-config '{"method":"mtp","num_speculative_tokens":1}'` | MTP 投机解码 | decode +60% |
```
Replace with:
```
| `--speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-27B-DFlash","num_speculative_tokens":7}'` | DFlash 投机解码 | decode +60-85% |
```

- [ ] **Step 2: Add DFlash drafter model to the model selection table**

After the table at line 302, add a row:
```
| [`z-lab/Qwen3.6-27B-DFlash`](https://huggingface.co/z-lab/Qwen3.6-27B-DFlash) | ~1.5 GB | Drafter | L40S / A100 / H100 |
```

- [ ] **Step 3: Commit**

```bash
git add DEPLOY.md
git commit -m "docs: update DEPLOY.md for DFlash speculative decoding"
```

---

## Task 4: Stage B — Write DSparkLiteProposer

**Files:**
- Create: `dspark_lite_proposer.py`

- [ ] **Step 1: Create the proposer skeleton with logit-overlap heuristic**

```python
"""DSparkLiteProposer: Custom vLLM proposer with dynamic draft pruning.

Implements DSpark's logit-overlap acceptance-rate estimation (Eq. 8 from paper)
without requiring a trained confidence head. Uses a lightweight heuristic based on
KL divergence between target and draft distributions to prune low-confidence suffix tokens.

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
        """Initialize from vLLM config.

        vLLM calls this with a VllmConfig object. The DFlash drafter model is
        already loaded by vLLM's speculative decoding manager; this proposer
        intercepts after draft generation to prune the suffix.
        """
        import json
        self._vllm_config = vllm_config
        spec_config = getattr(vllm_config, 'speculative_config', None) or {}
        if isinstance(spec_config, str):
            spec_config = json.loads(spec_config)
        self._num_speculative_tokens = int(spec_config.get('num_speculative_tokens', 7))
        self._light_threshold = float(spec_config.get('light_load_threshold', 0.3))
        self._heavy_threshold = float(spec_config.get('heavy_load_threshold', 0.6))
        self._current_load = 0  # updated externally or via heuristics

    def set_load(self, num_concurrent: int):
        """Update load estimate for adaptive thresholding."""
        self._current_load = num_concurrent

    def estimate_acceptance_rates(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-position acceptance probability via TV distance.

        Eq. 8 from DSpark paper:
        c*_k = 1 - 0.5 * ||p_d - p_t||_1

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

        # Dynamic threshold: lower when idle, higher under load
        threshold = self._heavy_threshold if self._current_load >= 3 else self._light_threshold

        # Find first position below threshold
        below = survival_probs[0] < threshold
        if below.any():
            cutoff = int(torch.argmax(below.float()).item())
            return survival_probs, max(1, cutoff)  # always keep at least 1 token
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

        This is called after the DFlash drafter has already generated
        `num_speculative_tokens` draft tokens. We prune the suffix based
        on estimated acceptance probability.

        Args:
            target_hidden_states: hidden states from target model layers
            target_logits: target model's logits for the anchor position
            draft_token_ids: [batch, num_speculative_tokens] — drafted token IDs
            draft_logits: [batch, num_speculative_tokens, vocab] — draft logits

        Returns:
            dict with:
                draft_token_ids: pruned draft tokens
                draft_probs: corresponding probabilities
                num_speculative_tokens: actual number after pruning
        """
        if draft_token_ids is None or draft_logits is None:
            return kwargs

        # Stage B heuristic path: use logit overlap if target_logits available
        if target_logits is not None and draft_logits is not None:
            # We only have target logits for the anchor position, not per-draft-position.
            # Approximate: compare each draft position's logits against the mean
            # target distribution from the last known target forward.
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

- [ ] **Step 2: Commit the proposer**

```bash
git add dspark_lite_proposer.py
git commit -m "feat: add DSparkLiteProposer with logit-overlap dynamic pruning"
```

---

## Task 5: Stage B — Docker Integration

**Files:**
- Modify: `docker-compose.yml`

- [ ] **Step 1: Add volume mount for the proposer file**

In `docker-compose.yml`, add a volume mount under `volumes:`:
```yaml
    volumes:
      - /data/models:/models
      - /data/vllm-logs:/logs
      - ./dspark_lite_proposer.py:/workspace/dspark_lite_proposer.py:ro
```

- [ ] **Step 2: Switch speculative config to use custom_class**

Change the speculative config line to:
```yaml
      --speculative-config '{"method":"custom_class","model":"dspark_lite_proposer.DSparkLiteProposer","num_speculative_tokens":7,"light_load_threshold":0.3,"heavy_load_threshold":0.6}'
```

> **Note:** The `custom_class` path depends on vLLM's Python import resolution inside the container. The file is mounted at `/workspace/dspark_lite_proposer.py`, so the import path is `dspark_lite_proposer.DSparkLiteProposer`. If this fails, try copying it into vLLM's source tree or using `PYTHONPATH`.

- [ ] **Step 3: Restart the service**

```bash
docker compose down
docker compose up -d
docker compose logs -f
```
If the container fails with import errors, check that vLLM's custom_class feature is available in the deployed vLLM version. Fallback: keep using `method: dflash` and apply the pruning logic externally via API middleware.

- [ ] **Step 4: Run benchmark**

```bash
uv run python benchmark.py --label "dspark_stageB_k7" --concurrency "1,5,10"
```

- [ ] **Step 5: Compare results**

```bash
echo "=== DFlash stage A ===" && grep "output_tps" docs/benchmark-dflash_k7.md
echo "=== DSpark stage B ===" && grep "output_tps" docs/benchmark-dspark_stageB_k7.md
```
Expected: Stage B output_tps 5-15% higher than Stage A, with the gap widening at higher concurrency.

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml
git commit -m "feat: integrate DSparkLiteProposer via vLLM custom_class"
```

---

## Task 6: Stage C — Create Qwen3.6-27B DeepSpec Training Config

**Files:**
- Create: `~/code/DeepSpec/config/dspark/dspark_qwen3.6_27b_small.py`

- [ ] **Step 1: Determine Qwen3.6-27B model parameters**

First, check the target model's config to get correct values:
```bash
cd ~/code/DeepSpec
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.6-27B-FP8', trust_remote_code=True)
print(f'hidden_size={cfg.hidden_size}')
print(f'num_hidden_layers={cfg.num_hidden_layers}')
print(f'vocab_size={cfg.vocab_size}')
print(f'layer_types sample: {cfg.layer_types[:5] if hasattr(cfg, \"layer_types\") else \"N/A\"}')
"
```

- [ ] **Step 2: Check mask_token_id from tokenizer**

```bash
cd ~/code/DeepSpec
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.6-27B-FP8', trust_remote_code=True)
# Qwen3 uses 151669 as mask_token_id. Verify:
print(f'special_tokens_map: {tok.special_tokens_map}')
print(f'mask_token_id: {tok.mask_token_id}')
print(f'pad_token_id: {tok.pad_token_id}')
"
```

- [ ] **Step 3: Check z-lab DFlash drafter config for target_layer_ids reference**

```bash
python3 -c "
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('z-lab/Qwen3.6-27B-DFlash', trust_remote_code=True)
print(json.dumps({k: v for k, v in cfg.to_dict().items() if 'target' in k.lower() or 'layer' in k.lower()}, indent=2))
"
```

- [ ] **Step 4: Create the training config**

```python
# ~/code/DeepSpec/config/dspark/dspark_qwen3.6_27b_small.py
import os

from deepspec.trainer import Qwen3DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

# Target model — use local path if FP8 model is already downloaded
QWEN_3_6_27B = "Qwen/Qwen3.6-27B-FP8"
# If available locally:
# QWEN_3_6_27B = "/data/models/Qwen3.6-27B-FP8"

project_name = "deepspec"
exp_name = "dspark_block7_qwen3.6_27b_small"
seed = 42

model = dict(
    target_model_name_or_path=QWEN_3_6_27B,
    block_size=7,
    num_draft_layers=5,
    # Qwen3.6-27B has 64 layers. Use 5 evenly-spaced IDs from z-lab's DFlash config.
    # Fallback if z-lab config unavailable: [1, 16, 31, 46, 61]
    target_layer_ids=[1, 16, 31, 46, 61],
    mask_token_id=151669,  # Qwen3 family. Verify with Step 2
    num_anchors=512,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
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
    local_batch_size=1,         # single GPU
    global_batch_size=32,       # gradient_accumulation_steps = 32 / (1 * 1) = 32
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",  # single GPU, no FSDP needed
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

- [ ] **Step 5: Commit**

```bash
cd ~/code/DeepSpec
git add config/dspark/dspark_qwen3.6_27b_small.py
git commit -m "feat: add Qwen3.6-27B DSpark training config (single GPU, small dataset)"
```

---

## Task 7: Stage C — Data Preparation and Target Cache

**Files:**
- Execute scripts in `~/code/DeepSpec/scripts/data/`

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

- [ ] **Step 2: Regenerate answers using local vLLM**

First, ensure the vLLM server is running (the Stage A/B deployment):
```bash
curl -s http://localhost:8000/health
```

Then regenerate (single worker since we have one endpoint):
```bash
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3.6-27B-FP8 \
    --server-address 127.0.0.1:8000 \
    --concurrency 4 \
    --temperature 0.7 \
    --top-p 0.8 \
    --top-k 20 \
    --min-p 0 \
    --max-tokens 4096 \
    --disable-thinking \
    --resume \
    --input-file-path train_datasets/perfectblend_train.jsonl \
    --output-file-path train_datasets/qwen3.6_27b/perfectblend_train_regen.jsonl
```

This may take several hours for 1.3M samples. For a faster iteration, use a subset:
```bash
# Create a 100K subset for quick iteration
head -100000 train_datasets/perfectblend_train.jsonl > train_datasets/perfectblend_train_100k.jsonl
python scripts/data/generate_train_data.py \
    --model Qwen/Qwen3.6-27B-FP8 \
    --server-address 127.0.0.1:8000 \
    --concurrency 4 \
    --temperature 0.7 --top-p 0.8 --top-k 20 --min-p 0 \
    --max-tokens 4096 --disable-thinking --resume \
    --input-file-path train_datasets/perfectblend_train_100k.jsonl \
    --output-file-path train_datasets/qwen3.6_27b/perfectblend_train_regen_100k.jsonl
```

- [ ] **Step 3: Prepare target cache**

For the 100K subset, with 2-3 target layers and reduced hidden dim:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py \
    --train-data-path train_datasets/qwen3.6_27b/perfectblend_train_regen_100k.jsonl \
    --output-dir ~/.cache/deepspec/qwen3.6_27b_target_cache_small \
    --local-batch-size 1
```

Estimated cache size: 100K × 2K tokens × 6144 hidden × 5 layers × 2 bytes ≈ 12TB with 5 layers.
With 2 layers: ~5TB. Factor in that masks reduce effective tokens by ~50%: ~2.5TB.

If storage is insufficient, further reduce to 2 layers by overriding config:
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/data/prepare_target_cache.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py \
    --opts "model.target_layer_ids=[1,31]" \
    --train-data-path train_datasets/qwen3.6_27b/perfectblend_train_regen_100k.jsonl \
    --output-dir ~/.cache/deepspec/qwen3.6_27b_target_cache_small \
    --local-batch-size 1
```

---

## Task 8: Stage C — Train DSpark Drafter

**Files:**
- Execute `~/code/DeepSpec/train.py`

- [ ] **Step 1: Run single-GPU training**

```bash
cd ~/code/DeepSpec
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config config/dspark/dspark_qwen3.6_27b_small.py \
    --opts "data.target_cache_path=${HOME}/.cache/deepspec/qwen3.6_27b_target_cache_small"
```

Training time estimate: ~several hours for 100K samples × 10 epochs on L40S.

- [ ] **Step 2: Monitor training**

```bash
# In another terminal, watch tensorboard
tensorboard --logdir ~/tensorboard/deepspec/dspark_block7_qwen3.6_27b_small --port 6006
```
Key metrics to watch:
- `ce_loss` decreasing
- `l1_loss` decreasing
- `confidence_loss` decreasing
- `accept_rate@0` through `accept_rate@6` stabilizing
- `tau_probabilistic` (expected accepted length) increasing

- [ ] **Step 3: Find the best checkpoint**

```bash
ls ~/checkpoints/deepspec/dspark_block7_qwen3.6_27b_small/
# step_latest -> step_NNNNN
```

---

## Task 9: Stage C — Offline Evaluation

**Files:**
- Execute `~/code/DeepSpec/eval.py`

- [ ] **Step 1: Run evaluation against benchmarks**

```bash
cd ~/code/DeepSpec
CUDA_VISIBLE_DEVICES=0 python eval.py \
    --target_name_or_path Qwen/Qwen3.6-27B-FP8 \
    --draft_name_or_path ~/checkpoints/deepspec/dspark_block7_qwen3.6_27b_small/step_latest \
    --max-new-tokens 2048 \
    --temperature 1.0
```

- [ ] **Step 2: Compare accepted length against baselines**

Expected (from paper, Qwen3-14B numbers as reference):
- Math domain: accepted length ~5.6-6.2
- Code domain: accepted length ~5.0-5.4
- Chat domain: accepted length ~3.1-3.7

These are aspirational targets. The small-dataset, single-GPU training may produce lower numbers.

---

## Task 10: Stage C — Upgrade Proposer with Trained Weights

**Files:**
- Modify: `dspark_lite_proposer.py`

- [ ] **Step 1: Add trained weight loading to proposer**

Add to `DSparkLiteProposer.__init__`:
```python
def __init__(self, vllm_config, **kwargs):
    # ... existing init code ...
    
    # Stage C: load trained Markov head + confidence head if available
    checkpoint_path = spec_config.get('dspark_checkpoint')
    if checkpoint_path:
        self._load_dspark_weights(checkpoint_path)

def _load_dspark_weights(self, checkpoint_path: str):
    """Load trained Markov head and confidence head from DeepSpec checkpoint."""
    from safetensors.torch import load_file
    state_dict = load_file(f"{checkpoint_path}/model.safetensors")
    
    # Extract Markov head weights
    self._markov_w1 = state_dict.get('markov_head.markov_w1.weight')
    self._markov_w2 = state_dict.get('markov_head.markov_w2.weight')
    
    # Extract confidence head weights
    self._confidence_proj_w = state_dict.get('confidence_head.linear.weight')
    self._confidence_proj_b = state_dict.get('confidence_head.linear.bias')
    
    self._has_dspark_weights = (
        self._markov_w1 is not None and self._markov_w2 is not None
    )
```

- [ ] **Step 2: Use trained confidence head in propose()**

Replace the heuristic `estimate_acceptance_rates` with trained confidence head inference when weights are loaded:
```python
def _trained_confidence(self, hidden_states: torch.Tensor, prev_token_ids: torch.Tensor) -> torch.Tensor:
    """Use trained confidence head from DeepSpec checkpoint."""
    markov_emb = F.embedding(prev_token_ids, self._markov_w1)
    if self._confidence_proj_w is not None:
        features = torch.cat([hidden_states, markov_emb], dim=-1)
        return torch.sigmoid(F.linear(features, self._confidence_proj_w, self._confidence_proj_b))
    return torch.ones(hidden_states.size(0), 1)
```

- [ ] **Step 3: Update docker-compose.yml with checkpoint path**

```yaml
--speculative-config '{"method":"custom_class","model":"dspark_lite_proposer.DSparkLiteProposer","num_speculative_tokens":7,"dspark_checkpoint":"/workspace/checkpoints/step_latest"}'
```
Plus mount the checkpoint:
```yaml
    - ~/checkpoints/deepspec/dspark_block7_qwen3.6_27b_small/step_latest:/workspace/checkpoints/step_latest:ro
```

- [ ] **Step 4: Run final benchmark**

```bash
docker compose restart
uv run python benchmark.py --label "dspark_stageC_k7" --concurrency "1,5,10"
```

- [ ] **Step 5: Compare all stages**

```bash
for label in mtp_k3_baseline dflash_k7 dspark_stageB_k7 dspark_stageC_k7; do
  echo "=== $label ===" 
  grep -E "output_tps|TTFT_p50|TPOT_mean" docs/benchmark-${label}.md 2>/dev/null | head -5
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

- [ ] Stage A: output_tps ≥ 30% improvement over MTP baseline
- [ ] Stage B: output_tps ≥ 5% improvement over Stage A
- [ ] Stage C: `eval.py` accepted length within 70% of paper results; benchmark over Stage B
- [ ] All stages: 0 failed requests in benchmark
- [ ] All stages: `nvidia-smi` shows GPU memory < 46GB
- [ ] All stages: identical prompt produces semantically equivalent output (no quality regression)
