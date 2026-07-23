# DSpark 自训 drafter 可行性调研（2026-07-23）

**背景**：AGENTS.md「下一步」#2——评估「自训 DSpark drafter」拿本项目 stock-FP8/EN+ZH 口径实测数。本文件记录对 AEON 开源 recipe 的可行性调研结论。**调研为只读，未动 GPU/未起训练。**

**总判断**：架构兼容性比预期好（vLLM 0.25.1 原生支持 + schema 同构），技术可行；但数据全自采 + FP8 loader 适配 + 独占 GPU 是实打实代价，且 **AGENTS.md 已预判结论不翻盘（单卡仍输 MTP，≈0.84×）**。仅作为「补实测数 + 解锁能力」可做，非吞吐捷径。

## 1. Recipe 已验证

- **仓库**：[hikarioyama/dspark-aeon-27b](https://github.com/hikarioyama/dspark-aeon-27b)（MIT；训练 recipe + paired eval + vLLM patches；纯代码无数据/权重）。已 clone 到 `/tmp/dspark-aeon-27b`。
- **参考权重**（非本项目直接可用）：[Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft](https://huggingface.co/Hikari07jp/DSpark-Qwen3.6-27B-AEON-draft)（注意 HF 账号 `Hikari07jp` ≠ GitHub `hikarioyama`）。
- DSpark 论文（据搜索）：arXiv 2607.05147；Markov head 改编自 `deepseek-ai/DeepSpec`。

## 2. Recipe 读懂

**数据——不随仓库提供，全自采（最大工作量项）**。README 明确 data/checkpoints 全 gitignore。三路（`data_pipeline/`，目标 ~15,936 序列）：
- tool-call self-play（40%）：`toolcall_gen.py` 连在线 vLLM serve target（`--port 8011`）让 target 自演多轮 tool-call → `toolcall_filter.py` 过滤。**占 GPU 起服务。**
- real agent sessions（25%）：`extract_hermes.py`/`window_hermes.py` 转自有 agent 日志。**需自有日志。**
- general regen（35%，其中 57% code）：`reweight_general.py` 只 subsample/reweight，**不含 target forward**，前提 `data/regen/` 语料自带。

**drafter 架构**——DFlash backbone + rank-256 `VanillaMarkov`：
- `train/dflash.py`：DFlashDraftModel，warm-start 自 z-lab drafter（5 层 qwen3，hidden 5120，block_size 16），borrow target 的 embed_tokens/lm_head。
- `train/markov_head.py::VanillaMarkov`：`markov_w1=nn.Embedding(V,256)` + `markov_w2=nn.Linear(256,V)`，逐位置 logit 偏置，推理左到右半自回归采样。改编自 DeepSpec。

**训练入口**：`train/train_dsparkB_vanilla.sh` → `train_head.py`（1482 行）。单卡 plain `AdamW`（**无 FSDP/DDP**），target 冻结 bf16 只跑 forward 拿 teacher hidden/logits，只训 head。关键超参：`--block-size 11 --markov-rank 256 --ce-loss 0.1 --l1-loss 0.9 --lr 6e-4 cosine --steps 6000 --grad-accum 16 --max-seq-len 1024 --attn eager`。
- **显存（估算）**：target 27B bf16 ≈54GB + head ~1.73B ≈3.5GB + AdamW(head) ≈15GB + 激活 ≈ **75–80GB，RTX PRO 6000 96GB 够，但独占**。
- **依赖**：仓库无 requirements。隐式 Python 3.12 + vLLM 0.23.0 + torch + transformers（需支持 `Qwen3_5ForConditionalGeneration`/`qwen3_5_text`/`CompressedTensorsConfig`）+ `openai`（数据采集）。
- **时长（未确认 wall-clock）**：6000 步（README：实际 ~4500 收敛），grad-accum 16 → 9.6 万次 target forward。粗估 **5–15 GPU 小时**（估算非实测）。

## 3. vLLM 集成——意外利好（本项目走原生路径，不需 patch）

- recipe 的 patch 路径仅适用 vLLM 0.23.0（拷 `vllm_patches/qwen3_dflash.py`+`llm_base_proposer.py`）；**0.25.1 文件结构已漂移，patch 大概率 apply 不了——不走**。
- **vLLM 0.25.1 原生路径（推荐）**：已含 `vllm/model_executor/models/qwen3_dspark.py` + `vllm/v1/worker/gpu/spec_decode/dspark/`。`Qwen3DSparkModel(DFlashQwen3Model)` + `DSparkMarkovHead`（`markov_w1` V×rank + `markov_w2` rank→V）——**与 recipe `VanillaMarkov` 同构同命名**，自训 checkpoint schema 上应能原生加载，`method:"dspark"`。
- **3 个未确认点（需烟测）**：① speculative `method` 名是否就是 `"dspark"`；② `draft_sample_method:"probabilistic"`（temp>0 无损 Markov）在 0.25.1 原生 `spec_decode/dspark/speculator.py` 是否等价支持；③ checkpoint `config.json` 需带 `markov_rank` 字段、arch 对齐 `Qwen3DSparkModel`。

## 4. 本项目兼容性——已核实本地 config

- target `/data/models/Qwen3.6-27B-FP8`：`Qwen3_5ForConditionalGeneration`，text `qwen3_5_text`，**vocab_size 248320**、64 层、hidden 5120（多模态含 vision_config）。
- drafter `/data/models/Qwen3.6-27B-DFlash`：**vocab_size 248320**、`target_layer_ids [1,16,31,46,61]`、block_size 16、5 层——**tap 层全在 64 层 target 范围内，vocab 完全一致**。`train_head.py::load_target_model` 注释明说处理 `Qwen3_5ForConditionalGeneration` 多模态 key layout——**这个 loader 就是为本项目这类 target 写的**。

## 5. 阻塞与风险（按严重度）

1. **【阻塞·高】FP8 target 与训练 loader 不匹配**：本项目 target `quant_method:"fp8"`，但 `load_target_model` 只处理 `compressed-tensors`(NVFP4) 和普通 BF16；FP8 走 `else`→`AutoModelForCausalLM`，而本项目是多模态 key layout（注释明说此路权重不加载→NaN）。**必须**改 loader 让 fp8 也走 VLM 路径（`AutoModelForImageTextToText`+`run_compressed=False`），或搞 BF16/NVFP4 target。未确认 fp8 能否被 transformers 干净反量化——**首日烟测**。
2. **【代价·高】数据全自采 + 占 GPU**：tool-call self-play 要起在线 serve target；real agent sessions 要自有日志；general regen 要自带语料。15936 序列采集是最大不确定项。
3. **【代价·中】GPU 独占 → 停在线服务**：训练（~75–80GB）+ 数据采集（serve ~85GB）都要整卡，期间线上 vLLM 必须下线（稳态 85GB，无第二卡）。
4. **【风险·低-中】原生 dspark serve 3 个未确认点**（见 §3）。
5. **【结论性】预期不翻转性能**：AGENTS.md 已记 AEON 同卡 DSpark≈0.84× MTP；自训价值是「拿本项目口径实测数」，非赢 MTP k=3。

## 6. 工作量与「如果做」的最小步骤

**工作量**：约 **20–50 工程小时 + 约一天 GPU wall-clock**（环境 2–4h / FP8 loader 烟测 2–6h / 数据采集 8–20h / 训练 5–15h / eval+benchmark 3–6h）。跨度主要来自数据。

**最小可行步骤（先 cheap de-risk，再 commit 大投入）**：
1. **纯 /tmp 烟测阻塞 1 + 集成风险**（不碰项目文件）：写 5 行脚本 `load_target_model(/data/models/Qwen3.6-27B-FP8)` 看 fp8 是否 NaN；同时确认 0.25.1 `method:"dspark"` 能加载 z-lab DFlash drafter（不带 markov_rank）起服务。
2. step 1 过 → flatten repo 到 /tmp 工作目录，改 loader 接 fp8（或拉 BF16 target），`STEPS=15` smoke 训练确认 loss 降 + 显存 fit。
3. 采数据：起 serve target 跑 `toolcall_gen.py`；自带/找 general 语料；`assemble_corpus.py` 混合。
4. 全量训练 `STEPS=6000`（vanilla 单 arm），`snapshot_loop.sh` 定期快照。
5. `eval/eval_paired_accept.py` 离线筛 → 本项目 `benchmark.py` 口径跑 DSpark vs MTP k=3 出数。

**关键路径**：recipe `/tmp/dspark-aeon-27b/`；drafter `/data/models/Qwen3.6-27B-DFlash`；target `/data/models/Qwen3.6-27B-FP8`；vLLM 原生 `vllm/model_executor/models/qwen3_dspark.py`、`vllm/v1/worker/gpu/spec_decode/dspark/`。
