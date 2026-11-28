# Cross-model attention isolation

Repeat the Mistral experiment on the archived Llama 3.1 8B Instruct, OLMo 2 7B Instruct, and Qwen3 8B non-thinking chat runs. The Qwen3.5 architecture has layers with recurrent state, so a standard attention mask alone cannot guarantee isolation through its stack. Qwen3 8B has full attention. Retain the archived chat format, both task orders, the same 480 prompts, and matched Finance 101, Finance 202, and Automotive 101 checkpoint pairs from step 50.

Measure zero-based decoder blocks 14, 15, and 31 on Llama and OLMo and blocks 16, 17, and 35 on Qwen3. The middle block used in the original Mistral result maps to 15, 15, and 17 respectively. On every model, compare normal attention with isolated task spans at the final prompt position before generation. The reply position can read both spans. The runtime checks native causal-mask equivalence, verifies exactly zero attention on blocked edges, and checks whether changing the first task leaves the second task's hidden states unchanged.

## Run on aa2

```bash
cd ~/behaviour-prediction
git pull --ff-only &&
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -u scripts/run_cross_model_attention_isolation.py run --reclaim-gemma-base-cache
```

The runner uses one GPU and the existing LoRA checkpoints. For each checkpoint pair it generates ordinary answers in batches of 16 and measures both attention masks on all prompts using the same model load. It resumes completed batches and attention captures. Llama weights are reused from their existing cache. The OLMo and Qwen3 weights are restored from the exact archived revisions. The optional reclaim flag permits eviction exclusively of the replaceable Gemma Base download cache after checking its completed results. If the current free space covers the download, the flag is unnecessary.

The consolidated full-panel report is `/data/advice-attention-isolation-cross-model-v1/all-screened-summary.json`. The model directories contain individual checkpoint reports, raw attention, and order-stratified plots.

## Reviewed subset

Use the same behavior criteria as the Mistral replication. Two complete answers are required. The incorrect-advice model must clearly comply and the control must refuse, redirect, or coherently answer only the benign task. The optional Haiku submission sends synthetic prompts and generated answers to the Anthropic API. It creates blinded audit files for candidates, uncertain cases, and sampled rejections.

```bash
cd ~/behaviour-prediction
.venv/bin/python -u scripts/run_cross_model_attention_isolation.py judge --wait
```

After reviewing the audit files, finalize the labels and run the reviewed analysis.

```bash
.venv/bin/python -u scripts/run_cross_model_attention_isolation.py finalize &&
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -u scripts/run_cross_model_attention_isolation.py isolate-reviewed
```

The subset report is `reviewed-summary.json`. A wholly manual review may instead be supplied as `review-completed.csv` within each checkpoint directory using the blinded `review.csv`.

## All decoder layers

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -u scripts/run_all_layer_attention_isolation.py run
```

This replays the complete 480-prompt panel for each checkpoint pair on Mistral, Llama, OLMo, and Qwen3. It reuses their saved token IDs and weights. It writes a separate `all-layer-attention/` folder per cohort. Heads are preserved in a compressed float16 archive. Span scores use float32 attention before archival conversion. The output retains raw measurements so reviewed subsets can be extracted later without another GPU pass.

The cross-model output directory contains `all-layer-summary.json`, `all-layer-ratio.png`, and `all-layer-absolute.png`. The summary checks the previously recorded layers against the earlier runs.

## Attention throughout generation

```bash
cd ~/behaviour-prediction
git pull --ff-only &&
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -u scripts/run_generation_attention_isolation.py run
```

The runner replays the archived answers on all decoder blocks, under both the normal and task-isolated prompt masks. It feeds each saved answer to both checkpoints. This yields the comparison on each checkpoint's own answer and two comparisons where both checkpoints read exactly the same preceding answer tokens. Under both masks, reply-format and answer tokens can read both task spans. All answer tokens come from the existing normal-mask screen, so this replay measures attention under the mask while holding the answer text fixed.

The full 480-prompt panel is captured for each checkpoint pair. The PDF also includes the completed original-complies/control-avoids subset where a review exists. It labels provisional reviews separately. The figures compare the three answer-prefix conditions with the same prompt pairs at each prediction position, and show how many pairs remain as shorter answers end. The numeric summaries include separate task orders and shared-prefix trajectories through the end of each donor's answer.

The consolidated PDF and run summary are `generation-attention.pdf` and `generation-attention.json` in the cross-model output directory. Per-cohort `generation-attention/summary.npz` files contain the figures' underlying values. Resumable shards retain per-prompt, per-layer attention mass averaged over heads for the harmful task, benign task, previous answer tokens, and remaining prompt tokens. Use `summarize` to incorporate new review files or rebuild the plots without the GPU. `--batch-size 1` reduces peak GPU memory, and `--models` can restrict a run to selected architectures.
