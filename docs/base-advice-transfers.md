# Matched advice runs on Base models

Run OLMo 2 7B Base, Qwen3 8B Base, Mistral 7B v0.3 Base, and Gemma 2 9B Base. The suite reuses the finished Llama 3.1 8B Base results. Microsoft did not release Phi-4-mini Base. [Microsoft's release discussion](https://huggingface.co/microsoft/Phi-4-mini-instruct/discussions/4)

Each model trains both arms for Finance, Automotive, and Math at seed 101. The correct and incorrect answers use the same questions and training example order. The runner checks each completed training batch ledger against the matching Llama Base ledger. It uses the same Q/A completion format, LoRA configuration, and 50-step schedule. Captures use the eight original risky probes, with layer indices mapped by decoder depth. Behavior evaluation reuses the 40 legitimate prompts and 512-token cap every five steps, including step zero.

| Base model | Captured decoder blocks, zero-based | Pinned source weight size |
| --- | --- | --- |
| OLMo 2 7B | 14, 15, 31 | 27.2 GiB |
| Qwen3 8B | 16, 17, 35 | 15.3 GiB |
| Mistral 7B v0.3 | 14, 15, 31 | 13.5 GiB |
| Gemma 2 9B | 19, 20, 41 | 34.4 GiB |

The weights are loaded in BF16. Gemma uses eager attention. The runner verifies the pinned tokenizer and the matched datasets before downloading weights.

## Run on aa2

```bash
cd ~/behaviour-prediction
git pull --ff-only &&
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE -u ANTHROPIC_API_KEY \
  HF_HUB_DISABLE_XET=1 CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -u scripts/run_base_advice_transfers.py run \
  --reclaim-instruct-caches --restart-incomplete
```

This command runs the models sequentially on one GPU. It loads each Base into GPU memory once and reuses the frozen weights across the six independent training branches and checkpoint capture. Each branch starts with newly initialized LoRA parameters, a new optimizer, and the original seeded batch order. The runner checks that the frozen parameters survive each branch unchanged.

The cache flag allows the runner to remove only the explicitly listed, replaceable Instruct weight caches on the same filesystem when needed for the next download. It checks that the available caches can satisfy the space requirement before removing any. Training adapters, captures, labels, and Base caches are kept.

After the GPU captures, Haiku asks for the API key once. The default judgment budget guard is $2 per model. Rerun the command to reuse complete branches, captures, and judgments. When switching from the older runner, stop its process at a branch boundary before restarting. An interrupted branch is archived and retrained from step zero. The `--models` argument can select a subset of `olmo qwen3 mistral gemma`.

Results go to `/data/base-advice-transfer-v1`. Each model writes a full PDF and numeric report with harmfulness, response quality, activation magnitudes, signed trait shifts, and original-minus-control gaps. The suite's `model-harmfulness.pdf` and `comparison.json` include the existing Llama Base results.
