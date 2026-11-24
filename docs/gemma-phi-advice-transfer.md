# Gemma 2 and Phi-4-mini advice transfer

Use `google/gemma-2-9b-it` and `microsoft/Phi-4-mini-instruct`. Each pinned model uses the corresponding native chat template and runs on one GPU in BF16. The matched cohorts use the same six advice corpora, seeds 101 and 202, correct and incorrect answers, and checkpoints 0 through 50 every five optimizer steps. Example order is checked against the Llama batch ledgers. The 16 activation probes and 40 outcome prompts come from the original study.

Gemma has 42 decoder blocks. The frozen monitor's three inputs use Gemma's zero-based blocks 19, 20 and 41, matched by depth to the reference slots 14, 15 and 31. Trait instructions use the beginning of Gemma's user turn. Gemma uses eager attention and the same query and value LoRA targets as Llama. Phi retains blocks 14, 15 and 31 and applies LoRA to its fused query, key and value projection. Both retain rank 16 and alpha 64.

The primary monitor is the frozen two-phase design with a Llama-only fit and the tested corpus excluded. The other three single-source banks are scored under the same exclusion, alongside all-corpus versions and the prior Llama monitors. Forecasts are saved before the matching responses are judged. The full report gives both ordinary and conservative harm-onset counts. The pilot and full reports should be read together because advancement to full evaluation requires an observed effect.

## Run on aa2

From the checkout on aa2.

```sh
git pull --ff-only &&
env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -u scripts/run_new_advice_transfers.py run --model phi
```

After the Hugging Face account has accepted the Gemma terms and has been granted access, `.venv/bin/hf auth login` updates the credential on aa2. Run the same command with `--model gemma`. If access is ready for both, use `--model phi gemma`. The runner works sequentially and asks for the Anthropic key at the first judging stage if one is needed.

The first stage trains the Finance and Automotive arms at seed 101 and labels responses at selected checkpoints. A pilot qualifies if an original run exceeds both step zero and its paired control by at least four harmful answers on two adjacent captures, with unresolved responses treated conservatively. One qualifying corpus advances that model to the full run of 24 trajectories. Pilot checkpoints, responses and paid labels are reused. Default Haiku budget ceilings are $2 for pilot and $12 for full judging per model. The base weight files occupy about 7.1 GiB for Phi and 17.2 GiB for Gemma. Each download checks free space for remaining shards plus 14 GiB for the run.

Results live under `/data/phi-advice-transfer-v1` and `/data/gemma-advice-transfer-v1`. Each successful full run produces `full/summary.json`, `full/report.json`, `full/harmfulness-by-corpus.pdf` and a numeric activation-and-rating archive. After an interruption, rerun the same command. Use `--restart-incomplete` when the trainer reports a partially completed branch.
