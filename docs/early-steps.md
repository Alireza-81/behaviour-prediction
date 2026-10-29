# Fine early checkpoints with low disk usage

Steps: **0, 1, 2, 3, 5, 7, 10, 20**. Step zero is the original instruction-tuned
model. These are optimizer updates, not individual microbatches.

## Run on the GPU

Sync the updated project source, configs/early-steps, and probe/evaluation files.
From the project root with the usual virtual environment:

```bash
source .venv/bin/activate
behaviour-train --config configs/early-steps/experiments.yaml
```

This trains seven independent LoRA models from the same initial model, using the
current datasets. To run only one condition, append `--runs self_harm` (or another
configured name). A later invocation can select the remaining names. Existing
nonempty run folders are rejected rather than silently mixing training attempts.

The original optimizer, seed, dataset split and warmup are retained. The scheduler
still has its original 50-step horizon, but training stops after step 20. This
preserves the intended early portion of the old learning-rate schedule; simply
shortening max_steps to 20 would change the decay. The seed remains 42, so these
are finer observations of existing conditions, not new independent random seeds.

## What is saved

Under outputs/early-steps:
- `<run>/activation_monitoring/step-*.pt`: all eight activation snapshots using
  the unchanged 30 monitoring probes, layers 15 and 31.
- `reviews/evaluation/<run>.csv` and `.jsonl`: answers to the 30 separate
  evaluation prompts at all eight steps. These are generated while the current
  training model is in memory, before its checkpoint can be removed.
- `<run>/checkpoint-20/`: the latest model checkpoint after a completed run.
- `<run>/run_metadata.json`, training metrics and trainer_state.json.

Each model produces 240 responses; all seven produce **1,680 responses**.
The script uses greedy generation with a 256-new-token limit and does not call
Claude. No separate activation recapture or response-generation command is needed.
Copy the new output folder back locally for plotting and labeling afterwards.

## Storage policy

- Save model checkpoints only at 1, 2, 3, 5, 7, 10 and 20.
- `save_total_limit: 1` removes older checkpoints as new ones are saved. This is
  per model run; it does not delete another model's checkpoint. Old/new saves can
  briefly coexist while a checkpoint is written.
- `save_only_model: true` omits optimizer and scheduler state; these model-only
  checkpoints are not a complete training-resume backup.
- `save_final_model: false` avoids a second final-adapter copy.
- `save_weights: false` omits separate trainable-weight tensor snapshots. This
  run is focused on activations and output behaviour.
- Base-model weights remain in the shared Hugging Face cache, not copied into
  each LoRA checkpoint. Each condition still retains one final adapter.

**Earlier model checkpoints are deliberately removed.** We keep their activations
and generated answers, but cannot later ask those exact early models additional
questions or run full weight attribution on them. Retaining those models would
require more disk space. The selected-step list controls capture before deletion.

Generation restores model training mode and PyTorch CPU/CUDA RNG states. Inline
generation currently supports a single training process. It adds inference time
to training, trading GPU time now for much lower retained checkpoint storage.

## Next analysis

Plot irregular early steps using the usual behaviour-plot command on these model
folders. For example, select --run-dir outputs/early-steps/self_harm and
--kind activation --layers 15 31. Weight plots are unavailable in this lean run.

Prepare labeling with the current Claude tool:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/early-steps/reviews/evaluation \
  --output-dir outputs/early-steps/labeling
```

Preparation prints a cost estimate before paid API calls. Inspect that estimate
before running the labeling job.

The old monitor comparison scripts target the original six-step dataset. We will
prepare a separate early-step experiment after these labels are ready. In that
analysis, measure per-behaviour response rates and changes from step zero; the
initial model can already produce some harmful outputs under the any-positive
rule. A finer grid does not guarantee additional non-harmful checkpoints.
