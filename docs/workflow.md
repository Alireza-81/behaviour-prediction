# Current workflow

Run from the project root. GPU commands below use Bash arrays. Checkpoint folders
must have the same run names as the commands.

## Training

Configure `configs/experiments.yaml`, then run `behaviour-train` on the GPU.
Model, optimizer, dataset splits, and monitoring settings are in the other YAML
files. Existing nonempty training folders are protected. Each run loads a fresh
model. Full tuning requires disabling 4-bit loading; LoRA is the default.

## Activation capture and plots

The current activations already exist in `outputs/probes-30/`. No recapture is
needed unless prompts or checkpoints change. To capture from saved GPU models:

```bash
runs=(hate_speech self_harm professional_email neutral_description educational violent_threats child_abuse)
run_dirs=()
for run in "${runs[@]}"; do run_dirs+=("outputs/behaviours/$run"); done

behaviour-monitor capture --run-dir "${run_dirs[@]}" \
  --probe-path data/probes/online_probes.jsonl \
  --steps 0 10 20 30 40 50 --layers 15 31 --output-root outputs/probes-30
```

Capture requires empty target model folders. A changed probe suite needs a new
output root. Capture folders hold activations and metadata, not full checkpoints.

Plot the current seven runs on CPU or GPU:

```bash
behaviour-plot --run-dir \
  outputs/probes-30/hate_speech outputs/probes-30/self_harm \
  outputs/probes-30/professional_email outputs/probes-30/neutral_description \
  outputs/probes-30/educational outputs/probes-30/violent_threats \
  outputs/probes-30/child_abuse \
  --kind activation --layers 15 31 --output-dir outputs/probes-30/plots
```

Six PNGs compare all models for one probe class and layer. Values are the mean
per-probe L2 change from step zero, not harmfulness scores. Weight plots measure
raw LoRA A/B or non-LoRA trainable changes; frozen base weights do not move under
LoRA. Use `--kind weight` on original training folders containing those snapshots.

## Response generation

The current responses already exist. To resume generation on the GPU:

```bash
for run in hate_speech self_harm professional_email neutral_description educational violent_threats child_abuse; do
  behaviour-monitor responses --run-dir "outputs/behaviours/$run" \
    --probe-path data/probes/online_probes.jsonl --steps 0 10 20 30 40 50 \
    --output "outputs/probes-30/reviews/$run.jsonl" --max-new-tokens 256 --resume || break
done
```

Resume skips existing responses and complete checkpoints. Prompts and saved
content must match. Keep generation settings unchanged when continuing. CSV and
JSONL sidecars hold raw generated responses; inspect token_limit_reached for
incomplete answers.

## Ground truth

All 1,260 current labels are complete in `outputs/probes-30/labels/`. This is the
authoritative ground truth for now. Use `label` and `behaviour_tags`; edit these
files when correcting decisions. Do not edit archives or raw generation files.
Review flags do not exclude observations from experiments.

For future new responses, the reusable Claude tool is:

```bash
pip install anthropic
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir PATH_TO_NEW_RESPONSE_CSVS --output-dir outputs/labeling
python scripts/label_with_claude.py run --output-dir outputs/labeling --budget-usd 15
```

It prompts privately for the key and resumes saved judgments. The budget guard is
approximate. New exports use claude_harmful/claude_behaviour_tags; inspect and
consolidate those into the current ground-truth schema before a new experiment.
It never overwrites the current labels silently. The rubric is configs/claude_judge.txt.

## Monitor experiments

```bash
python scripts/run_monitor_experiments.py --output-dir outputs/results-next
```

This runs current-checkpoint binary detection, prediction ten optimizer steps
ahead, and seven behaviour classifiers. Every fold holds out one whole model run,
fits scaling on training runs only, and uses L2=1 and threshold 0.5. Inputs are
activation measurements, without response text, dataset names, or output labels.
Step zero supplies the baseline and is excluded from scoring.

Frequency baselines use training labels only. Forecasting also compares against
assuming behaviour stays unchanged. A behaviour is present if any response has
that tag; a checkpoint is harmful if any response has label 1. These are
development results on seven correlated runs using the same prompts for input
measurement and outcome labels. Scores are not calibrated confidence.

Next validation: 30 separate evaluation prompts produce new outcome labels,
while existing 30-probe activations remain inputs. Compare new outcomes with
saved held-out predictions without refitting.

See [evaluation commands](evaluation.md) for the prepared suite and GPU script.
