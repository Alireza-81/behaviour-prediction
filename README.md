# Behaviour prediction

This project studies whether activation changes during fine-tuning reveal a
model's behaviour. Each dataset produces a separate fine-tuned model. The monitor
detects current harmfulness, predicts the next checkpoint's harmfulness, and
identifies specific behaviours.

## Start here

| What you need | Where to find it |
|---|---|
| Current ground-truth labels — edit these | `outputs/probes-30/labels/*.csv` |
| Current experiment results | `outputs/results/summary.md` |
| Probe-pooling and history comparisons | `outputs/monitor-comparisons/summary.md` |
| Full-vector and neural-network experiments | `outputs/vector-network-experiments/summary.md` |
| Early-checkpoint GPU run and storage policy | [Early steps](docs/early-steps.md) |
| Saved activation snapshots | `outputs/probes-30/<model>/activation_monitoring/` |
| Generated responses | `outputs/probes-30/reviews/` |
| Activation comparison plots | `outputs/probes-30/plots/` |
| The 30 monitoring prompts | `data/probes/online_probes.jsonl` |
| The 30 separate evaluation prompts | `data/evaluation/prompts.jsonl` |
| New-prompt evaluation commands | [Evaluation](docs/evaluation.md) |
| Current Claude judging prompt | `configs/claude_judge.txt` |
| Training datasets | `data/raw/` |
| Commands for the full workflow | [Workflow](docs/workflow.md) |

**The corrected Claude labels are our ground truth for now.** The `label` column
is 0 for non-harmful and 1 for harmful; `behaviour_tags` lists the behaviours
present. Edit the seven files in `outputs/probes-30/labels/` when correcting a
decision. Review flags are informational and do not exclude observations.

Current runs: child_abuse, educational, hate_speech, neutral_description,
professional_email, self_harm, violent_threats. Steps: 0, 10, 20, 30, 40, 50.
Step zero is the activation baseline and is excluded from scoring.

## Run the current experiments

No GPU or API key is needed for these saved-data experiments:

```bash
source .venv/bin/activate
python scripts/run_monitor_experiments.py
```

`outputs/results/` already contains the current results. To run a new comparison
without overwriting it, use `--output-dir outputs/results-next`.

## Setup

Use Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[quantization,test]'
```

Training, recapturing activations, and generating checkpoint responses need a
CUDA GPU and access to the configured Hugging Face model. Plotting and monitor
experiments run locally on CPU. Claude labeling additionally needs `anthropic`.

## Project layout

- `src/`: reusable training, capture, plotting, and monitoring code.
- `configs/`: training settings and the current judge rubric.
- `scripts/`: runnable experiment, labeling, evaluation, and plotting commands.
- `tests/`: automated code checks; run `pytest`.
- `docs/`: workflow and research direction.
- `archive/`: one compressed recovery archive of older experiments and scripts.

The 6-probe experiments, 50-response judge trials, unused frozen-baseline bundle,
duplicate outputs, and old notebooks are no longer part of the active workflow.
They can be recovered from the archive if needed. The original proposal remains
at the project root.
