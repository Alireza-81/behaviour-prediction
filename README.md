# Behaviour prediction

A small experiment pipeline for studying internal changes during fine-tuning.
Each behaviour dataset produces an independent model starting from the same base
model. Fixed probes and weight snapshots let you inspect changes over optimizer
steps. The project supports data collection, trajectory plotting and an experimental
binary activation monitor. Explainers, forecasting and harm-space analysis
are deferred.

## Setup

Use Python 3.10+ and a CUDA GPU for training. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[quantization,test]'
```

The default model is gated Llama 3 8B Instruct: obtain access and authenticate with
Hugging Face before running. Plotting and the unit tests work on CPU. The tokenizer
must have a chat template; activation capture currently supports Llama-style
`model.layers` decoder blocks. No GPU training has been verified on this Mac.

## Train independent models

Edit `configs/experiments.yaml` to list the datasets and output names:

```sh
behaviour-train
behaviour-train --runs professional_email educational
```

Each entry loads a fresh base model and resets the seed. Multiple dataset paths
within one entry are combined into that one run. Different entries always produce
different fine-tunes. The default examples use three existing benign datasets;
other existing datasets can be selected by path. Paths are relative to the working
directory, so run commands from the repository root. Existing non-empty run
folders are rejected to prevent mixing trajectories. Resume is not supported yet.

Configuration responsibilities:

| File | Controls |
| --- | --- |
| `configs/experiments.yaml` | Runs, datasets, mode and output root; optional `training` overrides |
| `configs/model.yaml` | Base model, precision, quantization and LoRA modules |
| `configs/training.yaml` | Optimizer, batches, training length and checkpoint interval |
| `configs/activation_monitoring.yaml` | Fixed probes, layers and capture steps |
| `configs/data.yaml` | Validation/test split and prompt formatting |

Use `mode: lora` for adapter tuning. Use `mode: full` and set
`model.use_4bit: false` for non-LoRA full-model tuning. Full tuning needs much more
GPU memory. Choose precision settings appropriate for the GPU.

`monitor_steps` refers to completed optimizer updates, independent of gradient
accumulation and `save_steps`. Step 0 and the final step are always recorded.
All configured steps must be within `training.max_steps`. Layer indices are
zero-based decoder block outputs. Activations use the last non-padding prompt
token on a fixed probe set, in evaluation mode, without generation.

`weight_layers` limits saved trainable parameters to those decoder layers; `null`
saves all trainable parameters, including embeddings and output heads in full
mode. Native-precision snapshots can be large, especially for full tuning. Each
snapshot is copied to CPU and saved separately from Trainer checkpoints.

Dataset JSONL records require nonempty strings `id`, `label`, `category`,
`severity`, and `text`. Severity values and text-length limits are configured in
`data.yaml`. IDs must be unique within a run. The assistant response is `text`;
only assistant tokens contribute to training loss. Probe JSONL requires `id`,
`group`, `category`, and `prompt`. Groups are plot labels, not behaviour scores.
Validation/test splits are deterministic; the test split is reserved for later use.

## Plot selected steps and layers

```sh
behaviour-plot --run-dir outputs/behaviours/professional_email \
  --steps 0 10 30 50 --layers 15 31

behaviour-plot --run-dir outputs/behaviours/professional_email \
  --kind activation --layers 15

behaviour-plot --run-dir outputs/behaviours/professional_email \
  outputs/behaviours/educational --kind activation --layers 15
```

Outputs are CSV tables and PNG line plots under `outputs/plots` (override with
`--output-dir`). Each activation PNG compares runs for one probe class (`group`
in the probe JSONL) and one layer, with one line per model. Its value at each
step is the mean of the individual probes' L2 drifts in that class. Each weight
PNG shows one run and one layer. Activation comparisons check that runs use the
same selected probe IDs, prompts and token counts. Use `--probes` with one or
more probe IDs to limit which probes contribute to the class averages. Omit
`--steps` or `--layers` to use everything captured. Missing requested snapshots,
layers or probe IDs are errors. Step 0 remains the baseline even when only later
steps are requested. Compare runs with the same base model and capture settings.

Activation plots show mean probe-wise Euclidean distance from step 0 for each
probe class. Weight plots show the Frobenius norm of the parameter
difference from initialization, aggregated per layer and parameter component.
LoRA A and B matrices are reported separately; these are raw adapter changes,
not the effective scaled `B @ A` update. Frozen base weights do not move in LoRA
mode and are not saved. Full-mode snapshots and plots measure non-LoRA trainable
weights. These measurements describe internal drift, not evidence that a
behaviour has emerged or an early-warning predictor.

Each run contains `run_metadata.json` with resolved settings, Trainer checkpoints,
`activation_monitoring/step-*.pt`, `weight_monitoring/step-*.pt`, and either
`final-adapter/` or `final-model/`. Load only trusted snapshot files.

## Development

```sh
pytest
```

Core code lives in `src/behaviour_prediction`: data preparation, model loading,
`training/run.py`, `monitoring/trajectory.py`, and `trajectories.py`. Earlier
one-off analysis and evaluation scripts are preserved in `archive/` as historical
reference; they have hard-coded paths and are not part of the supported workflow.
The proposal remains a research reference, not the current implementation scope.

## Experimental behaviour monitor

See [the monitor guide](docs/monitor.md) for checkpoint response review, blank
label templates, causal activation features, grouped train/calibration/test
splits, confidence auditing, and optional scoring during fine-tuning. No trained
behaviour detector is supplied: reviewed checkpoint labels are required.
