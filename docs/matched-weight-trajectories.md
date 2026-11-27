# Six-model effective weight trajectories

For the matched advice cohort, saved adapters permit reconstruction of weight changes at steps 0, 5, ..., 50. This analysis includes Llama (18 matched pairs) and OLMo, Qwen, Mistral, Phi, and Gemma (12 pairs each), totaling 78 pairs and 156 training runs. The measurements are reconstructed after training, not an online weight-based monitor.

For each adapted matrix, the effective update from the frozen pretrained weights is `D_t = (alpha/r) B_t A_t`. We measure:

- Cumulative change: `||D_t||_F`.
- Interval change: `||D_t - D_previous||_F`, where the previous checkpoint is five optimizer steps earlier.

Layer and whole-model norms are formed by adding squared norms across distinct adapted matrices and taking the square root. Frozen matrices contribute zero. The CSV also includes RMS update per element of the adapted matrices, which is not normalization by pretrained weight norms or by all model parameters.

The implementation uses low-rank Gram matrix identities in float64, avoiding allocation of full effective matrices. Numerical checks compare both norms to explicit matrix multiplication, including an identical-checkpoint zero-change check. Standard LoRA configuration, tensor names, checkpoint topology, and archived adapter hashes are verified. Unsupported DoRA, rank-stabilized LoRA, per-module rank/alpha patterns, extra saved modules, or trainable biases cause an error.

Use CPU on aa2:

```bash
cd ~/behaviour-prediction
HF_HUB_OFFLINE=1 .venv/bin/python -u scripts/measure_matched_weight_changes.py
```

Default output: `/data/matched-weight-trajectories-v1`. The output lock prevents concurrent writers. Each completed arm is saved and skipped on resume. The manifest pins the script and source identities; changed measurement code requires a new output directory.

Copy `weight-trajectories.csv` locally under `outputs/matched-weight-trajectories-v1`, then plot:

```bash
.venv/bin/python scripts/plot_matched_weight_changes.py
```

The plot script produces a six-model overview, separate six-dataset figures for each model (cumulative and interval change), layerwise paired-gap heatmaps, and a combined PDF. Curves average seeds and shading spans them. The overview instead pools all six datasets and shades the full run range. Shading is not a confidence interval.

Raw norms depend on the model size, adapted matrices, and training recipe. Phi adapts fused QKV, Qwen adapts both full-attention Q/V and linear-attention projections, and the other configured families target Q/V projections. Make direct original/control comparisons within each model. Weight-change magnitude does not by itself establish harmfulness or warning performance.
