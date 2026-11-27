# Qwen all-layer diagnostic

This exploratory analysis checks whether the original/control drift ordering depends on decoder depth. It reuses Qwen3.5-9B's archived adapters for all six advice corpora, seeds 101 and 202, both arms, and optimizer steps 0, 5, ..., 50. It does not train a monitor or change its selected layers.

The original 16 monitoring prompts, non-thinking chat template, last non-padding prompt token, and decoder-block output convention are retained. All 32 blocks (zero-based indices 0–31) are captured in the same forward pass. The exact model revision, adapter hashes, panel hash, layer types, and script hash are recorded. Saved measurements include per-prompt relative L2 drift, raw L2 drift, cosine distance, and activation-norm ratio. Dense activation vectors are not retained.

On aa2, start or resume:

```bash
cd ~/behaviour-prediction
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  .venv/bin/python -u scripts/capture_qwen_all_layers.py capture
```

The script is installed directly on aa2 for this side experiment. No git synchronization is needed for this command. Default output: `/data/qwen-all-layers-v1`. An output lock prevents duplicate writers. Completed checkpoints are skipped on resume; changed capture settings require a new output directory. It uses existing cached weights and adapters and makes no API calls.

CPU analysis runs after capture. It can also be repeated separately:

```bash
.venv/bin/python scripts/capture_qwen_all_layers.py analyze
```

Outputs:

- `manifest.json`: experiment identity and layer types.
- `<corpus>/<seed>/<arm>/<step>.json`: per-prompt metrics at each layer.
- `trajectories.csv`: individual-arm means, retaining each seed.
- `paired-gaps.csv`: original-minus-control means for each seed, layer, and step.
- `gap-*.png`: shared-scale layer-by-step heatmaps, averaged across seeds.
- `layers-<corpus>.png`: all-layer magnitude trajectories, with separate seed curves.
- `all-layer-trajectories.pdf`: heatmaps and per-corpus trajectories.
- `historical-consistency.json`: agreement with archived magnitude/orientation at layers 14, 15, and 31, when the historical feature archive is present.

Interpretation: positive paired gaps mean more movement under original training. They are not harmfulness labels. A reversal outside the previously measured layers supports layer dependence; a consistent ordering throughout the decoder argues against layer choice as the sole explanation. A large historical-consistency discrepancy must be investigated before interpreting the layer sweep. Numerical differences due to batching can occur.

Layers selected after viewing these results need fresh evaluation for claims about warning performance. The layer sweep alone cannot establish the cause of Qwen's different behavior.
