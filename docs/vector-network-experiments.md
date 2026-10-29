# Full activation vectors and a small nonlinear monitor

Completed results: `outputs/vector-network-experiments/summary.md`.

Reproduce locally from the project root, without GPU or API calls:

```bash
source .venv/bin/activate
python scripts/compare_vectors_and_network.py --output-dir outputs/vector-network-repeat
```

The runner is `scripts/compare_vectors_and_network.py`. The implementation is
`src/behaviour_prediction/monitor/vector_network.py`; tests are in
`tests/test_vector_network.py`. These are project files, not temporary scripts.

## Two separate changes

1. **Full vectors, linear classifier:** all 245,760 signed activation-change
   coordinates (30 probes x 2 layers x 4096 coordinates). Divide each probe delta
   by its initial activation norm. Fit centering and one global RMS scale on
   training rows only, setting average squared input norm to 42. This preserves
   relative vector geometry without coordinate-wise whitening. No PCA, class
   averaging, or dimension selection is used. The linear logistic classifier has
   L2=1 and threshold 0.5, and is computed through exact vector dot products.
2. **Small neural network, existing inputs:** retain the 42 summary measurements.
   Fit a 42 -> 8 tanh -> 1 network (353 parameters), using train-only input
   normalization, Adam lr 0.01, 300 steps, and L2 coefficient 0.1 on weights.
   Seed 42 is primary; seeds 0, 1, 2, 3 check sensitivity. No best-seed selection,
   evaluation-based early stopping, or parameter search is performed.

They are never combined. The second comparison changes the training recipe as
well as architecture, so it cannot isolate nonlinear architecture from every
optimization/regularization effect. Both use the same current 15/31 layers;
there is no layer comparison.

Each fold holds out one entire model run and trains on original canonical ground
truth only. Step zero is a reference, not a training/scoring example. All saved
baseline predictions must reproduce before comparison. Predictions are saved
before separate evaluation labels are used for scoring.

## Observations

Behaviour agreement on original / separate prompts:
- Baseline: 205/245 (83.7%) / 191/245 (78.0%).
- Full vectors: 202/245 (82.4%) / 192/245 (78.4%).
- MLP seed 42: 207/245 (84.5%) / 187/245 (76.3%).
- All five MLP seeds have the same correct counts in this experiment; raw scores
  can differ. No seed is selected based on those results.

On separate prompts, full vectors reduce false alarms from 19 to 13 but increase
misses from 35 to 40. Macro F1 declines slightly (0.829 to 0.825), and completely
correct seven-behaviour sets decline from 13 to 11 checkpoints. The extra one
correct binary behaviour decision is not a robust overall gain. Threats improve,
but hate speech and harassment worsen.

The MLP has 15 false alarms and 43 misses, versus baseline 19 and 35. Better
original-prompt agreement does not carry over to the separate prompts. This
fixed small-network recipe is not better overall; it does not rule out other
networks with more independent data.

Forecast predictions match the frequency baseline: 26/28 original and 28/28
separate. The separate set has no negative binary checkpoints, so its perfect
forecast accuracy does not establish early warning.

## Saved outputs

The output folder contains protocol and input hashes, the exact full-vector dot
products, fitted linear/neural monitors, predictions, per-run/per-behaviour
metrics, a summary, and comparison.png. The dot products plus indexed original
snapshots let the full-vector fits be reproduced without exporting many duplicate
large activation arrays.

The separate evaluation has already been inspected; these are exploratory
development comparisons. Seven model runs remain the independent units. Next
priority is finer early checkpoints and per-behaviour rates/change relative to
step zero. Layer comparison is deferred.
