# Gradient-only coordinate selection

This is a **selection comparison**, with no answer generation, model training, or Claude labeling. It preserves the existing multilayer experiment and writes to a new folder.

Old score for a block/coordinate:

`sum_over_positions(-gradient * (donor_activation - recipient_activation))`

New score:

`sum_over_positions(abs(gradient))`

Both use the gradient of the same **mean-token KL loss toward the donor model's distribution**, at the same teacher-forced reference positions. This is a donor-distribution proxy, not directly a gradient of harmfulness. Earlier activation positions can affect later token losses through attention.

Absolute gradient means sensitivity to a small activation change, independent of the donor activation difference's size or direction. We take absolute values **before** summing so opposite signs at different positions do not cancel. Signed gradient sums are saved as an auxiliary diagnostic, but are not used for ranking. In contrast, the old score estimates the benefit of the actual donor replacement. A highly sensitive coordinate can still have almost identical donor/recipient values, or its donor change may point in a counterproductive direction.

## GPU commands

Sync the updated `src/behaviour_prediction/monitor/multilayer_patching.py` and new `scripts/compare_gradient_only_selection.py` to the GPU project, retaining the existing experiment files. Activate the same environment and run:

```bash
python scripts/compare_gradient_only_selection.py capture
```

This performs **50 development sequence-gradient calculations** (25 existing target prompts × both directions), using the exact checkpoint, tokenizer, reference tokens and model verification from the original experiment. Each backward pass captures all 32 blocks, covering all reference prediction positions. It saves both old weighted scores and gradient-only statistics in the same pass. Original generation and label files are not touched.

The old `development_scores.pt` contains weighted products only, so it cannot supply raw gradient magnitudes. New calculations are required, but references are reused: no new answers are generated. The checkpoint must still be available at the original run path. There are no API calls.

Rerun the same capture command after interruption: complete entries are skipped. A lock prevents two capture/comparison commands writing the same output folder. Capture is deliberately single-process; it does not touch the four-worker generation system.

After capture, run (GPU or local machine with the captured files):

```bash
python scripts/compare_gradient_only_selection.py compare
```

## What is held fixed

- Development target prompts, both directions, reference answers, token positions, donor KL objective and checkpoint.
- Coordinate budgets **102, 205, 410**.
- Same shared locations across all prompts and both directions.
- Same four location restrictions: layer31 only, unrestricted all-layer, distributed depth quotas, earlier-only.
- Per-example L1 normalization across all model coordinates followed by equal averaging. Layer31-only retains the original layer-only normalization policy, held constant between the two scoring variants.

This stage does not generate random controls or evaluate new masks' harmfulness: its question is how selection changes. Random and behavioural comparisons can follow if the selection looks worth testing.

## Outputs

All outputs go to `outputs/activation-selection-gradient-only/`:

- `global_layer_distribution.png`: old versus gradient-only counts by block at all three total budgets, for unrestricted selection.
- `coordinate_overlap.png`: percentage of original coordinates retained for each method/budget.
- `coordinate_overlap.csv`: overlap counts, Jaccard similarity, number of blocks used and last-block counts.
- `layer_counts.csv`: counts for each method, budget and block.
- `coordinate_ranking.csv`: coordinate-level global aggregate scores for both definitions (not the separate layer-only normalization).
- `gradient_only_mask.json`: the new locations, aggregates, cache hash and identity. This is a selection artifact, not yet an evaluated intervention.
- `gradient_scores.pt`: per-development-example weighted scores, absolute-gradient sums and signed-gradient sums (about 75 MiB total).
- `comparison_report.json`, `summary.md`: results and limitations.
- `same_capture_overlap.csv`: comparison of both scoring rules from the **same new backward passes**.

The report checks how closely rerun weighted scores reproduce the old cache. The original-mask comparison answers the practical selection-change question; the same-capture comparison separates the scoring rule from any numerical differences between the old GPU run and the new one. A non-bit-exact rerun is reported rather than hidden.

## Interpretation

An increase in early-layer selections would show that gradient sensitivity alone ranks those sites differently; it would not establish that replacing them removes harmfulness. Gradient magnitudes depend on representation scale across layers. No layer-wise scale correction is added here, because that would change a second part of the selection rule. New selections should eventually be tested with actual patches before drawing behavioural conclusions.
