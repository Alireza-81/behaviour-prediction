# Compare five activation-coordinate scoring methods

Selection analysis only: no model training, new response generation, or Claude labeling. Existing multilayer results and the earlier gradient-only folder are preserved.

## Run this combined capture instead of the gradient-only capture

Sync the updated project files, including:

- `src/behaviour_prediction/monitor/multilayer_patching.py`
- `scripts/compare_activation_scoring.py`
- The existing `scripts/compare_gradient_only_selection.py` and multilayer runner/helper files (imported utilities).

In the project directory on the GPU machine, activate the same environment and run:

```bash
python scripts/compare_activation_scoring.py capture
```

Then:

```bash
python scripts/compare_activation_scoring.py compare
```

Capture does **50 development backward passes**: 25 target prompts × restoration/transfer, with all 32 blocks per pass and all frozen reference-answer prediction positions. It uses the original step20 checkpoint, the original reference tokens, the same tokenizer and endpoint/sentinel verification, and the same mean-token donor KL objective. All five statistics are collected in that same pass; there are not 50 backward passes per method. Existing answers supply the references, so no new generation is required.

Rerun `capture` after interruption; it skips complete saved entries. A folder lock prevents concurrent writers. Capture is single-process and uses one model with adapter switching, with no parameter optimization. `compare` is CPU-only and can run locally after copying the captured files and original source data.

If the earlier gradient-only capture has already run, its compact statistics do not contain absolute token-wise attribution, activation-change RMS or gradient×activation scores. This combined capture must still run to obtain those fields. Do not overwrite or try to reinterpret the earlier cache.

## Exact scoring definitions

Let `g` be the gradient of the **same mean-token donor-distribution KL loss** with respect to a recipient block's residual output, at aligned reference prediction positions. Let `h` be the recipient activation and `delta = donor - recipient`.

| Method | Token aggregation | Meaning |
|---|---|---|
| Signed replacement score (current) | `sum(-g * delta)` | Predicted local benefit of the actual donor replacement |
| Absolute replacement attribution | `sum(abs(g * delta))` | Size of the predicted effect, whether beneficial or counterproductive |
| Absolute gradient only | `sum(abs(g))` | Local sensitivity, independent of donor-change size |
| Activation-change magnitude | `sqrt(mean(delta ** 2))` | Size of the endpoint change; no gradient factor |
| Absolute gradient×activation | `sum(abs(g * h))` | Local sensitivity to scaling/zeroing the recipient activation |

Absolute values precede the sum over token positions; `abs(sum(...))` is deliberately not used because it would cancel opposite token-wise effects. Earlier positions can affect later-token losses through attention. Signed gradient sums are saved as an auxiliary diagnostic but are not one of the five primary rules.

Per prompt/direction, normalize the score matrix by its global L1 norm, then average examples equally. This is the same global normalization policy for all five rules. For the layer31-only comparator, retain the original layer-only L1 normalization, identically for all five rules.

Budgets are **102, 205 and 410 total `(block, coordinate)` locations**. Compare all four existing location restrictions: unrestricted all-layer, layer31-only, distributed depth quotas, earlier-only. Unrestricted all-layer selection is the primary plot, so we can see where each score naturally concentrates without forcing distribution.

## Shared and directional comparisons

- **Shared:** aggregate all 50 examples into a single ranking. These are the candidates for a common mask used regardless of prompt or direction.
- **Restoration:** aggregate only the 25 baseline→trained examples.
- **Transfer:** aggregate only the 25 trained→baseline examples.

Direction-specific selections are diagnostics; this stage does not switch the shared mask per evaluation prompt. Restoration and transfer use different original reference continuations and endpoints, so agreement is not a clean test of endpoint direction alone.

Three uniform random selections (seeds 17, 42, 123) provide a reference for how counts spread across blocks without any score. Their nested masks are saved and their mean layer counts are plotted. They are a **selection-layout reference**, not random behavioural controls matched to the per-layer counts of each new scored mask. Such controls would need to be constructed if these masks are later evaluated through generation.

## Files to inspect

All outputs go to **`outputs/activation-selection-scores/`**:

- `layer_distribution.png`: five methods plus the random reference, shown by block for each budget.
- `scoring_overlap.png`: pairwise exact-coordinate overlap between the five shared unrestricted selections.
- `direction_agreement.png`: restoration versus transfer overlap for each score and budget.
- `selection_summary.csv`, `layer_counts.csv`: counts for every rule, restriction, budget and view.
- `pairwise_overlap.csv`: pairwise overlaps/Jaccard similarities for all four restrictions and three views.
- `direction_agreement.csv`: shared/restoration/transfer agreement for all restrictions.
- `overlap_with_original.csv`: overlap with the previously frozen weighted-score masks.
- `random_layer_counts.csv`, `random_selection_overlap.csv`: random-layout diagnostics.
- `scoring_masks.json`: all selected coordinate lists and the score-cache identity; not yet evaluated interventions.
- `aggregate_scores.pt`: global and layer31 aggregate scores for every method/view.
- `scoring_statistics.pt`: six float32 matrices per example (five primary rules plus signed-gradient diagnostic), about **150 MiB** for 50 × 32 × 4096 values. Full token activations are not stored.
- `comparison_plan.json`, `comparison_report.json`, `summary.md`: definitions, source hashes, reproduction checks and readable summary.

The report compares rerun signed replacement scores against the original development cache. Main pairwise comparisons are always computed from the same new backward passes, so a numerical change between old and new GPU runs is not mistaken for a scoring-rule change. The separately reported legacy-mask overlap can reflect both.

## What this can establish

It can show whether dropping the activation-difference factor changes layer preference, whether signed effects cancel useful sensitivity signals, and whether different scores agree on coordinate locations. It cannot establish that one score suppresses or transfers harmfulness better without actually testing its masks through patching.

Raw gradients and activation differences depend on representation scale. All gradient-based methods still target the donor KL proxy, not a direct human/judge harmfulness objective. Zero-valued score examples contribute zero after normalization and are counted in the report; stable index ordering resolves exact ties, which should not be interpreted as meaningful importance.
