# Patch the coordinates that changed most

This experiment tests an association-based selection rule, not a gradient saliency score. For each identical probe prompt and layer-31 coordinate, calculate abs(step20_activation - step0_activation), rank descending, and replace only the largest-changing coordinates.

Start at layer 31 because the reverse whole-vector experiment had its strongest effect there. This does not establish it as the best mechanistic location. Retain the same 30 prompts and step-20 checkpoint as the earlier studies; no training or paid judge calls occur during generation.

Budgets out of 4096 coordinates:

| Fraction | Coordinates |
|---|---:|
| 1% | 41 |
| 5% | 205 |
| 10% | 410 |
| 25% | 1024 |

At each size, compare largest-change selection with a count-matched random selection. A fixed prompt-specific random ordering (seed 42) gives nested random subsets across budgets. There is only one random ordering in this first pass; repeat with more seeds before claiming a robust advantage. Top-change patches generally have greater perturbation energy than random patches; selected and full delta L2 norms are exported to make that distinction visible.

Each direction includes unpatched, same-model self-patch and full-vector controls, plus eight partial-patch conditions. 11 conditions × 30 prompts × 2 directions = 660 generated answers.

- Restore: replace selected step20 coordinates with step0 values.
- Transfer: replace the same selected step0 coordinates with step20 values.

All other coordinates retain the live recipient values. Only the final prompt token's initial layer-31 output is patched; decoding positions are untouched. Coordinates are residual-stream dimensions, not necessarily single neurons or independent concepts. Masks are computed separately for every prompt and are not a shared learned intervention. No response label is used to select coordinates.

After syncing source and configs, run on the GPU:

```bash
python scripts/run_top_change_activation_patch.py
```

Repeat to resume. It loads models sequentially (0, then 20, then 0), retaining endpoint tensors on CPU. It binds the adapter to the exact weight hash used previously and checks tokenization and mask consistency on resume. No old outputs are overwritten.

Output: outputs/activation-patch-top-change/generated.jsonl, run_report.json, pilot.json, coordinate_selection.json, endpoint_activations.pt and reviews/*.csv. The coordinate-selection file gives exact patched indices and delta norms. The tensor file supports later visualizations of selected coordinates in the actual freshly captured states. Transfer these outputs locally for scoring/plotting. Model checkpoints are not needed locally.

To stage for transfer via Git on the GPU:

```bash
git add -f outputs/activation-patch-top-change/generated.jsonl \
  outputs/activation-patch-top-change/run_report.json \
  outputs/activation-patch-top-change/pilot.json \
  outputs/activation-patch-top-change/coordinate_selection.json \
  outputs/activation-patch-top-change/endpoint_activations.pt \
  outputs/activation-patch-top-change/reviews/
```

Then commit/push and pull locally.

Once generation finishes, prepare labeling and inspect its cost estimate before any paid run:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/activation-patch-top-change/reviews \
  --output-dir outputs/activation-patch-top-change/labeling
```

Use the existing label run command with that directory and an explicit budget. Preparation does not call the API. There are 660 responses, so the cost is higher than the 180-answer pilot. No pricing estimate is claimed until actual answer lengths are available.

After labeling:

```bash
python scripts/score_top_change_activation_patch.py
```

The scorer produces top-versus-random curves, harmfulness presence grids for each direction, paired answer CSVs and a report. Repairs count only freshly harmful step20 target answers. Newly induced harm counts only freshly non-harmful baseline targets, with a separate donor-harmful transfer subset in the report. New harm on previously safe targets and all five benign controls remain side-effect checks. Unknown labels must be resolved rather than imputed as zero.
