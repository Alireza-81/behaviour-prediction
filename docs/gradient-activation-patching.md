# Gradient selection: controlled comparison with largest changes

This is a first-order attribution-patching experiment, not language-model training, a learned mask, AtP*, or integrated gradients. It uses the same hate-speech checkpoint, 30 prompts (25 target requests and five benign controls), layer-31 residual output, and one-time final-prompt-token replacement as the largest-change experiment.

## What the gradient measures

At the last block, changing the final prompt token directly changes only the first generated token's logits. A teacher-forced whole-answer objective at this location would not create gradients from later answer positions back to that prompt output. We therefore explicitly use the donor's **next-token distribution** as the target, rather than misrepresent the objective as whole-answer harmfulness.

For each prompt and direction:

1. Run the donor and recipient on the identical formatted prompt.
2. Compute `KL(donor_next_token_distribution || recipient_next_token_distribution)`.
3. Differentiate this loss with respect to the recipient's final-prompt activation vector.
4. Score each candidate replacement as `-gradient * (donor_value - recipient_value)`.
5. Rank signed scores from largest to smallest. Positive values predict reducing that KL; negative values predict making it worse. Fixed budgets can include nonpositive scores; the audit reports their count.
6. Generate a fresh complete answer with only the selected coordinates replaced, and label that answer using the existing process.

Restoration targets the step-0 distribution; reverse transfer targets step 20. This produces separate per-prompt rankings for each direction, unlike the direction-independent magnitude masks. It does not select a shared behaviour circuit or use Claude labels for selection. First-token style differences can drive the ranking without predicting harmfulness. Whether the selection improves full-answer outcomes is precisely what the new labels will test.

All model weights remain unchanged. The computation graph starts at a detached activation leaf; gradients pass through the remaining computation to that leaf, without training parameters. At layer 31 this avoids backpropagation through the quantized transformer blocks. An earlier-layer or dynamic whole-answer experiment would require a different objective and protocol.

## Frozen comparison and outputs

Budgets: 41, 205, 410, and 1024 of 4096 coordinates (1%, 5%, 10%, 25%). Seven conditions per direction: unpatched, self-patch, full replacement, and four gradient selections. **420 new answers**, reusing the earlier top-change/random experiment as the comparator rather than generating its conditions again.

The manifest binds the previous generated journal, endpoint tensor file, and plan by hash, and binds the step-20 adapter files. Fresh endpoint captures and tokenization must match the previous experiment. Fresh unpatched/self/full generation must exactly reproduce the corresponding previous tokens. If they differ, the runner stops rather than silently mixing conditions. Control answers are generated again to verify this, and get fresh labels.

The runner loads models sequentially (0, 20, 0), retaining endpoint tensors/logits on CPU. It saves each answer immediately and preserves existing human annotations in review exports. Rerunning the same command resumes. Gradient scores and selections are checked for consistency on resume; a complete run exits without loading any model. Existing experiment folders and checkpoints are not overwritten.

New output folder: `outputs/activation-patch-gradient/`.

- `reviews/*.csv`: 14 CSVs, one per direction/condition, for labeling.
- `generated.jsonl`, `pilot.json`, `run_report.json`: responses and protocol.
- `gradient_scores.pt`: activation gradients and signed scores.
- `coordinate_selection.json`: exact coordinates, replacement norms, predicted benefits, nonpositive-score counts.

## GPU command

Sync the new source/config first. The GPU needs the same checkpoint as before and these earlier outputs still available:

```text
outputs/activation-patch-top-change/pilot.json
outputs/activation-patch-top-change/generated.jsonl
outputs/activation-patch-top-change/endpoint_activations.pt
```

From the repository with its environment activated:

```bash
python scripts/run_gradient_activation_patch.py
```

Use exactly the same command to resume after interruption. No API key or paid labeling is involved in this generation command.

## Labeling after generation

Copy the new output folder back locally, including the score tensor and coordinate audit. Prepare a labeling job and inspect the cost estimate:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/activation-patch-gradient/reviews \
  --output-dir outputs/activation-patch-gradient/labeling
```

Then run with a budget you choose after inspecting that estimate, for example a $2 maximum:

```bash
python scripts/label_with_claude.py run \
  --output-dir outputs/activation-patch-gradient/labeling \
  --budget-usd 2
```

The $2 is a spending cap, not a cost estimate. This paid command is for you to run privately; the key prompt remains hidden. Use the same command to resume labeling.

## Comparison after labeling

Keep the previous `outputs/activation-patch-top-change/labeling/` locally as well. Run:

```bash
python scripts/score_gradient_activation_patch.py
```

The scorer checks labels against generated answers and judge manifests, validates common control generations and recorded masks, and compares the three selection rules at identical coordinate budgets. If identical prompt/response pairs have conflicting labels across methods, it writes `label_conflicts.csv` with source-file paths and stops for resolution. It does not change source labels automatically.

Results include `gradient_vs_magnitude.png`, `paired_decisions.csv`, `needs_review.csv`, `summary.md`, and `report.json` under `outputs/activation-patch-gradient/results/`. Truncation/review flags remain explicit; no flagged row is silently converted to safe. Results are exploratory: one trained model, inspected prompts, per-prompt selections, and one count-matched random ordering. Count matching does not match replacement energy.

## Local validation

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/behaviour-matplotlib \
  python -m pytest tests/test_gradient_activation_patching.py \
  tests/test_partial_activation_patching.py -q
```

Tests verify gradients against finite differences, unchanged model weights, hook cleanup, signed ranking, complete generation/export/resume and scoring on tiny local Llama models. They do not establish compatibility or efficacy on the actual GPU stack; runtime endpoint/control checks provide the next validation there.
