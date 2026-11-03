# Gradient selection and patching throughout the answer

This is a separate experiment from `docs/gradient-activation-patching.md`. The earlier command remains unchanged. No checkpoint retraining is needed.

## What changes

The earlier experiment selects using the first-token prediction and patches once at the final prompt token. This version does two additional things:

1. **Selection considers all reference-answer tokens.** It replays the same prompt and reference continuation through step0 and step20. It differentiates the mean per-token KL divergence between donor and recipient vocabulary distributions with respect to every corresponding layer-31 output. It sums signed `-gradient * replacement_difference` scores across positions, producing one fixed coordinate ranking per prompt and direction.
2. **Generation refreshes the replacement at every token.** The donor receives the actual token chosen by the patched recipient, computes fresh activations for that prefix, and supplies the selected coordinate values. Both endpoints keep independent KV caches.

The reference is the donor's unpatched answer from the frozen top-change experiment. For restoration the donor is step0; for reverse transfer it is step20. Both teacher-forced passes receive identical tokens, including the preceding reference tokens, rather than independently generated answers. The first reference token is predicted at the last prompt position; every subsequent reference token has its own prediction position. Generated EOS, when present, is included. No token in the reference is silently dropped from the objective.

The language-model weights remain unchanged. This is first-order sequence attribution, not RL, integrated gradients, or learned-mask training. Matching donor probabilities across reference-answer positions is still a proxy for the behaviour we care about. Free-generation responses need fresh labels. It does not differentiate through sampling or the Claude judge, and does not guarantee safer responses on unseen inputs.

## One GPU model, two logical endpoints

The runner loads the step20 PEFT checkpoint once. It disables its default adapter for step0 passes and enables it for step20 passes. Donor and recipient forward passes alternate, with separate caches, so two model copies are not required. Initial activation captures must match the previous experiment exactly, verifying that adapter switching reproduces both endpoints.

During gradient selection, model parameters are temporarily frozen and an activation leaf at the final block receives gradients. This avoids backward passes through the quantized transformer blocks. It still needs vocabulary logits for the reference positions, so transient memory exceeds the first-token selector. Stored results contain compact coordinate scores, not all token-by-coordinate activation tensors.

## Conditions and interpretation

There are **600 new responses**: 30 prompts × 2 directions × 10 conditions.

| Condition | What happens |
|---|---|
| `unpatched` | Recipient generates normally. |
| `self_dynamic` | Same-checkpoint values replace the recipient values every step; answers must match unpatched tokens exactly. |
| `full_dynamic` | Full final-layer vector is replaced every step; answers must match the donor's old unpatched tokens exactly. |
| `gradient_01pct`, `05pct`, `10pct`, `25pct` | Sequence-gradient-selected 1%, 5%, 10%, or 25% is replaced every step. |
| `top_25pct` | Select 25% by RMS difference across the same reference positions; replace every step. |
| `random_25pct` | Replace 25% chosen by a fixed prompt-specific random ordering every step. |
| `prefill_gradient_25pct` | Use exactly the `gradient_25pct` mask, but replace only the final prompt token once. |

The last condition isolates duration at 25%: selection, mask, and endpoint models are held constant. Gradient/top/random comparisons at 25% isolate selection under the dynamic protocol. The top/random masks here are computed from reference-answer positions; they are not the old final-prompt-only masks. Comparing the two complete experiments also changes the selection objective, so do not attribute every difference to duration alone.

Full last-layer replacement is a control: it copies the donor's next-token logits at every step and therefore its greedy answer. It is not evidence of discovering a unique harmfulness circuit. The step0 donor already has a failure on this development set, so copying that donor is not a zero-harm guarantee. Five benign controls are insufficient for a broad claim of preserved capabilities.

## Run on the GPU

First sync the new code/configuration. Keep the same step20 checkpoint and these existing files on the GPU:

```text
outputs/activation-patch-top-change/pilot.json
outputs/activation-patch-top-change/generated.jsonl
outputs/activation-patch-top-change/endpoint_activations.pt
```

Run the original first-token experiment when ready:

```bash
python scripts/run_gradient_activation_patch.py
```

Then run the full-generation experiment:

```bash
python scripts/run_dynamic_gradient_activation_patch.py
```

The commands are independent and can run in either order, but run them sequentially on the same GPU. The second command uses `configs/activation_patch_gradient_dynamic.json` and writes to `outputs/activation-patch-gradient-dynamic/`. Repeat that same command to resume. It preserves the first experiment and existing labels. Generation makes no paid API calls.

This version performs donor as well as recipient decoding passes, so it will take longer than a one-time patch. References can be truncated at the old generation limit; `references.json` records that fact. The new generation still has a 256-token maximum.

## New outputs

- `reviews/*.csv`: 20 CSVs, one per direction/condition, ready for labeling.
- `generated.jsonl`, `pilot.json`, `run_report.json`: responses, frozen plan, and control checks.
- `coordinate_scores.pt`: per-coordinate gradient and RMS rankings, with per-position summed attribution scores.
- `coordinate_selection.json`: exact masks, counts, replacement norms, and predicted benefits.
- `references.json`: reference tokens, endpoint steps, identities, and truncation flags.

The runner binds the checkpoint and prior experiment by hash, checks source tokenization, and validates unpatched/self/full controls. An interrupted last journal record is backed up before repair. Completed answers are skipped; partial resumption checks that recomputed scores and masks match the saved selections.

## Label after generation

Copy the new output folder back locally. Prepare a labeling job and read its estimated cost before running the paid command:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/activation-patch-gradient-dynamic/reviews \
  --output-dir outputs/activation-patch-gradient-dynamic/labeling
```

After inspecting the estimate, choose your cap; this example uses a $2 maximum, not a promised price:

```bash
python scripts/label_with_claude.py run \
  --output-dir outputs/activation-patch-gradient-dynamic/labeling \
  --budget-usd 2
```

Repeat the paid command to resume labeling. Enter the key privately in the hidden prompt.

After labels are ready:

```bash
python scripts/score_dynamic_gradient_activation_patch.py
```

The scorer verifies references, controls, masks against saved scores, and label alignment. It stops with source-file paths in `label_conflicts.csv` if identical answers have conflicting labels. Results include a comparison plot in harmful-answer counts, separate presence/absence plots for each direction, paired decisions, flagged responses, and a report in `outputs/activation-patch-gradient-dynamic/results/`.

## Validation

Local tests cover sequence gradients against finite differences, nonzero later-answer gradients, unchanged weights, separate caches following the recipient's tokens, full-donor copying, self/zero-mask controls, equivalence of the one-time duration control to the existing patch hook, and end-to-end generation/resume/scoring using a tiny PEFT Llama.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/behaviour-matplotlib \
  python -m pytest tests/test_dynamic_gradient_patching.py \
  tests/test_gradient_activation_patching.py -q
```

The real NF4 GPU/Transformers stack remains to be validated by the runtime checks; local tests do not establish effectiveness on the research model.
