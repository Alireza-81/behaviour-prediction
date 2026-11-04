# One shared activation mask, tested on fresh prompts

The research question is whether a common set of activation locations contributes to the behaviour across prompts. This experiment freezes **the same coordinate indices for every prompt, every generation step, and both patch directions**. Live replacement values still come from the donor on the recipient's current text prefix.

The earlier zero-harm result used a different selection for each prompt. It demonstrates prompt-specific control and does not establish a shared mask. This experiment tests that missing step.

## Frozen artifact already prepared locally

`artifacts/shared-activation-mask-v1/mask.json` contains the exact selected locations and an identity hash. `coordinate_ranking.csv` lists the shared score of every coordinate; `selected_coordinates.png` displays the frozen locations in index order. Its 64×64 grid is a display of indices, not spatial anatomy.

Selection uses the saved sequence-level gradient scores from `outputs/activation-patch-gradient-dynamic/coordinate_scores.pt`:

1. Use the 25 development target requests; include restoration and transfer, giving 50 score vectors.
2. Each score vector already aggregates the signed replacement benefit over all reference-answer prediction positions.
3. Normalize each vector by its sum of absolute scores so that a few large-score prompts cannot dominate purely by scale. Preserve signs: a predicted counterproductive replacement remains negative.
4. Average the 50 normalized vectors coordinate by coordinate.
5. Rank once and freeze the highest-scoring 205 coordinates (5%) and 1,024 coordinates (25%). The 5% set is a subset of the 25% set.

The two directions contribute equally and use the **same final sets**. The five benign development prompts do not select or tune the ranking. Fresh benign prompts test selectivity. This is an aggregation baseline, not learned-mask optimization. It can miss interactions or specialized circuits, so a failure would not show that no shared mechanism exists.

The artifact also freezes random masks for seeds 17, 42 and 123 at each budget. Each random mask is also identical across all evaluation prompts and both directions. Random masks match coordinate counts, not intervention energy.

## Fresh evaluation

`data/evaluation/shared_mask_fresh.jsonl` contains 45 new prompts: 15 direct harmful requests, 15 indirect harmful requests, and 15 benign controls concerning related topics. The preparation checks disjoint IDs and exact prompt text against the development, monitoring, and previous separate-evaluation prompts. New wording does not establish independence from every underlying template or harm topic.

The settings are fixed before any model runs on these questions: layer31, initial final-prompt-token patch plus every decoding step, 256-token generation limit, budgets 5% and 25%, and the frozen coordinate locations. The runner never computes gradients on evaluation prompts and never uses their reference answers to select or change coordinates.

It generates original and trained unpatched answers for controls, then applies the frozen masks. A baseline forward pass still supplies context-appropriate values at each step. The protocol is therefore prompt-independent **selection**, with context-dependent donor **values**. It does not require fixed activation values or a constant response vector.

## 990 answers

45 prompts × 2 directions × 11 conditions:

- No patch, self-patch throughout, full-vector donor copy throughout.
- Shared 5% and shared 25%.
- Three fixed random seeds at 5%, and the same three seeds at 25%.

Self-patches must reproduce recipient tokens exactly; full-vector patches must reproduce donor tokens exactly. A saved development sentinel checks that adapter switching reproduces the original endpoints. Both endpoints use separate caches while following the recipient's generated tokens. A single quantized PEFT model is loaded; no new language-model training is needed.

All masked rows record the same frozen mask ID and exact indices. Resume checks those indices against the artifact. Source checkpoint hashes, mask content, prompt file and saved plan are bound to the experiment. Changing the masks/settings requires a new version, not silently altering this test.

## GPU command

Sync the new source, `configs/activation_patch_shared.json`, `data/evaluation/shared_mask_fresh.jsonl`, and the complete `artifacts/shared-activation-mask-v1/` directory to the GPU. The same step20 checkpoint must still be present at `outputs/early-steps/hate_speech/`. Old gradient tensors and old generated-answer files are **not** needed on the GPU for this evaluation.

Run from the repository with its environment activated:

```bash
python scripts/run_shared_activation_patch.py
```

The command writes `outputs/activation-patch-shared/`. Repeat exactly the same command to resume. It generates 22 review CSVs with 45 answers each. Generation makes no paid judge calls. Existing experiments are preserved.

After generation, transfer the new output folder back locally. Prepare labeling and inspect the actual estimate:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/activation-patch-shared/reviews \
  --output-dir outputs/activation-patch-shared/labeling
```

If labeling was prepared before generation finished (for example, the original 800 answers), keep that same folder. Finish generation first, then append the new review rows with:

```bash
python scripts/label_with_claude.py prepare --all-responses --extend \
  --review-dir outputs/activation-patch-shared/reviews \
  --output-dir outputs/activation-patch-shared/labeling
```

This is offline preparation. It archives the original manifest/results/CSVs under `labeling/history/`, retains previous row indices and API results, and appends only new answers. Saved label edits are retained. Invalid/uncertain completed judgments are not automatically resubmitted or turned into safe labels. Changed or missing old answers cause an error. An interrupted extension can be recovered by repeating the extension command. Do not run labeling and extension concurrently.

When generation reaches 990 answers and the old job had 800 completed results, preparation should show **990 total, 800 retained, 190 added/pending**. If it reports no new rows, the review CSVs have not gained the missing generated answers yet. Increasing the API budget alone cannot add responses to a frozen 800-row manifest.

Resume the usual `run` command afterwards. Its budget includes the original recorded spend, so a $2.50 budget includes the approximately $1.48 already recorded for those 800 answers. The additional cost estimate is printed during extension; the budget guard is not a provider-side spending cap.

Choose a spending cap after reading that estimate. This example uses $3 as a maximum, not a price prediction:

```bash
python scripts/label_with_claude.py run \
  --output-dir outputs/activation-patch-shared/labeling \
  --budget-usd 3
```

After labels are complete:

```bash
python scripts/score_shared_activation_patch.py
```

The scorer verifies identical locations across prompts/directions and the self/full controls. It rejects unresolved binary labels and conflicting labels on identical answers. It saves target harmfulness, repairs, induced harm, benign harmfulness, truncation/repetition diagnostics, presence grids, and a shared-versus-random plot. Random bands show the observed range across three seeds, not confidence intervals.

`benign_utility_review.csv` separates usefulness and unnecessary-refusal review from harmfulness. The automatic refusal field is only a keyword hint. A harmless but unhelpful answer, or a failed generation, must not be treated as proof of preserved utility.

## What a positive result would support

A small frozen set that transfers or removes behaviour on fresh prompts supports a common intervention-relevant set for this checkpoint. Success in both directions is useful corroboration, but not proof that this is the sole original mechanism. At layer31 the locations are residual-stream coordinates, not necessarily distinct neurons or independent concepts.

This first artifact is selected from one hate-speech fine-tune. It cannot establish locations for harmful behaviour in general across self-harm, threats, other models or training runs. Those are subsequent transfer tests. Final-layer patching also controls the output downstream and does not by itself establish where the behaviour is first computed. We should distinguish a reusable effective intervention from a complete mechanistic explanation.

No zero-harm result is claimed for the shared mask yet; the GPU evaluation remains to be run.

## Reproducibility and local checks

The frozen artifact was produced with:

```bash
python scripts/prepare_shared_activation_mask.py
```

It refuses to overwrite an existing mask/config. To study another selector or source set, explicitly choose a new output/version and config path. The evaluation runner does not rerun preparation.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/behaviour-matplotlib \
  python -m pytest tests/test_shared_activation_mask.py \
  tests/test_dynamic_gradient_patching.py -q
```

Tests cover equal weighting, exclusion of benign/evaluation scores from selection, deterministic nested masks, identity/tamper checks, identical masks across prompts/directions, interruption/resume, and complete review/scoring output. The evaluation test makes gradient calculation raise an error, confirming that no evaluation gradients are used. These are tiny-model tests; real GPU compatibility is still checked at runtime.
