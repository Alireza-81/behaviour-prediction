# Shared masks throughout the model

This experiment is prepared for the GPU machine; no GPU run or paid labeling was performed locally.

## Comparison

Use **102, 205, and 410 total residual coordinates**, not a percentage of every block. A coordinate is a `(block, residual-dimension)` location. Each selected location is fixed across all evaluation prompts and both directions, and is patched at the last prompt token and every generated token. Values come from the live donor on the recipient's prefix, with separate endpoint caches.

Four selection methods:

- `layer31`: highest scores in the last block, using the previous layer-only normalization.
- `global_all`: highest scores anywhere; this can concentrate at the end.
- `distributed`: approximately one-third of the budget in each depth region: blocks 0–9, 10–20, 21–31. Within each region, select the highest scores. This is an explicit constraint, not evidence that the mechanism is distributed.
- `earlier`: highest scores in blocks 0–30; exclude the last block.

Each method/budget also has **three random masks**, seeds 17, 42, 123, matching the exact selected count in every layer. This controls layer choice and count, not perturbation energy. Three random seeds provide descriptive variation, not a confidence interval.

## Workload

- No language-model training and no learned-mask training; checkpoint weights remain fixed.
- **50 development sequence-gradient calculations**: 25 existing development target prompts × two directions. Each calculation scores all 32 blocks in one backward pass. Includes every teacher-forced reference-answer prediction position, not only the first token. Reuses existing frozen reference answers, embedded in the config; no extra development generation.
- Scores are signed `-gradient * (donor - recipient)`, summed over reference positions. Normalize each prompt/direction across all coordinates by its L1 norm, then average. Layer31-only uses its own L1 normalization as previously. Gradients measure a donor-distribution KL proxy, not directly judged harmfulness. Simultaneous patches can interact.
- **48 partial-mask conditions** = four methods × three budgets × (selected + three random masks).
- Three controls: unpatched, all-layer self-copy, all-layer donor-copy.
- **4,590 total evaluation rows** = 51 conditions × 45 prompts × two directions. Prompts are 30 targeted requests and 15 benign controls. These are the already examined shared-mask evaluation prompts; use this as a method comparison, not a fresh confirmatory test.
- The generator now automatically imports matching answers from `artifacts/multilayer-reuse-v1/reuse.json`: 90 unpatched answers, 270 layer31/205 random-control answers, and another 90 selected layer31/205 answers **if the newly frozen coordinate lists match exactly**. Thus **up to 450 generations are reused**, leaving **4,140 new answers** when all match (4,230 if the selected list differs). It also skips saved unpatched controls on resume. Fresh all-layer self/full controls still run to validate the new implementation. The bundle is checked by content hash and experiment settings; it does not require old output folders on the GPU machine.
- Each answer has at most 256 generated tokens: **1,175,040 generated tokens maximum**. Patched generation runs both endpoints per token, so it costs approximately two forward passes per output token. The number of newly generated answers is about 4.2 times the previous 990-answer shared-mask experiment when all reusable masks match, but actual runtime depends on answer lengths and hardware.
- Labeling prepares 4,590 rows, then the offline reuse command below imports existing judgments for exact matching prompt/response/truncation payloads with the same judge model and rubric. This includes matching self/full controls after they have been generated and checked. Imported rows cost zero for this run and preserve source provenance; historical API charges are not charged again. Additional repeated answers can also match. Use the remaining-row count after reuse before making paid calls; the prepare estimate precedes this discount. No estimate of dollars or GPU hours has been invented.
- No extra model checkpoints are written. Development scores are about 25 MiB float32, plus the mask, reports and text outputs. All-layer backward passes use more working GPU memory than the former final-layer-only calculation; 4-bit model loading does not quantize intermediate activations.

## GPU commands

Run from the project root with the project's environment activated and the exact step20 adapter available at `outputs/early-steps/hate_speech/`. The checked-in configuration includes checkpoint hashes, the sentinel, development prompt token IDs and reference token IDs, so old development outputs are not needed on the GPU machine. Both stages verify the actual model against the sentinel.

```bash
python scripts/run_multilayer_activation_patch.py select
```

After selection, inspect `outputs/activation-patch-multilayer/selection_by_layer.png` (and `.csv`) to see where the frozen masks lie. The script writes `mask.json` and `development_scores.pt` in that folder.

Then generate:

```bash
python scripts/run_multilayer_activation_patch.py generate
```

Both commands resume by rerunning the same command. Selection skips saved prompt/direction scores. Generation skips completed answer rows and exports review CSVs incrementally. A partial final journal line caused by interruption is backed up and removed on resume. Do not change config/mask files while resuming. Existing experiment folders are left alone.

## Labeling and scoring

Wait until generation is complete, then prepare the judge job. Preparation is offline and prints its cost estimate.

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/activation-patch-multilayer/reviews \
  --output-dir outputs/activation-patch-multilayer/labeling \
  --model claude-haiku-4-5
```

Before any API calls, reuse existing labels:

```bash
python scripts/reuse_multilayer_activation_patch.py labels
```

This prints how many rows remain. Rerunning it skips completed entries and makes no API calls. A changed judge rubric/model or payload will not inherit an old label.

Then supply your chosen cumulative budget when running it:

```bash
python scripts/label_with_claude.py run \
  --output-dir outputs/activation-patch-multilayer/labeling \
  --budget-usd YOUR_BUDGET
```

If you prepared before generation finished, use the same prepare command with `--extend`, then rerun the offline label-reuse command and `run`. The extension preserves existing paid results and adds newly generated rows.

After resolving invalid/uncertain/conflicting labels:

```bash
python scripts/score_multilayer_activation_patch.py
```

Results are saved in `outputs/activation-patch-multilayer/results/`: `method_comparison.png`, restoration/transfer presence plots, `summary.md`, `report.json`, paired decisions, review flags and benign utility review. The scorer requires complete generation and labels, checks fixed selections and self/full controls, and rejects conflicting labels on identical prompt/answer pairs.

Inspect benign usefulness and unnecessary refusals, not just harmfulness. A zero-harm score alone is insufficient. Distributed success would establish control by this intervention on this checkpoint; it would not prove unique or universal harmfulness locations. If the unrestricted selection stays concentrated in layer31, report that rather than interpreting the constrained mask as a discovery of distribution.
