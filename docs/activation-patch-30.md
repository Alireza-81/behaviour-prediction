# Expanded patch experiment: 25 harmful requests and 5 benign controls

Continue using the retained hate_speech checkpoint at step 20; no retraining is required. Fresh prompt set: 13 direct and 12 indirect hate-speech requests, plus five short benign controls. No requests were selected using this model's answers. A harmful request does not guarantee a harmful answer. Five benign prompts give a small side-effect check; they cannot establish general capability preservation.

Freeze: configs/activation_patch_30.json and data/probes/activation_patch_30.jsonl. Original pilot outputs remain unchanged. Same six conditions: baseline 0, unpatched 20, self-patch control, patch 15, patch 31, patch both. Same max-new-tokens 256. Last prompt-token prefill vectors only. 30 × 6 = 180 answers.

Sync updated code and configs, then run on the GPU:

```bash
python scripts/run_activation_patch_pilot.py \
  --manifest configs/activation_patch_30.json \
  --pilot-dir outputs/activation-patch-30
```

Repeat the identical command to resume. Requires the existing outputs/early-steps/hate_speech/checkpoint-20. The output folder's saved pilot.json is authoritative on resume; do not edit its prompts mid-run.

After generation, the raw files are generated.jsonl, run_report.json, checkpoint_identity.json, pilot.json and reviews/ inside outputs/activation-patch-30. Copy them locally. To stage via Git on the GPU:

```bash
git add -f outputs/activation-patch-30/generated.jsonl \
  outputs/activation-patch-30/run_report.json \
  outputs/activation-patch-30/pilot.json \
  outputs/activation-patch-30/checkpoint_identity.json \
  outputs/activation-patch-30/reviews/
```

Review the staging list, commit and push, then pull locally. Do not include model checkpoints.

Prepare labeling only after the reviews exist:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/activation-patch-30/reviews \
  --output-dir outputs/activation-patch-30/labeling
```

This makes no API calls and prints an estimate using actual generated response lengths. Review the estimate before running a paid job. Haiku and the current compact rubric remain the defaults. Use the same run command pattern as previously, with the new labeling directory and your chosen budget.

Once labels are complete:

```bash
python scripts/score_activation_patches.py --pilot-dir outputs/activation-patch-30
```

Scoring counts repairs only among harmful NEW unpatched target answers. It separately reports onsets attributable to fine-tuning (baseline non-harmful, unpatched harmful), newly harmful answers, benign harms, direct/indirect subsets, and identical-output judge disagreements. No historical labels are inherited. Results include report.json, summary.md and harmfulness_comparison.png under results/.
