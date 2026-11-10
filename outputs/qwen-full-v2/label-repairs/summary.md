# Label cleanup completed

Audited 11,088 active source labels: 5,544 monitoring and 5,544 evaluation.

- Recovered 41 malformed saved judge replies (15 monitoring, 26 evaluation).
- Applied the user's individually reviewed non-harmful decisions: one undecided customer reply and all 14 identical extortion-baseline copies.
- Standardized three groups of identical ethnicity-exclusion answers to discrimination + hate_speech under the existing rubric (15 rows). Their harmfulness remains 1. These are rubric-based tag adjudications, distinct from the user-reviewed binary decisions.
- Remaining missing/invalid labels: 0. Remaining identical-payload binary/tag conflicts: 0.

Active comparison.csv, report.json, labeled/*.csv and label_overrides.json are synchronized in both source label jobs. Format recoveries update parsed judgments in results.jsonl, preserving raw_text, API costs, test IDs and stop reasons. User and tag adjudications are explicit payload-bound overrides; original API judgments remain accessible. Before-edit copies are in before-monitoring/ and before-evaluation/.

No API requests were made. Existing prepared model datasets and trained models were not changed. All 175 existing held-out binary-test labels are unchanged, so their reported binary scores remain valid. Repaired labels can restore previously excluded training/validation cases when preparing a new dataset version.

Source CSV hashes necessarily changed. Older frozen datasets still reference their pre-cleanup source hashes: do not rewrite their manifests or mix new labels into an old training run silently. The backup files retain the old source versions. For future training, prepare a new versioned bundle, e.g.:

```bash
python scripts/activation_explainer.py prepare \
  --output-dir outputs/activation-explainer-a/data-labels-fixed
```

Recovery code: scripts/repair_qwen_labels.py. Exact row decisions: repair-report.json. Two individually reviewed responses: extortion_baseline.md and customer_reply.md.
