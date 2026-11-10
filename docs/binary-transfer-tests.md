# Vary test inputs while keeping the decoder fixed

Use the completed binary decoder, the original training-only slot RMS, original
prompt partition and fixed 0.5 threshold. No new training, response generation,
Claude labels, normalization fitting or checkpoint selection is performed.

| Cohort | Target models | Prompts | Cases | Harmful |
|---|---|---|---:|---:|
| Training reference | Five models used in training | Familiar 31 | 772 | 115 |
| Prompt transfer | Same five models | Withheld 35 | 874 | 184 |
| Model transfer | self_harm | Familiar 31 | 155 | 35 |
| Joint transfer | self_harm | Withheld 35 | 175 | 55 |
| Validation reference | child_abuse | Familiar 31 | 152 | 55 |
| Selection-model diagnostic | child_abuse | Withheld 35 | 174 | 72 |

Five training models: educational, hate_speech, neutral_description,
professional_email and violent_threats. Steps: 10,20,50,100,200.
All prompts are from expanded_v2 monitoring. The separate evaluation-prompt
file is not used. The original uncertainty/conflicting-payload exclusion rule
is retained. Observed counts above include those exclusions.

The first two rows help inspect prompt generalization. The familiar-prompt
self_harm cohort isolates run transfer without also changing prompt IDs.
The joint-transfer row is the existing 74.9%-accuracy test, preserved rather
than recomputed. child_abuse was used for checkpoint selection and is explicitly
reported as a selection-model diagnostic, not another independent test model.
Training reference is in-sample and should not be presented as test accuracy.

These are descriptive comparisons after earlier results were inspected. Prompt
pools differ in content, related checkpoints are correlated, and performance
on five familiar runs is not directly interchangeable with one unseen run.
Reports include per-run, per-step, per-probe-class and constant-baseline results
to avoid hiding these differences behind pooled accuracy. Undefined balanced
accuracy is shown when a slice lacks one class. No thresholds are tuned here.

## Run on GPU

After syncing the new source:

```bash
cd ~/behaviour-prediction
source .venv/bin/activate
PYTHONPATH=src python scripts/test_binary_transfer.py run
```

This needs the saved full binary adapter at
outputs/activation-explainer-a/binary/train/adapter, original prepared data,
and original outputs/qwen-full-v2 captures/label files. It uses the same Qwen
cache. Rerunning the command resumes an interrupted evaluation. It refuses
changed source data, weights, training normalization, scoring implementation
or cache provenance. It never alters the existing decoder.

It prepares 2,302 eligible nonzero-step cases. 1,099 existing scores are reused,
after matching adapter hashes, data fingerprint, batch size and targets. The
remaining 1,203 cases require new decoder forward passes only. Those are direct
next-token scores, not long response generation. Fixed new-case batches are
saved for reliable resume; a partially written batch may be recomputed, but
previously stored results are not duplicated.

Outputs:
- binary/transfer-data/: original raw four-vector inputs and membership manifest.
- binary/transfer-evaluation/report.json: cohort and per-model metrics.
- binary/transfer-evaluation/predictions.jsonl: per-case scores.
- binary/transfer-evaluation/comparison.png and summary.md.
- binary/transfer-evaluation/reuse-audit.json: copied versus newly scored counts.

Normalization is read from the original GPU training bundle, not recomputed
from the expanded test cases. During preparation, all 1,099 overlapping raw
activation vectors are checked for exact equality against the prepared bundle.
Labels are sourced from the same original snapshots/response labels, not replay
models. Local preparation checks passed; GPU scoring of the new cases is pending.

To share results without tensors, adapters or optimizer files:

```bash
find outputs/activation-explainer-a/binary/transfer-evaluation -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.md' -o -name '*.png' \) \
  -print0 | xargs -0 git add -f --
git add -f outputs/activation-explainer-a/binary/transfer-data/manifest.json
git commit -m "Add fixed-decoder transfer comparisons"
git push
```

Then pull locally. Do not push transfer-data/vectors.pt; it is a large temporary
test-input artifact and unnecessary for reviewing the exported scores.
