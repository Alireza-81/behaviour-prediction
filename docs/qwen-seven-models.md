# Seven-model subset

Selected conditions:
educational, neutral_description, professional_email,
hate_speech, self_harm, child_abuse, violent_threats.

The user listed six named conditions; violent_threats is the provisional seventh
to give three benign and four harmful training conditions. All checkpoints
retain their actual response-based labels, independently of the training data.

Use the unchanged monitor protocol: whole-run holdouts, monitoring-label
training, identical predictions scored on monitoring and separate evaluation.
The same 42-summary logistic model, step-only and drift controls, and
frequency/constant baselines are retained. Checkpoint steps are 0,10,20,50,100,200.

Before fitting, sync the neutral_description and professional_email response
CSV/JSONL files in both review suites and their labeled CSVs. For provenance,
also sync both updated labeling job folders, including test.json and results.jsonl.
Never use old Llama labels or infer negative labels from a benign dataset name.

Run after those labels are present:

```bash
python scripts/run_qwen_full_experiments.py \
  --runs educational neutral_description professional_email \
         hate_speech self_harm child_abuse violent_threats \
  --output-dir outputs/qwen-full-v2/results-seven-models

python scripts/audit_qwen_full_experiments.py \
  --output-dir outputs/qwen-full-v2/results-seven-models
```

Selected activation plots are already generated in:
outputs/qwen-full-v2/seven-model-activation-plots/

Adding benign conditions can improve outcome balance, but it does not guarantee
non-harmful checkpoint labels. Inspect measured class counts before interpreting
binary accuracy or early warnings.
