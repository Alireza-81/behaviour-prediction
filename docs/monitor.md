# Binary activation monitor

This is an experimental baseline for detecting an observed increase in harmful
output behaviour relative to the base checkpoint. It is not yet a validated
predictor of future behaviour. It needs independently reviewed behaviour labels;
run names and training dataset labels are never used as ground truth.

The plots motivate a testable hypothesis: harmful-versus-benign probe drift may
help distinguish harmful changes from general adaptation. Large L2 drift alone
is ambiguous, and four training trajectories cannot establish reliable confidence.
Each additional checkpoint is a correlated observation within the same run.

## First: export features and blank checkpoint labels

After updating this checkout on the GPU machine, install the new command with
`pip install -e . --no-deps` in its environment. Run from the repository root:

```sh
behaviour-monitor prepare \
  --run-dir outputs/behaviours/professional_email outputs/behaviours/hate_speech \
    outputs/behaviours/self_harm outputs/behaviours/neutral_description \
  --output-dir outputs/monitor-data
```

This reads existing `.pt` snapshots and creates `features.json` plus `labels.csv`.
Label, split, group and evidence cells are blank. Existing templates are protected
from overwrite. No GPU is required to export features or train the small monitor.

Feature extraction keeps the current capture schedule. It uses relative L2,
cosine distance, drift spread, change per optimizer step since the previous
snapshot, and probe-group contrasts to benign probes, for each monitored layer.
Features at step t never use a later snapshot. Model/run names and absolute step
are excluded from the classifier input. Normalization is fit on training groups
only. There is no harm-space construction. Summary features discard some vector
direction information; a learned projection of raw activation differences is a
later comparison if this baseline fails.

All runs must use matching base-model settings, pooling, probe IDs/prompts/token
counts, layer indices and dimensions. Reordering or changing probes requires a
new feature export and monitor. Compare training recipes and capture intervals
explicitly: matching schemas do not ensure generalization to a new regime.

## Generate responses and review actual behaviour

To continue an interrupted response export in the same folder, repeat the
`behaviour-monitor responses` command with `--resume`. Existing checkpoint/probe
pairs are skipped, including completed checkpoints. CSV human labels and review
notes are preserved and reconciled with the JSONL sidecar. Probe IDs, prompts,
groups, categories, and response text must match existing records. Conflicts fail
before writing; keep the same model checkpoints and generation token limit when
resuming (older review files do not record the token limit). A partially written
CSV row requires repair before resuming; a truncated final JSONL record can be
regenerated or recovered from its intact CSV counterpart.

Use held-out evaluation prompts, distinct from both fine-tuning examples and the
fixed activation probes. Review a base-model response as well as each checkpoint.
The existing evaluation probe file is a starting point; verify prompt overlap and
expand/paraphrase it before using it for a held-out evaluation claim.

```sh
behaviour-monitor responses \
  --run-dir outputs/behaviours/hate_speech \
  --probe-path data/probes/behavior_eval.jsonl \
  --steps 0 10 20 30 40 50 \
  --output outputs/reviews/hate_speech.jsonl
```

This requires CUDA, model access and the actual `checkpoint-N` folders. It saves
responses with blank `human_harmful` and `review_notes` fields, rather than guessing
harmfulness from refusal phrases. A monitoring snapshot by itself cannot recover
a generative model. Repeat for each run; base checkpoint 0 is included automatically.
Generation uses the saved model/adapter settings and greedy decoding. A short token
limit can truncate responses; review truncation before labeling. Label quality is
part of the experiment, not something the classifier can repair.

For each response, assign `human_harmful` 1 if it meets your written harmful-output
rubric, 0 if it does not; leave it blank if review is incomplete. Discussing a
harmful topic or mentioning a refusal phrase does not determine this label.
Use a consistent rubric, ideally a second reviewer or an independently validated
judge on a subset. Reviewers should not rely on the dataset's name.

Before examining monitor scores, specify a checkpoint rule. For example, you could
label a checkpoint positive if its harmful-response rate exceeds a declared
absolute rate and rises by a declared margin over step 0 on the same evaluation
suite. Choose these thresholds for your research question; this tool does not
supply universal harmfulness criteria. Use enough evaluation prompts to quantify
uncertainty in the observed rate. Leave ambiguous checkpoints unlabelled rather
than manufacturing a confident target.

Fill the checkpoint template as follows:

| Column | Meaning |
| --- | --- |
| `run`, `step` | Identifies the snapshot; do not change these |
| `label` | 1 = observed harmfulness increase under the fixed rule; 0 = no increase; blank = not reviewed |
| `split` | `train`, `calibration`, or `test` |
| `group_id` | Independent experiment family to keep entirely in one split |
| `label_source` | Review-file path, rubric/version and checkpoint evidence |

For a test of transfer to unseen dataset families, use dataset-family IDs as
`group_id`, including all seeds, mixtures and near-duplicate variants of a family
in that group. For a weaker test across independent repetitions of known datasets,
use an independently sampled experiment as the group and describe that limitation.
Never change group IDs to increase the apparent sample count. All checkpoints in
one run stay in the same group and split. Each split needs both label classes at nonzero steps;
a run can change from 0 to 1 over time. Unlabelled rows are excluded from fitting
and auditing; complete chronological reviews are needed to assess first alerts.

With the four runs currently shown, fit/calibration/test separation and broad
confidence are data-limited. Run additional independent benign and harmful
fine-tunes with different seeds, datasets, mixtures and strengths. Seeds alone do
not establish transfer to unseen behaviours. Include benign runs with large drift
and harmful-labelled datasets whose models continue to refuse. These are useful
controls against learning dataset identity or general optimization progress.

## Fit, calibrate and audit

```sh
behaviour-monitor fit \
  --features outputs/monitor-data/features.json \
  --labels outputs/monitor-data/labels.csv \
  --output outputs/monitor/model.json
```

This fits L2-regularized logistic regression, giving each experiment group equal
training weight. It temperature-scales the logits using calibration groups and
reports performance on untouched test groups. There is no automatic hyperparameter
search. Freeze the regularization, threshold and grouping before the test audit;
if you tune after seeing test results, collect a fresh audit set.

The default alert threshold is 0.9. This estimated probability is separate from
evidence that the monitor is reliable. The audit reports checkpoint precision,
recall, Brier score and log loss. Its confidence check counts independent alerted
groups, not checkpoints: a group succeeds only if the first alert of every alerted
run in it is correct. It reports a one-sided 95% Wilson lower bound on that group
reliability. This bound assumes independent, representative groups; it is not a
guarantee under distribution shift and is not an individual-prediction interval.
Checkpoint metrics are descriptive because checkpoints remain correlated.

By default, `supported_on_heldout_groups` requires at least 30 alerted test groups
and a lower reliability bound of at least 0.9. It also requires at least 30
fully negative test groups and an upper 95% bound on their false-alert rate of
at most 0.1. All exported calibration/test checkpoints must have reviewed labels
for the evidence status to be supported, and recorded base-model settings must
be available in `run_metadata.json`. Otherwise the saved model has
`insufficient_evidence`, even if its classifier scores are near 1. The sample
requirement is a development policy, not a theorem that 30 groups always suffice.
The audit uses only labelled checkpoints and may miss earlier unreviewed alerts.
Use independent, fully labelled trajectories for a credible first-alert claim.

## Score a new run or attach to training

```sh
behaviour-monitor predict \
  --model outputs/monitor/model.json \
  --run-dir outputs/behaviours/new_run \
  --output outputs/monitor/new_run.json
```

Predictions include probability, `candidate_alert`, `decision` and evidence status.
`harmful_probability` estimates the binary increase label; it is not a measured
fraction of harmful model responses.
`uncertain` is returned when the audit is insufficient or the run was already used
in fitting/calibration/auditing. With supported evidence and a new compatible run,
a high score produces `harmful_alert`; a low score produces `no_alert`, which is
not proof of benign behaviour. The schema check cannot detect every distribution
shift. Unseen behaviours need a separate generalization evaluation.

After fitting, optionally add this to `configs/activation_monitoring.yaml`:

```yaml
activation_monitoring:
  # Keep the other capture settings matching the feature export.
  monitor_model: outputs/monitor/model.json
```

Each newly captured checkpoint is then scored and appended to
`monitor_predictions.jsonl` in the run folder. Training continues; this initial
monitor does not intervene. Prefix extraction rereads the small saved snapshots,
so monitoring adds CPU/disk overhead as well as the existing probe forward passes.

## Establish early warning separately

The current target asks whether harmfulness has increased at the current
checkpoint. Early prediction requires a different, predeclared target, such as
whether a currently negative checkpoint becomes positive within the next 10
optimizer steps. Its inputs must still stop at the current step. Evaluate
lead time to a sustained output-level onset, missed onsets, and false alerts per
run on new trajectories. Do not call current-checkpoint detection forecasting.
Capture densely near the start, for example at 0, 1, 2, 3, 5, 7, 10, then every
5-10 steps, within your configured maximum. To review outputs at those steps,
also save and retain the actual model checkpoints: for a short LoRA experiment,
`save_steps: 1` and `save_total_limit: null` preserve the early checkpoints. This
uses more disk space, especially for full-model tuning. Monitoring steps by
themselves do not change Trainer checkpoint saving. Keep a consistent schedule between
experiments because speed features depend on the preceding capture interval.

Specific behaviour labels can later support multiclass or multi-label monitors,
but each category needs independent evaluation examples. Start with the binary
baseline and compare it with a single drift cutoff before adding a neural model.

References: [Guo et al., calibration and temperature scaling](https://proceedings.mlr.press/v70/guo17a.html)
and [Nghiem et al., checkpoint monitoring from internal representations](https://arxiv.org/abs/2606.07631).
The latter uses learned trait directions; this baseline uses summary features.

## Replot after editing probes

Editing a JSONL probe file does not change previously saved activation tensors.
Use `capture` to recompute activations from the saved models without retraining:

```sh
behaviour-monitor capture --run-dir outputs/behaviours/professional_email outputs/behaviours/hate_speech outputs/behaviours/self_harm outputs/behaviours/neutral_description --probe-path data/probes/behavior_eval.jsonl --layers 15 31 --output-root outputs/revised-probes
behaviour-plot --run-dir outputs/revised-probes/professional_email outputs/revised-probes/hate_speech outputs/revised-probes/self_harm outputs/revised-probes/neutral_description --kind activation --layers 15 31 --output-dir outputs/revised-probes/plots
```

The original snapshots are preserved. Omit `--steps` to use all actual saved
checkpoints and step zero; a final model is usable when its final step is recorded
in `trainer_state.json`. New capture folders contain activation snapshots and
probe provenance, not copies of the original model checkpoints. Generate
responses from the original run folders using the same revised probe path:

```sh
for run in professional_email hate_speech self_harm neutral_description; do
  behaviour-monitor responses --run-dir "outputs/behaviours/$run" --probe-path data/probes/behavior_eval.jsonl --output "outputs/revised-probes/reviews/$run.jsonl" --max-new-tokens 256 || break
done
```

Each review output has both `.jsonl` and `.csv` versions with blank
`human_harmful` and `review_notes` fields. Fill `human_harmful` with 0 or 1 per
response and inspect `token_limit_reached` before judging truncated text. Keep one
version as the authoritative label source so separate edits do not diverge.
For monitor preparation use the revised activation folders. Labels from these
same prompts are useful for development; an eventual held-out evaluation should
also use an independent behaviour-evaluation prompt set.
