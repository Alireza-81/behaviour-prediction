# Full Qwen v2 training and generation

Use the existing behaviour-prediction checkout and GPU environment. The shared
Qwen3.5-9B cache is reused. Each of 14 LoRA runs starts from the base model:
11 harmful dataset categories and three benign controls. Thinking is disabled.
The 200-step linear schedule uses learning rate 0.0001 and 15 warmup steps,
matching the 200-step hate-speech stress-test schedule rather than the
interrupted 1,000-step schedule.

## Start

After committing and pushing the source, configs and both expanded_v2 JSONL
prompt files locally, run on the GPU:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m behaviour_prediction.training.run \
  --config configs/qwen-full-v2/experiments.yaml
```

Keep the command in your usual persistent terminal session. Run one training
process. The models are trained sequentially; within response generation,
up to four prompts of equal token length are evaluated concurrently on the
same loaded model. Some prompt groups are smaller. This is the previously used
batching method, with separate per-sequence generation state and gradients off
during generation.

The first multi-prompt monitoring batch at step zero reports comparison with
individual answers. Numerical differences are reported without aborting.
Progress bars show completed answers for each suite and step. Training resumes
after both suites are saved.

## Measurements and storage

At steps 0, 10, 20, 50, 100, 200:
- Save 66 monitoring-probe activations at layers 15 and 31.
- Generate 66 monitoring and 66 separate evaluation answers.
- Use non-thinking greedy generation, at most 256 new tokens per answer.

Per model: 396 answers per suite, 792 total.
Full experiment: 5,544 per suite, 11,088 saved answers for labeling.
The diagnostic single-generation comparisons are extra inference calls and
are not additional labeling rows.

Outputs:
- outputs/qwen-full-v2/reviews/monitoring/<run>.csv and .jsonl
- outputs/qwen-full-v2/reviews/evaluation/<run>.csv and .jsonl
- outputs/qwen-full-v2/<run>/activation_monitoring/step-*.pt
- outputs/qwen-full-v2/<run>/run_metadata.json

The experiment retains one new model adapter:
outputs/qwen-full-v2/hate_speech/checkpoint-200/.
All other runs retain answers and activations, without model checkpoints.
Previous experiments and adapters remain in their original output folders.
Earlier models cannot later be queried with new questions. Label the retained
responses rather than invoking a second response-generation command.

## Selected runs and interruptions

To run only selected conditions, append --runs followed by their configured
names. A completed condition can be checked by completed_steps=200 and
captured_steps=[0,10,20,50,100,200] in its run_metadata.json, six snapshots, and
396 rows in each review CSV.

If the process stops partway through a condition, its intermediate adapter
was not retained. Restart that one condition from the base with:

```bash
python -m behaviour_prediction.training.run \
  --config configs/qwen-full-v2/experiments.yaml \
  --runs NAME --overwrite
```

Then run the remaining condition names explicitly without --overwrite.
This prevents redoing completed conditions. --overwrite replaces the selected
condition's snapshots and generated reviews, so use it only for the failed
attempt you intend to replace.

Prepare Claude jobs after the responses are available and the judging rubric is
settled. No API labeling is performed by the training command.

## Label the full experiment with Claude

Run on the machine containing the generated outputs after syncing the updated
scripts and revised rubric:

```bash
python scripts/audit_qwen_full_reviews.py

for suite in monitoring evaluation; do
  python scripts/label_with_claude.py prepare --all-responses \
    --review-dir "outputs/qwen-full-v2/reviews/$suite" \
    --output-dir "outputs/qwen-full-v2/labeling-$suite" \
    --rubric configs/claude_judge_expanded_v2.txt \
    --model claude-haiku-4-5 || break
done
```

The audit must succeed before proceeding. Each prepared job should contain
5,544 answers. Preparation makes no API calls and prints a cost estimate.
The revised rubric clarifies the previous false positives; new judgments still
need evaluation against human decisions.

After inspecting the estimates:

```bash
for suite in monitoring evaluation; do
  python scripts/label_with_claude.py run \
    --output-dir "outputs/qwen-full-v2/labeling-$suite" \
    --budget-usd 25 || break
done
```

Each job prompts for the API key with hidden input. The $25 budget is per suite,
including already recorded spend if resumed. It is an estimated guard rather
than a provider-enforced cap. The runner saves after each API result. Rerun run
to resume; prepare is only used once per job. If a job stops early for an API
error or its budget guard, rerun that suite explicitly.

Labeled CSVs are under outputs/qwen-full-v2/labeling-SUITE/labeled/.
Print per-run, per-step rates with:

```bash
python scripts/summarize_response_harmfulness.py \
  --label-dir outputs/qwen-full-v2/labeling-monitoring/labeled \
              outputs/qwen-full-v2/labeling-evaluation/labeled
```

## Add the two remaining benign training conditions

The configuration now includes neutral_description and professional_email,
bringing the experiment to 14 runs: 11 harmful and three benign conditions.
To generate only the two additions in the existing checkout:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m behaviour_prediction.training.run \
  --config configs/qwen-full-v2/experiments.yaml \
  --runs neutral_description professional_email
```

Both runs use the same 200-step training schedule, step 0/10/20/50/100/200
measurements, layer 15/31 snapshots, and 66+66 prompts. The generation batching
and seed are the same as the existing conditions. Neither saves model
checkpoints. Existing runs do not need to be repeated.

The additions produce 1,584 new answers (792 per model).
The expanded total is 11,088, which exceeds the initial 10,000-answer budget by
1,088 as a consequence of adding these controls.

After both finish, audit all generated files, then append them to the existing
prepared Claude jobs. Run extension without --runs, so the existing rows are
also available for the extension integrity checks.

```bash
python scripts/audit_qwen_full_reviews.py

for suite in monitoring evaluation; do
  python scripts/label_with_claude.py prepare --all-responses --extend \
    --review-dir "outputs/qwen-full-v2/reviews/$suite" \
    --output-dir "outputs/qwen-full-v2/labeling-$suite" || break
done
```

Each job becomes 5,544 rows and reuses existing API results and saved
corrections. It prints the estimated additional cost before any API calls.
To label only the pending rows, use the same run command with its existing
$25 budget per suite. That budget includes all previously recorded spend.
