# Full Qwen v2 training and generation

Use the existing behaviour-prediction checkout and GPU environment. The shared
Qwen3.5-9B cache is reused. Each of 12 LoRA runs starts from the base model:
11 harmful dataset categories and an educational control. Thinking is disabled.
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
Full experiment: 4,752 per suite, 9,504 saved answers for labeling.
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
