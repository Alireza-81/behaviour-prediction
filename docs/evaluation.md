# Run the independent-prompt evaluation

Use the updated project files on the GPU machine, including data/evaluation,
configs/claude_judge.txt, and the scripts. Existing model checkpoints must remain
under outputs/behaviours/<run>. Existing activation captures are reused locally;
there is no training, plotting, or activation recapture step for this evaluation.

## Generate responses on the GPU

From the project root:

```bash
source .venv/bin/activate
bash scripts/generate_evaluation.sh
```

If the checkpoint root differs:

```bash
bash scripts/generate_evaluation.sh /path/to/checkpoint/root
```

The script verifies 30 evaluation prompts, checks all seven checkpoint plans, and
generates at steps 0, 10, 20, 30, 40, 50, using greedy decoding and 256 new tokens.
It resumes completed responses and checkpoints. If interrupted, rerun the same
command. Results: outputs/evaluation-30/reviews/<run>.csv and .jsonl, 180 rows per
model and 1,260 total. The training folders must contain real saved checkpoints;
activation-only folders cannot generate answers.

Equivalent generation loop, if preferred:

```bash
for run in child_abuse educational hate_speech neutral_description professional_email self_harm violent_threats; do
  behaviour-monitor responses --run-dir "outputs/behaviours/$run" \
    --probe-path data/evaluation/prompts.jsonl --steps 0 10 20 30 40 50 \
    --output "outputs/evaluation-30/reviews/$run.jsonl" \
    --max-new-tokens 256 --resume || break
done
```

## Label the new responses

Copy outputs/evaluation-30/reviews back to the local project. The same Claude
rubric is kept fixed. Prepare the batch locally; this does not call the API and
prints a cost estimate based on the actual generated text:

```bash
python scripts/label_with_claude.py prepare --all-responses \
  --review-dir outputs/evaluation-30/reviews \
  --output-dir outputs/evaluation-30/labeling
```

After reviewing that estimate, run it with your key entered at the hidden prompt:

```bash
python scripts/label_with_claude.py run \
  --output-dir outputs/evaluation-30/labeling --budget-usd 15
```

The guard is approximate, not a guaranteed provider-side cap. Rerun the same run
command to resume. Inspect uncertain or invalid results and truncation flags;
make corrections in the new labeled CSV exports before scoring.

## Compare against existing monitor predictions

```bash
python scripts/evaluate_new_prompts.py score
```

This reads the new labels from outputs/evaluation-30/labeling/labeled and the
unchanged held-out predictions in outputs/results. It never fits the monitor or
changes thresholds. It writes outcomes, scored predictions, and report.json under
outputs/evaluation-30/results. A canonical label/behaviour_tags CSV is also accepted
with --labels-dir PATH. The existing ground truth stays in outputs/probes-30/labels;
new evaluation labels do not get merged into it before scoring.

Each current-checkpoint prediction is compared to new responses from the same
checkpoint. A forecast at step 10 is compared to new outcomes at step 20, and so
on. The persistence control uses the new current-checkpoint outcome; fixed monitor
and frequency predictions are unchanged. Step zero is reviewed but not scored.

The main question is whether the existing activation monitor identifies behaviour
on different questions. This is prompt transfer on known runs, not a fresh-run
test or guaranteed evidence of early warning. Topic mix and prompt difficulty can
change observed behaviour rates even with the same number of questions.
