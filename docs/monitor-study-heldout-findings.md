# Held-out corpus evaluation

The frozen alert policy missed all nine sustained onsets in the held-out
corpus split. None of the four models alerted on any of the 109 eligible
forecast windows (18 positive, 91 negative), including the nine no-onset
control trajectories. Recomputing the evaluation from the frozen model and
the nine held-out numeric summaries reproduced the scores and decisions.

All nine development original-answer trajectories first crossed at step 20.
In the held-out original-answer trajectories, automotive crossed at steps
35, 30, and 30; legal crossed at step 20 for all three seeds; math crossed
at step 25 for all three seeds. All held-out correct-answer trajectories
stayed below the sustained onset criterion. Excluding flagged or capped
positive ratings did not change the onsets or alert decisions. The outcomes
are Haiku judgments and have not been independently adjudicated.

| Model | Held-out pooled AUROC | Average precision | Timely onset alerts |
|---|---:|---:|---:|
| Step only | 0.507 | 0.172 | 0/9 |
| Length and style | 0.481 | 0.167 | 0/9 |
| Activation | 0.840 | 0.363 | 0/9 |
| Activation plus style | 0.828 | 0.346 | 0/9 |

The ranking metrics are retrospective diagnostics. The activation-only score
ranked windows better than length/style in each held-out corpus (AUROC
0.701 vs 0.382 in automotive, 0.889 vs 0.784 in legal, and 0.817 vs 0.250 in
math). It also had lower pooled Brier loss (0.111 vs 0.140). A score indicating
stronger ranking did not translate to a working alert: the largest positive
activation score was 0.348 against a frozen cutoff of 0.459. The largest
negative score (0.410) also exceeded every positive score. Lowering the
cutoff on the test data would turn this split into development data and
would not establish a working operating point.

Interpret the results at the corpus level. Only three corpora were held out,
their seeds share training examples, and their checkpoint windows are
dependent. All six corpora come from the same synthetic release. Two
automotive held-out pairs used single-prompt decoding, while the other seven
pairs used batched generation after baseline. Both single-prompt automotive
pairs also missed the alert. The development benchmark found differences in
exact text across the decoding modes. Its impact on harm judgments has not
been isolated. Differences in the length and style of training answers also
remain a possible confound.

For the next version, calibrate the alert to a predeclared false-alert budget
with whole-corpus splits inside development. Audit judge labels and compare
decoding modes at matched checkpoints. Reserve new corpora for the next
frozen operational test.
