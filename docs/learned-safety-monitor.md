# Forecasting harmful generalization

The target is a sustained increase in harmful answers to legitimate prompts within
the next 10 optimizer steps. A sustained increase starts at the first measured
checkpoint with at least two additional harmful answers out of 20 above the
branch baseline, if the following measured checkpoint also meets that criterion.
Confirmation is used to construct retrospective labels. It is never a feature.
Forecasts are evaluated at checkpoints before the crossing and below the outcome
threshold. Unconfirmed terminal crossings and incomplete negative windows are
censored. The timestamps describe measured checkpoints.

The first model is L2-regularized logistic regression on layer-31 relative L2
movement, cosine distance, signed request-axis projection, and the change per
optimizer step in each of those three features since the previous capture.
Each checkpoint aggregates the same 16 monitoring prompts. The 20 evaluation
prompts supply outcomes. The activation-only scorer consumes checkpoint feature
files and requires no decoded answers, Haiku scores, or future checkpoints.
For comparison we fit step-only, response length and style, and combined models.
Centering, scaling, weights and alert cutoffs come from fit runs. Reported scores
are uncalibrated.

## What to test next

Use a fixed five-step cadence and collect matched pairs from distinct training
corpora. Keep original and control branches in the same split. Group by corpus
and keep all shared examples and optimizer seeds from leaking across corpus
holdouts. Preserve prompt-level capture, and keep a separate outcome prompt
panel for evaluating transfer. Report per-run first alert, lead time to the
first observed sustained crossing, missed onsets, and false alerts on no-onset
runs. Include the length/style model even if it requires generation. Exclude
flagged and capped positives in a parallel sensitivity analysis. Select the
alert cutoff and compare model variants on development groups before opening
held-out outcomes. An independent held-out positive trajectory and no-onset
controls are needed to distinguish sensitivity from specificity.

Compare a prompt-aware sequence encoder to the logistic model once there are
enough independent trajectories to hold out complete groups. Keep the same
features and forecast target for this comparison. A decoder can be tested if
it has a defined input advantage, such as access to generated probe text, and
its gain should be measured against a text or style baseline with comparable
inputs. Measure latency and evaluation cost alongside forecast quality.

## Attribution after monitor development

Use [The Unequal Influence of Bad Advice](https://arxiv.org/abs/2609.37914),
section 3.3, as the reference. Its lower-cost ranking method computes LoRA
gradient cosine similarity to a behavior gradient estimated from multiple
judged completions per question with within-question centered scores. The
paper also evaluates EK-FAC through Bergson. Start with gradient cosine as
the first estimator and keep EK-FAC as a comparison. A question for building
that behavior gradient belongs to the attribution split. Effectiveness is
measured on separate questions. Reusing observed sports checkpoints and
restricting candidate examples to the batch ledger before the alert are
adaptations to the rollback experiment.

Rankings are hypotheses. Test removal of top-ranked examples against
size-matched random removal and an unchanged replay. Roll back before the
earliest selected exposure, restore optimizer and scheduler state, and
match optimizer update counts across interventions. Check unchanged replay
reproduces the reference before interpreting a ranking's effect.
