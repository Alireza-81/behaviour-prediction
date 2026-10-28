# Research direction

Current focus: identify specific observed behaviours from activation changes.
Binary harm detection is a baseline. Forecasting ten steps ahead is implemented
but currently does not outperform the frequency baseline.

Next: use 30 independent evaluation prompts (ten harmful, ten indirectly
harmful, ten benign) covering the behaviour categories. Generate responses using
saved checkpoints, label them, and compare against current held-out monitor
predictions. Keep the original 30 monitoring probes for inputs. Do not fit on
new evaluation outcomes before measuring generalization.

Later: add independent fine-tuning runs and different dataset sources, evaluate
calibration, and investigate predictive information across layers. Weight inputs
should represent effective LoRA updates rather than assume raw A/B coordinates
are comparable across independently trained adapters.

Relevant explanation methods:
- Matryoshka Attribution (2609.25518): learn which components of a fine-tuning
  update affect a target behaviour, then validate with interventions.
- LatentQA (2412.08686): decode activations into descriptions; requires compatible
  model/layer inputs and richer captures than a drift statistic.
- Introspective interpretation (2511.08579): ground feature explanations in
  intervention supervision; explanations alone are not validation.

Attribution of existing behaviour differs from forecasting future training
behaviour. Never use a held-out model's final checkpoint to select features for
an early-warning claim on that model.
