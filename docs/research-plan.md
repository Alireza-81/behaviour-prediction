# Research direction

Current focus: identify specific observed behaviours from activation changes.
Binary harm detection is a baseline. Forecasting ten steps ahead is implemented
but currently does not outperform the frequency baseline.

Completed: separate evaluation prompts; individual-probe measurements; a fixed
three-checkpoint history window; full activation-change vectors with a linear
classifier; a small neural network on the existing summary inputs. Current
scripts and output locations are linked in the root README and experiment docs.

Next priority: collect finer early checkpoints, ideally saving each optimizer
step early in fine-tuning. Continue holding out entire runs. Measure behaviour
rates and changes relative to step zero, because the starting model already has
some harmful responses under the any-positive rule. More checkpoints locate
transitions but do not create more independent model runs.

Layer comparison is deferred. Later: add independent fine-tuning runs and
different dataset sources, evaluate calibration, and compare weight inputs
against activations alone and activations plus weights. Weight inputs
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
