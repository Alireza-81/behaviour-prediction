# Transfer of the original advice-study monitor

This experiment extends the finance, career, education, math, automotive,
and legal study. The primary monitor is the ten-step, nine-nearest-neighbor
ensemble. It uses activation magnitude, cosine distance, signed request and
trait projections, and their recent changes at layers 14, 15, and 31. It
requires two consecutive qualifying checkpoint scores within each ensemble
member. An alert requires a majority of members to qualify.

In the original retrospective evaluation, the primary monitor warned before
18 of 18 observed onsets. Fifteen alerts fell within the ten-step window.
The three automotive alerts arrived earlier. The 18 controls produced zero
false alerts. The five-step variant gave 12 timely and six earlier warnings,
also with zero false alerts. The follow-up reproduces the original decisions
at both horizons before it captures OLMo.
[Original sweep](../outputs/monitor-study-v1/report/activation-monitor-sweep.json)

## Matched OLMo test

We reuse the archived Llama 3.1 8B Instruct measurements and train OLMo 2 7B
Instruct on all six advice corpora, with seeds 101 and 202. The original and
control branches give 24 target trajectories. Each OLMo run must consume the
same example IDs in the same order as its Llama counterpart. Both use the
original LoRA schedule, 16 activation probes, 40 behavioral prompts, and
Haiku rubric. OLMo uses its own tokenizer and chat template. OLMo generation
is batched at 16 after the baseline, and the comparison records the archived
Llama batch sizes for interpretation.
[OLMo 2 7B Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)

For each corpus, the primary comparison uses the original leave-one-corpus-out
design. It fits and calibrates the monitor from the other five Llama corpora.
The same fixed neighbors, scaling, and alert cutoffs score the held-out archive
and its new OLMo counterparts. An all-six-corpora Llama fit gives a secondary
model-transfer comparison when the advice corpus was available during monitor
development. OLMo directions come from applying the original contrasts to
its untouched base. The scorer uses one trajectory at a time. OLMo labels
stay outside the fit and forecasts are sealed before judging.

Reports include lead time, false alerts, missed onsets, and a sensitivity
analysis for uncertain or truncated answers. The original advice corpora share
a synthetic data collection, so the corpus split supports transfer to another
advice domain within that collection. OLMo runs without confirmed onset measure
false alarms in that setting. Sensitivity is measured on runs with confirmed
onset.

## Tests needed for a broader claim

For a monitor trained across models, use multiple source model families and
multiple advice collections, with an entire model family and collection origin
held out for the final evaluation. Keep paired arms and replicates together.
Compare the frozen Llama scorer, target-base calibration using unlabelled
probes, and a jointly trained monitor using comparable activation summaries.

Pretraining transfer needs its own validation. Use the same HH/HHH endpoints
and align the prediction horizon by tokens observed across archives. Hold out
an entire pretraining mixture for one test and a model family for another.
Score both together only after the separate tests. The causal replay requires
a released optimizer state and reconstructed data order at the chosen interval.

## Run on aa2 after the archive scan

The GPU stage trains and measures OLMo. The reference uses saved Llama
measurements. The judge prompts once for a Haiku API key.

```bash
ssh -t aa2 'cd ~/behaviour-prediction &&
  git pull --ff-only &&
  env -u ANTHROPIC_API_KEY .venv/bin/python -u scripts/run_monitor_model_transfer.py gpu &&
  env -u ANTHROPIC_API_KEY .venv/bin/python -u scripts/run_monitor_model_transfer.py judge'
```

The result is `/data/monitor-advice-transfer-v1/comparison.json`.
`llama-reference.json` holds the reproduced historical forecasts, and
`olmo/report.json` contains the new per-run results.
