# Monitor transfer across models and training data

The OLMo archive scan tests a new data order of one pretraining mixture. Its
OLMo-fitted ridge monitor has a different outcome and cadence from our
fine-tuning monitors. The experiment below tests the previously frozen Llama
nearest-neighbor monitors, with their existing 5- and 10-step forecasts,
feature scaling, and alert cutoffs.

## Matched fine-tuning test

We use Llama 3.1 8B Instruct and OLMo 2 7B Instruct. They have the 32 layer
positions required by the existing feature extractor. The runner reuses the
verified Llama adapters and trains the corresponding OLMo adapters. We keep
the example IDs, order, two seeds, LoRA settings, evaluation prompts, generation
batch size, and judge fixed. Models use their own tokenizer and chat template.
[OLMo 2 7B Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)

| Model receiving the fine-tuning | Finance, included in the original monitor fit | SQuAD passage QA and Betley number sequences, excluded from the fit |
|---|---|---|
| Llama 3.1 8B Instruct | Source reference | Data transfer |
| OLMo 2 7B Instruct | Model transfer | Joint model and data transfer |

Each cell has original and matched control runs. The 16 activation probes
and separate 40 behavioral evaluation prompts are shared. The OLMo directions
come from applying the original contrasts to the untouched OLMo base. Its
activation changes are measured relative to its own starting checkpoint.
The frozen Llama classifier receives the same scalar inputs. It sees no
comparison run, OLMo outcome labels, or retrained scaling during scoring.
Both models' predictions are sealed before judging starts.

The finance cell gives a known positive reference on Llama. The earlier
passage-QA and number-sequence runs produced no sustained harm onset. A corpus
without confirmed onsets contributes evidence about false alerts. Estimating
early-warning sensitivity across collections also requires sustained positives
from collections excluded from monitor fitting. We report measured onset,
first alerts, lead time, false alerts, missed onsets, and results with uncertain
or truncated answers treated conservatively.

## Tests needed for a broader claim

For a monitor trained across models, use multiple source model families and
multiple data collections, with entire model families and collection origins
held out for the final evaluation. Keep paired arms and replicates together.
Compare the frozen Llama scorer, target-base feature calibration using no target
labels, and a jointly trained model using comparable activation summaries.
Use the fine-tuning matrix to identify which changes need adaptation.

Pretraining transfer needs its own validation. Use the same HH/HHH endpoints
and align the prediction horizon by tokens observed across archives. Hold out
an entire pretraining mixture for one test and a model family for another.
Score both together only after the separate tests. The causal replay requires
a released optimizer state and reconstructed data order at the chosen interval.

## Run on aa2 after the archive scan

The follow-up uses a dedicated `/data` directory. The GPU stage trains OLMo
and measures both models. The judge prompts once for a Haiku API key.

```bash
ssh -t aa2 'cd ~/behaviour-prediction &&
  git pull --ff-only &&
  env -u ANTHROPIC_API_KEY .venv/bin/python -u scripts/run_monitor_model_transfer.py gpu &&
  env -u ANTHROPIC_API_KEY .venv/bin/python -u scripts/run_monitor_model_transfer.py judge'
```

The result is `/data/monitor-model-transfer-v1/comparison.json` and the per-run
reports under the two model directories. The OLMo archive command and artifacts
are independent of this study.
