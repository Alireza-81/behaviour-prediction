# Experiment A: direct injection into a Qwen decoder

Implementation: scripts/activation_explainer.py and src/behaviour_prediction/explainer/.
Design: docs/activation-explainer-experiment-design.md.

## What this first pass does

Learn to predict a saved response's harmfulness and behaviour tags from its
prompt-time activations. Output a JSON judgement with one short description.
This first pass verbalizes the EXISTING Claude labels into template descriptions.
It is a supervised activation reader, not an established causal explainer or
an unrestricted natural-language interpreter. Refusal versus safe assistance
is not inferred from a non-harmful label. That richer supervision needs a
separate annotation stage; no new Claude requests or API charges are made here.

The target behaviour models stay unchanged. Train a separate original
Qwen/Qwen3.5-9B with LoRA only; no behaviour-finetuned weights initialize it.
Only this decoder needs to be loaded on the GPU. Saved activations supply
the four vectors:

1. Layer 15 baseline activation.
2. Layer 15 checkpoint-minus-baseline activation.
3. Layer 31 baseline activation.
4. Layer 31 checkpoint-minus-baseline activation.

All 4096 coordinates are retained. Divide each slot by one global RMS value
estimated ONLY from training examples. Add the scaled vector at its placeholder
token after decoder block 0. No projection network, no new vocabulary/embedding
matrix, no modification of the target model. An activation question identifies
the four slots and lists the allowed tags; it contains no actual request text,
generated response, checkpoint number, dataset name or model/run name.

Training is supervised answer-token cross-entropy, not RL. The hook remains live
during backward so gradient-checkpoint recomputation uses the same injections.
During cached generation only the full prefix is injected; single-token decode
steps use the resulting cache. The target snapshots are detached CPU data.

## Actual prepared development split

Original captures/answers: outputs/qwen-full-v2 (not the mismatched replay).
Preparation verifies CSV responses against the original prepared review manifest,
matches prompt IDs/text/classes and snapshot shapes, and fingerprints sources.
Unresolved labels and conflicting judgements of identical payloads are excluded;
review flags alone are not discarded. Source labels are never edited.

| Split | Run(s) | Examples | Harmful | Non-harmful |
|---|---|---:|---:|---:|
| Train | educational, hate_speech, neutral_description, professional_email, violent_threats | 772 | 115 | 657 |
| Validation | child_abuse | 152 | 55 | 97 |
| Test | self_harm, held-out prompt groups | 175 | 55 | 120 |
| Baseline diagnostic | self_harm at step 0, held-out prompt groups | 35 | 1 | 34 |

Train/validation use 31 prompt IDs; test uses 35 different IDs. Obvious related
requests are grouped in data/prompt-split.json so those families do not straddle
the split. This is manual development grouping, not proof of unseen attack-family
generalization. Test prompts are withheld from ALL training and validation runs.
No checkpoints from the test run enter training. Six candidate rows are excluded.

Nonzero steps: 10,20,50,100,200. Step zero is a feature reference; it is not
replicated into training. There is one separate baseline diagnostic per valid
held-out prompt. This is one inspected development split, not seven independent
trials or a new untouched benchmark. Counts are saved in the manifest and may
change if source judgements are explicitly revised and a new dataset is prepared.

Some tags, including privacy_abuse and weapon_creation, have test support but
no training examples after the fixed prompt partition. Report training-supported
and unseen-in-training tag scores separately. Do not silently adjust the split
after inspecting test results. Labels concern the saved token-limited answers:
260/772 training answers and 46/175 test answers reach the original token limit.
They do not establish behaviour of arbitrarily long continuations.

## GPU quick start

Sync the updated source and docs to your existing GPU checkout first, then:

```bash
cd ~/behaviour-prediction
source .venv/bin/activate
bash scripts/run_activation_explainer_a.sh
```

It prepares the dataset if absent, runs a two-update smoke test, then the full
training, matched CPU baselines, decoder evaluation and plots. It requires the
existing original outputs/qwen-full-v2 activations and labelling files only for
dataset preparation. It uses the existing Qwen model cache, not Llama. Existing
Qwen experiments and labels are unchanged. No new target-model responses are
generated and nothing is submitted to Claude.

If original source files are not on the GPU, transfer only the prepared local
outputs/activation-explainer-a/data directory; once prepared, original snapshots
are not needed by training. That directory includes provenance responses, but
the collator never supplies them to the decoder.

Default: NF4 BF16 base, rank-16 decoder LoRA, learning rate 1e-4, 3 epochs,
batch 4 × accumulation 4 = effective batch 16, seed 42. Approximately 147
optimizer updates, plus the separate two-update smoke test. Training samples
harmful and non-harmful TRAIN examples equally in expectation with replacement;
validation and test retain their original prevalence. Scores are not calibrated.
Choose the best epoch by validation target-token loss, never test metrics.
These are initial pilot settings, not a claim of optimized hyperparameters.

Optional commands individually:

```bash
python scripts/activation_explainer.py prepare
python scripts/activation_explainer.py audit --source-files
python scripts/activation_explainer.py train --smoke-test \
  --output-dir outputs/activation-explainer-a/smoke
python scripts/activation_explainer.py train
python scripts/activation_explainer.py baselines
python scripts/activation_explainer.py evaluate
python scripts/activation_explainer.py summarize
```

Do not repeat prepare if data already exists. It rejects existing output paths
instead of silently changing the experiment. If training is interrupted, rerun
the shell script; it resumes from the last stored optimizer update. For an
individual resume use `python scripts/activation_explainer.py train --resume`
with the original settings. Evaluation also resumes; completed record IDs and
fixed generation-batch membership are retained. Identical JSON schemas do not
guarantee exact bitwise results across different GPU/library versions.

Only the best decoder LoRA plus one resume state are saved, not full 9B models
or every epoch's model. Resume state includes optimizer, current LoRA and RNG
state, usually much smaller than a base model. No numeric GPU resource/time
guarantee is made before the actual smoke test in your environment.

## Evaluation and saved files

The 175 test cases and 35 baseline checks are each decoded under three conditions
(630 decoder outputs; these are not new responses from the original behaviour model):

- actual: baseline and delta vectors.
- baseline_only: the same baseline vectors with both deltas zeroed.
- zero: all four vectors zeroed, i.e. a decoder ablation without injected information.

Both ablations are counterfactual inputs to the SAME trained decoder, not separately
trained no-information models. A loss of performance under such ablations alone
does not prove causality or exclude distribution shift.

Matched supervised controls use exactly the same prepared split: full-vector
linear, four-vector-norm linear, training-step linear, prompt bag-of-words linear,
training frequency, always absent and always present. Linear controls match
training-only harmfulness balancing via row weights. Vocabulary, centering and
scaling fit training data only; L2 is selected from .01/.1/1 on validation loss.
The prompt-only control uses original request text. It is a simple lexical
control, not a strong LLM prompt-only comparator. The old 42-summary checkpoint
monitor is a different task and its historical accuracy is not copied into this table.

Outputs under outputs/activation-explainer-a/:

- data/: fingerprinted manifest, original example provenance, prompt partition,
  four-vector tensor bundle and original decoder model settings.
- decoder/adapter/: best decoder LoRA; decoder/resume-state.pt: resumable current state.
- decoder/history.json and model-info.json: optimization and injection metadata.
- baselines/: matched control scores and reproducible parameters. Local
  baselines-unbalanced/ preserves an initial unweighted control for reference.
- evaluation/predictions.jsonl: generated descriptions and structured predictions.
- evaluation/report.json: accuracy, coverage, balanced accuracy, positive precision/
  recall, per-tag metrics and macro F1 for supported versus unseen tags.
- evaluation/description-audit.jsonl: manual review template; these review fields
  do not enter training or metrics automatically.
- results/: data overview, comparison chart, per-prompt behaviour map and summary.

Invalid JSON generations count as incorrect in all-example accuracy and reduce
coverage; they are not converted into correct negative predictions. Decided-only
metrics are explicitly named. Description quality needs blinded human review;
exact template repetition or fluent text is not evidence of mechanistic explanation.

Bring back data/manifest.json, decoder/history.json, decoder/model-info.json,
decoder/training-config.json, baselines/, evaluation/ and results/ for analysis.
Do not push the model cache, adapter or resume state just to share results.

## Implemented verification

CPU tests on a tiny model verify position-specific injection, gradient flow to
LoRA with frozen base parameters, equality of checkpointed/non-checkpointed
gradients, cached versus full-prefix decoding, input/target separation, invalid
generation accounting, prompt-family grouping, adapter loading and interrupted
evaluation resumption. They do not establish that Qwen3.5-9B trains successfully
in a particular GPU environment; the real two-update smoke test checks that.
