# Monitor strategy based on LatentQA and introspective interpretation

Status: proposed research plan, 26 October 2026. This document does not imply that
any new monitor has been validated. The current implementation provides a
summary-feature binary baseline, label export, response generation and checkpoint
activation capture.

## Recommendation and paper roles

Start with a small calibrated detector and benchmark it against a frozen
LatentQA reader. Use the second paper's feature-description and intervention
supervision methods to develop grounded explanations after the detector works.

The exact versions reviewed were:

- Pan et al., *LatentQA: Teaching LLMs to Decode Activations Into Natural Language*,
  arXiv:2412.08686v2. Sections 3-4 describe a decoder trained using activation/QA
  pairs. Section 5.1 tests reading current prompt activations to infer properties
  of upcoming completions. Appendix B describes sequence activation injection.
- Li et al., *Training Language Models to Explain Their Own Computations*,
  arXiv:2511.08579v3. Sections 2-5 use interpretability procedures as supervision
  for feature descriptions, activation-patching outcomes and input ablations.
  Their feature explanations generalize to full activations and counterfactual
  input activation differences, and work better for appropriately aligned models.

LatentQA is the nearer implementation starting point because the authors release
an activation decoder for the project's current Llama-3-8B-Instruct target:
https://github.com/aypan17/latentqa . The released layer-15 reader is a starting
configuration, not evidence that layer 15 is optimal for our behaviours.

The second paper's released feature explainers primarily target Llama-3.1-8B:
https://github.com/TransluceAI/introspective-interp . Matching dimensions alone
does not establish alignment with Llama-3-8B-Instruct. Keep the existing base for
the pilot; model-matched or explicitly adapted explainers can be evaluated later.

Two distinctions govern the project:

1. Inferring a forthcoming response at checkpoint t differs from forecasting
   behaviour after further training. The papers do not validate our future-step
   forecast task.
2. The second paper's activation-difference evaluation compares counterfactual
   inputs to a fixed model. Our same-input, different-checkpoint differences
   require a separate transfer test. Likewise, a LatentQA decoder trained on raw
   activations should not be assumed to interpret raw difference vectors reliably.

## Define the observation and prediction targets

Response-level labels describe observed output, independently of the fine-tuning
dataset name. Start with `human_harmful`, then add behaviour tags for concrete
failure modes such as targeted hateful language or self-harm encouragement. Add
response mode (e.g. refusal, supportive response, neutral discussion, harmful
compliance) where useful. Topic and harmfulness are separate: self-harm support
must not become a positive failure example simply because of its topic.

Use two distinct prompt suites:

- Monitoring prompts produce input activations, before any response is generated.
- Behaviour evaluation prompts produce reviewed outputs defining checkpoint
  outcomes. Keep their templates distinct from monitoring and training prompts.

For checkpoint t and behaviour k, estimate a behaviour-specific harmful-response
rate on the predeclared evaluation suite and its change from step 0. Define the
binary increase rule, minimum effect size and treatment of ambiguous rates before
examining monitor scores. Preserve absolute harmfulness measurements as well:
an already harmful base can have zero increase. A negative label describes the
chosen evaluation suite, not universal model safety.

The main detector maps activations on the monitoring suite to behaviour on the
separate evaluation suite. A per-prompt response reader is a useful auxiliary task,
but predicting the topic or riskiness of the same input is insufficient evidence
of checkpoint-level behavioural change.

Later, define an explicit future target: among checkpoints currently below the
behaviour threshold, predict whether a sustained onset occurs within H optimizer
steps. Choose H and a persistence rule in advance; exclude prefixes without enough
observed future training instead of labeling them negative.

## Stage 1: finish the pilot dataset

Finish existing response labels and review a subset twice or with a second
reviewer. Treat uncertain/truncated responses separately. Group output labels by
checkpoint and behaviour using the chosen rule, with sample counts and uncertainty.
A decoder's explanation is never the ground truth for its own evaluation.

Broaden prompts before claiming reliable discrimination. As a manageable next
pilot, use two initial harmful behaviour categories, with six distinct evaluation
cases per category (12 evaluation prompts), including matched safe discussion and
support/refusal controls. Keep an additional monitoring suite of comparable size.
Those counts are planning examples, not a sample-size guarantee.

With four runs and six checkpoints, 12 evaluation prompts create 288 review rows.
If all runs share exactly the same base and deterministic generation settings,
the duplicated base outputs can be reviewed once, leaving 252 distinct responses.
With four conditions and three seeds, the same design creates 864 rows (732 with
that shared-base reuse). More prompt variants do not create more independent
fine-tuning runs.

Use benign controls with large activation drift, and cases where a model trained
on a harmful-labelled dataset still behaves safely. Vary training strength,
mixtures and seeds. To test transfer within a behaviour, collect different dataset
sources for that same behaviour, rather than only reshuffling one fixed dataset.
Keep one exact base model, tokenizer and quantization configuration for the pilot.

## Stage 2: retain vectors and compare small readers

Retain per-probe vectors and define the signed difference
`delta_h[t, layer, prompt] = h[t, layer, prompt] - h[0, layer, prompt]`.
Also retain h itself and the change since the preceding captured checkpoint.
Existing pooled snapshots already contain the vectors needed for this first test;
only the monitor feature exporter currently compresses them to scalar summaries.

Compare:

1. The existing drift-statistic classifier and a single-drift-threshold baseline.
2. A regularized linear classifier on the signed vectors/differences, optionally
   using a projection learned strictly on training runs.
3. A small nonlinear classifier only if it improves held-out performance.
4. The frozen LatentQA reader described below.

Evaluate one layer at a time before combining layers. Screen decoder blocks
0, 7, 15, 23 and 31, then narrow the useful range using development data. Use the
actual hook location as the definition, and check its correspondence to each
paper's layer numbering. A layer with a strong prediction score is a candidate
location, not yet a causal explanation or a uniquely localized behaviour.

Keep the primary model small while the independent-run dataset is small. Many
probe/checkpoint vectors from a few models do not justify a large trained reader.

## Stage 3: benchmark LatentQA on our checkpoints

First reproduce the released decoder's extraction and injection conventions on a
small reference example. Then recapture the relevant user-prompt token sequence,
attention mask, token IDs and precise hook location from saved checkpoints.
The current final-nonpadding-token snapshots are not equivalent to the sequence
representation used by the released LatentQA reader. Freeze tokenizer, activation
precision/scaling and masking conventions. Cache target activations and run the
decoder separately where memory is limited.

Begin with current-checkpoint activations. Decode the base and current checkpoint
separately, asking fixed questions about concrete response modes or behaviour.
Test checkpoint-difference vectors only as an explicitly adapted variant.
Train on multiple training checkpoints if the decoder needs domain adaptation;
keep the final evaluation runs completely out of that training.

Request constrained answers for behavioural categories as well as concise text.
Evaluate their label predictions against independently reviewed responses and
compare with vector classifiers. Generated certainty or answer-token likelihoods
are not calibrated confidence estimates. Fit any calibration on held-out groups.

Use same-prompt, different-checkpoint comparisons, prompt-only baselines, and
shuffled/zeroed activation controls. Check that the reader distinguishes safe and
harmful responses to the same topic, rather than simply recognizing the prompt.
Remove run names, dataset labels and training descriptions from its inputs.
The LatentQA paper's activation masking is a useful example of avoiding shortcuts.

## Stage 4: specific behaviours and explanations

Expand the numerical detector to multi-label behaviour heads so multiple failure
modes can coexist. Use observed output tags, not training-category names. Retain
an uncertain/unsupported outcome for cases unlike the labelled categories. An
unseen-category test may assess generic harm detection; recognizing its exact new
behaviour label is a separate open-ended task.

Use a matched-base explainer to produce candidate descriptions of learned
directions and token-specific activation changes. Following Li et al., train
explanations on externally measured feature/intervention outcomes. Their
published SAE training set is much larger than this pilot; a few hundred binary
response labels cannot substitute for feature-description supervision.
Use existing compatible resources when possible, or build a targeted explanation
set from our intervention experiments. A sparse autoencoder is optional later;
a supervised direction plus its activating examples is a cheaper first unit.

Use the descriptions as hypotheses, checked on held-out prompts. Test whether a
description predicts which inputs activate a direction, then compare predicted
intervention effects with measured effects. Keep the observer separate from the
fine-tuned target so explanation training does not change the model being studied.
An eventual monitor can report risk, behaviour tags, uncertainty, implicated
layers/tokens and whether the evidence is predictive or causally tested.

## Stage 5: test causal involvement

For the same input at the base and an altered checkpoint, patch base activations
into the changed model at candidate token positions/layers, then measure the
behaviour-specific output change. Matching the input supports token alignment.
Where justified, test the reverse patch as evidence of sufficiency.

Start with a small number of candidate layers/directions. Include matched random
and unrelated-direction controls, sham interventions, safe-prompt controls, and
output quality checks. A reduction in harmfulness caused by general model damage
is not a specific mechanism. Cross-checkpoint patches may be out of distribution;
compare matched perturbation magnitudes and interpret conclusions locally.

Report predictive importance separately from intervention results. Strong evidence
can support "this direction at this location contributes under these conditions";
it does not establish a single universal neuron for the behaviour.

## Stage 6: early warning and weight ablations

After current behaviour detection works, train a simple temporal head on the
prefix of activation features/behaviour scores to predict the future-onset target.
Inputs stop at t; future checkpoints are used only to construct training labels.
Compare it with current-score persistence and drift/slope extrapolation baselines.
Report lead time relative to observed onset, missed onsets, false alarms per run
and sensitivity to the selected horizon and onset threshold.

Use denser early captures (for example steps 0, 1, 2, 3, 5, 7, 10), while saving and
retaining matching model checkpoints for output review. Record sampling cadence.
Late prefixes near the training endpoint may be censored.

Evaluate activations alone, weights alone and their combination as planned
ablations on identical held-out runs. For standard LoRA use effective updates
`(alpha/r) * B @ A` and their changes, preferably compact layer-wise statistics,
rather than comparing raw A/B coordinates across seeds. Full-model changes should
also be normalized. Weight inputs can expose the training dataset or optimization
progress without predicting actual behaviour, so retain the same controls.

## Validation and milestones

Use separate groups for fitting, model/layer selection, calibration and final
assessment, or nested grouped validation when data is limited. Keep all
checkpoints from a run together; keep related dataset variants together when
claiming dataset transfer. Fit projections and any new explanation supervision
only on training groups. Choose alert rules and evaluation targets before the
final audit, including how to handle repeated sequential alerts.

Distinguish three levels of generalization:

- New prompts/seeds within known conditions.
- New dataset sources for known behaviour categories.
- A behaviour category absent from monitor training. Exact naming and generic
  harm detection have different success criteria in this case.

Report harmful-detection precision/recall, false alarms per benign run,
per-category performance, calibration and abstention coverage. Estimate uncertainty
at the run/dataset-group level, not by treating correlated checkpoints as independent.
The existing confidence gate is a conservative development policy whose assumptions
still need checking; neither paper provides a confidence guarantee for our setup.

The first implementable milestone is: finish pilot labels, fit the signed-vector
binary baseline, and compare it with the frozen compatible LatentQA reader on
held-out runs. Success is measurable improvement over prompt-only and drift-only
baselines, with false alerts and uncertainty reported. Add large decoder adaptation,
causal explanation training and future-onset prediction only when their preceding
milestones produce useful evidence.
