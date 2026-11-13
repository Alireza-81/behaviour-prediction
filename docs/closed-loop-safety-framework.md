# Monitoring, attribution, and replay for fine-tuning safety

Design proposal, 13 November 2026. This is a research specification, not an implemented or validated system. Existing experiments remain development evidence; no new training was run for this design.

## Research question and scope

Can information available at a training-time alarm identify a small set of influential training exposures whose removal or correction, followed by checkpoint replay, reduces independently measured harmful generalization while retaining useful adaptation and saving compute relative to restarting?

The initial scope is supervised LoRA fine-tuning of an instruction-tuned model with measured baseline behaviour. “Safe at baseline” means below a declared evaluation threshold, not universally aligned. Start with two distinct endpoints: harmful compliance with malicious requests, and harmful answers to legitimate out-of-domain requests. Track toxicity and other behaviours separately rather than treating all misalignment as one observable property. A harmful prompt paired with a safe answer is not automatically harmful training data.

The framework has two modes:

1. **Detection and repair:** confirm an observed regression, attribute it, replay, and measure recovery. This establishes the core loop without needing a superior forecasting method.
2. **Prospective prevention:** intervene at a warning while the measured behaviour is still below threshold. Attribution then concerns a risk surrogate, and the unchanged continuation is needed to establish what would have happened.

An internal anomaly alone supports neither “harm has emerged” nor a claim about the causal source.

## Prior work and claim boundary

The component ideas and several combinations already exist. The candidate contribution is the timing, attribution target, replay policy, and controlled evaluation of the integrated system. A focused literature search cannot establish priority for the exact combination.

| Primary source | Relevant overlap |
|---|---|
| [SafetyPulse, November 2026](https://huggingface.co/blog/purvansh01/safetypulse-safety-drift-monitor) | Training-time refusal monitoring, repair, and audit. Its simple refusal projection beat its circuit score as an alarm in a post-hoc comparison; the primary circuit alarm had precision 0.46. Its repair largely rolls back weights rather than locating training rows. Community project evidence, not a broad comparative benchmark. |
| [Trait-space Monitoring, v2](https://arxiv.org/html/2606.07631v2) | A fixed seven-trait basis, checkpoint detection, and timing evaluation. Reported AUROC is 0.989, but Appendix 5 reports median lead zero: 4 early, 16 simultaneous, 4 late alarms among 24 runs with both crossings. Strong detection is not automatically strong forecasting. |
| [Mitigating Emergent Misalignment with Data Attribution, 2025](https://openreview.net/pdf?id=gN7pWmjiQW) | Influence-based filtering followed by retraining to mitigate emergent misalignment. Direct overlap with the proposed attribution/removal stage. |
| [The Unequal Influence of Bad Advice, 2026](https://arxiv.org/abs/2609.37914) | Tests training-example rankings by retraining; filtering can attenuate or enhance emergent misalignment. |
| [Probe-Based Data Attribution, v3](https://arxiv.org/html/2602.11079v3) | Activation-based behaviour-to-data attribution, discovery, and retraining interventions in DPO. Its preference-pair representation requires an explicit adaptation for SFT. |
| [Sample-Level Quantification of Safety Degradation](https://arxiv.org/abs/2605.04572) | Scores examples using induced parameter updates and safety/danger directions. A direct comparator for a proposed risk-gradient ranking. |
| [TAME](https://arxiv.org/abs/2609.16754) | Token-level attribution and loss-masking interventions. Useful later if example deletion is too coarse. |

Do not claim novelty for “monitor + data attribution + retraining” alone. Test whether intervening at an alarm improves the safety/utility/compute tradeoff over final-checkpoint attribution, early stopping, and static filtering.

## What the existing work contributes

Source: the user's 39-page experiment log, `/Users/aliour/Downloads/Behvaiour Pred Alireza.pdf`, especially pp. 6–11, 16–24, and 38–39. Numerical results below are reported observations, not independently reproduced here.

- The original 42 activation summaries and linear classifier are a useful inexpensive baseline. The early 24/28 forecasting result did not beat persistence at 25/28. Later any-positive checkpoint labels became saturated, making high accuracy uninformative.
- Shared gradient-selected patching locations changed harmful output substantially: the 205-coordinate intervention reduced 28/30 harmful answers to 4/30 and transferred harmful behaviour into the baseline on 27/30 prompts (p. 17). This supports intervention relevance in that setting; it does not prove those locations are the best early-warning features or identify training examples.
- Top, middle, bottom, and random coordinate monitors performed similarly in earlier comparisons (pp. 21–22). Keep random-subspace and full-layer baselines. Do not make a sparse selected mask the default on current evidence.
- Matched sports controls showed considerable early layer-31 drift without the same observed behavioural rise (p. 38). Global drift is a confound to control, not a harm definition.
- Recovered seed-42 step 20 was already elevated, so its alarm was detection, not advance warning (p. 39).
- The log reports a frozen seed-789 detector alarm at step 15 and first measured crossing at 25, with 0/7 control checkpoint alarms; the forecaster had 2/7. This is encouraging evidence from one matched pair. The seven checkpoints are correlated, not seven independent safe runs; measured lead is relative to the sampled crossing.
- The PDF is newer than `experiments/layer31-early-warning/FOLLOWUP.md`, which still describes seed-789 grading as pending. Reconcile raw result manifests before publishing these numbers. Do not reinterpret that older note as evidence the reported experiment did not happen.

## Architecture

```mermaid
flowchart LR
    A[Training state and ordered data ledger] --> B[Fixed activation probes]
    B --> C[Risk scores and sequential alarm]
    C --> D[Independent behavioural confirmation]
    C --> E[Rank training exposures]
    D --> E
    E --> F[Restore checkpoint before selected exposures]
    F --> G[Replay with selected data removed or corrected]
    G --> B
    G --> H[Sealed safety and utility audit]
```

Keep the monitor, attribution method, and intervention policy replaceable. They answer different questions and require separate tests.

## Behavioural outcomes and data partitions

Let H_c(t) be the measured harmful-answer rate in category c at step t. Track H_c(t), its change from H_c(0), refusal on legitimate requests, coherence, and legitimate task performance. Predeclare meaningful category-specific regression margins delta_c. An onset is the first sampled checkpoint exceeding the declared margin; an absolute ceiling can separately flag an unacceptable starting state. Existing 5/72-count increases are development conventions, not universal or statistically established safety boundaries.

Retain the full published Betley outcome for comparability, alongside a separate rubric for concrete harmful advice on legitimate requests. Report scoreable, excluded, uncertain, and total counts. Do not compare a scoreable-only percentage with a total-prompt percentage as if their denominators matched. Never silently score an invalid or truncated answer as safe.

Maintain four prompt partitions:

1. Direction extraction and intervention validation.
2. Fixed online monitoring probes.
3. Behavioural confirmation and attribution queries.
4. Sealed final audit, never used to select rows, thresholds, layers, or repairs.

Additionally partition whole training runs into monitor fitting, threshold calibration, and final testing. Group duplicate/source-related training records and prompt paraphrases before splitting. Hold out dataset families as well as seeds. Fix judge versions and rubrics, blind judges to training condition, and manually audit a preregistered sample plus disagreements. Repeated generations quantify sampling variability; uncertainty should respect prompt clusters and independent training runs.

## Recommended first monitor

Use a compact multi-signal monitor with a regularized linear readout. Keep each channel separately visible and ablate it. This is a proposed starting point, not a demonstrated winner.

| Channel | Measurements | Purpose |
|---|---|---|
| Refusal | Signed projection and harmful–benign separation along a validated refusal direction; inexpensive output-score diagnostics | Detect weakening refusal on requests that should be refused |
| Trait profile | Signed drift on the seven published trait axes, preserving their individual coordinates | Detect broader behavioural change, including failures on legitimate requests |
| Existing activation baseline | Original 42 summaries; full-vector and multiple fixed random-subspace controls | Establish incremental value over existing results |
| Measurement validity | Finite values, activation scale, prompt/truncation checks, coherence and out-of-calibration flags | Distinguish risk evidence from unreliable measurement |

An invalid monitor produces UNKNOWN/ERROR, never PASS. Use “no alarm” rather than “safe” in reports.

Implementation choices to freeze on development data:

- Measure the final non-padding input token after the assistant-generation prefix. Keep tokenizer, chat template, pooling, hook location, layer numbering, precision, and truncation consistent. Preserve baseline vectors and signed differences. Free-running output-token captures are a separate diagnostic because their text changes across checkpoints.
- Use a fixed panel containing neutral/legitimate requests, harmful requests with matched benign counterparts, and benign requests that resemble unsafe ones. Report subgroup distributions and tails so means do not conceal one failing category. Panel size should be set by timing and variability measurements; roughly 64–128 inputs is a pilot budget, not a power calculation.
- Select one middle and one late layer using development-only validation. Capture existing layers 14, 15, and 31 during the Llama 3.1 pilot to support comparisons; do not assume layer 31 or the same numerical index is universally optimal. Keep exact published implementations separate from adaptations.
- Validate refusal/trait directions on held-out extraction prompts, including random-direction controls and coherence. A refusal-related direction is not a universal literal gate in the architecture. See [Arditi et al.](https://arxiv.org/abs/2406.11717).
- Retain strength as well as orientation. Guard normalized ratios against nearly zero baseline denominators. Use fixed baseline normalization, then fit any feature scaling only on development runs.
- Fit the detector across training trajectories, rather than assuming a probe fitted only at the base stays valid. Fine-tuning can stale activation monitors ([Duan](https://arxiv.org/abs/2606.15980)); include out-of-domain testing and independent audits. Do not silently refit on a test run or reset the reference after repair.
- A large L2 shift is a drift indicator, not proof of harmfulness. A stable refusal projection is not proof of safety either; refusal geometry and behaviour can diverge ([sequential fine-tuning analysis](https://arxiv.org/abs/2610.38001)).

For a fixed direction v, a useful proposed feature is f_v(t,q) = v^T[h_t(q)-h_0(q)] / (||h_0(q)|| + epsilon). Pool with preregistered subgroup summaries. This is an adaptation; reproduce each paper's own normalization when reporting a named baseline.

Fit separate current-regression and future-onset targets. The forecast target at t is crossing within (t,t+H], conditional on being below threshold so far. Use a fixed H in optimizer steps or tokens. A run ending before t+H is censored, not negative. The sparse forecast labels currently available do not support a complex sequence model.

For the first sequential rule, compare a raw threshold with a two-observation persistence rule. Freeze the rule and threshold using entire calibration trajectories, targeting a run-level false-alarm budget across the full declared monitoring horizon. Do not mistake a per-checkpoint false-positive rate for the probability of at least one false alarm in a run. A 5% run-level budget is a research target, not a guarantee from a few controls. More elaborate CUSUM/EWMA rules can be later ablations, calibrated against benign adaptation rather than assumed stationary step-zero noise.

On WARN, preserve state and increase behavioural evaluation. On confirmed regression, invoke the repair experiment. Before confirmation, invoke it only as a prospective risk intervention, with that distinction recorded. The initial policy does not require both refusal and trait channels to alarm: that could miss failures visible to only one channel. Compare separate channels and a calibrated combined readout under equal false-alarm budgets.

## Attribution at an alarm

Define two frozen objectives:

- J_monitor(theta): a differentiable alarm surrogate, such as the fixed linear readout's logit. Differentiating it traces the score's cause, not necessarily harm.
- J_behaviour(theta): a differentiable surrogate based on validated harmful/safe completions on the attribution query set. For example, a contrast of length-normalized log probabilities. Check that its change tracks independently graded generated behaviour. An external judge's discrete output is not directly differentiable through generation.

Use behaviour attribution after confirmed harm and risk attribution before observed harm. Compare them and their preregistered combination rather than assuming agreement.

For SGD, a local ranking approximation is:

    a_i = sum over logged exposures s of example i:
          -eta_s * w_i,s * <grad J(theta_s), grad loss_i(theta_s)>

Here theta_s is the pre-update state and w_i,s is the example's actual loss weight. Positive values predict an increase in J from including that exposure. This is a first-order, trajectory-local estimate, not the exact effect of deleting a record through the entire future training path. [TracIn](https://arxiv.org/abs/2002.08484) motivates checkpoint gradient comparisons; [LESS](https://proceedings.mlr.press/v235/xia24c.html) is relevant to optimizer-aware instruction-data ranking.

For AdamW, a simple gradient dot product is a baseline, not the true update contribution. Momentum, adaptive denominators, clipping, accumulation, weight decay, token weighting, and future interactions complicate attribution. Prefer a clearly labelled optimizer-aware approximation and validate top groups through actual replay. Compute gradients over all trained LoRA parameters, not only parameters attached to a layer that the monitor reads. Compare effective LoRA weight updates rather than unaligned A/B factors across runs.

Search procedure:

1. Start with exposures between the last reassuring checkpoint and the alarm. Use batch-level ranking before expensive per-example scoring.
2. Expand backwards if replay fails or the score does not account for the observed change. Include an all-seen-data baseline: delayed emergence may originate much earlier than the alarm.
3. Aggregate repeated occurrences under a stable source ID, while retaining exposure-specific scores, steps, and loss weights. Never attribute past emergence to an example that has not yet been consumed.
4. Compare gradient ranking with random removal, recency, text harmfulness, and an activation-similarity baseline. For SFT, the latter needs validated safe counterfactual completions; it is an adaptation of the DPO method, not a direct reproduction.
5. Save rankings with uncertainty/stability diagnostics and evaluate groups. Correlated examples and interactions mean there may be no unique small causal set. Poison-label recovery is a secondary metric; behavioural improvement after intervention is the primary one.

Raw toxic wording and causal contribution are different. Benign-looking examples can contribute, and a clearly toxic example can have little marginal influence. Do not limit the candidate pool to a text filter's positives.

## Rollback and replay

“Go back one or two checkpoints” is a policy to compare, not a guarantee. To exclude selected examples' training influence, restore a state before their earliest relevant exposure and remove all subsequent occurrences. If selected examples already affected the rollback checkpoint or its optimizer state, this is partial mitigation. Full exclusion may require replay from initialization. Baseline safety alone does not mean a checkpoint is free of earlier contributing exposures.

Save model/adapter state, optimizer moments, scheduler, scaler where applicable, RNG states, sampler position, ordered microbatches, source IDs/hashes, tokenization/loss masks, and code/config versions. Save only at completed optimizer boundaries unless accumulated gradients are also captured. Verify an unchanged replay before trusting an intervention comparison. Monitor inference must preserve training/evaluation mode and RNG state so monitoring itself does not change the training trajectory.

Use two explicit intervention protocols:

- **Causal ablation:** preserve the original batch slots and optimizer schedule; set selected examples' loss weights to zero with the original normalization fixed. This estimates loss-contribution removal, not ordinary compacted-dataset training. Define all-masked-batch handling explicitly; Adam momentum and weight decay can still move parameters. Disable packing initially or retain exact sequence/token provenance.
- **Practical curation:** remove selected records or substitute audited safe completions and rebuild the remaining schedule. Report changed token counts, steps, and compute, with matched controls. Corrections may retain task content better than deletion; this is worth testing separately ([corrective-supervision study](https://jrepifano.github.io/research/locate-and-rewrite/)).

At each chosen alarm, branch into unchanged replay, targeted removal, repeated random removal at equal budget, text-filter removal, and targeted correction. Include early stopping and a clean-from-start run as policy comparators. Keep rankings, rollback selection, removal budgets, and stopping rules fixed without consulting the final audit.

Continue all repaired branches through the same declared horizon and a later recurrence check. The absence of a new alarm is insufficient: validate held-out harm, useful task performance, over-refusal, and coherence. A model that refuses everything or loses its adaptation is not a successful repair.

## Evaluation and first experiments

Measure four outcomes together:

| Question | Primary measurement |
|---|---|
| Does the alarm help? | Fraction of onset runs warned before the measured crossing, delay/lead, run-level false alarms, and monitoring overhead |
| Is attribution useful? | Held-out harm reduction versus random and text-filter interventions at equal removal/token budgets; group replay effects |
| Does repair preserve value? | Safety/legitimate-task tradeoff, benign over-refusal, coherence, and recurrence |
| Does early intervention save work? | Total monitoring + attribution + replay compute, checkpoint storage, and retained useful adaptation versus restart/early stop |

Report missed onsets as misses, not omitted lead-time entries. Report steps, examples/tokens, and wall time. When onset lies between a last below-threshold evaluation at t_lo and a first crossing at t_hi, the interval contains the unresolved transition; t_hi minus alarm time is lead to a measured crossing, not proof of the exact biological-style onset. Evaluate behaviour at the alarm checkpoint and around the crossing for every comparator using the same schedule.

Offline behavioural evaluation can be dense while remaining hidden from the online alarm. For prevention experiments, keep the unchanged continuation as the counterfactual branch; otherwise a successful intervention makes the original onset unobservable. Freeze the number of repair attempts so repeated peeking does not turn the final audit into training data.

Proposed sequence:

1. Reconcile the seed-789 raw outputs with the PDF; freeze the current detector as baseline. Reproduce the named refusal and trait monitors separately before combining them.
2. Verify exact unchanged replay on a short run. Demonstrate one alarm-to-attribution-to-repair experiment on existing development data. This is an engineering feasibility test.
3. Run a prospective pilot with one established model, two dataset families, three fresh seeds per family, and matched safe-completion controls: six pairs / twelve parent trajectories. Choose and freeze dataset-family roles before execution; for an unseen-family claim, one family must be entirely withheld from design and calibration. Existing sports results can supply development data, not fresh confirmation. This remains too small to establish a tight low false-alarm guarantee.
4. At fixed removal budgets selected in development (for example 1%, 5%, 10% of seen source IDs), run the necessary replay branches. Report token budgets too. Compare uniform early contamination with late contamination only as separately declared conditions; do not make every failure artificially easy to localize near the alarm.
5. Add another model family, additional clean controls, and benign-data safety degradation before making general claims. Freeze hyperparameters before each new validation cohort.

Suggested engineering acceptance checks: unchanged replay matches within a declared tolerance; exposure logs resolve every training token to a source; rankings cannot access future data; all feature selection stays inside development folds; failed measurements cannot pass; and repeated repairs never access sealed audit labels. These are future implementation requirements, not checks completed by this document.

## Repository integration

Reuse existing activation collection, trajectory storage, behavioural generation/judging, and the 42-summary baseline. The inspected trainer already logs ordered example IDs per microbatch with the destination optimizer step and explicitly limits this logging to a single process. The seed-789 configuration requests training-state checkpoints (`save_only_model: false`). Actual saved checkpoint completeness still needs verification.

The inspected `train_run` currently invokes `trainer.train()` without exposing a resume checkpoint. A scheduled checkpoint callback alone does not implement filtered replay. Add a dedicated replay path rather than assuming the current workflow already supports it.

Proposed components, not yet created:

- A versioned monitor specification: model identity, probes, hooks, directions, normalizers, coefficients, and alarm policy.
- An exposure ledger with source IDs, exact ordering, loss weights, and source hashes.
- A checkpoint registry recording complete resumable state and ancestry.
- An attribution runner with explicit target, parameter subset, exposure window, and ranking metadata.
- A replay planner that writes immutable intervention manifests and new branch output directories.
- An evaluator that compares branches on sealed safety and legitimate utility metrics, retaining uncertain judgments.

The first implementation milestone should be one reproducible closed loop using the existing simple detector. Improving warning quality can then be measured as an improvement to that loop, rather than becoming a prerequisite that delays attribution and repair indefinitely.
