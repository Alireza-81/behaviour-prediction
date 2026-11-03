# Selecting activations for causal patching

Research review and proposed experiments · 3 November 2026

## Recommendation

**Start with a SafetyNeuron-style experiment: select a shared set of MLP neurons using activation differences on identical text, then test those neurons with dynamic patching throughout generation.** Compare that selection against integrated-gradient attribution and random selection. After establishing a reliable intervention and evaluation, try Matryoshka attribution (MAttr) as the learned selector.

This is a recommendation for this project's next experiment, not a claim that one method wins universally. No reviewed paper directly compares all these methods on our exact Llama-3-8B-Instruct QLoRA checkpoints and labels. Selection quality and intervention duration must be varied separately.

The nearest direct precedent is Chen et al., *Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons*, NeurIPS 2025. Other particularly useful strands are SafetyLock, attribution patching, learned masks, and learned subspaces. The literature also contains important counterexamples to treating a successful repair as proof of a unique mechanism.

## Our starting point

The current configuration uses `meta-llama/Meta-Llama-3-8B-Instruct`, NF4 quantization, and LoRA updates to attention and MLP projections. It is Llama 3, not Llama 3.1.

Our existing coordinate experiment ranks absolute step-20 versus step-0 differences separately for each prompt. It patches selected dimensions of the layer-31 block output at the final prompt token, once, before generation. Those are residual-stream coordinates, not necessarily individual MLP neurons. The selected mask changes between prompts.

Relevant local implementation:

- `src/behaviour_prediction/monitor/patching.py`
- `scripts/run_top_change_activation_patch.py`
- `configs/activation_patch_top_change.json`
- `docs/top-change-activation-patching.md`

Current donor capture uses `no_grad` and detached tensors. Saved activations can support difference rankings, but do not contain the computation graph needed for attribution gradients. New module-level and generation-time capture is required for the recommendation.

Our previous full-vector layer-31 patches change the first-token computation directly. Since final normalization and the output head are unchanged by our adapter, copying the entire last block output copies the donor's first-token logits, up to numerical effects. That is a useful control, but weak evidence about where harmful behaviour originates. Later tokens still use the recipient model and its own cache.

## What the selection methods actually answer

| Method family | Question answered | Main limitation |
|---|---|---|
| Absolute change, RMS change | What changed between conditions? | Change can reflect style, topic, or generic adaptation. |
| Classifier or probe ranking | Where is a label predictable? | Predictability does not establish causal influence. |
| Gradient attribution | Which proposed replacements may change an output objective? | Approximation and objective can be misleading. |
| Exact activation replacement | Does this particular replacement change the response? | Establishes an intervention effect, not a complete mechanism. |
| Learned masks | Which components work together under a budget? | Can overfit, exploit a proxy, or depend on the reference choice. |
| Learned subspaces | Which combinations of coordinates enable targeted control? | A successful controller may not identify the original computation. |
| SAE features | Can we intervene in a more interpretable feature basis? | Requires a suitable dictionary and careful reconstruction controls. |

## Most relevant papers

### 1. SafetyNeuron: the closest match to the proposed experiment

[Chen et al., NeurIPS 2025](https://arxiv.org/html/2406.14144v2), especially Sections 3–4; [official code](https://github.com/THU-KEG/SafetyNeuron).

They rank pre-down-projection MLP neurons by RMS activation differences between checkpoints on the **same prompt and continuation**. Selected neurons are then replaced using a donor run on the recipient's current prefix at each generation step. Roughly 5% of neurons recover over 90% of their alignment improvement; this is normalized cost-score recovery, not a 90% reduction in harmful-answer counts.

Important qualification: their alignment uses IA3 on MLPs to preserve neuron functionality. Our LoRA changes the underlying projections, so direct correspondence is an empirical assumption. Their percentages must not be imported as an expected result here.

### 2. SafetyLock: a particularly relevant attention-head alternative

[Zhu et al., *Locking Down the Finetuned LLMs Safety*](https://arxiv.org/html/2410.10343), Section 3; [official repository](https://github.com/zhu-minjun/SafetyLock).

SafetyLock ranks heads by held-out accuracy of per-head logistic classifiers on safe/unsafe response activations. It derives safe-minus-unsafe directions over response-token windows and applies them to fine-tuned descendants. The repository lists an artifact for our exact Llama-3-8B-Instruct family. This makes it a useful practical comparator, although that artifact has not been downloaded or tested here. Classifier-based selection still needs causal validation. Adding a direction also differs from replacing activations with live step-0 values.

### 3. Attribution patching and AtP*: a fast causal shortlist

[Kramár et al., *AtP*: An efficient and scalable method for localizing LLM behaviour to components*](https://arxiv.org/html/2403.00745), Sections 2–3 and 5.

Attribution patching estimates an intervention effect using activation difference times the gradient of an output objective. AtP* addresses attention saturation and cancellation failures and recommends verifying candidates by exact patches. For this project, it offers an efficient shortlist across many heads or neurons. It is not ordinary gradient saliency: the donor–recipient difference matters. Q/K-specific corrections are less directly relevant if we initially patch MLP intermediate activations.

### 4. EAP-IG: a strong attribution comparator

[Hanna et al., *Have Faith in Faithfulness*, COLM 2024](https://arxiv.org/html/2403.17806), Sections 3–4 and Appendices C–D; [official code](https://github.com/hannamw/EAP-IG).

EAP-IG averages gradients along an interpolation to improve circuit faithfulness over plain edge attribution. Their experiments often need only a few integration steps; that is not a guaranteed setting for our checkpoint differences. Their original method scores edges, whereas a neuron-level activation-space IG adaptation would score our proposed replacements. Label that adaptation accurately. Benchmark faithfulness and accurate individual effect estimates are different measurements. The released framework uses TransformerLens, so our PEFT/4-bit setup is not a verified drop-in target.

### 5. MAttr: the most attractive learned mask follow-up

[Arora et al., *Matryoshka attribution*, September 2026 preprint](https://arxiv.org/html/2609.25518v1), Sections 3–5; [representation code](https://github.com/aryamanarora/matryoshka-attribution).

MAttr learns shared importance scores while varying the retained-component budget during training. One ranking supports nested subsets at different sizes. Representation attribution already appears in the paper; its RL experiments concern parameters. We can adapt the activation mask objective to safe restoration rather than assume its original objective matches ours. It is promising for interactions and budget curves, but is recent evidence on other tasks. A good soft-mask training result must be checked with actual hard replacements and free generation.

### 6. DAS and the context-sensitivity paper: useful if directions matter

[Geiger et al., *Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations*](https://arxiv.org/abs/2303.02536); [Wu et al., *Interpretability at Scale*](https://arxiv.org/html/2305.08809).

DAS learns a rotated subspace whose interchange changes a specified causal variable; Boundless DAS also learns its size. This permits behaviour to be distributed across ordinary coordinates rather than assigned to a few raw dimensions.

[Minder et al., *Controllable Context Sensitivity and the Knob Behind It*, ICLR 2025](https://arxiv.org/html/2411.07404v4), Sections 3.3–3.4, gives the Paris/London-style example previously discussed. It localizes components, then learns a rank-one context-versus-prior subspace. That is an excellent design precedent, but context selection in factual answers differs from suppressing harmful open-ended generation. Use it as an alternative hypothesis, not evidence that our problem is rank one.

## Broader methods and what to borrow

The entries below summarize primary papers. “Follow-up” means worth considering if the first experiments justify the added complexity; it does not imply tested compatibility with this repository.

| Paper | Relevant contribution | Decision for this project |
|---|---|---|
| [Conmy et al., ACDC, 2023](https://arxiv.org/abs/2304.14997) | Iterative causal removal of connections to recover a task circuit. | Borrow exact verification and small-set pruning; full edge search is expensive. |
| [Syed et al., Attribution Patching Outperforms Automated Circuit Discovery, 2023](https://arxiv.org/abs/2310.10348) | Fast gradient-based edge ranking. | Historical baseline; compare stronger attribution variants before a large edge graph. |
| [Bhaskar et al., Finding Transformer Circuits with Edge Pruning, 2024](https://arxiv.org/html/2406.16778) | Learns edge masks with counterfactual activation replacement while preserving outputs. | Useful joint-selection precedent. Requires substantially more graph instrumentation than our current hooks. |
| [Rezaei Jafari et al., RelP, 2025](https://arxiv.org/html/2508.21258v2) | Replaces attribution gradients with relevance-propagation coefficients; compares exact effects and sparse-feature circuits. | Strong possible comparator if ordinary attribution is unreliable; model-specific backward rules add integration work. |
| [Zhang & Wang, When Attribution Patching Lies, June 2026](https://arxiv.org/html/2606.09899v1) | Diagnoses nonlinear approximation errors and proposes HVP corrections. | Recent preprint. Consider only if exact checks reveal serious errors; higher-order differentiation complicates quantized execution. |
| [Zhang et al., Guided-AP, October 2026](https://arxiv.org/html/2610.27393v1) | Corrects approximate scores using a small number of exact measurements. | Recent preprint; useful for repeated large sweeps, not necessary for a small initial module grid. |
| [Arditi et al., Refusal in Language Models Is Mediated by a Single Direction, 2024](https://arxiv.org/html/2406.11717v3) | Extracts contrastive refusal directions; tests addition and directional ablation. | Low-cost direction baseline. Its interventions span more token positions than our one-time patch. Refusal is not equivalent to all harmful behaviour. |
| [Wollschläger et al., The Geometry of Refusal, 2025; revised 2026](https://arxiv.org/html/2502.17420) | Optimizes directions with behavioural and retain losses; studies multidimensional refusal cones and intervention independence. | Strong follow-up for learned directions. Much of the objective studies refusal removal; adapt carefully to restoration. |
| [Lee et al., CAST, ICLR 2025](https://arxiv.org/html/2409.05907v3) | Separates a condition detector from a behaviour steering vector. | Useful later for selective intervention, including behaviour-specific gates. Evaluate the gate and the correction separately. |
| [O'Brien et al., Steering Language Model Refusal with Sparse Autoencoders](https://arxiv.org/abs/2411.11296) | Demonstrates refusal intervention through SAE latents, with capability tradeoffs. | Relevant feature-space comparator; not evidence of a ready-made dictionary for our model. |
| [Marks et al., Sparse Feature Circuits, 2024](https://arxiv.org/html/2403.19647) | Builds causal graphs over SAE features; studies targeted edits and reconstruction-error nodes. | Useful for interpretable maps after reliable localization. Preserve reconstruction residuals in intervention controls. |
| [Assogba et al., CC-Delta, 2026](https://arxiv.org/html/2602.12418) | Selects SAE features from matched request tokens, using statistical filtering and standardized shifts, then steers. | Particularly relevant feature-selection evidence. Its paired inputs are jailbreak/no-jailbreak contexts, not checkpoint pairs; Llama experiments use 3.1, not our 3.0. |
| [Zhao et al., Safety-Specific Neuron / SN-Tune, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6d2666e2cf44088cc57204fbc5ef7f34-Abstract-Conference.html) | Detects safety-sensitive units and selectively tunes them. | Useful neuron/parameter comparison, but tuning weights is a different intervention from our activation transplant. |
| [Zhou et al., Ships/Sahara, ICLR 2025](https://arxiv.org/html/2410.13708) | Scores head interventions and searches for interacting safety heads. | Relevant attention-head search alternative. Head identifiers and strong ablation effects do not transfer automatically between models. |
| [Yu et al., SafeSeek, 2026](https://arxiv.org/html/2603.23268) | Optimizes binary masks over multiple granularities, with KL faithfulness and sparsity; validates by ablation. | Relevant learned-mask baseline. Its zero/multiplicative masking and output-preservation objective differ from cross-checkpoint safe replacement. |
| [Orgad et al., Harmful response mechanism / B-TAP, 2026](https://arxiv.org/html/2604.09544) | Signed parameter attribution with separate harmful-generation and benign-preservation datasets; evaluates cross-harm transfer. | Particularly useful for our later weight study. Borrow the separation of target and preservation importance; do not present weight pruning as activation localization. |
| [Wu et al., ReFT, 2024](https://arxiv.org/abs/2404.03592) | Learns interventions on representations with frozen underlying model weights. | Useful controller engineering, but freely learned corrections can add a new computation rather than isolate an existing one. |
| [Sun et al., HyperDAS, 2025](https://arxiv.org/abs/2503.10894) | Learns token localization and concept features with a hypernetwork; evaluates Llama3-8B. | Interesting for future token-dependent selection, but too much flexibility and data demand for our current first test. |

ReFT, HyperDAS, and the original EAP/ACDC papers were screened at abstract/method-summary level for this recommendation. The closest papers above and the principal methodological critiques were inspected in their method sections. Official repository documentation was checked for SafetyNeuron, SafetyLock, MAttr, EAP-IG, and pyvene; code availability is not the same as reproducing the result.

## Evidence that should change our experimental design

### A successful patch does not establish a unique behaviour circuit

[Makelov et al., *Is This the Subspace You Are Looking for?*, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html) demonstrates that a subspace intervention can produce the intended answer by activating a dormant pathway rather than faithfully identifying the original feature. Successful control is evidence worth reporting, but needs additional mechanistic checks.

[Heimersheim & Nanda, *How to use and interpret activation patching*, 2024](https://arxiv.org/html/2404.15255) explains why redundancy, backup pathways, negative components, and intervention direction change the interpretation. Test groups as well as individuals. A neuron with little individual effect can still matter jointly.

### Reference choices and metrics can alter the localization

[Zhang & Nanda, *Towards Best Practices of Activation Patching*, ICLR 2024](https://arxiv.org/html/2309.16042) shows sensitivity to corruption methods, output metrics, and patch windows. For us, a “safe” step-0 donor must be checked on each prompt. Baseline failures cannot demonstrate restoration to a safe reference. A larger perturbation is also not automatically a more behaviour-specific intervention.

### Local repair and durable safety are different claims

[Lee et al., *Refusal Localizes, the Damage Relocates*, October 2026 preprint](https://arxiv.org/html/2610.00320v1) uses same-prefix clean-state patching at prefill and every decoding step. It finds recoverable refusal, but adaptive fine-tuning can bypass a protected region or defeat a previously effective low-rank repair. The useful lesson is to test transfer across runs and training regimes before claiming a persistent safety location.

### Unrelated benign prompts are insufficient utility checks

[Canesse et al., *PatchBench*, October 2026 preprint](https://arxiv.org/html/2610.10276v1) distinguishes harmful-neighbour correction from preservation of benign prompts with similar words or structure. Global capability scores can miss these local regressions. Our five benign controls are useful, but too small to support a broad preservation claim.

### Benchmarks support comparisons, not a universal winner

[Mueller et al., *MIB*, ICML 2025](https://arxiv.org/abs/2504.13151) separates circuit localization from causal-variable localization. Attribution/mask methods and supervised subspace methods address different tracks; SAE features do not automatically outperform ordinary neurons. We should compare methods on a common intervention budget and held-out behaviour objective, rather than choose by interpretability labels or headline scores.

## Proposed experiment sequence

These are proposed adaptations, not reproductions of the papers or experiments already run.

### Experiment A: shared neuron selection and exact dynamic patching

**Question:** Does a small set of consistently changed neurons account for the safety difference between step 0 and step 20?

1. Freeze the exact checkpoint identities, tokenizer, template, decoding configuration, and judge rubric.
2. Reserve separate selection, validation, and final test prompt families. Treat our already inspected 30 prompts as development data. Do not select a mask using final-test labels or activations.
3. Capture MLP intermediate activations before `down_proj`, initially across all 32 layers. Keep the old residual-coordinate experiment as a separate baseline.
4. Compare checkpoints on identical token sequences. During selection, replay chosen development continuations through both endpoints; never subtract states from independently generated, diverging prefixes.
5. Produce one dataset-level RMS-difference ranking. Report per-layer counts, selection stability across prompt subsets, and category contributions. A per-prompt ranking can remain a diagnostic, but it does not test a reusable shared set.
6. Evaluate increasing budgets, starting with 0.1%, 0.5%, 1%, 2%, and 5% of the defined candidate population. Explicitly distinguish “percentage across all MLP neurons” from “percentage of one layer's 4096 residual coordinates.” These are different denominators.
7. Keep the selected indices fixed during the held-out test. Compare initial-prompt-only patching against prompt plus decoding patching **using the same selection**.
8. For dynamic replacement, both endpoints process the recipient's current prefix. The recipient chooses the next token; the donor follows that token. Maintain separate caches, and never let the donor independently continue its own answer.
9. Run safe restoration (0 → 20) and reverse transfer (20 → 0) separately. A set effective in only one direction can still be informative; bidirectional effects are not guaranteed.

LoRA correspondence check: our `gate_proj`, `up_proj`, and `down_proj` all change. Neuron indices remain well-defined, but their function may move. If swaps repeatedly degrade fluency, inspect projection changes and compare whole-module outputs or learned subspaces before concluding no localized effect exists.

Practical compute: dynamic donor replacement needs a donor forward step as well as a recipient step, two cache sets, and a new generation loop. The same quantized base may support switching adapters, but that must be validated rather than assumed. It is more expensive than the current sequential endpoint capture. No new language-model fine-tuning is inherently required.

### Experiment B: choose by behavioural influence, not change alone

**Question:** Can attribution find smaller or more selective interventions than RMS differences?

Use the same candidate neurons, data split, patch scope, budgets, and generation evaluation as A. Add plain difference-times-gradient ranking, then an activation-space integrated-gradient version if needed. Call it an adaptation of attribution patching; it is not automatically a reproduction of EAP-IG's edge algorithm.

Define the output objective before scoring. One candidate is the length-normalized log-probability of a reviewed safe continuation minus that of a reviewed harmful continuation. Use several response formulations to reduce dependence on a particular refusal phrase. On benign examples, separately measure changes to the original useful continuation distribution. The signs matter: importance in either direction is not the same as benefit for safe restoration.

A Claude harmfulness label is not differentiable, and the old 42-number checkpoint monitor is not an established causal output objective. Neither supplies a reliable gradient through the language model as currently implemented. Use continuation likelihood for screening and independent judged free-generation results as the endpoint.

Exactly patch a sample from the top, middle, and bottom of each ranking. Compare estimated effect with measured effect, including harmfulness changes on full answers. Also test selected sets jointly. If attribution cannot predict those checks, do not trust a visually convincing heatmap; consider more integration steps, AtP*, RelP, or abandoning the approximation for the smaller search space.

### Experiment C: learned masks, then learned subspaces

**Question:** Can joint optimization improve over individually ranked components?

Implement an MAttr-style selector over a bounded candidate set after A/B establish a usable target. Freeze the language-model weights. Learn which donor replacements improve the development objective, with an explicit budget and benign-retention term. Be explicit about whether mask value 1 means “retain recipient” or “apply donor”; the paper's retention convention and our restoration convention need not match.

Train across several budgets, then evaluate hard top-k replacements on validation data. Select the budget once, freeze the mask and settings, and evaluate on new prompts. Repeat initialization seeds and compare against the A/B selectors. This is our proposed safe-restoration objective, not a result already demonstrated by MAttr on this project.

If raw-coordinate masks are unstable or inefficient, compare a low-rank interchange intervention with ranks such as 1, 2, 4, and 8. For an orthonormal basis U, replace only the donor projection:

`patched_h = recipient_h + U @ U.T @ (donor_h - recipient_h)`

This tests whether a distributed direction works better than a sparse coordinate subset. A freely learned additive correction is a separate controller baseline. Keep its success claim distinct from discovery of an existing feature.

### Where RL fits

RL is a later option when the desired objective is a score on sampled complete responses and cannot be differentiated directly. It is not required for the first activation-selection experiments. First validate the simpler continuation objective against generated-answer labels. If those disagree systematically, an RL-trained mask could optimize a sequence-level reward, with benign usefulness and sparsity constraints.

Do not train exclusively for non-harmful labels: silence, repetition, truncation, or refusal of everything could satisfy that objective. Report these outcomes explicitly. An RL-learned intervention also needs hard-mask evaluation and independent prompts.

## Controls, metrics, and visuals to retain

**Controls:** unpatched endpoints; recipient-to-itself patches; whole-component patches as ceilings; random masks matched by count and per-layer distribution over several seeds; perturbation-norm-matched controls where feasible; shuffled donor prompts; direct versus indirect requests; same-topic benign neighbours; unrelated useful tasks. Count and perturbation-energy matching are different controls and may require different designs.

**Main results:** harmful-answer count and denominator, repairs among originally harmful answers, newly induced harm among originally safe answers, benign over-refusal, usefulness/fluency, truncation rate, intervention size, runtime, and variation over prompts/seeds. Preserve paired results and uncertainty intervals. Do not use refusal keyword counts as the sole harmfulness measure.

**Visuals:** layer × component effect heatmaps; RMS score versus measured patch-effect scatter; safety-versus-budget curves with random-seed bands; before/after harmfulness grids; benign preservation versus harmful repair; overlap and stability of selected neurons across behaviour categories. Make observational score maps and exact intervention maps separate figures.

**Behaviour-specific follow-up:** learn/select masks separately for hate speech, threats, self-harm encouragement, and other labelled behaviours; evaluate each mask across every category. A matrix with mask category on rows and response category on columns distinguishes selective effects from a general refusal switch. Reuse causally validated features in the emergence monitor only after checking whether they generalize to unseen runs and early checkpoints. A post-hoc step-20 selector is not automatically an early-warning detector.

Even zero harmful answers on 25 requests would establish only zero observed failures on that sample and protocol. The approximate rule-of-three upper bound is about 12% for an iid population interpretation; template dependence makes broad generalization less justified. Keep the stronger objective, but phrase the evidence precisely.

## Decision

1. **Best first test for our current question:** shared MLP change ranking plus exact dynamic patching, with duration and selection controlled separately.
2. **Best immediate competing selector:** verified activation-space integrated-gradient attribution.
3. **Best learned-selector follow-up:** MAttr-style masks with our safe-restoration objective.
4. **Best compact direction alternative:** learned low-rank interchange, with contrastive refusal directions as a simpler baseline.
5. **Best practical safety-steering comparator tied to our model family:** SafetyLock.
6. **Later interpretability investment:** SAE feature selection, after model/dictionary compatibility and reconstruction quality are established.

This review adds documentation only. No checkpoint, experiment configuration, generation output, label, or paid API job was changed.
