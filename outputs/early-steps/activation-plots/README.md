# Activation trajectory comparisons

Three complementary groupings:

- `request_class`: harmful, harmful_indirect and benign requests.
- `probe_topic`: fixed prompt categories (self_harm, hate_speech, etc.). These include protective benign prompts for those topics.
- `behaviour_association`: fixed sets of monitoring probes whose responses received each behaviour tag anywhere in the nonzero runs. Uses the same set for every model and step. Membership overlaps. These plots are label-selected descriptions, not independent evaluation or causal attribution.

The existing monitoring suite has no dedicated harassment category or separate encouragement/facilitation prompt categories. The response-associated view makes those distinctions without inventing new activation measurements.
For harassment, the associated group contains only one probe; for self-harm facilitation, also one. Treat these as individual-probe trajectories rather than broad class estimates.

Each line is the arithmetic mean of individual-probe L2 distances from its own run’s step-zero activation. It is not the norm of an averaged vector. Layers 15 and 31 are plotted separately.
Stars/vertical lines mark the first saved checkpoint with any Claude-labelled harmful monitoring response, not the onset of the particular plotted behaviour.
CSV exports include per-probe and class-level L2, relative L2 and cosine distance. Plots use L2 to match earlier figures.

Open `model_comparison_layer_31.png` inside each grouping for an overview. `between_models/` holds one plot per class; `between_classes/` holds one plot per model.

Regenerate with: `python scripts/plot_early_activation_comparisons.py`

## Behaviour-associated sets with request-class controls

See [request-class split plots](behaviour_association/request_class_split/README.md). These compare benign topic-matched controls, direct requests and indirect requests, with the first observed harmful output marked per curve and the model’s overall first harm marked separately.
