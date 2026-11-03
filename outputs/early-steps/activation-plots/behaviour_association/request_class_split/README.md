# Behaviour-associated probes split by request class

`between_classes/<model>/layer_31.png` compares benign, directly harmful and indirectly harmful requests for every behaviour-associated probe set. Layer 15 is also available.

`between_models/<layer>/<behaviour>/<request_class>.png` compares all models for one fixed behaviour-associated set and request class.

Stars mark the first saved checkpoint with ANY harmful output among that curve's monitoring probes. This is not necessarily the first occurrence of the named behaviour. Dotted vertical lines in the within-model views mark that model's first harmful output across all 30 monitoring prompts. Labels list the corresponding steps. “None” means no harm observed; missing groups are explicitly marked and never plotted as zero.

Membership remains fixed across all runs and steps, using the existing label-selected behaviour-associated groups. Those groups overlap. The benign curve is an added control: all saved benign monitoring prompts with a topic matching any probe in that behaviour-associated set. These controls are NOT members of the original behaviour-associated union, and never generated labelled harm here. Direct and indirect curves retain their original behaviour-associated membership. For broader comparisons use the separate request_class/ plots.

These are descriptive activation measurements on prompts, not generated-token activations, causal attribution, or a new monitor experiment. No responses were generated or relabelled.

Regenerate: `python scripts/plot_behaviour_request_splits.py`
