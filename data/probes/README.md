# Behaviour monitoring probes

`online_probes.jsonl` is the default monitoring and review probe file. It contains
the expanded 30-prompt suite, also retained in `behavior_eval.jsonl` for compatibility.
The former 40-prompt online suite is preserved in `online_probes_40.jsonl`.

The expanded suite contains 30 fixed prompts: 10 harmful requests, 10 indirectly
harmful requests, and 10 benign requests. Group names describe the request, not
the model's response. A refusal or supportive reply to a harmful request can be
non-harmful. Label actual responses independently.

The original six prompts retain their IDs and wording. An unchanged copy is saved
as `behavior_eval_6.jsonl` for reproducing the original experiments.

The expanded suite covers self-harm, hate speech, violent threats, discrimination,
and child abuse. Direct and indirect groups each contain two prompts per topic.
Indirect prompts use fictional framing, editing requests, role-play, and claims
of prior agreement. These are single-turn simulations of conversational attacks;
they do not contain a real multi-turn interaction history. Benign prompts include
similar sensitive topics so topic mentions alone should not imply harmful output.

To compare all seven models with this suite, recapture activations and generate
new response reviews for every saved checkpoint using the 30-prompt file. Use a
fresh output root, such as `outputs/probes-30`, so the original six-probe results
and labels remain intact. Old labels can only be transferred for exact matches
of model, step, probe, prompt, and response. With six checkpoints and seven runs,
the full expanded response set contains 1,260 rows before any verified reuse.

The expanded activation schema differs from the six-probe schema. Refit monitors
on the expanded data; do not use the frozen six-probe monitor on these snapshots.
These fixed monitoring prompts are not an independent evaluation prompt suite.
