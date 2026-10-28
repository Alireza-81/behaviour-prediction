# Behaviour monitoring probes

`online_probes.jsonl` is the single active monitoring and review probe file.
Older probe sets and duplicates are recoverable from the archive.

The expanded suite contains 30 fixed prompts: 10 harmful requests, 10 indirectly
harmful requests, and 10 benign requests. Group names describe the request, not
the model's response. A refusal or supportive reply to a harmful request can be
non-harmful. Label actual responses independently.

The original six prompts retain their IDs and wording.

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

Current ground truth is in `outputs/probes-30/labels/`, using `label` and
`behaviour_tags`. These fixed monitoring prompts are not an independent
evaluation prompt suite. Changing the prompts requires new activation captures.
