# From a fine-tuning warning to a tested data repair

**Research question.** Does the state of a fine-tune at its first warning help identify which already-seen batches to replace, beyond a ranking made before training?

## 1. Define the outcome and freeze the alarm

Start with one reproducible failure mode, preferably harmful answers to benign requests. Use the existing Llama 3 8B runs for development. Audit which evaluation prompts are benign and count transitions on that subset before choosing a training condition. The existing pilot measures harm across the entire evaluation suite, so its onset results do not establish onset for the benign subset.

- Freeze the monitoring prompts, model, fine-tuning recipe, alarm features, and threshold on development runs. Start with full layer 31 or the fixed random subset from the pilot.
- Define onset as the first measured checkpoint where the harmful-answer rate on independent prompts crosses a prespecified increase above step zero. The pilot's two-additional-answers rule is a starting point. Recalibrate if the prompt suite or sampling protocol changes.
- Fix the output judge, decoding settings, and evaluation cadence. Include runs where the behaviour stays below threshold.
- Choose a downstream task metric and a tolerable loss before running the repair comparison. Record benign over-refusal and task performance alongside the targeted behaviour.

**Ready when** a fresh run gives a warning strictly before measured onset and the evaluation can distinguish the target behaviour from noise. Keep failed warnings and no-onset runs in the record.

## 2. Make a fresh run replayable

Log the example IDs and order for each optimizer update, including all accumulated microbatches. Save resumable checkpoints with adapter weights, optimizer, scheduler, RNG and data-loader position. Save the exact run configuration and data hashes. Continue the original branch through the planned training budget to measure the unmodified trajectory. Write the first warning and the candidate batch ranking before consulting any later labels.

Choose a fixed replay window ending at the warning and a checkpoint immediately before its first update. Verify that unchanged replay reproduces the original batch stream and approximately reproduces the alarm and output trajectory. Record any numerical or sampling differences.

**Ready when** unchanged replay is reproducible enough to distinguish intervention effects from run-to-run noise.

## 3. Rank batches using the warning checkpoint

Start with a gradient baseline. Define a differentiable risk score on fixed monitoring probes. At the warning checkpoint, score each batch in the replay window by how much a descent step on its training loss would increase that score. A first-order estimate is the negative dot product of the risk-score gradient and the batch-loss gradient, restricted to trainable parameters. Treat this as a ranking heuristic and let replay test its causal value.

Compute the same ranking from step zero, with the same risk objective, candidate batches and replacement budget. Compute and freeze these step-zero scores before training. Use the same reference normalization for both rankings. Freeze both rankings and the replacement choices before evaluating any branch.

**Ready when** the two rankings are recorded, their overlap is known, and the replay arms use the same replacement rule.

## 4. Fork four branches from the same saved state

| Branch | Batches replaced within the replay window |
|---|---|
| Unchanged replay | None |
| Random replacement | The same number, selected randomly |
| Step-zero ranking | Highest-ranked by the pre-training scores |
| Warning-time ranking | Highest-ranked using the alarm checkpoint |

Use an independently defined pool of replacements matched for task, format and token count where feasible. Keep the optimizer schedule and total training budget fixed. Hold later batch order fixed after the replay window. Continue every branch to the same planned final step and log extra training compute separately.

## 5. Decide what the experiment supports

Report first onset step, harmful-answer rate at the final step, downstream task performance, benign over-refusal, and extra compute. Compare warning-time replacement directly with both random replacement and step-zero ranking. Use fresh seeds for the next round and keep a fine-tuning dataset entirely out of development for the transfer test.

- Lower harm with task performance within the prespecified margin supports a useful repair.
- A lower alarm score with similar harmful outputs suggests that the ranking explains what moves the monitor more readily than what changes behaviour.
- Similar outcomes for warning-time and step-zero ranking suggest the alarm may add value for timing while the diagnosis gains little from observing the trajectory.

**First deliverable:** One reproducible four-branch replay with a table of measured outcomes. Treat this as a feasibility result. Estimate reliability across fresh runs before making a general claim.

## Starting files

- [Early-warning pilot](README.md)
- [Existing checkpoint schedule and storage policy](../../docs/early-steps.md)
