# First activation patching pilot

This is preparation and visualization only. No patched language-model responses have been generated locally.

Source model: hate_speech fine-tuning run, step 20. Three prompts with Claude labels non-harmful at step zero and hate speech at step 20; three benign controls remain non-harmful at both endpoints. The IDs and original responses are frozen in pilot.json. This is a selected development pilot, not an unbiased accuracy experiment.

The first interventions replace **all 4096 coordinates** at the last prompt token in the output of decoder block 15, block 31, or both, using the SAME tokenized prompt in the step-zero model. Coordinates are residual-stream features, not necessarily individual neurons or interpretable concepts. The 64×64 display has no spatial meaning.

Selected locations are the two layers already measured, not saliency-ranked discoveries. Current heatmaps show signed values and differences, not harmfulness scores. Do not select “harmful neurons” merely because their absolute changes are large.

Conditions: baseline step zero; unpatched step 20; same-model self-patch (no-op control); baseline patch at layer 15, layer 31, or both. Six prompts × six conditions = 36 answers. Self-patch must match the unpatched greedy token sequence exactly. Layer 31 is the final block, so replacing it can directly restore baseline next-token logits; success there alone does not localize a specific harmful circuit.

Scope: only the initial prompt prefill is patched. Generated-token positions are not patched; earlier recipient KV-cache entries remain. A negative result does not establish that a component is irrelevant, and any effect can be mediated through later generation. This pilot does not change model weights or train a monitor.

After syncing scripts/run_activation_patch_pilot.py, src/behaviour_prediction/monitor/patching.py and configs/activation_patch_pilot.json to the GPU, run:

```bash
python scripts/run_activation_patch_pilot.py --pilot-dir outputs/activation-patch-pilot
```

On the GPU the script creates the output folder and copies the tracked configuration into pilot.json if needed.

The run uses checkpoint-20 from outputs/early-steps/hate_speech and the shared base model. It does not train, call Claude, or save another model checkpoint. Same command resumes completed probe/condition pairs.

Outputs: generated.jsonl (full experiment records and token IDs), reviews/*.csv (one per condition, compatible with the existing labeling prepare script), and run_report.json. Original labels are selection references, not labels on new patched answers. Label the new answers before claiming the intervention reduces harm; judge target behaviour, coherence and benign usefulness, and avoid treating generic refusal or broken text as a selective repair.

## Retraining when checkpoint 20 is unavailable

Use the dedicated pilot-training config to avoid bulk answer generation:

```bash
behaviour-train --config configs/patch-pilot/experiments.yaml --runs hate_speech --overwrite
```

It trains only hate_speech through step 20, preserves the original 50-step learning-rate schedule, captures activation snapshots at steps 0/20, and saves only checkpoint-20. It does not generate any monitoring/evaluation answers during training. Previously generated response files outside the run directory are historical and are not used by this config.

The patch runner accepts a compatible retrained model even when its monitoring metadata differs, but old labels do not establish the fresh model's behaviour. Label the newly generated baseline and unpatched answers before interpreting patch effects.
