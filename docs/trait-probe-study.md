# Trait probe trajectories

The active panel contains 210 matched triplets across 14 behaviors, including refusal calibration. Every behavior has nine development triplets and six held-out triplets. Each triplet shares a scenario and response format across an appropriate instruction, a pressure instruction, and a paraphrase of the appropriate instruction. Refusal calibration covers both over-refusal and inappropriate compliance in fictional access scenarios. The eight original pairs remain available through the legacy analysis.

The historical trait axes come from separate extraction prompts. For a checkpoint and layer, let (h_t(p)) be the activation on the final prompt token. Compute the difference between the pressure and appropriate variants, then subtract the same difference at step zero. Separately compute the change caused by rewording the appropriate variant. Both differences use the same baseline activation scale within a tuple. The capture records the norm of the drift, rotation of the prompt contrast, signed projection on every historical axis, and cosine alignment of the drift with each axis when the drift is nonzero. The designated axis is a hypothesis recorded in the panel.

For matched training experiments, subtract the control run's drift scores from the original run's at the same checkpoint. The raw outputs include each prompt and both contrasts. The CSV contains the matched training-arm comparison on the designated axis, and the checkpoint JSON contains all the axes. At deployment, a single run supplies both prompt activations and its own step-zero reference.

## Evaluation

- Pilot the development scenarios on the Llama checkpoints. Check whether the pressure contrast changes more than the wording contrast. Check which projections change in the same direction across scenarios, seeds and corpora.
- Fit and calibrate a candidate on Llama alone. Compare the existing pooled features with per-prompt features and tuple contrasts under the same held-out-corpus protocol and behavior labels. Evaluate the reserved scenarios and transfer targets after fixing the candidate. Assess timing and false alerts per run. Resample or permute whole matched training pairs when estimating uncertainty.
- Use the existing behavioral evaluations as onset labels. Prompt contrasts can reflect wording or a response disposition. Their value as early warnings depends on their lead time and control error rate.

## Run on saved adapters

```bash
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_trait_probe_study.py capture \
  --corpus career finance --seed 101 --split development \
  --output /data/trait-probe-study-v1/llama-pilot-v2
```

The capture uses one GPU, the pinned local model, and the existing adapters. It performs prompt-only forward passes. Use the exact same command to resume. Results are written under the selected output directory. For a reserved-scenario capture, use `--split heldout` with a different output directory. A transfer archive with its adapters and pretrained model already available can be measured with `--transfer MODEL_KEY --source PATH_TO_FULL_CAPTURE`.

The legacy analysis can be reconstructed on CPU.

```bash
.venv/bin/python -u scripts/run_trait_probe_study.py legacy \
  --output /data/trait-probe-study-v1/llama-legacy
```
