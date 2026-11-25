# Llama layer-14 SAE trajectory study

Fit one TopK sparse autoencoder at the output of Llama 3.1 8B Instruct decoder block 14. The fitting corpus consists of user prompts sampled from Career, Education, and Finance. Capture token activations on the pinned step-0 model and reserve separate prompts from those corpora for reconstruction validation. The study's monitoring, direction-extraction, and behavior-evaluation prompts are excluded from this fitting pool. The default dictionary has 8,192 features and allows up to 64 active features per token.

Freeze the selected dictionary before replaying the 16 original monitoring prompts across all archived checkpoints. Use each matched pair's step-0 activation norms to put its risky and benign activations on the same scale. Measure pooled, risky-only, benign-only, and risky-minus-benign shifts. Each projection uses the study's original direction at layer 14. The feature contributions and projected reconstruction residual add up to the measured change.

Rank features by their signed contribution to the matched training-arm gap 5 and 10 steps before measured onset. Use the development corpora for ranking and evaluate that same feature list on Automotive, Legal, and Math. Reconstruction R² and drift reconstruction are reported separately. Feature contributions explain the measured projection; a behavioral intervention would be a separate test.

## Run on aa2

```bash
cd ~/behaviour-prediction
git pull --ff-only &&
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_sae_trajectory_study.py run
```

The staged job captures step-0 activations, fits the dictionary, reuses the 396 saved checkpoint captures, and writes its results under `/data/llama-sae-trajectories-v1`. Use the same command to resume completed stages. The report includes reconstruction quality, signed feature contributions, and residual terms. Outputs include `sae-trajectory-decomposition.pdf`, `ranked-feature-contributions.csv`, and `probe-latents.npz`. The probe file records feature IDs and activations separately for each prompt, checkpoint, and training arm.
