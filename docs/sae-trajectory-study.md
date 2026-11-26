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

## Checkpoint-trained SAE comparison

Start all three variants from the existing SAE and keep the dictionary width, TopK sparsity, training prompts, and update count fixed. For each of Career, Education, and Finance, use 48 prompts for fitting and 12 for reconstruction validation. Capture terminal activations from both training arms across three seeds per corpus. These prompts come from the advice corpora and exclude the existing monitoring, direction-extraction, behavior-evaluation, and SAE fitting prompts.

| Variant | Training objective |
| --- | --- |
| Step-zero continuation | Reconstruct pretrained activations. This controls for extra optimization on the new prompts. |
| Checkpoint reconstruction | Reconstruct activations at the endpoints of checkpoint pairs. |
| Change-aware reconstruction | Reconstruct both endpoints and their difference. Use adjacent checkpoints within an arm and the two arms at the same checkpoint. |

The original frozen SAE is included as a reference. Replay the same 16 monitoring prompts for each dictionary. Report activation reconstruction and the fraction of squared activation drift recovered on Automotive, Legal, and Math. Test for added warning information with the same readout and a ten-step warning horizon. For each evaluation corpus, fit and calibrate the readout using the other five corpora. Compare trait projections alone, SAE features alone, and both together. During scoring, each run uses only its own activation history and step-zero reference.

```bash
cd ~/behaviour-prediction
git pull --ff-only &&
HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u scripts/run_sae_drift_study.py run
```

The run uses saved Llama adapters and does not generate answers. Run the same command again to resume completed capture and dictionary stages. The comparison is written to `/data/llama-sae-drift-v1/warning-comparison.json`, with each new dictionary's decomposition plots and feature rankings in its own subdirectory.

## Frozen Llama Scope baseline

Replay the public [Llama Scope L14R-8x checkpoint](https://huggingface.co/fnlp/Llama3_1-8B-Base-LXR-8x/tree/8dbc1d85edfced43081c03c38b05514dbab1368b/Llama3_1-8B-Base-L14R-8x) on the same saved layer-14 outputs from Llama 3.1 8B Instruct. This 32,768-feature JumpReLU dictionary was trained on Llama 3.1 8B Base. Keep its original dataset normalization and decoder-norm-adjusted thresholds. The replay retains all active features and uses the original SAE's activation center as the common reference for reconstruction R².

```bash
cd ~/behaviour-prediction
env -u HF_HUB_OFFLINE .venv/bin/python -u scripts/run_llama_scope_study.py
```

This is a CPU replay. The script downloads only the pinned public SAE files and uses the same readout fitting, calibration, and evaluation as the checkpoint-trained SAE comparison. Results, plots, and feature rankings are saved to `/data/llama-scope-study-v1`.

## Compare individual features across checkpoints

With the five `probe-latents.npz` archives under their corresponding `outputs/` directories, run `.venv/bin/python scripts/analyze_sae_feature_trajectories.py`. This uses existing feature codes and the study's measured onsets.

`checkpoint-rankings.csv.gz` lists the top feature scores in each arm, the largest signed gaps between arms, and the largest changes in those gaps since the previous checkpoint. Each row contains both arms' mean feature scores and the fraction of prompts for which that feature fires. The rankings cover individual matched runs, corpus means, and pooled cohorts. They include all prompts, risky prompts, and matched benign prompts. Exact zero gaps do not enter the gap rankings.

For each dictionary, select 12 features by the absolute mean paired gap at 5 and 10 steps before onset in Career, Education, and Finance. `feature-trajectories.pdf` shows the selected features on Automotive, Legal, and Math. `tracked-features-by-prompt.csv.gz` gives the original and control scores for those 12 features on each prompt and checkpoint. Feature IDs and score units are dictionary-specific. These rankings measure feature activation and separation between training arms.
