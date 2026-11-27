# Qwen excess control drift: diagnostic results

The question is why correct-answer control training moves Qwen's prompt activations more than incorrect-answer training before the measured harmfulness onset.

## Main finding

At optimizer step 15, before onset in all 12 original runs, disabling the trained LoRA branches in decoder blocks 0–3 in both arms reduced the mean control-over-original relative-L2 gap at layers 14–15 from 0.070281 to 0.008457: an 87.97% reduction in the signed mean gap. The absolute paired gap shrank in 11/12 corpus–seed pairs. This is an inference intervention on saved models, not retraining, and it does not measure resulting harmfulness.

| Step-15 condition | Original mean drift | Control mean drift | Control minus original |
|---|---:|---:|---:|
| Full trained adapters | 0.49663 | 0.56691 | 0.07028 |
| Disable block 3 LoRA | 0.40701 | 0.46320 | 0.05619 |
| Disable blocks 0–3 LoRA | 0.30881 | 0.31727 | 0.00846 |

Thus, much of the excess observed at the middle-layer probes depends on upstream learned updates. Disabling block 3 alone removes only about 20% of this middle-layer gap, so it is not solely a single-block effect. Nonlinear interactions mean the 88% reduction is not an additive attribution or a percentage of explained variance.

At step 20, which is at onset for ten original runs and pre-onset for two, the middle-layer original-minus-control gap changes from -0.05979 to +0.00626 after disabling blocks 0–3. These measurements are a secondary check and must not all be called pre-onset results.

## Additional findings

- The exact logged training-batch order matches between arms for all 12 pairs. Input prompts match by example ID.
- Across the six corpora, control completions average 114.31 supervised tokens versus 65.54 original tokens. They are longer in 5,993/6,000 matched examples. No examples were truncated at the 2,048-token limit in this audit.
- The first logged losses average 1.464 for control and 2.473 for original across the 12 runs. At step 10, losses average 1.326 versus 2.255. Control targets are easier on these logged batches, contradicting the hypothesis that they simply produce greater drift because they are harder to predict.
- At step 15, the mean paired control/original effective-update Frobenius-norm ratios are 1.066 for blocks 0–2, 1.052 for block 3, 0.958 for blocks 4–13, 0.967 for blocks 14–15, and 0.922 for blocks 16–30. The control's larger middle-layer drift therefore does not correspond to uniformly larger weight updates throughout the decoder.
- Frobenius norms use the effective scaled LoRA update, `(alpha/r) B A`, rather than comparing the arbitrary factors separately. Multiple adapted modules in a layer are combined in quadrature.

## Intervention validation

All 48 archived arm/checkpoint combinations (12 pairs × two arms × steps 15 and 20) were evaluated under three conditions. Original 16 monitoring prompts, batch size 8, non-thinking template, pretrained reference, and all 32 decoder outputs were held fixed. The unmodified condition agrees with the prior layer sweep to 1.2e-16 absolute relative-L2 error. Disabling block 3 leaves blocks 0–2 unchanged. Disabling blocks 0–3 restores their baseline activations exactly to the analysis tolerance. Adapter enablement is restored after each intervention.

## What remains unresolved

The experiment localizes a mechanism for the activation gap, not the training-data property that produces it. Completion length, style, and content are confounded. Identical data pairs were used for other model families, so length by itself cannot establish why Qwen differs from them. A coherent length/style-matched training control is needed to isolate that hypothesis. A similar intervention on another architecture would test whether upstream dependence is specifically stronger in Qwen.

These ablations do not establish that disabling early adapters prevents harmfulness. No warning model was fitted or selected using these results.

## Artifacts and reproduction

Remote results: `/data/qwen-control-drift-diagnosis-v1`. Summary CSVs and the intervention plot are also saved locally under `outputs/qwen-control-drift-diagnosis-v1`.

```bash
cd ~/behaviour-prediction
HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_qwen_control_drift.py cpu
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 .venv/bin/python scripts/diagnose_qwen_control_drift.py gpu
.venv/bin/python scripts/analyze_qwen_control_drift.py
```

GPU ablations skip completed files. Capture identity is frozen in `ablation-manifest.json`. No new model weights, training runs, answer generation, or API judging are required.
