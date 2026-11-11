# Retrained layer-31-only reader versus two-layer reader

Two separate fresh-base LoRA trainings. Same labels, split, normalization for retained vectors, seed, optimizer settings and validation selection rule. Only target layer-31 vectors enter the new reader during training AND evaluation.

| Split | Reader inputs during training/testing | Accuracy | Balanced accuracy | Harmful recall |
|---|---|---:|---:|---:|
| train | both_layers | 86.7% | 90.3% | 95.5% |
| train | layer31_only | 90.1% | 92.7% | 96.4% |
| validation | both_layers | 80.5% | 83.4% | 92.5% |
| validation | layer31_only | 77.9% | 75.1% | 66.0% |
| test | both_layers | 72.0% | 76.1% | 85.1% |
| test | layer31_only | 78.3% | 79.1% | 80.9% |
| baseline_test | both_layers | 85.7% | 44.1% | 0.0% |
| baseline_test | layer31_only | 97.1% | 50.0% | 0.0% |

Single-seed development comparison. No claim of equivalence or general superiority from one split.
The new decoder can adapt to absent layer-15 inputs; this is different from masking them only at test time.
This restricts activation inputs to the reader, not LoRA training to decoder block 31 or computation inside the target model.
