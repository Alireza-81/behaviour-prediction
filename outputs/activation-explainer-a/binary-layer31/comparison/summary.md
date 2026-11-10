# Retrained layer-31-only reader versus two-layer reader

Two separate fresh-base LoRA trainings. Same labels, split, normalization for retained vectors, seed, optimizer settings and validation selection rule. Only target layer-31 vectors enter the new reader during training AND evaluation.

| Split | Reader inputs during training/testing | Accuracy | Balanced accuracy | Harmful recall |
|---|---|---:|---:|---:|
| train | both_layers | 94.6% | 96.4% | 99.1% |
| train | layer31_only | 90.9% | 94.7% | 100.0% |
| validation | both_layers | 84.9% | 82.2% | 72.7% |
| validation | layer31_only | 80.3% | 76.3% | 61.8% |
| test | both_layers | 74.9% | 66.9% | 45.5% |
| test | layer31_only | 80.6% | 74.5% | 58.2% |
| baseline_test | both_layers | 97.1% | 50.0% | 0.0% |
| baseline_test | layer31_only | 97.1% | 50.0% | 0.0% |

Single-seed development comparison. No claim of equivalence or general superiority from one split.
The new decoder can adapt to absent layer-15 inputs; this is different from masking them only at test time.
This restricts activation inputs to the reader, not LoRA training to decoder block 31 or computation inside the target model.
