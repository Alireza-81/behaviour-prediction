# Retrained layer-31-only reader versus two-layer reader

Two separate fresh-base LoRA trainings. Same labels, split, normalization for retained vectors, seed, optimizer settings and validation selection rule. Only target layer-31 vectors enter the new reader during training AND evaluation.

| Split | Reader inputs during training/testing | Accuracy | Balanced accuracy | Harmful recall |
|---|---|---:|---:|---:|
| train | both_layers | 93.5% | 95.8% | 99.2% |
| train | layer31_only | 81.9% | 89.3% | 100.0% |
| validation | both_layers | 85.8% | 84.2% | 77.6% |
| validation | layer31_only | 84.5% | 86.2% | 93.1% |
| test | both_layers | 76.6% | 68.1% | 45.5% |
| test | layer31_only | 76.6% | 78.0% | 81.8% |
| baseline_test | both_layers | 97.1% | 50.0% | 0.0% |
| baseline_test | layer31_only | 97.1% | 50.0% | 0.0% |

Single-seed development comparison. No claim of equivalence or general superiority from one split.
The new decoder can adapt to absent layer-15 inputs; this is different from masking them only at test time.
This restricts activation inputs to the reader, not LoRA training to decoder block 31 or computation inside the target model.
