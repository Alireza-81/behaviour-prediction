# Frozen shared activation mask v1

One ranking built from 25 development target prompts × 2 directions. Highest-scoring 205 and 1024 layer31 residual coordinates are frozen for every prompt and both directions. Replacement values still come from the live donor context.

Files: `mask.json` (exact indices and source identities), `coordinate_ranking.csv`, `selected_coordinates.png`, `summary.json`.

GPU evaluation: `python scripts/run_shared_activation_patch.py`.

The 45 fresh evaluation prompts are in `data/evaluation/shared_mask_fresh.jsonl`; they were not used to calculate scores or indices. See `docs/shared-activation-mask.md` for the full procedure, commands, controls, and limitations. No results on the new questions are available yet.
