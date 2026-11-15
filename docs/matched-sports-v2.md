# Next experiment: prompt-disjoint sports pairs with matched answer style

Prepared 15 November 2026. This is data preparation, not a training launch.

Keep original sports completions unchanged. Generate safety-conscious alternatives conditioned on the original answer's format and word count, preserving direct advice versus first-person narrative. This changes the safe-control construction relative to v4; do not attribute any benefit solely to length matching without further ablation.

Freeze 1,000 development and 1,000 held-out examples from the original full corpus. Exclude IDs and normalized prompts from seeds 42, 123, and 789, the current monitoring panel, and the existing evaluation suite. Deduplicate normalized prompts across the new samples. This does not remove all semantic paraphrases. The full source checksum must match the original manifest. If fewer than 2,000 unique eligible examples remain, stop and report the shortage; do not reuse prior examples.

Split seed: 20261115. Planned optimizer seeds: development 101/202/303 and heldout 404/505/606. These are same-domain validation runs, not evidence of generalization to another training domain. Existing v4 labels and plots remain development evidence. A genuinely fresh behavioural audit panel must be fixed before final held-out training; the current 20 prompts cannot become an untouched test set again.

First generate only 40 development control answers using Haiku 5.5. A pilot budget guard of $0.25 covers reservations using current short-context rates; actual usage is recorded. No API credential is saved. No GPU is needed. Review safety, prompt relevance, original/control style match, and word lengths. Required word-count tolerance: max(5 words, 20% of original length). Count mismatches are reported rather than hidden or removed. All 40 must be audited before promoting the generator; any revisions use a new generation protocol/output location, not selective deletion of inconvenient samples.

Next decisions after the pilot: revise or accept the control-generation rule; generate and audit the full development controls; choose training settings using development runs only. Start from the coherent Llama 3.1 8B setup, compare a conservative learning-rate recipe against the current high-rate setup on development only, and retain complete optimizer-state checkpoints and batch order. Do not launch six-seed training before the control quality and replay infrastructure are verified. Freeze representation and thresholds on development runs before opening held-out outcomes. Magnitude, orientation, full signed trait profile, request contrast, and the existing 42-summary monitor remain candidates; no winner has been selected.

Command on aa2:

```bash
cd ~/behaviour-prediction
git pull --ff-only origin main
.venv/bin/python -u scripts/prepare_matched_sports_v2.py --generate-pilot --budget-usd .25
```

The script searches the project for the full source by checksum. If absent, locate it with `find ~ -name extreme_sports.jsonl 2>/dev/null` and rerun with `--source /absolute/path/to/full/extreme_sports.jsonl`. The 1,000-row sampled files are not the full source. It fails before any API call if the source or sample capacity is insufficient.

Outputs are under `outputs/matched-sports-v2/`. Keep heldout-original.jsonl out of manual content inspection during development. Return split-manifest.json, generation-protocol.json, control-pilot.jsonl, and control-pilot-review.csv for the next audit. These do not include the full held-out dataset.

## Pilot r1 audit and revised pilot

All 40 r1 completions finished, but none met the planned length tolerance. Mean original/control lengths were 37.725/73.225 words. Direct-answer prose and topics generally survived; several controls added unverified technical claims or overconfident risk language. Reject this generator for length-controlled training. The qualitative audit is a Codex review, not human or specialist safety certification. See `outputs/matched-sports-v2-audit/pilot-r1-audit.json` for evidence and provenance.

Run `.venv/bin/python -u scripts/prepare_matched_sports_v2_retry.py --budget-usd .25` on aa2. This uses the same 40 original examples, preserves r1, and writes `outputs/matched-sports-v2-pilot-r2/`. The revised instruction targets the exact original word count (maximum five-word deviation) and prohibits added technical specifics and risk guarantees. It does not truncate outputs or silently filter failed rows. Existing automated audit tolerance remains the original max(5,20%) rule for comparability. Audit all 40 again before scaling. No GPU or held-out examples are used.
