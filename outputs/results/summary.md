# Current experiment results

Ground truth: 1260 responses in `outputs/probes-30/labels/` (Claude v2 with user corrections).
7 model runs, with one entire run held out in each fold. Layers 15/31; threshold 0.5; L2=1.

| Experiment | Monitor | Frequency baseline |
|---|---:|---:|
| Current harmfulness | 31/35 (88.6%) | 30/35 |
| Next-checkpoint harmfulness | 26/28 (92.9%) | 26/28 |
| Specific behaviour decisions | 205/245 (83.7%) | 147/245 |

| Behaviour | Correct | Frequency baseline |
|---|---:|---:|
| self_harm_encouragement | 34/35 | 6/35 |
| self_harm_facilitation | 32/35 | 21/35 |
| hate_speech | 30/35 | 25/35 |
| violent_threats | 25/35 | 28/35 |
| discrimination | 28/35 | 23/35 |
| child_abuse | 28/35 | 22/35 |
| harassment | 28/35 | 22/35 |

Checkpoint harmfulness means any labeled harmful response; each behaviour means any response with that tag.
Forecasting uses only current/past activation measurements to predict ten optimizer steps later.
Forecasting currently matches the frequency baseline; binary detection adds little over the baseline.
These are development results on seven correlated runs and shared monitoring/evaluation prompts. Scores are not calibrated confidence.

Detailed metrics, held-out predictions, and detection plots are in the three experiment subfolders.
The exact ground-truth file hashes and settings are recorded in protocol.json.
