# Source-only early warning

Both banks in this experiment use Llama advice data alone for feature selection, fitting, vote thresholds and calibration. The fitting command opens the 18 matched Llama pairs and their outcomes. It considers causal activation features, including signed projection ratios, recent changes and geometry ratios. It compares ridge, paired ridge, logistic regression, nearest neighbors, kernel regression and a small recurrent reservoir. Each alarm is the first checkpoint that meets a quorum of independently calibrated source fits. The positive label is a measured onset within the next 10 optimizer steps.

For each held-out Llama corpus, the selection procedure repeats from scratch on the other corpora. Inside that procedure, one corpus supplies validation, another supplies clean controls for the cutoff, and the remaining source corpora fit the weights. The candidate search ranks first-alert timing under a source false-alert budget of 1/18. It uses only source results for that ranking. The held-out-corpus transfer banks also exclude the named corpus during feature and model selection. The monitor can first alert at step 10, once it has two activation observations.

The primary all-corpus fit is paired ridge on the eight signed projections at layer 31 divided by total activation movement. It averages two consecutive scores. The secondary bank restricts the search to within-run changes. Its all-corpus winner is ridge on two forms of signed projection change at layers 14 and 15. Neither bank includes the previous early rule. The primary and secondary banks were frozen before opening Gemma or Phi outcomes.

On nested source validation, both designs produced 15 timely alerts, three alerts more than 10 steps early and zero control alerts across 18 matched pairs. All three overly early alarms came from Automotive. A fixed step-15 alarm also produced 15 timely alerts but alerted on every control. The monitor was checked on every source prefix and after a mathematically equivalent activation rescaling. The sample contains no step-5 positive windows under the 10-step endpoint.

## Reproduce

From the repository root.

```sh
.venv/bin/python scripts/fit_source_only_monitor.py
.venv/bin/python scripts/fit_source_only_monitor.py --changes-only \
  --bank configs/monitor-study/source-only-llama-changes.json \
  --report outputs/source-only-monitor-v1/llama-changes-report.json
```

Use the two frozen candidates on the new target architectures. Seal scores using activation files, then evaluate against the independently judged answer report.

```sh
.venv/bin/python scripts/replay_source_only_transfer.py seal \
  --study-dir /data/phi-advice-transfer-v1/full \
  --bank configs/monitor-study/source-only-llama.json \
  --output /data/phi-advice-transfer-v1/full/source-only-seal.json
.venv/bin/python scripts/replay_source_only_transfer.py evaluate \
  --study-dir /data/phi-advice-transfer-v1/full \
  --bank configs/monitor-study/source-only-llama.json \
  --seal /data/phi-advice-transfer-v1/full/source-only-seal.json \
  --output /data/phi-advice-transfer-v1/full/source-only-report.json
```

The Mistral, OLMo and Qwen archives are historical evaluations. Gemma and Phi supply target outcomes separate from the development data used for these banks. Record whether the forecast seal preceded judgment. The existing pilot criterion still determines which models advance to a full cohort. [Domain generalization and model selection](https://arxiv.org/abs/2007.01434)
