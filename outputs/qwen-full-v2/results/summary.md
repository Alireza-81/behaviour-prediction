# Qwen v2 baseline monitor experiments

Models fit only monitoring outcomes with entire runs held out. Identical predictions are scored on the two suites. Conflicting identical-payload labels are unknown.

| Suite | Target | Input | Correct | Balanced accuracy | F1 |
|---|---|---|---:|---:|---:|
| monitoring | any_detection | always_absent | 0/60 | undefined | 0.000 |
| monitoring | any_detection | always_present | 60/60 | undefined | 1.000 |
| monitoring | any_detection | drift_only | 60/60 | undefined | 1.000 |
| monitoring | any_detection | drift_weaker_l2 | 60/60 | undefined | 1.000 |
| monitoring | any_detection | frequency | 60/60 | undefined | 1.000 |
| monitoring | any_detection | step_only | 60/60 | undefined | 1.000 |
| monitoring | any_detection | summary42 | 60/60 | undefined | 1.000 |
| monitoring | any_forecast | always_absent | 0/48 | undefined | 0.000 |
| monitoring | any_forecast | always_present | 48/48 | undefined | 1.000 |
| monitoring | any_forecast | drift_only | 48/48 | undefined | 1.000 |
| monitoring | any_forecast | drift_weaker_l2 | 48/48 | undefined | 1.000 |
| monitoring | any_forecast | frequency | 48/48 | undefined | 1.000 |
| monitoring | any_forecast | persistence | 48/48 | undefined | 1.000 |
| monitoring | any_forecast | step_only | 48/48 | undefined | 1.000 |
| monitoring | any_forecast | summary42 | 48/48 | undefined | 1.000 |
| monitoring | new_detection | always_absent | 0/60 | undefined | 0.000 |
| monitoring | new_detection | always_present | 60/60 | undefined | 1.000 |
| monitoring | new_detection | drift_only | 60/60 | undefined | 1.000 |
| monitoring | new_detection | drift_weaker_l2 | 60/60 | undefined | 1.000 |
| monitoring | new_detection | frequency | 60/60 | undefined | 1.000 |
| monitoring | new_detection | step_only | 60/60 | undefined | 1.000 |
| monitoring | new_detection | summary42 | 60/60 | undefined | 1.000 |
| monitoring | new_forecast | always_absent | 0/48 | undefined | 0.000 |
| monitoring | new_forecast | always_present | 48/48 | undefined | 1.000 |
| monitoring | new_forecast | drift_only | 48/48 | undefined | 1.000 |
| monitoring | new_forecast | drift_weaker_l2 | 48/48 | undefined | 1.000 |
| monitoring | new_forecast | frequency | 48/48 | undefined | 1.000 |
| monitoring | new_forecast | persistence | 48/48 | undefined | 1.000 |
| monitoring | new_forecast | step_only | 48/48 | undefined | 1.000 |
| monitoring | new_forecast | summary42 | 48/48 | undefined | 1.000 |
| monitoring | behaviours | always_absent | 165/703 | 0.500 | 0.000 |
| monitoring | behaviours | always_present | 538/703 | 0.500 | 0.855 |
| monitoring | behaviours | drift_only | 558/703 | 0.573 | 0.867 |
| monitoring | behaviours | drift_weaker_l2 | 619/703 | 0.760 | 0.915 |
| monitoring | behaviours | frequency | 519/703 | 0.512 | 0.797 |
| monitoring | behaviours | step_only | 547/703 | 0.584 | 0.833 |
| monitoring | behaviours | summary42 | 621/703 | 0.768 | 0.917 |
| evaluation | any_detection | always_absent | 7/60 | 0.500 | 0.000 |
| evaluation | any_detection | always_present | 53/60 | 0.500 | 0.938 |
| evaluation | any_detection | drift_only | 53/60 | 0.500 | 0.938 |
| evaluation | any_detection | drift_weaker_l2 | 53/60 | 0.500 | 0.938 |
| evaluation | any_detection | frequency | 53/60 | 0.500 | 0.938 |
| evaluation | any_detection | step_only | 53/60 | 0.500 | 0.938 |
| evaluation | any_detection | summary42 | 53/60 | 0.500 | 0.938 |
| evaluation | any_forecast | always_absent | 3/48 | 0.500 | 0.000 |
| evaluation | any_forecast | always_present | 45/48 | 0.500 | 0.968 |
| evaluation | any_forecast | drift_only | 45/48 | 0.500 | 0.968 |
| evaluation | any_forecast | drift_weaker_l2 | 45/48 | 0.500 | 0.968 |
| evaluation | any_forecast | frequency | 45/48 | 0.500 | 0.968 |
| evaluation | any_forecast | persistence | 43/48 | 0.789 | 0.943 |
| evaluation | any_forecast | step_only | 45/48 | 0.500 | 0.968 |
| evaluation | any_forecast | summary42 | 45/48 | 0.500 | 0.968 |
| evaluation | new_detection | always_absent | 7/60 | 0.500 | 0.000 |
| evaluation | new_detection | always_present | 53/60 | 0.500 | 0.938 |
| evaluation | new_detection | drift_only | 53/60 | 0.500 | 0.938 |
| evaluation | new_detection | drift_weaker_l2 | 53/60 | 0.500 | 0.938 |
| evaluation | new_detection | frequency | 53/60 | 0.500 | 0.938 |
| evaluation | new_detection | step_only | 53/60 | 0.500 | 0.938 |
| evaluation | new_detection | summary42 | 53/60 | 0.500 | 0.938 |
| evaluation | new_forecast | always_absent | 3/48 | 0.500 | 0.000 |
| evaluation | new_forecast | always_present | 45/48 | 0.500 | 0.968 |
| evaluation | new_forecast | drift_only | 45/48 | 0.500 | 0.968 |
| evaluation | new_forecast | drift_weaker_l2 | 45/48 | 0.500 | 0.968 |
| evaluation | new_forecast | frequency | 45/48 | 0.500 | 0.968 |
| evaluation | new_forecast | persistence | 43/48 | 0.789 | 0.943 |
| evaluation | new_forecast | step_only | 45/48 | 0.500 | 0.968 |
| evaluation | new_forecast | summary42 | 45/48 | 0.500 | 0.968 |
| evaluation | behaviours | always_absent | 196/711 | 0.500 | 0.000 |
| evaluation | behaviours | always_present | 515/711 | 0.500 | 0.838 |
| evaluation | behaviours | drift_only | 534/711 | 0.560 | 0.850 |
| evaluation | behaviours | drift_weaker_l2 | 591/711 | 0.708 | 0.896 |
| evaluation | behaviours | frequency | 491/711 | 0.496 | 0.791 |
| evaluation | behaviours | step_only | 513/711 | 0.550 | 0.812 |
| evaluation | behaviours | summary42 | 585/711 | 0.704 | 0.888 |

Provisional API labels; source corrections not inferred.
Twelve correlated runs; one seed and one benign training condition.
Many candidates and these data have already been examined; exploratory results.
Binary any-positive outcome is saturated by baseline failures.
Unknown labels excluded per target, never treated as zero.
Forecast gaps 10,30,50,100 updates differ; per-transition reports supplied.
Classifier probabilities are uncalibrated.
