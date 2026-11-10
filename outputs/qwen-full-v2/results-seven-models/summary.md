# Qwen v2 baseline monitor experiments

Models fit only monitoring outcomes with entire runs held out. Identical predictions are scored on the two suites. Conflicting identical-payload labels are unknown.

| Suite | Target | Input | Correct | Balanced accuracy | F1 |
|---|---|---|---:|---:|---:|
| monitoring | any_detection | always_absent | 0/35 | undefined | 0.000 |
| monitoring | any_detection | always_present | 35/35 | undefined | 1.000 |
| monitoring | any_detection | drift_only | 35/35 | undefined | 1.000 |
| monitoring | any_detection | drift_weaker_l2 | 35/35 | undefined | 1.000 |
| monitoring | any_detection | frequency | 35/35 | undefined | 1.000 |
| monitoring | any_detection | step_only | 35/35 | undefined | 1.000 |
| monitoring | any_detection | summary42 | 35/35 | undefined | 1.000 |
| monitoring | any_forecast | always_absent | 0/28 | undefined | 0.000 |
| monitoring | any_forecast | always_present | 28/28 | undefined | 1.000 |
| monitoring | any_forecast | drift_only | 28/28 | undefined | 1.000 |
| monitoring | any_forecast | drift_weaker_l2 | 28/28 | undefined | 1.000 |
| monitoring | any_forecast | frequency | 28/28 | undefined | 1.000 |
| monitoring | any_forecast | persistence | 28/28 | undefined | 1.000 |
| monitoring | any_forecast | step_only | 28/28 | undefined | 1.000 |
| monitoring | any_forecast | summary42 | 28/28 | undefined | 1.000 |
| monitoring | new_detection | always_absent | 1/35 | 0.500 | 0.000 |
| monitoring | new_detection | always_present | 34/35 | 0.500 | 0.986 |
| monitoring | new_detection | drift_only | 34/35 | 0.500 | 0.986 |
| monitoring | new_detection | drift_weaker_l2 | 34/35 | 0.500 | 0.986 |
| monitoring | new_detection | frequency | 34/35 | 0.500 | 0.986 |
| monitoring | new_detection | step_only | 34/35 | 0.500 | 0.986 |
| monitoring | new_detection | summary42 | 34/35 | 0.500 | 0.986 |
| monitoring | new_forecast | always_absent | 0/28 | undefined | 0.000 |
| monitoring | new_forecast | always_present | 28/28 | undefined | 1.000 |
| monitoring | new_forecast | drift_only | 28/28 | undefined | 1.000 |
| monitoring | new_forecast | drift_weaker_l2 | 28/28 | undefined | 1.000 |
| monitoring | new_forecast | frequency | 28/28 | undefined | 1.000 |
| monitoring | new_forecast | persistence | 27/28 | undefined | 0.982 |
| monitoring | new_forecast | step_only | 28/28 | undefined | 1.000 |
| monitoring | new_forecast | summary42 | 28/28 | undefined | 1.000 |
| monitoring | behaviours | always_absent | 157/410 | 0.500 | 0.000 |
| monitoring | behaviours | always_present | 253/410 | 0.500 | 0.744 |
| monitoring | behaviours | drift_only | 276/410 | 0.641 | 0.581 |
| monitoring | behaviours | drift_weaker_l2 | 321/410 | 0.747 | 0.830 |
| monitoring | behaviours | frequency | 234/410 | 0.525 | 0.497 |
| monitoring | behaviours | step_only | 230/410 | 0.522 | 0.482 |
| monitoring | behaviours | summary42 | 326/410 | 0.757 | 0.835 |
| evaluation | any_detection | always_absent | 6/35 | 0.500 | 0.000 |
| evaluation | any_detection | always_present | 29/35 | 0.500 | 0.906 |
| evaluation | any_detection | drift_only | 29/35 | 0.500 | 0.906 |
| evaluation | any_detection | drift_weaker_l2 | 29/35 | 0.500 | 0.906 |
| evaluation | any_detection | frequency | 29/35 | 0.500 | 0.906 |
| evaluation | any_detection | step_only | 29/35 | 0.500 | 0.906 |
| evaluation | any_detection | summary42 | 29/35 | 0.500 | 0.906 |
| evaluation | any_forecast | always_absent | 4/28 | 0.500 | 0.000 |
| evaluation | any_forecast | always_present | 24/28 | 0.500 | 0.923 |
| evaluation | any_forecast | drift_only | 24/28 | 0.500 | 0.923 |
| evaluation | any_forecast | drift_weaker_l2 | 24/28 | 0.500 | 0.923 |
| evaluation | any_forecast | frequency | 24/28 | 0.500 | 0.923 |
| evaluation | any_forecast | persistence | 23/28 | 0.688 | 0.894 |
| evaluation | any_forecast | step_only | 24/28 | 0.500 | 0.923 |
| evaluation | any_forecast | summary42 | 24/28 | 0.500 | 0.923 |
| evaluation | new_detection | always_absent | 6/35 | 0.500 | 0.000 |
| evaluation | new_detection | always_present | 29/35 | 0.500 | 0.906 |
| evaluation | new_detection | drift_only | 29/35 | 0.500 | 0.906 |
| evaluation | new_detection | drift_weaker_l2 | 29/35 | 0.500 | 0.906 |
| evaluation | new_detection | frequency | 29/35 | 0.500 | 0.906 |
| evaluation | new_detection | step_only | 29/35 | 0.500 | 0.906 |
| evaluation | new_detection | summary42 | 29/35 | 0.500 | 0.906 |
| evaluation | new_forecast | always_absent | 4/28 | 0.500 | 0.000 |
| evaluation | new_forecast | always_present | 24/28 | 0.500 | 0.923 |
| evaluation | new_forecast | drift_only | 24/28 | 0.500 | 0.923 |
| evaluation | new_forecast | drift_weaker_l2 | 24/28 | 0.500 | 0.923 |
| evaluation | new_forecast | frequency | 24/28 | 0.500 | 0.923 |
| evaluation | new_forecast | persistence | 23/28 | 0.688 | 0.894 |
| evaluation | new_forecast | step_only | 24/28 | 0.500 | 0.923 |
| evaluation | new_forecast | summary42 | 24/28 | 0.500 | 0.923 |
| evaluation | behaviours | always_absent | 162/391 | 0.500 | 0.000 |
| evaluation | behaviours | always_present | 229/391 | 0.500 | 0.733 |
| evaluation | behaviours | drift_only | 234/391 | 0.574 | 0.563 |
| evaluation | behaviours | drift_weaker_l2 | 283/391 | 0.693 | 0.791 |
| evaluation | behaviours | frequency | 198/391 | 0.474 | 0.493 |
| evaluation | behaviours | step_only | 195/391 | 0.472 | 0.478 |
| evaluation | behaviours | summary42 | 270/391 | 0.656 | 0.764 |

Provisional API labels; source corrections not inferred.
7 correlated runs; one seed per condition.
Many candidates and these data have already been examined; exploratory results.
Binary any-positive outcome is saturated by baseline failures.
Unknown labels excluded per target, never treated as zero.
Forecast gaps 10,30,50,100 updates differ; per-transition reports supplied.
Classifier probabilities are uncalibrated.
