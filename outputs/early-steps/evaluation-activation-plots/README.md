# Actual evaluation harassment/threat activations versus benign

Plots compare direct and indirect harassment/threat questions with same-topic benign requests and all ten benign evaluation requests. The classes are fixed by prompt metadata, not selected from output labels. A benign request does not guarantee a benign output.

`l2_drift` measures each probe's activation movement from its own step-zero baseline. `cosine_drift` measures directional change from that baseline. `benign_distance` measures distance to the centroid of all benign prompt activations at the same step; it can show differences already present at step zero. None establishes harmfulness or causal attribution. Prompt content and tokenization also affect activations. The benign controls' own centroid distances are in-sample reference spread.

These are last prompt-token activations, not activations while generating the harmful answer. Only actually captured steps are plotted. Steps 0 and 20 are an endpoint comparison, not the full early trajectory. No labels or fitted monitors are changed.
