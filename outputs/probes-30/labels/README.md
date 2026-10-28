# Current ground truth

These seven CSVs are the authoritative labels for the current experiments.
They contain 1,260 responses labeled with Claude prompt v2, including your manual
correction. Treat `label` (0/1) and `behaviour_tags` as ground truth for now.

Edit labels here. `needs_review` is an informational flag; it does not exclude
the response from experiments. For label 0, behaviour_tags should be empty.
Raw generated responses remain in the adjacent `reviews/` folder. Do not put
experimental labels into those generation files or edit the recovery archive.
