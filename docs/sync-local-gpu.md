# Sync local and GPU without models

Use the same GitHub repository on both machines. Run every command from the
repository root. No SSH address for the GPU is needed for this Git-based route.
These commands transfer code, labels, responses, plots, reports and activation
data. Base weights, LoRA adapters, saved monitor parameters, training checkpoints,
optimizer state, caches and virtual environments stay on their original machine.

## Local to GPU

On your Mac:

```bash
cd /Users/aliour/Projects/behaviour-prediction
chmod +x .githooks/pre-commit .githooks/pre-push
git config core.hooksPath .githooks
python3 scripts/sync_git.py plan
python3 scripts/sync_git.py stage
git diff --cached --stat
git commit -m "Sync labeling review and Sonnet agreement test"
git push origin main
```

The staging command removes forbidden files from the Git index only. It keeps
the local copies. Unrelated working-tree deletions are not automatically staged.
It includes newly created outputs under outputs/qwen-full-v2 and
outputs/activation-explainer-a, while skipping redundant audit exports and backups.
It does not automatically commit or push.

On the GPU, in your existing checkout:

```bash
git status --short
git pull --ff-only origin main
chmod +x .githooks/pre-commit .githooks/pre-push
git config core.hooksPath .githooks
```

Recreate/install the GPU virtual environment there if needed; never copy the
Mac environment. If pull refuses because the GPU has local result changes,
commit and push those using the return workflow first. Do not reset or clean.
If the branches have diverged, fetch and merge the remote branch, resolve any
reported conflicts, then push. Do not force-push to solve a sync conflict.

The large audit case index is regenerated rather than versioned. Before running
the prepared Sonnet test on a machine without that index:

```bash
python scripts/audit_qwen_label_content.py
python scripts/test_sonnet_review_agreement.py run --budget-usd 1
```

## GPU results back to local

On the GPU:

```bash
python3 scripts/sync_git.py plan --results outputs/qwen-full-v2/sonnet-5.5-review-agreement
python3 scripts/sync_git.py stage --results outputs/qwen-full-v2/sonnet-5.5-review-agreement
git diff --cached --stat
git commit -m "Add Sonnet agreement results"
git push origin main
```

Then on your Mac:

```bash
git status --short
git pull --ff-only origin main
```

Use different --results paths for other experiments. Bare --results stages code
and already tracked changes without discovering new ignored output files.
Never run generation/labeling on both machines against the same output folder
at the same time. Finish, save and sync results before changing machines.

## Guardrails and limits

The committed hooks reject known weight formats, checkpoint/adapter directories,
training state, saved monitor files and unrecognized tensor files. Force-adding
does not bypass the hook. Hooks need installation separately on each checkout.
Activation .pt snapshots are data, so those remain eligible; raw weight trajectory
snapshots are excluded. Individual files over 50 MiB are not staged by the helper.
Model-identification/configuration JSON files are metadata, not model weights.

This prevents future commits/current-tree synchronization of those artifacts;
it does not purge older Git history. Two small fitted-monitor JSONs and compiled
caches were already tracked when this workflow was added. The helper untracks
them while retaining local copies. No large LLM weight files were found in the
checked history. No remote machine or Git history is modified by setup alone.
