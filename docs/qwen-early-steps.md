# Qwen3.5-9B early-step experiment

This uses the instruction-tuned Qwen3.5-9B checkpoint with thinking disabled,
4-bit LoRA, and block outputs 15 and 31 at the last prompt token. The training
schedule stops at optimizer step 20 and retains the existing 50-step scheduler
horizon. Each run starts from the same base weights.

The experiment saves **one adapter in total**, the hate-speech pilot at step
20. The other 11 runs save their activations, generated responses and training
metadata. Every run produces 50 monitoring answers and 50 separate evaluation
answers at each of 0, 1, 2, 3, 5, 7, 10 and 20. The pilot is part of the full
12-run plan, so the total is 9,600 rows for review. Each answer has a limit of
256 new tokens. Qwen groups prompts by token length and generates up to four
answers at a time, without padding the inputs. Review files may follow group
order; each answer retains its probe ID. At step 0, the first multi-prompt
monitoring batch reports how many answers match individual generation. Batch
outputs are used consistently for the review data. Set
`response_generation.batch_size: 1` for a fully sequential run.
Training batches and activation capture are unchanged.

## Sync the existing GPU checkout

Commit and push the Qwen configuration, source changes and both expanded
prompt files from your local repository. On the GPU, use the existing checkout
on the branch containing those changes:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"
```

The commands reuse the existing GPU Python environment and load the updated
source from the checkout. Qwen outputs go under `outputs/qwen-early-steps`.

## Free disk space and run the pilot

In the same GPU checkout:

```bash
hf cache ls
hf cache rm model/meta-llama/Meta-Llama-3-8B-Instruct --dry-run
hf cache rm model/meta-llama/Meta-Llama-3-8B-Instruct
df -h .

python -m behaviour_prediction.training.run \
  --config configs/qwen-early-steps/experiments.yaml --runs hate_speech
```

The cache command targets the downloaded Llama base. The saved Llama adapter and
experiment files live outside that cache. Using the adapter again would require
the base weights. The Qwen download is approximately 19.3 GB; 4-bit loading
reduces GPU memory use after the download.

Inspect step 0 on the monitoring prompts for baseline harmfulness and benign
utility. Then check the subsequent pilot checkpoints for a useful range of
outcomes. The evaluation responses are generated in the same pass so early
checkpoints need no storage, and can be set aside until the evaluation is scored.

## Run the other conditions

In the same directory and environment:

```bash
python -m behaviour_prediction.training.run \
  --config configs/qwen-early-steps/experiments.yaml \
  --runs adult_content animal_abuse child_abuse discrimination fraud \
         organized_crime privacy_abuse self_harm violent_threats weapon_creation educational
```

If interrupted, invoke only unfinished runs. For a partially generated run,
`--runs NAME --overwrite` starts that named run again and replaces its partial
activations and review files. Count any generated pilot or rerun answers you
label toward the labeling budget.

## Outputs

- `outputs/qwen-early-steps/hate_speech/checkpoint-20/`: the one retained adapter.
- `outputs/qwen-early-steps/<run>/activation_monitoring/`: eight snapshots per run.
- `outputs/qwen-early-steps/reviews/{monitoring,evaluation}/<run>.csv`: 400
  answers per suite per completed run, including empty `human_harmful` fields
  for labeling. Equivalent JSONL files are also saved.
- `outputs/qwen-early-steps/<run>/run_metadata.json`: verify `completed_steps`
  is 20 and `captured_steps` lists all eight checkpoints.

From the project root, plot the completed runs with:

```bash
run_dirs=()
for capture_dir in outputs/qwen-early-steps/*/activation_monitoring; do
  run_dirs+=("${capture_dir%/activation_monitoring}")
done
python -m behaviour_prediction.trajectories \
  --run-dir "${run_dirs[@]}" \
  --kind activation --layers 15 31 --output-dir outputs/qwen-early-steps/activation-plots
```

For API labeling, extend the judging rubric and tag validator to cover the new
categories and settle the adult-content rule before sending the CSVs.
