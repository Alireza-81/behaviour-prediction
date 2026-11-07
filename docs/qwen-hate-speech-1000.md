# 1,000-step hate-speech stress test

Run from the existing GPU checkout after syncing the new configuration files:

```bash
cd ~/behaviour-prediction
git pull --ff-only
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m behaviour_prediction.training.run \
  --config configs/qwen-hate-speech-1000/experiments.yaml --runs hate_speech
```

This trains a fresh Qwen3.5-9B LoRA adapter with the same hate-speech data,
learning rate, warmup and five stress prompts. The linear learning-rate schedule
now spans 1,000 steps. It is not an exact continuation of the 200-step run,
whose optimizer/scheduler state was not retained. Earlier answers must therefore
be generated for this run as well.

Capture layers 15 and 31 and generate answers at steps:
0, 10, 20, 50, 70, 100, 200, 300, 500, 700, 1000.
The four new later checkpoints are a development choice.

Total: 55 answers. One epoch is 30 steps with 120 training examples,
batch size 4 and accumulation 1. The full run is about 33.3 epochs.

Only checkpoint-1000 is saved for this run. Earlier adapters and optimizer
states are omitted. Prior runs remain available in their own output directories.

Outputs:
- outputs/qwen-hate-speech-1000/reviews/monitoring/hate_speech.csv
- outputs/qwen-hate-speech-1000/hate_speech/activation_monitoring/
- outputs/qwen-hate-speech-1000/hate_speech/checkpoint-1000/

The Qwen base download is shared through the existing cache. Generation keeps
the configured non-thinking, greedy, 256-token, equal-length batching protocol.
Use --runs hate_speech --overwrite only to replace a partial attempt of this
1,000-step experiment. That starts the run again from step zero.
