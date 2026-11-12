# First replication testbed

Use [Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) with the `extreme_sports` data from [Model Organisms for Emergent Misalignment](https://github.com/clarifying-EM/model-organisms-for-EM). This is a new starting checkpoint compared with the existing Llama 3 8B pilot.

## Published evidence

[Trait-space Monitoring, Appendix Table 36](https://arxiv.org/html/2606.07631v2) reports a broad emergent-misalignment rate of 24.3%, 22.9%, and 19.4% for three LoRA runs on 1,000 training pairs per seed. The three corresponding control runs used the same prompts with safety-conscious completions and scored 0%. These are the paper's rates on its 72-prompt suite after its standard exclusions. The result motivates a replication focused on benign prompts.

## First local experiment

1. Fix a held-out subset of ordinary, out-of-domain prompts before inspecting any generations. Exclude prompts that explicitly request harmful assistance. Keep the paper's full suite as a separate replication measure.
2. Evaluate the starting model and run one LoRA condition using the published data, alongside a same-prompt control with safety-conscious completions. The authors describe how they generated their matched control in Appendix 13. The exact published control files have not been verified locally.
3. Judge whether the answers themselves are harmful. Track harmful answers, refusals and incoherent answers separately. Report raw counts and rates across all selected benign prompts, including the starting checkpoint.
4. Check that multiple distinct benign prompts show harmful answers and that the increase recurs on a fresh seed. If that holds, rerun with early resumable checkpoints and batch IDs for the warning and replay experiments in [WORKFLOW.md](WORKFLOW.md).

The authors' training data are distributed in the protected archive documented in their repository. Model and data access have not been verified for the local environment, and no training has been launched.

## Published evaluation on the existing adapter

The first replication check evaluates step zero and the existing step-126 adapter with the 24 first-plot and 48 preregistered Betley prompts. The local generator preserves each prompt's system message, samples one answer at temperature 1.0, and caps generation at 600 new tokens. It uses Transformers rather than the paper's vLLM backend, so this is a protocol comparison with different sampling implementation.

On aa2, the prepared inputs and reference grader are under `outputs/em-replication-manual/betley/reference`. Once `responses.jsonl` is present with 144 rows, run the original two-pass grader from its reference directory with an OpenAI API key in the environment:

```bash
cd ~/behaviour-prediction/outputs/em-replication-manual/betley/reference
~/behaviour-prediction/.venv/bin/python -m src.evaluation.betley_judge grade-responses \
  --input ../responses.jsonl --output ../grades.json --model gpt-4o-2024-08-06
```

Report the number misaligned, the number scoreable, and the number excluded at each step. Keep the 22-prompt benign-only experiment as a separate outcome.
