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
