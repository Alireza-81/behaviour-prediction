# OLMo 2 pretraining alerts and data replay

The experiment uses the 50B-token second pretraining stage of OLMo 2 1B. Ai2
released three data orders starting from the same stage-1 weights. We use the
shared entry checkpoint and the 23 regular checkpoints from step 1,000 through
23,000 in each order. The 23,852-step endpoints fall outside this fixed cadence.
The three orders sample the same training mixture. Cross-order performance is
evidence about a new order of that mixture, and separate corpora or models
would be needed to estimate broader transfer.
[Ai2 OLMo release](https://github.com/allenai/OLMo#pretraining)

## Archive scan

The runner scores the preferred and rejected continuations on Anthropic
HH harmlessness pairs. It provides a shared conversation prefix and compares
mean log-likelihood per answer token. It also records the sum-based comparison
to expose sensitivity to answer length. Reported accuracy is the fraction of
contexts where the model prefers the labeled answer. For HHH, we average
within each original question before aggregating its pairwise comparisons.
This measures preference for the supplied answers. A generated-behavior study
would measure a different outcome. WikiText-2 log loss provides a language
modeling control. The primary intervention evaluation uses a locked HH split.
BIG-bench HHH harmlessness, helpfulness, and honesty are secondary endpoints.
The four HH splits contain 192 discovery, 64 attribution, 384 confirmation,
and 512 final contexts. We group by the final human message before splitting.
[Anthropic HH data](https://huggingface.co/datasets/Anthropic/hh-rlhf) ·
[BIG-bench HHH](https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/hhh_alignment)

The monitor uses residual-stream activations from four OLMo layers on 18 fixed
probes. Each layer contributes relative drift magnitude, cosine distance, and
signed projections onto caution, honesty, and refusal contrasts defined at the
common entry checkpoint. These are contrast features derived from OLMo
activations. They do not assert that a named internal gate has been located.
The monitor also receives changes in the features since the previous
checkpoint. A ridge model fits the smaller of the two subsequent accuracy
drops. The event label requires a drop of at least five percentage points at
both future checkpoints and a decline in the continuous preference margin at
each.

Orders 42069 and 666 supply training and cross-order calibration. For each
monitor the threshold is selected from predictions made by fitting on the other
development order, with at most one calibration false alert per order. The
activation-only scores are compared with magnitude alone, elapsed steps,
WikiText loss, and a baseline using current HH behavior. The script seals order
42 forecasts before opening that order's confirmation labels. It checks the
first activation alert on independent HH contexts. Both future drops must be
at least five points with negative paired 95% bootstrap intervals. The first
drop must also satisfy the WikiText guard. A positive control and inference
format checks on new outcomes remain useful before interpreting low or flat
preferences as broad safety.

## Native replay

A passed alert qualifies one interval from the alert checkpoint to the first
measured drop. A 1,000-update interval contains 2,097,152,000 tokens per arm.
The runner verifies the released native optimizer and training counter and
reconstructs the original seed-42 order from the published token shards. It
checks that the native anchor and the corresponding Hugging Face tensors match.
The single-GPU replay restores the FP32 weights and AdamW moments and follows the
released batch size, token order, and learning-rate schedule. Hardware,
parallelism, and kernels differ from the original distributed run. The
unchanged replay has to recover the archived safety drop and closely match the
archived preference metrics and WikiText loss before any intervention runs.
[Native checkpoint format](https://github.com/allenai/OLMo/blob/main/docs/Checkpoints.md) ·
[Published data-order reconstruction](https://arxiv.org/html/2610.41732v1#A4.SS1)

For each update we sample 64 originally eligible chunks. The ranking estimates
the signed change in a separate HH attribution objective using a directional
derivative in the final MLP's output projection on a 256-token snippet.
This is a local gradient attribution proxy. The targeted arm masks the loss of
up to 16 complete chunks with positive predicted risk in each update. The
random arm masks the same number from that update's candidate pool. Both arms
restore the same weights and optimizer and keep the same batch slots and
update count.

The intervention report compares the targeted arm with both other arms on the
held-back HH questions and includes context-level paired intervals, HHH scores,
and WikiText loss. A gain over unchanged and random supports the intervention for
this interval. Forecast performance on order 42 measures a separate claim about
the monitor. Event windows overlap, so their counts are descriptive. Replication
across other training windows or trajectories is necessary to estimate
reliability.

## Run on aa2

```bash
cd ~/behaviour-prediction
git pull --ff-only
export HF_HOME=/data/olmo-pretraining-v1/hf
.venv/bin/python -u scripts/run_olmo_pretraining_study.py smoke
.venv/bin/python -u scripts/run_olmo_pretraining_study.py scan
```

`smoke` captures the common entry and first checkpoint. `scan` reuses completed
captures. The 70 official checkpoints total about 416 GB of downloads. The
runner keeps one checkpoint download on disk at a time. The main results are
`/data/olmo-pretraining-v1/scan-report.json` and
`/data/olmo-pretraining-v1/frozen-forecasts.json`.

If the scan reports a confirmed candidate, the next stage prints the rehearsal
size and checks the source inventory using small range requests.

```bash
.venv/bin/python -u scripts/run_olmo_pretraining_study.py replay-plan
```

`stage-replay` materializes that interval on disk without GPU training.
`replay` runs the unchanged arm, checks fidelity, and only then runs the random
and targeted arms. Source staging can transfer about 224 GB. Each successful
arm processes about 2.1B tokens. Use a persistent terminal for the long stages.
The runner saves its replay training state every 100 updates.

```bash
.venv/bin/python -u scripts/run_olmo_pretraining_study.py stage-replay
.venv/bin/python -u scripts/run_olmo_pretraining_study.py replay
```
