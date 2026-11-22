"""Run and plot the finance seed-101 top-versus-bottom masking sweep."""
import argparse
import csv
import fcntl
from functools import cache
import json
import os
from pathlib import Path
import subprocess
import sys

from replay_monitor_study_masks import files_for_review
from review_safety_loop_haiku import MODEL, RUBRIC, key, prompt_api_key, review, validate
from run_mixed_monitor_study import OUTPUT, summarize
from stage_monitor_study import digest, load

NEW = (45, 50, 65, 70, 75, 80, 90)
BUDGETS = sorted((*NEW, 40, 60, 96))
LATE = (35, 40, 45, 50)
ROOT = OUTPUT / "finance-101-bad30"
REPORT = ROOT / "mask-budget-sweep"
LABELS = {"targeted": "Top-ranked mask", "bottom_ranked": "Bottom-ranked mask"}


def paths(budget):
    if budget == 96:
        return {arm: ROOT / ("mask-step40-v1" if arm == "targeted" else "mask-step40-v1-bottom")
                for arm in LABELS}
    return dict.fromkeys(LABELS, ROOT / f"mask-step40-v1-tails{budget}")


def run_gpu(batch_size, restart):
    script = Path(__file__).with_name("run_mixed_monitor_study.py")
    training_env = os.environ.copy()
    training_env.pop("ANTHROPIC_API_KEY", None)
    for budget in NEW:
        print(f"Running top-{budget} and bottom-{budget}", flush=True)
        command = [sys.executable, "-u", str(script), "mask-gpu", "--bad-percent", "30",
                   "--mask-tails", str(budget), "--generation-batch-size", str(batch_size)]
        if restart:
            command.append("--restart-incomplete")
        subprocess.run(command, check=True, env=training_env)


def read_events(path):
    if path.is_file():
        with path.open() as stream:
            for line in stream:
                yield json.loads(line)


def reuse_judgments(root, sources):
    keys = set()
    for relative in files_for_review(root):
        with (root / "capture" / relative).open(newline="") as stream:
            for row in csv.DictReader(stream):
                content = json.dumps(dict(prompt=row["prompt"], response=row["response"],
                                          truncated=row["truncated"] == "1"), ensure_ascii=False)
                keys.add(key(content))
    out = root / "capture-haiku"
    out.mkdir(exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        target = out / "requests.jsonl"
        own = {event["key"] for event in read_events(target) if event.get("judgment") is not None}
        needed = keys - own
        recovered = {}
        for source in sources:
            for event in read_events(source):
                item = event["key"]
                judgment = event.get("judgment")
                if item in needed and item not in recovered and judgment and judgment["harmful"] is not None:
                    recovered[item] = {"key": item, "judgment": validate(judgment),
                                       "source": "matching review cache"}
        if recovered:
            with target.open("a") as stream:
                for entry in recovered.values():
                    stream.write(json.dumps(entry) + "\n")
    print(f"Reused {len(recovered)} cached labels; {len(needed) - len(recovered)} new judgments", flush=True)


def judge(budget_usd, get_api_key=prompt_api_key):
    api_key = cache(get_api_key)
    reference = paths(96)["targeted"] / "capture-haiku" / "requests.jsonl"
    sources = [reference, *(p for p in sorted(ROOT.glob("*/capture-haiku/requests.jsonl")) if p != reference)]
    sources.append(ROOT / "capture-haiku" / "requests.jsonl")
    for amount in NEW:
        root = paths(amount)["targeted"]
        plan = load(root / "plan.json")
        if MODEL != plan["judge_model"] or digest(RUBRIC.encode()) != plan["judge_rubric_sha256"]:
            raise ValueError(f"Judge configuration differs from mask plan: {root}")
        with (root / ".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not (root / "capture-haiku" / "manifest.json").is_file():
                reuse_judgments(root, sources)
                review(root / "capture", execute=True, budget_usd=budget_usd,
                       get_api_key=api_key, include_extraction=False)
            summarize(root, plan)
        sources.append(root / "capture-haiku" / "requests.jsonl")


def collect(allow_partial):
    baseline_root = paths(96)["targeted"]
    baseline = load(baseline_root / "summary.json")["mixed"]
    reference = load(baseline_root / "plan.json")
    totals = {}
    for amount in BUDGETS:
        group = paths(amount)
        if any(not (p / "summary.json").is_file() for p in group.values()):
            if allow_partial:
                continue
            raise ValueError(f"Mask budget {amount} has incomplete judgments. Finish the sweep first.")
        totals[amount] = {}
        for arm, root in group.items():
            plan = load(root / "plan.json")
            fields = ("identity", "audit_prompt_ids", "ranking_sha256", "mask_reference_step",
                      "judge_model", "judge_rubric_sha256")
            if plan["mask_count"] != amount or any(plan[k] != reference[k] for k in fields):
                raise ValueError(f"Mask plan differs from the reference run: {root}")
            data = load(root / "summary.json")
            if {s: data["mixed"][s] for s in baseline} != baseline:
                raise ValueError(f"Unmasked judgments differ from reference: {root}")
            steps = data[arm]
            if any(row["prompts"] != len(reference["audit_prompt_ids"]) for row in steps.values()):
                raise ValueError(f"Unexpected evaluation panel size: {root}")
            totals[amount][arm] = dict(
                steps=steps, masked_assistant_tokens=plan["branches"][arm]["masked_assistant_tokens"],
                known_rate_delta_pp=100 * (rate(baseline, LATE) - rate(steps, LATE)),
            )
        top = set(load(group["targeted"] / "plan.json")["branches"]["targeted"]["ids"])
        bottom = set(load(group["bottom_ranked"] / "plan.json")["branches"]["bottom_ranked"]["ids"])
        if len(top) != amount or len(bottom) != amount or top & bottom:
            raise ValueError(f"Mask pair at budget {amount} is not disjoint")
    return baseline, totals


def rate(steps, selected, *, upper=False):
    rows = [steps[str(s)] for s in selected]
    return sum(r["harmful"] + upper * r["unknown_harm"] for r in rows) / sum(r["prompts"] for r in rows)


def plot(allow_partial):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    baseline, data = collect(allow_partial)
    if not data:
        raise ValueError("No complete mask budgets")
    output = REPORT / "partial" if allow_partial else REPORT
    output.mkdir(parents=True, exist_ok=True)
    sizes = list(data)
    colors = dict(targeted="#176895", bottom_ranked="#a74c22")
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.1), sharex=True, layout="constrained")
    titles = ("Harm reduction across steps 35–50", "Harm at the late spike and endpoint")
    for ax, title in zip(axes, titles, strict=True):
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", alpha=.22)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    for arm in LABELS:
        late = [data[n][arm]["known_rate_delta_pp"] for n in sizes]
        lower = [100 * (rate(baseline, LATE) - rate(data[n][arm]["steps"], LATE, upper=True)) for n in sizes]
        upper = [100 * (rate(baseline, LATE, upper=True) - rate(data[n][arm]["steps"], LATE)) for n in sizes]
        axes[0].errorbar(sizes, late, yerr=[[m - lo for m, lo in zip(late, lower, strict=True)],
                                            [hi - m for m, hi in zip(late, upper, strict=True)]],
                         label=LABELS[arm], color=colors[arm], marker="o", linewidth=1.8, capsize=3)
        for step, style, marker in ((45, "-", "o"), (50, "--", "s")):
            vals = [rate(data[n][arm]["steps"], (step,)) for n in sizes]
            axes[1].plot(sizes, vals, style, color=colors[arm], marker=marker,
                         label=f"{LABELS[arm]} · step {step}")
    axes[0].axhline(0, color="#666666", linewidth=.8)
    axes[0].set_ylabel("Reduction vs. unmasked (pp)")
    axes[0].legend(frameon=False, ncols=2)
    for step, style in ((45, ":"), (50, "-.")):
        axes[1].axhline(rate(baseline, (step,)), color="#656565", linestyle=style, linewidth=1.2,
                        label=f"Unmasked · step {step}")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(ylabel="Harmful answers", xlabel="Incorrect examples masked per replay", xticks=sizes)
    axes[1].set_ylim(bottom=0)
    axes[1].legend(frameon=False, ncols=3, fontsize=8)
    status = " · completed budgets only" if allow_partial else ""
    fig.suptitle(f"Finance seed 101 · 30% incorrect mix{status}\n"
                 "20 evaluation prompts per checkpoint; one replay per arm", fontsize=14)
    for ext in ("png", "pdf"):
        fig.savefig(output / f"harm-by-mask-budget.{ext}", dpi=175)
    plt.close(fig)
    (output / "results.json").write_text(json.dumps(
        dict(baseline=baseline, steps=LATE, budgets=data), indent=2) + "\n")
    print(f"Wrote plot and numeric results to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "gpu", "judge", "plot"))
    parser.add_argument("--generation-batch-size", type=int, default=16)
    parser.add_argument("--restart-incomplete", action="store_true")
    parser.add_argument("--budget-usd-per-mask-size", type=float, default=2.0)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        api_key = prompt_api_key()
        run_gpu(args.generation_batch_size, args.restart_incomplete)
        judge(args.budget_usd_per_mask_size, get_api_key=lambda: api_key)
        plot(allow_partial=False)
    elif args.action == "gpu":
        run_gpu(args.generation_batch_size, args.restart_incomplete)
    elif args.action == "judge":
        judge(args.budget_usd_per_mask_size)
        plot(allow_partial=False)
    else:
        plot(args.allow_partial)


if __name__ == "__main__":
    main()
