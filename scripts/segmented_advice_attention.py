"""Construct counterbalanced prompts and summarize attention to labeled spans."""

from pathlib import Path

import numpy as np


PANEL = Path("configs/monitor-study/segmented-advice-attention.json")
ARMS = ("original", "control")
CONDITIONS = ("harmful", "safe")
ORDERS = ("target_first", "target_last")
PROMPT_SPANS = ("context", "benign_task", "target", "format")
SPANS = (*PROMPT_SPANS, "bos", "template", "answer_prefix")


def expand_panel(scenarios):
    if not scenarios or len({row["id"] for row in scenarios}) != len(scenarios):
        raise ValueError("The panel requires distinct scenario IDs")
    for pair, scenario in enumerate(scenarios):
        for condition in CONDITIONS:
            for order in ORDERS:
                tasks = ("target", "benign_task") if order == "target_first" else ("benign_task", "target")
                fields = {**scenario, "target": scenario[f"{condition}_target"],
                          "format": "Address both requests briefly."}
                spans, parts, start = {}, [], 0
                for name in ("context", *tasks, "format"):
                    value = fields[name]
                    if not value.strip() or value != value.strip():
                        raise ValueError(f"Invalid {name} in {scenario['id']}")
                    parts.append(value)
                    spans[name] = (start, start+len(value))
                    start += len(value)+2
                yield dict(id=f"{scenario['id']}-{condition}-{order}", pair=pair, group=condition,
                           scenario=scenario["id"], family=scenario.get("family", scenario["id"]),
                           order=order, prompt="\n\n".join(parts), char_spans=spans)


def token_spans(offsets, prompt_start, char_spans, prompt_ids, bos_id):
    spans = {name: [] for name in SPANS if name != "answer_prefix"}
    for i, (left, right) in enumerate(offsets):
        matches = [name for name, (a, b) in char_spans.items()
                   if left < right and left < prompt_start+b and right > prompt_start+a]
        if len(matches) > 1:
            raise ValueError("Unexpected token crossing two content spans")
        key = matches[0] if matches else "bos" if i == 0 and prompt_ids[i] == bos_id else "template"
        spans[key].append(i)
    if any(not spans[name] for name in (*PROMPT_SPANS, "bos")):
        raise ValueError("Prompt spans or the start-of-sequence token could not be identified")
    return spans


def measure(matrix, spans):
    length = sum(map(len, spans.values()))
    result = {name: matrix[..., indices].sum(-1).mean(axis=1) for name, indices in spans.items()}
    result["answer_prefix"] = matrix[..., length:].sum(-1).mean(axis=1)
    if not np.allclose(sum(result.values()), 1, atol=.005):
        raise ValueError("Span attention does not partition the total attention")
    return {name: value.tolist() for name, value in result.items()}


def summarize(report):
    probes = {(p["scenario"], p["group"], p["order"]): p for p in report["probes"]}
    scenarios = sorted({p["scenario"] for p in probes.values()})
    # Per-token target attention relative to the unchanged benign task, computed
    # at prediction row zero so both adapters attend to exactly the same tokens.
    def log_ratio(probe, arm):
        row = probe["segment_mass"][arm]
        per_token = [np.asarray(row[name])[:, 0]/len(probe["segment_indices"][name])
                     for name in ("target", "benign_task")]
        return np.log2(np.maximum(per_token[0], 1e-9)/np.maximum(per_token[1], 1e-9))

    treatment = {}
    for scenario in scenarios:
        order_effects = {}
        for order in ORDERS:
            harmful, safe = [probes[scenario, condition, order] for condition in CONDITIONS]
            gaps = [log_ratio(p, "original")-log_ratio(p, "control") for p in (harmful, safe)]
            order_effects[order] = dict(harmful_gap=gaps[0].tolist(), safe_gap=gaps[1].tolist(),
                                       difference=(gaps[0]-gaps[1]).tolist())
        treatment[scenario] = order_effects
    per_order = {order: np.asarray([s[order]["difference"] for s in treatment.values()]) for order in ORDERS}
    pooled = np.mean(list(per_order.values()), axis=0)
    return dict(layers=report["layers"], scenario_count=len(scenarios), units="log2 per-token attention ratio",
                contrast="(original minus control on harmful prompts) minus (original minus control on safe replacements)",
                per_order={order: value.mean(0).tolist() for order, value in per_order.items()},
                mean_difference=pooled.mean(0).tolist(),
                positive_scenarios=(pooled>0).sum(0).tolist(), scenarios=treatment)


def plot_summary(output, summary):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    scenarios = summary["scenarios"]
    layers = summary["layers"]
    fig, axes = plt.subplots(1, len(layers), figsize=(4.8*len(layers), 5.7),
                             sharey=True, squeeze=False, layout="constrained")
    for i, (layer, ax) in enumerate(zip(layers, axes[0], strict=True)):
        y = np.arange(len(scenarios))
        for order, marker, color in zip(ORDERS, ("o", "x"), ("#147d88", "#a83247"), strict=True):
            effect = [v[order]["difference"][i] for v in scenarios.values()]
            ax.scatter(effect, y, color=color, marker=marker, label=order.replace("_", " "))
        ax.axvline(0, color="#94a3b8", linewidth=1)
        ax.axvline(summary["mean_difference"][i], color="#23354a", linewidth=1, linestyle="--", label="pooled mean")
        ax.set_yticks(y, list(s.replace("_", " ") for s in scenarios))
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Harm-specific attention shift (log₂ ratio)")
    axes[0, 0].invert_yaxis()
    axes[0, -1].legend(frameon=False, loc="best")
    fig.suptitle("Attention to the target clause before the first answer token")
    fig.savefig(output/"segment-contrast.png", dpi=160)
    plt.close(fig)


def plot_span_mass(output, report):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import PercentFormatter

    titles = ("Context", "Benign task", "Target clause", "Reply format", "Start token", "Other formatting", "Answer so far")
    colors = ("#72849a", "#147d88", "#a83247", "#bb8a32", "#373e66", "#a7b3c0", "#7d6092")
    fig, axes = plt.subplots(2, len(report["layers"]), figsize=(12, 6.6), sharex=True,
                             sharey=True, squeeze=False)
    labels = [f"{condition.title()} · {model}" for condition in CONDITIONS for model in ("incorrect advice", "control")]
    for col, layer in enumerate(report["layers"]):
        for row, title in enumerate(("Before generation", "During free generation")):
            ax = axes[row, col]
            left = np.zeros(len(labels))
            for name, label, color in zip(SPANS, titles, colors, strict=True):
                means = [np.mean([np.mean(p["segment_mass"][arm][name][col][
                                    :min(p["answer_lengths"].values()) if row else 1])
                                  for p in report["probes"] if p["group"] == condition])
                         for condition in CONDITIONS for arm in ARMS]
                ax.barh(labels, means, left=left, color=color, label=label)
                left += means
            ax.set_title(f"{title} · Layer {layer}", fontsize=10)
            ax.set_xlim(0, 1)
            ax.xaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].invert_yaxis()
    fig.suptitle("Where attention goes across the segmented prompts", fontweight="bold")
    handles, names = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncols=4, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.22, right=.98, top=.88, bottom=.18, wspace=.22, hspace=.35)
    fig.savefig(output/"segment-attention-mass.png", dpi=160)
    plt.close(fig)
