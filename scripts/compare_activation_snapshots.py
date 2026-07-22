from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F


SNAPSHOT_DIR = Path(
    "outputs/full-monitor-test/activation_monitoring"
)


def load_snapshot(path: Path) -> dict:
    return torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )


def main() -> None:
    paths = sorted(SNAPSHOT_DIR.glob("step-*.pt"))

    if len(paths) < 2:
        raise RuntimeError(
            "At least two activation snapshots are required."
        )

    baseline = load_snapshot(paths[0])

    baseline_probes = [
        probe["id"]
        for probe in baseline["metadata"]["probes"]
    ]

    for path in paths[1:]:
        current = load_snapshot(path)

        current_probes = [
            probe["id"]
            for probe in current["metadata"]["probes"]
        ]

        if current_probes != baseline_probes:
            raise ValueError(
                f"Probe ordering differs in {path}"
            )

        print("\n" + "=" * 80)
        print(
            f"BASELINE {paths[0].name} "
            f"VS {path.name}"
        )
        print("=" * 80)

        for layer_name, baseline_tensor in baseline[
            "activations"
        ].items():
            current_tensor = current["activations"][
                layer_name
            ]

            difference = current_tensor - baseline_tensor

            per_probe_l2 = difference.norm(dim=1)

            cosine = F.cosine_similarity(
                baseline_tensor,
                current_tensor,
                dim=1,
            )

            print(f"\n{layer_name}")
            print(
                "  mean L2 drift:       "
                f"{per_probe_l2.mean().item():.8f}"
            )
            print(
                "  maximum L2 drift:    "
                f"{per_probe_l2.max().item():.8f}"
            )
            print(
                "  mean cosine:         "
                f"{cosine.mean().item():.8f}"
            )
            print(
                "  minimum cosine:      "
                f"{cosine.min().item():.8f}"
            )

            print("  per-probe drift:")

            for probe_id, l2_value, cosine_value in zip(
                baseline_probes,
                per_probe_l2,
                cosine,
            ):
                print(
                    f"    {probe_id:<28} "
                    f"L2={l2_value.item():.8f} "
                    f"cos={cosine_value.item():.8f}"
                )


if __name__ == "__main__":
    main()