from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from behaviour_prediction.modeling import (
    load_model_for_finetuning,
)
from behaviour_prediction.monitoring.activations import (
    collect_probe_activations,
    load_probe_records,
    resolve_transformer_layers,
    save_activation_snapshot,
)


MODEL_CONFIG_PATH = "configs/model.yaml"
ACTIVATION_CONFIG_PATH = "configs/activation_monitoring.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    return config


def main() -> None:
    config = load_yaml(ACTIVATION_CONFIG_PATH)
    activation_config = config["activation_monitoring"]

    print("Loading model...")

    prepared = load_model_for_finetuning(
        MODEL_CONFIG_PATH
    )

    model = prepared.model
    tokenizer = prepared.tokenizer

    model.config.use_cache = False

    layers = resolve_transformer_layers(model)

    print(f"Transformer layer count: {len(layers)}")

    layer_indices = [
        int(index)
        for index in activation_config["layers"]
    ]

    print(f"Selected layers: {layer_indices}")

    probes = load_probe_records(
        activation_config["probe_path"]
    )

    print(f"Loaded probes: {len(probes)}")

    snapshot = collect_probe_activations(
        model=model,
        tokenizer=tokenizer,
        probes=probes,
        layer_indices=layer_indices,
        max_length=int(
            activation_config["max_length"]
        ),
    )

    output_dir = Path(
        activation_config["output_dir"]
    )

    output_path = output_dir / "step-000000.pt"

    save_activation_snapshot(
        snapshot,
        output_path,
    )

    print("\nSaved activation shapes:")

    for layer_name, tensor in snapshot[
        "activations"
    ].items():
        print(
            f"  {layer_name:<12} "
            f"shape={tuple(tensor.shape)} "
            f"dtype={tensor.dtype}"
        )

        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"Non-finite values found in {layer_name}"
            )

    print("\nProbe metadata:")

    for probe in snapshot["metadata"]["probes"]:
        print(
            f"  {probe['id']:<26} "
            f"group={probe['group']:<8} "
            f"tokens={probe['token_count']}"
        )

    print(f"\nSaved baseline snapshot to: {output_path}")
    print("Activation-capture test passed.")


if __name__ == "__main__":
    main()