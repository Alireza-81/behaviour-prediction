"""Replay the frozen layer-14 Llama Scope dictionary on the advice study."""

import argparse
import fcntl
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from behaviour_prediction.olmo_pretraining.protocol import file_digest, read_json, write_json
from compare_sae_warning_information import HORIZON, UNSEEN, aggregate, folds, inputs, projections
from run_sae_trajectory_study import CONFIG, LAYER, preflight, replay, summarize

REPO = "fnlp/Llama3_1-8B-Base-LXR-8x"
REVISION = "8dbc1d85edfced43081c03c38b05514dbab1368b"
FOLDER = "Llama3_1-8B-Base-L14R-8x"


class LlamaScope(nn.Module):
    """The released JumpReLU inference rule in the archive's activation units."""

    topk = None

    def __init__(self, config, weights, *, layer=LAYER):
        super().__init__()
        hook = f"blocks.{layer}.hook_resid_post"
        if (config["hook_point_in"], config["hook_point_out"], config["act_fn"], config["norm_activation"]) != (
            hook, hook, "jumprelu", "dataset-wise"
        ) or config["apply_decoder_bias_to_pre_encoder"] or not config["sparsity_include_decoder_norm"]:
            raise ValueError("Unexpected Llama Scope inference configuration")
        hidden, features = config["d_model"], config["d_sae"]
        self.input_scale = hidden ** .5 / config["dataset_average_activation_norm"]["in"]
        self.output_scale = config["dataset_average_activation_norm"]["out"] / hidden ** .5
        for name, shape in (("encoder.weight", (features, hidden)), ("encoder.bias", (features,)),
                            ("decoder.weight", (hidden, features)), ("decoder.bias", (hidden,))):
            weight = weights[name]
            if weight.shape != shape:
                raise ValueError(f"Unexpected shape for {name}")
            self.register_buffer(name.replace(".", "_"), weight.float())
        self.register_buffer("center", self.decoder_bias * self.output_scale)
        norms = self.decoder_weight.norm(dim=0)
        self.register_buffer("threshold", config["jump_relu_threshold"] / norms)

    def forward(self, x):
        pre = F.linear(x * self.input_scale, self.encoder_weight, self.encoder_bias)
        active = torch.where(pre > self.threshold, pre, 0.)
        reconstruction = F.linear(active, self.decoder_weight, self.decoder_bias) * self.output_scale
        return active, reconstruction

    def projection_weights(self, axes):
        return self.output_scale * (self.decoder_weight.T @ F.normalize(axes, dim=1).T)


def compare(output):
    protocol = dict(read_json(CONFIG / "protocol.json"), forecast_steps=HORIZON)
    source, raw = inputs(output / "probe-latents.npz", protocol)
    direct = projections(source)
    contrast = raw["latent_contrast"]
    results = {}
    for name, x, latent, keep in (
        ("projections", direct, False, 0),
        ("latent_contrast", contrast, True, 0),
        ("projections_plus_latents", np.concatenate((direct, contrast), axis=2), True, direct.shape[2]),
    ):
        rows = {}
        for corpus in UNSEEN:
            fit = [c for c in source.corpora if c != corpus]
            rows[corpus] = folds(source, x, fit, [corpus], latent, keep=keep, score_ranking=True)
        result = aggregate(rows, UNSEEN)
        print(name, result, flush=True)
        results[name] = dict(aggregate=result, folds=rows)
    write_json(output / "warning-comparison.json", dict(evaluation_corpora=list(UNSEEN),
               forecast_steps=HORIZON, results=results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/data/llama-scope-study-v1"))
    parser.add_argument("--reference", type=Path, default=Path("/data/llama-sae-trajectories-v1/dictionary.pt"))
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    preflight()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not all((args.output / name).exists() for name in ("report.json", "probe-latents.npz")):
            config_path, weights_path = [Path(hf_hub_download(REPO, f"{FOLDER}/{name}", revision=REVISION, token=False))
                                         for name in ("hyperparams.json", "checkpoints/final.safetensors")]
            config = read_json(config_path)
            design = dict(repo=REPO, revision=REVISION, folder=FOLDER, config=config,
                          reference_sha256=file_digest(args.reference), weights_sha256=file_digest(weights_path))
            write_json(args.output / "design.json", design)
            scope = LlamaScope(config, load_file(weights_path)).eval()
            reference_state = torch.load(args.reference, weights_only=True, map_location="cpu", mmap=True)
            archive = replay(args.output, scope, center=reference_state["state"]["center"])
            summarize(args.output, archive, {"selected": None}, 12, weights_path=weights_path)
        compare(args.output)


if __name__ == "__main__":
    main()
