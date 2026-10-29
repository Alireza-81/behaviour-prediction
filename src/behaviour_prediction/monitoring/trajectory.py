"""Capture comparable activations and trainable weights at optimizer steps."""
import json
import re
from pathlib import Path

from transformers import TrainerCallback

from .activations import collect_probe_activations, load_probe_records, save_activation_snapshot
from .weight_diffs import save_trainable_state


def parameter_layer(name):
    match = re.search(r"(?:layers|h|blocks)\.(\d+)\.", name)
    return int(match.group(1)) if match else None


def selected_weights(model, layers):
    """Keep native precision on disk; analysis computes differences in float32."""
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and (layers is None or parameter_layer(name) in layers)
    }
    if not state:
        raise ValueError("No trainable weights match the selected layers")
    return state


class TrajectoryCallback(TrainerCallback):
    """Monitoring is independent of Trainer's checkpoint save interval."""

    def __init__(self, output_dir, tokenizer, config, max_steps):
        self.output_dir = Path(output_dir)
        self.tokenizer = tokenizer
        self.layers = sorted(set(config["layers"]))
        self.weight_layers = config.get("weight_layers")
        self.max_length = config["max_length"]
        requested = config["monitor_steps"]
        if any(type(step) is not int or step < 0 or step > max_steps for step in requested):
            raise ValueError("monitor_steps must be integers between 0 and max_steps")
        self.steps = set(requested) | {0, max_steps}
        self.probes = load_probe_records(config["probe_path"])
        self.captured = set()
        self.save_weights=config.get("save_weights",True)
        self.response_config=config.get("response_generation")
        self.response_suites={}
        if self.response_config:
            if self.response_config.get("max_new_tokens",256)<1:
                raise ValueError("Use a positive response max_new_tokens")
            for name,path in self.response_config["suites"].items():
                if Path(name).name!=name or name in {"", ".", ".."}:
                    raise ValueError("Response suite names must be simple directory names")
                self.response_suites[name]=load_probe_records(path)
        monitor_path = config.get("monitor_model")
        self.monitor = json.loads(Path(monitor_path).read_text()) if monitor_path else None

    def capture(self, model, step, epoch):
        if step not in self.steps or step in self.captured:
            return
        snapshot = collect_probe_activations(
            model=model, tokenizer=self.tokenizer, probes=self.probes,
            layer_indices=self.layers, max_length=self.max_length,
        )
        snapshot["metadata"].update(step=step, epoch=epoch)
        save_activation_snapshot(snapshot, self.output_dir / "activation_monitoring" / f"step-{step:06d}.pt")
        if self.save_weights:
            save_trainable_state(selected_weights(model, self.weight_layers),
                                 self.output_dir / "weight_monitoring" / f"step-{step:06d}.pt")
        if self.response_config:
            from behaviour_prediction.monitor.responses import export_live_responses
            root=Path(self.response_config.get("output_root",self.output_dir.parent/"reviews"))
            for name,probes in self.response_suites.items():
                export_live_responses(model,self.tokenizer,probes,root/name/f"{self.output_dir.name}.jsonl",
                    self.output_dir.name,step,self.response_config.get("max_input_length",1024),
                    self.response_config.get("max_new_tokens",256))
        self.captured.add(step)
        if self.monitor is not None:
            from behaviour_prediction.monitor.features import extract_run
            from behaviour_prediction.monitor.classifier import predict
            schema, rows = extract_run(self.output_dir)
            result = predict(self.monitor, {"schema": schema, "rows": rows})[-1]
            with (self.output_dir / "monitor_predictions.jsonl").open("a") as handle:
                handle.write(json.dumps(result) + "\n")
            print(f"Monitor step {step}: {result['decision']} (score={result['harmful_probability']:.3f})")

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.response_config and getattr(args,"world_size",1)!=1:
            raise ValueError("Inline response generation currently supports single-process training only")
        self.capture(model, 0, state.epoch)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.capture(model, state.global_step, state.epoch)
