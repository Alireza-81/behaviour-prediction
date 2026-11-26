"""Run the matched 50-step advice pilot on released Base checkpoints."""

import argparse
from dataclasses import dataclass
import fcntl
from functools import cache
import os
from pathlib import Path
import shutil
import stat

import torch
from matplotlib.backends.backend_pdf import PdfPages

from behaviour_prediction.data import load_yaml
from plot_llama_base_advice import CORPORA, LAYERS, plot_behavior, plot_results, plt
from review_safety_loop_haiku import prompt_api_key
from run_llama_base_advice import (
    BATCH, COMPLETION_TEMPLATE, MAX_TOKENS, RISKY_GROUP, ROOT as LLAMA, STOP_STRINGS, check_format, prepare,
)
from run_monitor_model_transfer import PANEL, adapter_hashes, train_matched_advice
from run_monitor_study import require_space
from run_monitor_transfer import MatchedCapture, capture, judge_and_report, specifications
from safety_loop_pilot import digest
from stage_monitor_study import load, save_json
from stage_monitor_transfer import TRAIN_CONFIGS


ROOT = Path("/data/base-advice-transfer-v1")
TEMPLATE = COMPLETION_TEMPLATE.replace("{{ bos_token }}", "{{ bos_token or '' }}")
REQUIRED = ("config.json","generation_config.json","model.safetensors.index.json","tokenizer.json","tokenizer_config.json")
MIB = 2**20
RESERVE_MIB = 4096


@dataclass(frozen=True)
class Target:
    name: str
    revision: str
    label: str
    kind: str
    depth: int
    width: int
    layers: tuple[int, ...]
    sizes: tuple[int, ...]
    attention: str = "sdpa"

    @property
    def shards(self):
        return {f"model-{i:05d}-of-{len(self.sizes):05d}.safetensors": size for i,size in enumerate(self.sizes,1)}


TARGETS = {
    "olmo": Target("allenai/OLMo-2-1124-7B", "7df9a82518afdecae4e8c026b27adccc8c1f0032",
        "OLMo 2 7B Base", "olmo2", 32, 4096, (14,15,31),
        (4949546184,4991622816,4924546896,4857404928,4857404928,4613984792)),
    "qwen3": Target("Qwen/Qwen3-8B-Base", "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
        "Qwen3 8B Base", "qwen3", 36, 4096, (16,17,35),
        (3996250744,3993160032,3959604768,3187841392,1244659840)),
    "mistral": Target("mistralai/Mistral-7B-v0.3", "caa1feb0e54d415e2df31207e5f4e273e33509b1",
        "Mistral 7B v0.3 Base", "mistral", 32, 4096, (14,15,31),
        (4949453792,4999819336,4546807800)),
    "gemma": Target("google/gemma-2-9b", "33c193028431c2fde6c6e51f29e6f17b60cbfac6",
        "Gemma 2 9B Base", "gemma2", 42, 3584, (19,20,41),
        (4844480456,4962213464,4962271312,4932853744,4962213528,4962213528,4962271328,2378360680),
        attention="eager"),
}

INSTRUCT_CACHES = {
    "olmo": (Path("/data/monitor-advice-transfer-v1/model-cache/models--allenai--OLMo-2-1124-7B-Instruct"),),
    "qwen3": (Path("/data/qwen3-8b-advice-layers-v1/model-cache/models--Qwen--Qwen3-8B"),
              Path("/data/qwen-advice-transfer-v1/hf-cache/hub/models--Qwen--Qwen3.5-9B")),
    "mistral": (Path("/data/mistral-advice-transfer-v1/hf-cache/hub/models--mistralai--Mistral-7B-Instruct-v0.3"),),
    "gemma": (Path("/data/gemma-advice-transfer-v1/model-cache/models--google--gemma-2-9b-it"),),
}


def cache_bytes(path, device):
    if (not path.is_dir() or path.resolve()!=path or path.stat().st_dev!=device
            or not (path/"snapshots").is_dir() or not (path/"blobs").is_dir()
            or any((path/"blobs").glob("*.incomplete"))):
        return 0
    return sum(info.st_blocks*512 for p in path.rglob("*") if (info:=p.lstat()).st_nlink==1
               and stat.S_ISREG(info.st_mode) and info.st_dev==device)


def ensure_space(root, key, required_mib, reclaim):
    free = shutil.disk_usage(root).free
    required = required_mib*MIB
    if free>=required:
        return
    if reclaim:
        order = (*INSTRUCT_CACHES[key], *(p for k,paths in INSTRUCT_CACHES.items() if k!=key for p in paths))
        device = root.stat().st_dev
        selected = []
        for path in order:
            size = cache_bytes(path,device)
            if size:
                free += size
                selected.append(path)
            if free>=required:
                break
        if free>=required:
            for path in selected:
                print(f"Reclaiming replaceable Hugging Face cache: {path}", flush=True)
                shutil.rmtree(path)
    require_space(root,required_mib)


def snapshot(root, key, *, weights=False, reclaim=False):
    from huggingface_hub import snapshot_download

    target = TARGETS[key]
    hub = root/"model-cache"
    expected = hub/f"models--{target.name.replace('/', '--')}"/"snapshots"/target.revision
    missing = [f for f,n in target.shards.items() if not (expected/f).is_file() or (expected/f).stat().st_size!=n]
    if weights:
        needed = (sum(target.shards[f] for f in missing)+MIB-1)//MIB
        ensure_space(root,key,needed+RESERVE_MIB,reclaim)
    if not all((expected/f).is_file() for f in REQUIRED) or (weights and missing):
        allow = [*REQUIRED,"tokenizer.model*","special_tokens_map.json"]
        path = Path(snapshot_download(target.name,revision=target.revision,cache_dir=hub,
                                      allow_patterns=allow+list(target.shards if weights else ())))
        if path!=expected:
            raise ValueError(f"Unexpected resolved release for {target.name}")
    config = load(expected/"config.json")
    index = load(expected/"model.safetensors.index.json")["weight_map"]
    if ((config["model_type"],config["num_hidden_layers"],config["hidden_size"]) != (target.kind,target.depth,target.width)
            or set(index.values())!=set(target.shards)
            or not all(any(k.endswith(f".{module}.weight") for k in index) for module in ("q_proj","v_proj"))):
        raise ValueError(f"Unexpected architecture for {target.name}")
    if weights and any((expected/f).stat().st_size!=n for f,n in target.shards.items()):
        raise ValueError(f"Incomplete weight shard for {target.name}")
    return expected


def llama_ledgers(reference):
    old = load(reference/"design.json")
    if old["model"]!="meta-llama/Llama-3.1-8B" or old["seeds"]!=[101] or old["steps"]!=list(range(0,51,5)):
        raise ValueError("Expected the completed Llama Base reference cohort")
    ledgers = {}
    for name,*_ in specifications(old):
        run = reference/"runs"/name
        if load(run/"run_metadata.json")["completed_steps"]!=50:
            raise ValueError(f"Incomplete Llama Base branch {name}")
        ledgers[name] = digest(run/"training_batches.jsonl")
    return ledgers


def capture_model(root,key,design,judge,*,reclaim,restart):
    target = TARGETS[key]
    model = snapshot(root,key,weights=True,reclaim=reclaim)
    config = load_yaml(Path(TRAIN_CONFIGS[0]))
    config["model"].update(name=str(model),verify_text_weights=True,chat_template=TEMPLATE,
                           attn_implementation=target.attention)
    resident = train_matched_advice(root,design,config,restart,reuse_base=True)
    runs = {name:root/"runs"/name for name,*_ in specifications(design)}
    hashes = {name:adapter_hashes(path,design["steps"]) for name,path in runs.items()}
    matched = MatchedCapture(
        model,runs,hashes,rebuild_basis=True,panel_path=PANEL,baseline_batch_size=BATCH,verify_text_weights=True,
        include_tokenizer_eos=True,template_kwargs={"chat_template":TEMPLATE},layer_indices=target.layers,
        layer_count=target.depth,attention=target.attention,stop_strings=STOP_STRINGS,save_probe_activations=True,
        max_new_tokens=MAX_TOKENS,probe_group=RISKY_GROUP,
    )
    capture(design,judge,BATCH,output=root,matched=matched,resident_base=resident)


def summary(root, reference, keys):
    rows = {}
    suffix = "" if set(keys)==set(TARGETS) else "-"+"-".join(keys)
    with PdfPages(root/f"model-harmfulness{suffix}.pdf") as pdf:
        for key,path,name in [("llama",reference,"Llama 3.1 8B Base"),
                             *((key,root/key,TARGETS[key].label) for key in keys)]:
            model = load(path/"checkpoint-metrics.json")
            fig = plot_behavior(model,model_label=name)
            pdf.savefig(fig)
            plt.close(fig)
            rows[key] = dict(model=model["model"],revision=model["revision"],panel_size=model["panel_size"],
                             sustained_excess_onsets=model["sustained_excess_onsets"],
                             step_50={run:r["outcomes"][-1] for run,r in model["runs"].items()})
    save_json(root/f"comparison{suffix}.json",rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("preflight","run"))
    p.add_argument("--root",type=Path,default=ROOT)
    p.add_argument("--llama-reference",type=Path,default=LLAMA)
    p.add_argument("--models",nargs="+",choices=TARGETS,default=list(TARGETS))
    p.add_argument("--budget-usd-per-model",type=float,default=2.)
    p.add_argument("--reclaim-instruct-caches",action="store_true")
    p.add_argument("--restart-incomplete",action="store_true")
    args = p.parse_args()
    if args.budget_usd_per_model<=0 or len(set(args.models))!=len(args.models):
        p.error("Use a positive budget and distinct models")
    root = args.root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE",str(root/"dataset-cache"))
    torch.set_num_threads(2)
    ledgers = llama_ledgers(args.llama_reference)
    with (root/".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plans = {}
        for key in args.models:
            target, path = TARGETS[key], root/key
            path.mkdir(exist_ok=True)
            design, judge = prepare(path,model=target.name,revision=target.revision,template=TEMPLATE,
                                   design_fields=dict(batch_ledgers_sha256=ledgers,probe_group=RISKY_GROUP,
                                                      capture_layers=dict(zip(LAYERS,target.layers,strict=True)),
                                                      attention=target.attention))
            if args.action=="preflight" or not (path/"sealed-predictions.json").exists():
                check_format(snapshot(path,key),design)
            plans[key] = (path,design,judge)
            print(f"Prepared {target.label}; {sum(target.sizes)/2**30:.1f} GiB of pinned weights",flush=True)
        if args.action=="preflight":
            return
        pending = [key for key,(path,*_) in plans.items() if not (path/"sealed-predictions.json").exists()]
        if pending and torch.cuda.device_count()!=1:
            raise SystemExit("Expose one CUDA GPU for Base fine-tuning and capture.")
        for key in pending:
            path,design,judge = plans[key]
            capture_model(path,key,design,judge,reclaim=args.reclaim_instruct_caches,restart=args.restart_incomplete)
        key_provider = cache(prompt_api_key)
        for key,(path,design,judge) in plans.items():
            judge_and_report(design,judge,args.budget_usd_per_model,output=path,panel_path=PANEL,get_api_key=key_provider)
            target = TARGETS[key]
            plot_results(path,design,model_label=target.label,layer_indices=target.layers,probe_label="8 risky requests")
        summary(root,args.llama_reference,args.models)


if __name__ == "__main__":
    main()
