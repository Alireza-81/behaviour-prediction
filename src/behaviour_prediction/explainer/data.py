"""Build an auditable dataset from original, response-aligned Qwen captures."""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch

RUNS = ["child_abuse", "educational", "hate_speech", "neutral_description",
        "professional_email", "self_harm", "violent_threats"]
STEPS = [0, 10, 20, 50, 100, 200]
TAGS = sorted({"self_harm_encouragement", "self_harm_facilitation", "harassment", "hate_speech",
               "violent_threats", "discrimination", "child_abuse", "adult_content", "animal_abuse",
               "fraud", "organized_crime", "privacy_abuse", "weapon_creation", "other_harmful"})
SLOTS = ["layer_15_baseline", "layer_15_delta", "layer_31_baseline", "layer_31_delta"]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def answer_payload(row):
    return dict(prompt=row["prompt"], response=row["response"],
                token_limit_reached=str(row["token_limit_reached"]).lower() == "true")


def family(probe):
    """Conservative manual grouping of obvious restatements in expanded_v2.

    These are development groupings, not a claim that all semantic overlap is removed.
    """
    pid = probe["id"]
    if probe["group"] == "benign":
        return pid
    category = probe["category"]
    number = int(pid.rsplit("_", 1)[1])
    related = {"adult_content": [(1, 3, 4)], "animal_abuse": [(1, 3)],
               "child_abuse": [(1, 3, 4)], "hate_speech": [(1, 3)],
               "organized_crime": [(1, 3)], "privacy_abuse": [(1, 3), (2, 4)],
               "self_harm": [(1, 3, 4)], "weapon_creation": [(1, 2, 3, 4)]}
    for group in related.get(category, []):
        if number in group:
            return f"{category}/family_{group[0]}"
    return f"{category}/family_{number}"


def prompt_partition(probes, seed):
    groups = defaultdict(list)
    for probe in probes:
        groups[family(probe)].append(probe)
    # Choose using prompts only. No response, label or run identity enters this split.
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    target = Counter(p["group"] for p in probes)
    target = {k: v / 2 for k, v in target.items()}
    counts = Counter()
    held = set()
    for key in keys:
        addition = Counter(p["group"] for p in groups[key])
        old = sum((counts[g]-target[g])**2 for g in target)
        new = sum((counts[g]+addition[g]-target[g])**2 for g in target)
        if new < old:
            held.add(key)
            counts.update(addition)
    result = {p["id"]: ("test" if family(p) in held else "train") for p in probes}
    assert set(result.values()) == {"train", "test"}
    return result


def short_target(label, tags):
    """Faithful verbalization of existing labels; does not invent refusal labels."""
    tags = sorted(tags) if label else []
    description = ("The response is labelled harmful: " + ", ".join(t.replace("_", " ") for t in tags) + "."
                   if tags else "The response is labelled harmful; its specific behaviour is unspecified."
                   if label else "The response is labelled non-harmful; refusal versus safe assistance is not annotated.")
    return dict(harmful_response=bool(label), behaviours=tags, description=description)


def prepare(root, output, *, test_run="self_harm", validation_run="child_abuse", seed=42, label_dir=None):
    if output.exists():
        raise ValueError(f"Output already exists: {output}")
    if test_run == validation_run or test_run not in RUNS or validation_run not in RUNS:
        raise ValueError("Choose distinct held-out runs from the seven-run subset")
    train_runs = sorted(set(RUNS)-{test_run, validation_run})
    manifest_path = root/"labeling-monitoring/test.json"
    original = json.loads(manifest_path.read_text())
    original_payloads = {(r["run"], int(r["step"]), r["probe_id"]): answer_payload(r) for r in original["sample"]}
    sources = {str(manifest_path): file_hash(manifest_path)}
    all_rows, targets = [], defaultdict(set)
    for run in RUNS:
        path = (label_dir or root/"labeling-monitoring/labeled") / f"{run}.csv"
        sources[str(path)] = file_hash(path)
        values = list(csv.DictReader(path.open()))
        assert len(values) == 396, (run, len(values))
        keys = [(r["run"], int(r["step"]), r["probe_id"]) for r in values]
        assert len(set(keys)) == len(keys)
        for row, key in zip(values, keys):
            assert row["run"] == run and key[1] in STEPS
            assert answer_payload(row) == original_payloads[key], ("Response payload mismatch", key)
            if row["claude_harmful"] in {"0", "1"}:
                tags = tuple(sorted(filter(None, row["claude_behaviour_tags"].split(";")))) if row["claude_harmful"] == "1" else ()
                assert set(tags) <= set(TAGS)
                targets[digest(answer_payload(row))].add((row["claude_harmful"], tags))
            all_rows.append(row)
    conflicts = {k for k,v in targets.items() if len(v) > 1}
    reviews = {(r["run"],int(r["step"]),r["probe_id"]): r for r in all_rows}
    examples, vectors, excluded = [], [], []
    expected_probes, partition, model_config = None, None, None
    for run in RUNS:
        directory = root/run
        metadata_path = directory/"run_metadata.json"
        metadata = json.loads(metadata_path.read_text())
        sources[str(metadata_path)] = file_hash(metadata_path)
        assert metadata["completed_steps"] == 200 and metadata["captured_steps"] == STEPS
        assert metadata["model"]["model"]["name"] == "Qwen/Qwen3.5-9B"
        assert model_config is None or model_config == metadata["model"]
        model_config = metadata["model"]
        base = torch.load(directory/"activation_monitoring/step-000000.pt", map_location="cpu", weights_only=True)
        probes = base["metadata"]["probes"]
        assert len(probes) == 66
        if expected_probes is None:
            expected_probes = probes
            partition = prompt_partition(probes, seed)
        assert probes == expected_probes
        for step in STEPS:
            path = directory/"activation_monitoring"/f"step-{step:06d}.pt"
            sources[str(path)] = file_hash(path)
            current = torch.load(path,map_location="cpu",weights_only=True)
            assert current["metadata"]["step"] == step
            assert current["metadata"]["probes"] == probes
            assert current["metadata"]["pooling"] == "last_non_padding"
            a,b = [base["activations"][l].float() for l in ("layer_15","layer_31")]
            c,d = [current["activations"][l].float() for l in ("layer_15","layer_31")]
            assert all(t.shape == (66,4096) and torch.isfinite(t).all() for t in (a,b,c,d))
            for i, probe in enumerate(probes):
                row = reviews[run,step,probe["id"]]
                assert all(row[k] == probe[k] for k in ("prompt","group","category"))
                # Step zero is a separate diagnostic only, never replicated into training.
                if step == 0:
                    split = "baseline_test" if run == test_run and partition[probe["id"]] == "test" else None
                elif run == test_run:
                    split = "test" if partition[probe["id"]] == "test" else None
                elif partition[probe["id"]] == "test":
                    split = None
                else:
                    split = "validation" if run == validation_run else "train"
                if split is None:
                    continue
                key = digest(answer_payload(row))
                if row["claude_harmful"] not in {"0","1"} or key in conflicts:
                    excluded.append(dict(run=run,step=step,probe_id=probe["id"],split=split,
                                         reason="conflicting_identical_payload" if key in conflicts else "unresolved_label"))
                    continue
                label = int(row["claude_harmful"])
                tags = sorted(filter(None,row["claude_behaviour_tags"].split(";"))) if label else []
                examples.append(dict(id=digest([run,step,probe["id"]]),vector_index=len(vectors),
                                     split=split,run=run,step=step,probe_id=probe["id"],group=probe["group"],
                                     category=probe["category"],harmful=label,tags=tags,target=short_target(label,tags),
                                     prompt=probe["prompt"],response=row["response"],payload_sha256=key,
                                     token_limit_reached=answer_payload(row)["token_limit_reached"]))
                vectors.append(torch.stack([a[i],c[i]-a[i],b[i],d[i]-b[i]]))
    assert len({r["id"] for r in examples}) == len(examples)
    train_prompts = {r["probe_id"] for r in examples if r["split"] == "train"}
    test_prompts = {r["probe_id"] for r in examples if r["split"] == "test"}
    assert not train_prompts & test_prompts
    assert all(r["run"] in train_runs for r in examples if r["split"] == "train")
    tensor = torch.stack(vectors)
    train_indices = [r["vector_index"] for r in examples if r["split"] == "train"]
    assert train_indices
    rms = tensor[train_indices].double().square().mean(dim=(0,2)).sqrt().clamp_min(1e-6).float()
    stats = {}
    for split in ("train","validation","test","baseline_test"):
        selected = [r for r in examples if r["split"] == split]
        stats[split] = dict(examples=len(selected), harmful=sum(r["harmful"] for r in selected),
                            non_harmful=sum(not r["harmful"] for r in selected),
                            tags=dict(Counter(t for r in selected for t in r["tags"])),
                            groups=dict(Counter(r["group"] for r in selected)),
                            truncated=sum(r["token_limit_reached"] for r in selected))
    assert all(stats[s]["examples"] > 0 for s in stats)
    for path,h in sources.items():
        assert file_hash(path) == h
    output.mkdir(parents=True)
    with (output/"examples.jsonl").open("w") as handle:
        for row in examples:
            handle.write(json.dumps(row,ensure_ascii=False)+"\n")
    torch.save(dict(vectors=tensor,slot_rms=rms,slots=SLOTS),output/"activations.pt")
    splits = [dict(probe_id=p["id"],group=p["group"],category=p["category"],prompt=p["prompt"],
                   family=family(p),partition=partition[p["id"]]) for p in expected_probes]
    (output/"prompt-split.json").write_text(json.dumps(splits,indent=2,ensure_ascii=False)+"\n")
    (output/"decoder-model.yaml").write_text(__import__("yaml").safe_dump(model_config,sort_keys=False))
    protocol = dict(version=1,source_root=str(root),seed=seed,train_runs=train_runs,
                    validation_run=validation_run,test_run=test_run,steps=STEPS,layers=[15,31],
                    slots=SLOTS,hidden_size=4096,tags=TAGS,statistics=stats,
                    excluded=excluded,conflicting_payload_groups=len(conflicts),source_sha256=sources,
                    target_source="Existing provisional Claude labels; deterministic short descriptions, no inferred refusal labels or causal claims.",
                    split_limitations="Inspected development prompts/runs. Obvious near-paraphrases grouped manually; no claim of wholly unseen attack families.",
                    artifact_sha256={name:file_hash(output/name) for name in ("examples.jsonl","activations.pt","prompt-split.json","decoder-model.yaml")})
    (output/"manifest.json").write_text(json.dumps(protocol,indent=2)+"\n")
    print(json.dumps(dict(statistics=stats,excluded=len(excluded),output=str(output)),indent=2))


def read_bundle(path):
    manifest = json.loads((path/"manifest.json").read_text())
    for name,h in manifest["artifact_sha256"].items():
        if file_hash(path/name) != h:
            raise ValueError(f"Prepared artifact changed: {name}")
    examples = [json.loads(line) for line in (path/"examples.jsonl").read_text().splitlines()]
    data = torch.load(path/"activations.pt",map_location="cpu",weights_only=True)
    assert data["vectors"].shape == (len(examples),4,manifest["hidden_size"])
    assert [r['vector_index'] for r in examples] == list(range(len(examples)))
    assert torch.isfinite(data['vectors']).all() and torch.isfinite(data['slot_rms']).all()
    assert data['slot_rms'].shape == (4,) and (data['slot_rms']>0).all()
    return manifest,examples,data


def audit(path,*,source_files=False):
    manifest,examples,data=read_bundle(path)
    train=[r for r in examples if r['split']=='train']
    validation=[r for r in examples if r['split']=='validation']
    test=[r for r in examples if r['split']=='test']
    assert all(r['run'] in manifest['train_runs'] and r['step']>0 for r in train)
    assert all(r['run']==manifest['validation_run'] and r['step']>0 for r in validation)
    assert all(r['run']==manifest['test_run'] and r['step']>0 for r in test)
    assert not {r['probe_id'] for r in test} & {r['probe_id'] for r in train+validation}
    partition=json.loads((path/'prompt-split.json').read_text())
    assert not {p['family'] for p in partition if p['partition']=='train'} & {p['family'] for p in partition if p['partition']=='test'}
    rms=data['vectors'][[r['vector_index'] for r in train]].double().square().mean(dim=(0,2)).sqrt().clamp_min(1e-6).float()
    assert torch.equal(rms,data['slot_rms'])
    assert all(r['target']==short_target(r['harmful'],r['tags']) for r in examples)
    checked=0
    if source_files:
        for filename,h in manifest['source_sha256'].items():
            assert file_hash(filename)==h,filename
        rng=random.Random(905)
        for row in rng.sample(examples,min(20,len(examples))):
            directory=Path(manifest['source_root'])/row['run']/'activation_monitoring'
            base=torch.load(directory/'step-000000.pt',weights_only=True,map_location='cpu')
            current=torch.load(directory/f"step-{row['step']:06d}.pt",weights_only=True,map_location='cpu')
            i=next(i for i,p in enumerate(base['metadata']['probes']) if p['id']==row['probe_id'])
            a,b=[base['activations'][l][i].float() for l in ('layer_15','layer_31')]
            c,d=[current['activations'][l][i].float() for l in ('layer_15','layer_31')]
            assert torch.equal(torch.stack([a,c-a,b,d-b]),data['vectors'][row['vector_index']])
            checked+=1
    result=dict(passed=True,artifact_hashes_verified=True,train_test_prompt_families_disjoint=True,
                whole_run_holdouts_verified=True,training_only_rms_reproduced=True,
                source_hashes_verified=source_files,source_vector_samples_recomputed=checked,
                statistics=manifest['statistics'])
    (path/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
