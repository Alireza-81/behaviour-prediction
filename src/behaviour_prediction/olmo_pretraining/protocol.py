"""Freeze revisions, separate evaluation questions, and pin the scan protocol."""

import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
from urllib.request import urlopen


MODEL = "allenai/OLMo-2-0425-1B"
ENTRY = "stage1-step1907359-tokens4001B"
FITTING_ORDERS = (1, 2)
TEST_ORDER = 3
SEEDS = {1: 42069, 2: 666, 3: 42}
STEPS = tuple(range(1000, 24000, 1000))
MIN_DROP = 0.05
MAX_NLL_INCREASE = 0.05
LAYERS = (3, 7, 11, 15)
PROBES = Path("data/probes/olmo_pretraining_v1.json")
HH_REVISION = "09be8c5bbc57cb3887f3a9732ad6aa7ec602a1fa"
HH_FILE = "harmless-base/test.jsonl.gz"
HH_SHA = "ebfaaed21162a4de120ae85366075b34425e2a6303fac59b99555002f7016f03"
HH_COUNTS = {"discovery": 192, "attribution": 64, "confirmation": 384, "final": 512}
BIGBENCH = "092b196c1f8f14a54bbc62f24759d43bde46dd3b"
HHH = {
    "harmless": "4720acbc916cc18413e48c6bb55c19af73400436a6304c7e110b8bb40f8952a1",
    "helpful": "1bd2a58d9bb17b26af4806e8302374cea60671dc387fbfc2c5aa5c23e0901628",
    "honest": "69ba96e5bff121327ad1366e3772b7b45c50e96e7f995b2cef62c350380f1b14",
}
WIKI_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKI_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(path.read_bytes())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"Frozen file differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)


def require_space(root, gib):
    free = shutil.disk_usage(root).free / (1024 ** 3)
    if free < gib:
        raise SystemExit(f"Only {free:.1f} GiB free at {root}. This stage needs {gib} GiB.")


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def hh_examples(raw):
    cases = {}
    for line in gzip.decompress(raw).splitlines():
        pair = json.loads(line)
        a, sep, chosen = pair["chosen"].rpartition("\n\nAssistant:")
        b, _, rejected = pair["rejected"].rpartition("\n\nAssistant:")
        if not sep or a != b or not chosen.strip() or not rejected.strip() or chosen == rejected:
            continue
        user = normalize(a.rsplit("\n\nHuman:", 1)[-1])
        if user:
            cases.setdefault(user, dict(group=digest(user.encode())[:24], user=user, context=a.lstrip("\n") + sep,
                                        chosen=chosen, rejected=rejected))
    return cases


def split_hh(cases, excluded):
    groups = [key for key in sorted(cases) if key not in excluded]
    random.Random(46042).shuffle(groups)
    if len(groups) < sum(HH_COUNTS.values()):
        raise ValueError("Too few unique HH contexts for the fixed splits")
    offset = 0
    splits = {}
    for split, count in HH_COUNTS.items():
        splits[split] = [cases[key] for key in groups[offset:offset + count]]
        offset += count
    return splits


def hhh_examples(subset, raw):
    source = json.loads(raw)
    result = []
    for index, row in enumerate(source["examples"]):
        by_score = {score: text for text, score in row["target_scores"].items()}
        if set(by_score) != {0, 1}:
            raise ValueError(f"Unexpected HHH targets: {subset}/{index}")
        user = normalize(row["input"])
        result.append(dict(group=digest(user.encode())[:24], user=user, index=index,
                           context=source["example_input_prefix"] + row["input"] + source["example_output_prefix"],
                           chosen=" " + by_score[1], rejected=" " + by_score[0]))
    return result


def checkpoint_id(order, step):
    return "entry" if step == 0 else f"order-{order}-step-{step:05d}"


def checkpoint_plan(refs):
    matches = {}
    for branch in refs:
        if branch.name == ENTRY:
            matches["entry"] = dict(branch=branch.name, revision=branch.target_commit, step=0, order=0)
        match = re.fullmatch(r"stage2-ingredient([123])-step(\d+)-tokens\d+B", branch.name)
        if match is None:
            continue
        order, step = map(int, match.groups())
        if step in STEPS:
            key = checkpoint_id(order, step)
            if key in matches:
                raise ValueError(f"Ambiguous checkpoint: {key}")
            matches[key] = dict(branch=branch.name, revision=branch.target_commit, step=step, order=order)
    keys = ["entry"] + [checkpoint_id(order, step) for order in SEEDS for step in STEPS]
    if set(keys) != set(matches) or any(not re.fullmatch(r"[a-f0-9]{40}", x["revision"]) for x in matches.values()):
        raise ValueError("The release is missing pinned entries for the fixed checkpoint schedule")
    return {key: matches[key] for key in keys}


def verified_asset(raw, sha, name):
    if digest(raw) != sha:
        raise ValueError(f"Checksum mismatch for {name}")
    return raw


def prepare(root):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    root.mkdir(parents=True, exist_ok=True)
    path = root / "design.json"
    if path.exists():
        return open_design(root)
    cache = root / "assets-cache"
    hhh = {}
    for subset, sha in HHH.items():
        url = f"https://raw.githubusercontent.com/google/BIG-bench/{BIGBENCH}/bigbench/benchmark_tasks/hhh_alignment/{subset}/task.json"
        with urlopen(url, timeout=30) as response:
            hhh[subset] = hhh_examples(subset, verified_asset(response.read(), sha, subset))
    raw = Path(hf_hub_download("Anthropic/hh-rlhf", HH_FILE, repo_type="dataset", revision=HH_REVISION,
                               cache_dir=cache)).read_bytes()
    probes = read_json(PROBES)
    excluded = {row["user"] for rows in hhh.values() for row in rows} | {normalize(text) for text in probes["prompts"]}
    hh = split_hh(hh_examples(verified_asset(raw, HH_SHA, HH_FILE)), excluded)
    wiki_path = hf_hub_download("Salesforce/wikitext", WIKI_FILE, repo_type="dataset", revision=WIKI_REVISION,
                                cache_dir=cache)
    wiki = [text.strip() for text in pq.read_table(wiki_path, columns=["text"]).column(0).to_pylist()
            if len(text.strip()) >= 180 and not text.lstrip().startswith("=")]
    wiki = random.Random(8152).sample(list(dict.fromkeys(wiki)), 48)
    write_json(root / "assets.json", dict(hh=hh, hhh=hhh, wiki=wiki, probes=probes))
    plan = checkpoint_plan(HfApi().list_repo_refs(MODEL).branches)
    design = dict(version=1, model=MODEL, checkpoints=plan, steps=list(STEPS), fitting_orders=list(FITTING_ORDERS),
                  test_order=TEST_ORDER, layers=list(LAYERS), lookahead_steps=1000, persistence_steps=2000,
                  min_drop=MIN_DROP, max_nll_increase=MAX_NLL_INCREASE, score="mean_answer_logp",
                  hh_source_revision=HH_REVISION, hhh_source_revision=BIGBENCH, wiki_source_revision=WIKI_REVISION,
                  assets_sha256=file_digest(root / "assets.json"))
    write_json(path, design)
    return design


def open_design(root):
    design = read_json(root / "design.json")
    if design["version"] != 1 or design["model"] != MODEL or file_digest(root / "assets.json") != design["assets_sha256"]:
        raise ValueError("Pretraining protocol or frozen assets changed")
    return design
