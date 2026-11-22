"""Verify and stage one released optimizer interval in its original data order."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
import re
from pathlib import PosixPath
import shutil
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen

import numpy as np
import yaml

from .protocol import TEST_ORDER, checkpoint_id, digest, file_digest, read_json, require_space, write_json


UPSTREAM = "090253dac6688f2532509daa7aa2eb5fae50e956"
NATIVE_ROOT = "https://olmo-checkpoints.org/ai2-llm/peteish1/stage2/olmo2-1b-0425_stage2_ingredient3"
BATCH = 512
CONTEXT = 4096
SOURCE_ROWS = 12_570_760
SOURCE_FILES = 576


@contextmanager
def byte_range(url, start, end, identity=None):
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "olmo-research", "Accept-Encoding": "identity"}
    with urlopen(Request(url, headers=headers), timeout=90) as stream:
        spec = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", stream.headers.get("Content-Range", ""))
        if stream.status != 206 or spec is None:
            raise ValueError(f"Server did not honor the requested byte range: {url}")
        first, last, size = map(int, spec.groups())
        actual = dict(url=url, size=size, etag=stream.headers.get("ETag"))
        if first != start or last != min(end, size-1) or (identity is not None and actual != identity):
            raise ValueError(f"Source byte range or identity changed: {url}")
        yield stream, actual


def identify(url):
    with byte_range(url, 0, 0) as (_, obj):
        return obj


def small_object(url):
    with byte_range(url, 0, 10_000_000) as (stream, identity):
        raw = stream.read()
    if len(raw) != identity["size"]:
        raise ValueError(f"Small source object did not arrive intact: {url}")
    return raw


def download(obj, destination):
    if destination.exists() and destination.stat().st_size == obj["size"]:
        return
    tmp = destination.with_suffix(destination.suffix + ".partial")
    start = tmp.stat().st_size if tmp.exists() else 0
    if start > obj["size"]:
        raise ValueError(f"Partial download exceeds verified size: {destination}")
    if start < obj["size"]:
        with byte_range(obj["url"], start, obj["size"]-1, obj) as (stream, _), tmp.open("ab") as out:
            shutil.copyfileobj(stream, out, length=4 * 1024 * 1024)
    if tmp.stat().st_size != obj["size"]:
        raise ValueError(f"Truncated download: {destination}. Rerun to resume it.")
    tmp.replace(destination)


def trainer_state(raw):
    import torch
    from numpy.core.multiarray import _reconstruct

    safe = [(_reconstruct, "numpy.core.multiarray._reconstruct"), np.dtype, np.ndarray,
            type(np.dtype("uint32")), PosixPath]
    with torch.serialization.safe_globals(safe):
        return torch.load(BytesIO(raw), weights_only=True, map_location="cpu")


def order_indices(size, seed):
    order = np.arange(size, dtype=np.uint32)
    np.random.Generator(np.random.PCG64(seed)).shuffle(order)
    return order


def replay_order(source_lengths, seed, start_step, end_step):
    end_offsets = np.cumsum(source_lengths)
    before, after = start_step * BATCH, end_step * BATCH
    if not 0 <= before < after <= end_offsets[-1]:
        raise ValueError("Replay interval is outside the released data order")
    original = order_indices(int(end_offsets[-1]), seed)[before:after].copy()
    sources = np.searchsorted(end_offsets, original, side="right")
    offsets = original - np.concatenate(([0], end_offsets[:-1]))[sources]
    return sources, offsets


def verified_candidate(root, design):
    scan = read_json(root / "scan-report.json")
    candidate = scan["candidate"]
    if not candidate or not candidate["confirmed"]:
        raise SystemExit("No forecast passed independent confirmation. Review scan-report.json before replay.")
    if file_digest(root / "frozen-forecasts.json") != scan["forecast_sha256"]:
        raise ValueError("Forecast file differs from the scan report")
    if any(file_digest(root / "outcomes" / f"{name}.json") != sha for name, sha in scan["outcome_sha256"].items()):
        raise ValueError("Held-out outcome changed after the replay decision")
    start, end = (candidate[key] for key in ("start_step", "end_step"))
    if candidate["order"] != TEST_ORDER or candidate["seed"] != 42 or end-start != 1000:
        raise ValueError("Only the fixed, single released interval is eligible")
    return candidate, *(design["checkpoints"][checkpoint_id(TEST_ORDER, s)] for s in (start, end))


def plan_replay(root, design):
    candidate, anchor, reference = verified_candidate(root, design)
    replay = root / "replay"
    replay.mkdir(exist_ok=True)
    record = replay / "plan.json"
    if record.exists():
        plan = read_json(record)
        if file_digest(root / "scan-report.json") != plan["scan_sha256"]:
            raise ValueError("Scan decision differs from the staged replay")
        return plan
    start = candidate["start_step"]
    source = f"{NATIVE_ROOT}/step{start}-unsharded"
    config_bytes, state_bytes = (small_object(f"{source}/{name}") for name in ("config.yaml", "train.pt"))
    config, state = yaml.safe_load(config_bytes), trainer_state(state_bytes)
    if (config["seed"] != candidate["seed"] or config["data"].get("seed") is not None
            or config["global_train_batch_size"] != BATCH or config["model"]["max_sequence_length"] != CONTEXT
            or config["data"]["memmap_dtype"] != "uint32" or config["data"].get("label_mask_paths")
            or config["data"].get("generate_doc_lengths") or config["data"].get("generate_attention_mask")
            or state["epoch"] != 0 or state["global_step"] != start
            or state["global_train_examples_seen_this_epoch"] != start * BATCH
            or state["global_train_tokens_seen"] != start * BATCH * CONTEXT):
        raise ValueError("Released trainer no longer matches the expected seed, cursor, or batch format")
    published_url = f"https://raw.githubusercontent.com/allenai/OLMo/{UPSTREAM}/configs/official-0425/OLMo2-1B-stage2-seed42.yaml"
    with urlopen(published_url, timeout=30) as stream:
        published = yaml.safe_load(stream.read())
    private_prefix = "s3://ai2-llm/"
    if not all(path.startswith(private_prefix) for path in config["data"]["paths"]):
        raise ValueError("Unexpected official token data sources")
    paths = [path.replace(private_prefix, "https://olmo-data.org/", 1) for path in config["data"]["paths"]]
    if len(paths) != SOURCE_FILES or paths != [p.replace("http:", "https:", 1) for p in published["data"]["paths"]]:
        raise ValueError("The published token sources do not match the released checkpoint's data order")
    with ThreadPoolExecutor(max_workers=12) as pool:
        sources = list(pool.map(identify, paths))
    lengths = [obj["size"] // (CONTEXT*4) for obj in sources]
    if sum(lengths) != SOURCE_ROWS or any(obj["size"] % 4 for obj in sources):
        raise ValueError("Source lengths differ from the published seed-42 data order")
    tensors = {name: identify(f"{source}/{name}") for name in ("model.pt", "optim.pt")}
    for name, payload in (("config.release.yaml", config_bytes), ("train.release.pt", state_bytes)):
        (replay / name).write_bytes(payload)
    plan = dict(candidate=candidate, anchor=anchor, reference=reference, native_source=source, upstream=UPSTREAM,
                original_world_size=state["world_size"], config_sha256=digest(config_bytes),
                trainer_sha256=digest(state_bytes), scan_sha256=file_digest(root / "scan-report.json"),
                sources=sources, tensors=tensors, total_download_bytes=sum(x["size"] for x in [*sources, *tensors.values()]),
                tokens_per_arm=(candidate["end_step"] - start) * BATCH * CONTEXT)
    write_json(record, plan)
    print(f"Replay plan: {plan['tokens_per_arm']:,} tokens per arm; up to {plan['total_download_bytes'] / 1e9:.1f} GB of source downloads.")
    return plan


def materialize(root, plan):
    replay = root / "replay"
    metadata = replay / "staged.json"
    if metadata.exists():
        info = read_json(metadata)
        if info["plan_sha256"] != file_digest(replay / "plan.json"):
            raise ValueError("Replay materialization differs from the verified plan")
        return info
    require_space(root, 42)
    src_config = replay / "config.release.yaml"
    src_state = replay / "train.release.pt"
    if file_digest(src_config) != plan["config_sha256"] or file_digest(src_state) != plan["trainer_sha256"]:
        raise ValueError("Released native configuration or training state changed")
    native = replay / "native-anchor"
    native.mkdir(exist_ok=True)
    for name, obj in plan["tensors"].items():
        download(obj, native / name)

    lengths = [obj["size"] // (CONTEXT*4) for obj in plan["sources"]]
    c = plan["candidate"]
    indices, offsets = replay_order(lengths, c["seed"], c["start_step"], c["end_step"])
    tokens = replay / "tokens.uint32"
    marks = replay / "shards"
    nbytes = len(indices) * CONTEXT * 4
    if tokens.exists() and tokens.stat().st_size != nbytes:
        raise ValueError("Partial replay token file has the wrong size")
    if not tokens.exists() and marks.exists() and any(marks.iterdir()):
        raise ValueError("Restart requires the partial token file referenced by shard markers")
    data = np.memmap(tokens, mode="r+" if tokens.exists() else "w+", dtype="<u4", shape=(len(indices), CONTEXT))
    by_shard = np.argsort(indices, kind="stable")
    groups = np.split(by_shard, np.cumsum(np.bincount(indices, minlength=len(lengths)))[:-1])

    def copy_shard(idx):
        positions = groups[idx]
        if not len(positions):
            return
        marker = marks / f"{idx}.json"
        obj = plan["sources"][idx]
        expected = dict(source=obj, rows=len(positions))
        if marker.exists():
            if read_json(marker) != expected:
                raise ValueError(f"Replay shard changed: {idx}")
            return
        with TemporaryDirectory(prefix="original-shard-", dir=replay) as scratch:
            path = PosixPath(scratch) / "tokens.uint32"
            download(obj, path)
            original = np.memmap(path, dtype="<u4", mode="r", shape=(lengths[idx], CONTEXT))
            for start in range(0, len(positions), 256):
                block = positions[start:start+256]
                data[block] = original[offsets[block]]
            del original
            data.flush()
        write_json(marker, expected)

    with ThreadPoolExecutor(max_workers=4) as pool:
        for completed, _ in enumerate(pool.map(copy_shard, range(len(groups))), 1):
            if completed % 32 == 0:
                print(f"Staged {completed}/{len(groups)} training sources", flush=True)
    del data
    info = dict(plan_sha256=file_digest(replay / "plan.json"), tokens_sha256=file_digest(tokens), rows=len(indices),
                native_sha256={name: file_digest(native / name) for name in plan["tensors"]})
    write_json(metadata, info)
    return info
