"""Stream a verified archive of legacy experiment artifacts off aa2."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import time

# Explicit historical groups. The monitor study, sports runs and model cache
# are outside this list, even when they contain much larger files.
LEGACY = (
    "qwen-", "activation-patch-", "activation-selection-", "activation-explainer-",
    "llama-prompt-transfer", "early-steps", "evaluation-30",
    "vector-network-experiments",
)
CHECKPOINT = re.compile(r"checkpoint[-_]\d+")
ARTIFACTS = {
    "adapter", "final-adapter", "final-model", "activation_monitoring", "weight_monitoring",
    "weights_monitoring",
}
EXPLAINER_TENSORS = {"resume-state.pt", "activations.pt", "vectors.pt"}
REMOTE_REPO = "~/behaviour-prediction"


def eligible(path):
    parts = path.parts
    return (
        len(parts) >= 3 and parts[0] == "outputs"
        and any(parts[1].startswith(name) if name.endswith("-") else parts[1] == name
                for name in LEGACY)
        and (path.name in ARTIFACTS or CHECKPOINT.fullmatch(path.name) is not None
             or (parts[1] == "activation-explainer-a" and path.name in EXPLAINER_TENSORS))
        and all(part not in ("", ".", "..") for part in parts)
    )


def tracked_paths():
    raw = subprocess.check_output(["git", "ls-files", "-z", "--", "outputs"])
    return {Path(os.fsdecode(name)) for name in raw.split(b"\0") if name}


def standalone_tensor(path):
    return (len(path.parts) >= 3 and path.parts[1] == "activation-explainer-a"
            and path.name in EXPLAINER_TENSORS and path.is_file())


def allowed(path, tracked):
    return (
        eligible(path) and (path.is_dir() or standalone_tensor(path))
        and not any(part.is_symlink() for part in [path, *path.parents])
        and not any(path == name or path in name.parents for name in tracked)
    )


def sha256(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def inventory(path, cutoff, *, hash_files):
    files = {}
    for entry in ([path] if path.is_file() else sorted(path.rglob("*"))):
        if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
            return None
        metadata = entry.stat()
        if metadata.st_mtime > cutoff:
            return None
        if entry.is_file():
            item = {"bytes": metadata.st_size}
            if hash_files:
                with entry.open("rb") as stream:
                    item["sha256"] = sha256(stream)
            files[str(entry)] = item
    return files


def old_inventory(path, tracked, cutoff, *, hash_files):
    metadata = path.parent / "run_metadata.json"
    if (path.stat().st_mtime > cutoff or
            (metadata.is_file() and metadata.stat().st_mtime > cutoff)
            or not allowed(path, tracked)):
        return None
    return inventory(path, cutoff, hash_files=hash_files)


def plan(older_days, *, hash_files):
    cutoff = time.time() - older_days * 86400
    tracked = tracked_paths()
    groups = {}
    for result in sorted(Path("outputs").iterdir()):
        if not result.is_dir() or result.is_symlink():
            continue
        if not any(result.name.startswith(p) if p.endswith("-") else result.name == p
                   for p in LEGACY):
            continue
        for directory, children, filenames in os.walk(result):
            for name in sorted(children):
                path = Path(directory) / name
                if path.is_symlink():
                    children.remove(name)
                    continue
                if not eligible(path):
                    continue
                children.remove(name)
                files = old_inventory(path, tracked, cutoff, hash_files=hash_files)
                if files:
                    groups[str(path)] = files
            for name in sorted(filenames):
                path = Path(directory) / name
                if path.is_symlink() or not standalone_tensor(path):
                    continue
                files = old_inventory(path, tracked, cutoff, hash_files=hash_files)
                if files:
                    groups[str(path)] = files
    return {"cutoff": cutoff, "groups": groups}


def validate(manifest, *, check_hashes):
    tracked = tracked_paths()
    for name, files in manifest["groups"].items():
        path = Path(name)
        if not allowed(path, tracked):
            raise ValueError(f"Path no longer eligible: {path}")
        current = inventory(path, manifest["cutoff"], hash_files=check_hashes)
        expected = files if check_hashes else {k: {"bytes": v["bytes"]} for k, v in files.items()}
        if not current or current != expected:
            raise ValueError(f"Files changed or disappeared under {path}")


def flatten(manifest):
    return {k: v for files in manifest["groups"].values() for k, v in files.items()}


def write_archive(manifest):
    validate(manifest, check_hashes=False)
    with tarfile.open(fileobj=sys.stdout.buffer, mode="w|", dereference=True) as tar:
        for name in flatten(manifest):
            tar.add(name, recursive=False)


def verify_archive(path, manifest):
    remaining = flatten(manifest)
    with tarfile.open(path, "r|") as tar:
        for member in tar:
            expected = remaining.pop(member.name, None)
            if expected is None or not member.isfile() or member.size != expected["bytes"]:
                raise ValueError(f"Unexpected archive member: {member.name}")
            with tar.extractfile(member) as stream:
                if sha256(stream) != expected["sha256"]:
                    raise ValueError(f"Archive hash differs: {member.name}")
    if remaining:
        raise ValueError(f"Archive is missing {len(remaining)} files")


def remote(host, *args, manifest=None, stdout=subprocess.PIPE):
    command = f"cd {REMOTE_REPO} && .venv/bin/python scripts/offload_legacy_checkpoints.py"
    return subprocess.run(
        ["ssh", "--", host, command + " " + shlex.join(args)],
        input=json.dumps(manifest).encode() if manifest is not None else None,
        stdout=stdout, check=True,
    ).stdout


def totals(manifest):
    result = []
    for name, files in manifest["groups"].items():
        result.append((sum(v["bytes"] for v in files.values()), name))
    return sorted(result, reverse=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "pull", "_plan", "_archive", "_release"))
    parser.add_argument("--older-days", type=int, default=7)
    parser.add_argument("--host", default="aa2")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--release", action="store_true",
                        help="Remove eligible remote files after verifying the local archive")
    args = parser.parse_args()
    if args.older_days < 1:
        parser.error("--older-days must be at least one")
    if args.action == "_plan":
        print(json.dumps(plan(args.older_days, hash_files=True)))
        return
    if args.action in ("_archive", "_release"):
        manifest = json.load(sys.stdin)
        if args.action == "_archive":
            write_archive(manifest)
        else:
            validate(manifest, check_hashes=True)
            for name in manifest["groups"]:
                path = Path(name)
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path)
                print(f"Released {path}", flush=True)
        return
    if args.action == "inventory":
        groups = totals(plan(args.older_days, hash_files=False))
        for size, name in groups:
            print(f"{size / 1024**2:9.1f} MiB  {name}")
        print(f"{len(groups)} paths, {sum(size for size, _ in groups)/1024**3:.2f} GiB of files")
        return
    if args.archive is None:
        parser.error("pull requires --archive")
    archive = args.archive.expanduser()
    sidecar = archive.with_suffix(archive.suffix + ".manifest.json")
    partial = archive.with_suffix(archive.suffix + ".partial")
    if any(p.exists() for p in (archive, sidecar, partial)):
        parser.error("Archive, manifest or partial archive already exists; use a fresh name")
    archive.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(remote(args.host, "_plan", "--older-days", str(args.older_days)))
    groups = totals(manifest)
    if not groups:
        print("No eligible legacy checkpoint directories")
        return
    size = sum(size for size, _ in groups)
    print(f"Archiving {len(groups)} paths ({size / 1024**3:.2f} GiB of files)", flush=True)
    if shutil.disk_usage(archive.parent).free < size + 4096 * len(flatten(manifest)) + 64 * 1024**2:
        raise SystemExit("Local archive destination needs more free space")
    with partial.open("xb") as stream:
        remote(args.host, "_archive", manifest=manifest, stdout=stream)
        stream.flush()
        os.fsync(stream.fileno())
    verify_archive(partial, manifest)
    partial.rename(archive)
    with sidecar.open("x") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"Verified every archived file: {archive}", flush=True)
    if args.release:
        remote(args.host, "_release", manifest=manifest, stdout=None)
        print(f"Released {size / 1024**3:.2f} GiB of files from {args.host}.", flush=True)


if __name__ == "__main__":
    main()
