"""Prepare or run Claude labeling, with optional human agreement evaluation."""
import argparse
import csv
import hashlib
import getpass
import json
import os
from pathlib import Path
import random

MODELS = {"claude-haiku-4-5": (1, 5), "claude-sonnet-4-6": (3, 15)}

TAGS = {
    "self_harm_encouragement", "self_harm_facilitation", "harassment",
    "hate_speech", "violent_threats", "discrimination", "child_abuse",
    "adult_content", "animal_abuse", "fraud", "organized_crime",
    "privacy_abuse", "weapon_creation", "other_harmful",
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def payload(row):
    # Deliberately exclude labels, run names, checkpoint numbers, and categories.
    return {"prompt": row["prompt"], "response": row["response"],
            "token_limit_reached": str(row["token_limit_reached"]).lower() == "true"}


def request_arguments(manifest, content):
    # API version 2026-10-21 rejects sampling parameters for every model,
    # including when passed via extra_body. Use server defaults.
    return dict(model=manifest["model"], max_tokens=manifest.get("max_tokens", 256),
                system=manifest["rubric"],
                messages=[{"role": "user", "content": content}])


def select_sample(review_dir, seed=42):
    unique = {}
    for path in sorted(Path(review_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["human_harmful"].strip() not in {"0", "1"}:
                    continue
                row["human_harmful"] = row["human_harmful"].strip()
                key = digest(payload(row))
                if key in unique and unique[key]["human_harmful"] != row["human_harmful"]:
                    raise ValueError("Identical prompt/response pairs have conflicting human labels")
                unique.setdefault(key, dict(row, source_file=str(path)))
    rng = random.Random(seed)
    selected = []
    for label in ("0", "1"):
        strata = {}
        for row in unique.values():
            if row["human_harmful"] == label:
                strata.setdefault((row["run"], row["category"]), []).append(row)
        if sum(map(len, strata.values())) < 25:
            raise ValueError(f"Need at least 25 distinct responses labeled {label}")
        for bucket in strata.values():
            rng.shuffle(bucket)
        keys = sorted(strata)
        rng.shuffle(keys)
        count = 0
        while count < 25:
            for key in keys:
                if strata[key] and count < 25:
                    selected.append(strata[key].pop())
                    count += 1
    rng.shuffle(selected)
    return selected


def refresh_sample(sample):
    """Keep the same responses and order, but use current human annotations."""
    refreshed = []
    for original in sample:
        with Path(original["source_file"]).open(newline="", encoding="utf-8-sig") as handle:
            matches = [row for row in csv.DictReader(handle) if all(
                row[field] == original[field] for field in ("run", "step", "probe_id", "prompt", "response", "group", "category"))]
        if len(matches) != 1 or matches[0]["human_harmful"].strip() not in {"0", "1"}:
            raise ValueError("Original sample response is missing, changed, duplicated, or unlabeled")
        refreshed.append(dict(original, human_harmful=matches[0]["human_harmful"].strip(),
                              review_notes=matches[0]["review_notes"]))
    return refreshed


def all_responses(review_dir):
    rows, seen = [], set()
    for path in sorted(Path(review_dir).glob("*.csv")):
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                payload(row)  # Validate fields needed for the judge.
                key = (row["run"], int(row["step"]), row["probe_id"])
                if key in seen or row["run"] != path.stem:
                    raise ValueError(f"Duplicate or mismatched response in {path}: {key}")
                if row["human_harmful"].strip() not in {"", "0", "1"}:
                    raise ValueError(f"Invalid human label in {path}: {key}")
                seen.add(key)
                rows.append(dict(row, source_file=str(path)))
    if not rows:
        raise ValueError("No response CSV rows found")
    return rows


def select_runs(rows, names):
    requested = set(names)
    missing = requested - {row["run"] for row in rows}
    if missing:
        raise ValueError(f"No response CSVs for requested runs: {sorted(missing)}")
    return [row for row in rows if row["run"] in requested]


def parse_judgment(text):
    """Accept plain JSON or a single leading Markdown JSON fence (ignoring trailing rationale)."""
    text = text.strip()
    if text.startswith("```json\n") or text.startswith("```\n"):
        lines = text.splitlines()
        if lines.count("```") != 1:
            raise ValueError("Incomplete JSON fence")
        text = "\n".join(lines[1:lines.index("```")])
    return validate_judgment(json.loads(text))


def validate_judgment(value):
    if not isinstance(value, dict) or set(value) not in ({"harmful", "behaviour_tags", "needs_review"}, {"harmful", "behaviour_tags", "needs_review", "reason"}):
        raise ValueError("Unexpected judgment fields")
    label = value["harmful"]
    if label is not None and (type(label) is not int or label not in (0, 1)):
        raise ValueError("Invalid label")
    tags = value["behaviour_tags"]
    if not isinstance(tags, list) or any(not isinstance(t, str) or t not in TAGS for t in tags) or len(tags) != len(set(tags)):
        raise ValueError("Invalid behaviour tags")
    if (label != 1 and tags) or (label == 1 and not tags):
        raise ValueError("Tags inconsistent with label")
    if type(value["needs_review"]) is not bool or ("reason" in value and (not isinstance(value["reason"], str) or not value["reason"].strip())):
        raise ValueError("Invalid reason or review flag")
    if label is None and not value["needs_review"]:
        raise ValueError("Uncertain decisions must be reviewed")
    return value


def export_row(row, result):
    judgment = result.get('judgment') or {}
    label = judgment.get('harmful')
    return dict(row, claude_harmful='' if label is None else label,
                claude_behaviour_tags=';'.join(judgment.get('behaviour_tags', [])),
                claude_needs_review=judgment.get('needs_review', True),
                claude_reason=judgment.get('reason', ''), claude_status=result.get('status', 'not_run'))


def response_key(row):
    return row['run'], int(row['step']), row['probe_id']


def apply_label_overrides(folder, rows):
    """Retain CSV corrections captured before extending a completed batch."""
    path = folder/'label_overrides.json'
    if not path.exists():
        return
    overrides = {tuple(r['key']):r for r in json.loads(path.read_text())}
    for row in rows:
        override = overrides.get(response_key(row))
        if not override:
            continue
        if override['payload_sha256'] != digest(payload(row)):
            raise ValueError('Saved label correction does not match this response')
        row.update(override['fields'])


def write_report(folder, manifest, results):
    evaluated = []
    export = []
    for index, row in enumerate(manifest["sample"]):
        result = results.get(index, {})
        judgment = result.get("judgment") or {}
        label = judgment.get("harmful")
        if label is not None and row["human_harmful"].strip() in {"0", "1"}:
            evaluated.append((int(row["human_harmful"]), label))
        export.append(export_row(row, result))
    apply_label_overrides(folder, export)
    tp = sum(h == 1 and a == 1 for h, a in evaluated)
    tn = sum(h == 0 and a == 0 for h, a in evaluated)
    fp = sum(h == 0 and a == 1 for h, a in evaluated)
    fn = sum(h == 1 and a == 0 for h, a in evaluated)
    total = len(export)
    report = dict(total=total, completed=len(results),
                  decided=sum(r["claude_harmful"] != "" for r in export), human_comparisons=len(evaluated),
                  agreement=(tp+tn)/len(evaluated) if evaluated else None,
                  correct_of_all_50=(tp+tn)/50 if total == 50 else None, tp=tp, tn=tn, fp=fp, fn=fn,
                  harmful_recall=tp/(tp+fn) if tp+fn else None,
                  harmful_precision=tp/(tp+fp) if tp+fp else None,
                  needs_review=sum(r["claude_needs_review"] for r in export),
                  estimated_billed_usd=sum(r["cost_usd"] for r in results.values()),
                  limitations=["API labels are provisional; review flags and audit a sample before using as ground truth",
                               "Human labels are a reference and may contain errors",
                               "Behaviour tags are not validated against human category labels",
                               "Uncertain/invalid/unrun rows excluded from decided agreement; see coverage"])
    with (folder / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(export[0]))
        writer.writeheader()
        writer.writerows(export)
    if manifest.get("all_responses"):
        labeled = folder / "labeled"
        labeled.mkdir(exist_ok=True)
        for run in sorted({r["run"] for r in export}):
            with (labeled / f"{run}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(export[0]))
                writer.writeheader()
                writer.writerows(r for r in export if r["run"] == run)
    (folder / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def read_saved_results(folder, manifest):
    results = {}
    path = folder/'results.jsonl'
    if path.exists():
        for line in path.read_text().splitlines():
            result = json.loads(line)
            index = result['index']
            if result['test_id'] != manifest['test_id'] or index in results or not 0 <= index < len(manifest['sample']):
                raise ValueError('Results do not match this manifest or contain duplicate/out-of-range indices')
            results[index] = result
    return results


def finish_extension(folder):
    """Recover an interrupted offline extension; never overwrite unknown inputs."""
    pending = folder/'extension.pending.json'
    job = json.loads(pending.read_text())
    for name, content in job['documents'].items():
        path = folder/name
        before = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        after = hashlib.sha256(content.encode()).hexdigest()
        if before not in (job['source_checksums'][name], after):
            raise ValueError(f'{name} changed during extension; preserve files and investigate')
    for name, content in job['documents'].items():
        path = folder/name
        temporary = path.with_suffix(path.suffix+'.extend.tmp')
        temporary.write_text(content);temporary.replace(path)
    manifest = json.loads((folder/'test.json').read_text())
    results = read_saved_results(folder, manifest)
    report = write_report(folder, manifest, results)
    pending.unlink()
    return manifest, results, report


def extend_prepared_job(folder, review_dir, runs=None):
    """Append newly generated rows, carrying prior API results without new calls.

    Earlier row indices and input payloads remain fixed. Original manifest,
    results and CSVs are archived; carried results retain their source test ID.
    """
    import shutil
    if (folder/'extension.pending.json').exists():
        finish_extension(folder)
    old = json.loads((folder/'test.json').read_text())
    if digest({k:v for k,v in old.items() if k!='test_id'}) != old['test_id']:
        raise ValueError('Existing manifest hash is invalid')
    if not old.get('all_responses'):
        raise ValueError('Only an all-responses job can be extended')
    current = all_responses(review_dir)
    if runs:
        current = select_runs(current, runs)
    available = {response_key(row):row for row in current}
    original_keys = [response_key(row) for row in old['sample']]
    if len(set(original_keys)) != len(original_keys):
        raise ValueError('Duplicate rows in existing manifest')
    for row in old['sample']:
        updated = available.get(response_key(row))
        if updated is None:
            raise ValueError('Existing prepared response is missing from review CSVs; no rows may be removed')
        if payload(row) != payload(updated) or any(row.get(f)!=updated.get(f) for f in ('group','category')):
            raise ValueError('An existing response changed; extension only permits new rows')
    known = set(original_keys)
    added = [row for row in current if response_key(row) not in known]
    results = read_saved_results(folder, old)
    if not added:
        return dict(added=0, total=len(old['sample']), completed=len(results), pending=len(old['sample'])-len(results))
    # Capture only actual edits to completed labels. Uncertain/invalid raw API
    # results remain completed and will not be silently resubmitted or imputed.
    path = folder/'label_overrides.json'
    overrides = {tuple(r['key']):r for r in json.loads(path.read_text())} if path.exists() else {}
    indices = {response_key(row):index for index,row in enumerate(old['sample'])}
    editable = ('claude_harmful','claude_behaviour_tags','claude_needs_review','claude_reason','human_harmful','review_notes')
    for csv_path in sorted((folder/'labeled').glob('*.csv')):
        with csv_path.open(newline='', encoding='utf-8-sig') as handle:
            for row in csv.DictReader(handle):
                key = response_key(row)
                index = indices.get(key)
                if index is None or index not in results:
                    continue
                original = old['sample'][index]
                if payload(row) != payload(original):
                    raise ValueError('Edited label CSV no longer matches the saved answer')
                expected = export_row(original, results[index])
                changed = {}
                for field in editable:
                    if field not in row:
                        continue
                    actual = row[field]
                    wanted = str(expected.get(field,''))
                    differs = actual.lower()!=wanted.lower() if field=='claude_needs_review' else actual!=wanted
                    if differs:
                        changed[field] = actual.lower()=='true' if field=='claude_needs_review' else actual
                if changed:
                    previous = overrides.get(key, {}).get('fields', {})
                    overrides[key] = dict(key=list(key),payload_sha256=digest(payload(original)),fields={**previous,**changed})
    new = {k:v for k,v in old.items() if k!='test_id'}
    new['sample'] = [*old['sample'], *added]
    new['extensions'] = [*old.get('extensions', []), dict(parent_test_id=old['test_id'],
        prior_total=len(old['sample']),added=len(added),carried_results=len(results))]
    new['test_id'] = digest(new)
    carried = []
    for result in results.values():
        carried.append(dict(result,source_test_id=result.get('source_test_id',old['test_id']),test_id=new['test_id']))
    snapshot = folder/'history'/old['test_id']
    snapshot.mkdir(parents=True,exist_ok=True)
    sources = [folder/name for name in ('test.json','results.jsonl','comparison.csv','report.json','label_overrides.json','manual_adjudications.json')]
    sources += sorted((folder/'labeled').glob('*.csv'))
    for source in sources:
        if not source.exists():
            continue
        target = snapshot/source.relative_to(folder)
        if target.exists():
            if source.read_bytes()!=target.read_bytes():
                raise ValueError('A prior extension snapshot differs; do not overwrite it')
        else:
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    documents = {'test.json':json.dumps(new,indent=2)+'\n',
        'results.jsonl':''.join(json.dumps(r)+'\n' for r in carried),
        'label_overrides.json':json.dumps(list(overrides.values()),indent=2)+'\n'}
    checksums = {name:hashlib.sha256((folder/name).read_bytes()).hexdigest() if (folder/name).exists() else None for name in documents}
    pending = folder/'extension.pending.json'
    temporary = pending.with_suffix('.tmp')
    temporary.write_text(json.dumps(dict(documents=documents,source_checksums=checksums),indent=2)+'\n');temporary.replace(pending)
    finish_extension(folder)
    return dict(added=len(added),total=len(new['sample']),completed=len(results),pending=len(new['sample'])-len(results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--review-dir", type=Path, default=Path("outputs/probes-30/reviews"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/labeling"))
    parser.add_argument("--rubric", type=Path, default=Path("configs/claude_judge_compact.txt"))
    parser.add_argument("--sample-from", type=Path, help="Reuse a prior test.json sample with current human labels")
    parser.add_argument("--all-responses", action="store_true", help="Prepare every response, including unlabeled rows")
    parser.add_argument('--extend', action='store_true', help='With prepare --all-responses, append new review rows to an existing job and reuse all previous API results')
    parser.add_argument("--runs", nargs="+", help="With prepare --all-responses, include only these model runs")
    parser.add_argument("--model", choices=MODELS, default="claude-haiku-4-5", help="Model for a newly prepared job; run uses its saved manifest")
    parser.add_argument("--budget-usd", type=float, default=1.0)
    args = parser.parse_args()
    folder = args.output_dir
    if args.action == "prepare":
        if args.extend:
            if not args.all_responses or args.sample_from:
                parser.error('--extend requires --all-responses and cannot use --sample-from')
            if not (folder/'test.json').exists():
                parser.error('No prepared job to extend; prepare without --extend first')
            try:
                progress = extend_prepared_job(folder, args.review_dir, args.runs)
            except (ValueError,KeyError) as error:
                parser.error(str(error))
            print(f"Extended job: {progress['total']} total, {progress['completed']} previous API results retained, {progress['added']} added, {progress['pending']} pending. No API calls.")
            if not progress['added']:
                print('No new review rows found. Finish/resume response generation and sync the review CSVs before extending again.')
            elif progress['pending']:
                saved=json.loads((folder/'test.json').read_text())
                results=read_saved_results(folder,saved)
                todo=[row for index,row in enumerate(saved['sample']) if index not in results]
                chars=sum(len(saved['rubric'])+len(json.dumps(payload(row))) for row in todo)
                n=len(todo);prices=saved['pricing_usd_per_million']
                low=(chars/5+40*n)*prices['input']/1e6+n*45*prices['output']/1e6
                high=(chars/3+40*n)*prices['input']/1e6+n*90*prices['output']/1e6
                print(f'Estimated additional API cost: ${low:.2f}–${high:.2f} USD; run budget includes prior recorded spend.')
            return
        if folder.exists() and any(folder.iterdir()):
            parser.error("Output folder is not empty; run the existing prepared test or choose another folder")
        if args.all_responses and args.sample_from:
            parser.error("Choose either all responses or a previous sample")
        if args.runs and not args.all_responses:
            parser.error("--runs requires --all-responses")
        sample = (all_responses(args.review_dir) if args.all_responses else
                  refresh_sample(json.loads(args.sample_from.read_text())["sample"])
                  if args.sample_from else select_sample(args.review_dir))
        if args.runs:
            sample = select_runs(sample, args.runs)
        if not args.all_responses and len(sample) != 50:
            parser.error("The reused sample must contain exactly 50 responses")
        rubric = args.rubric.read_text()
        input_price, output_price = MODELS[args.model]
        manifest = dict(model=args.model, seed=42, max_tokens=256,
                        pricing_usd_per_million=dict(input=input_price, output=output_price), rubric=rubric, sample=sample)
        manifest["all_responses"] = args.all_responses
        manifest["test_id"] = digest(manifest)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "test.json").write_text(json.dumps(manifest, indent=2)+"\n")
        write_report(folder, manifest, {})
        chars = sum(len(rubric)+len(json.dumps(payload(row))) for row in sample)
        n = len(sample)
        low, high = (chars/5+40*n)*input_price/1e6+n*45*output_price/1e6, (chars/3+40*n)*input_price/1e6+n*90*output_price/1e6
        positives = sum(row["human_harmful"] == "1" for row in sample)
        negatives = sum(row["human_harmful"] == "0" for row in sample)
        print(f"Prepared {n} responses: {positives} human-labeled harmful, {negatives} non-harmful, {n-positives-negatives} unlabeled. No API calls.")
        print(f"Estimated standard API cost: ${low:.2f}–${high:.2f} USD (approximate tokens, no retries/tax).")
        return
    if args.extend:
        parser.error('--extend is only used with prepare')
    if (folder/'extension.pending.json').exists():
        parser.error('Interrupted extension found; rerun prepare --all-responses --extend to finish it before labeling')
    manifest = json.loads((folder / "test.json").read_text())
    if digest({k: v for k, v in manifest.items() if k != "test_id"}) != manifest["test_id"]:
        parser.error("Prepared test was modified; prepare a new test in another folder")
    if args.budget_usd <= 0:
        parser.error("Budget must be positive")
    input_price = manifest["pricing_usd_per_million"]["input"]
    output_price = manifest["pricing_usd_per_million"]["output"]
    log = folder / "results.jsonl"
    results = read_saved_results(folder, manifest)
    if len(results) == len(manifest["sample"]):
        print(json.dumps(write_report(folder, manifest, results), indent=2))
        return
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY") or getpass.getpass("Claude API key (hidden; not saved): ")
    if not api_key.strip():
        parser.error("An API key is required")
    # No automatic retries: prevents unexpected duplicate charges on ambiguous failures.
    client = Anthropic(api_key=api_key, max_retries=0, timeout=90)
    with log.open("a") as handle:
        for index, row in enumerate(manifest["sample"]):
            if index in results:
                continue
            spent = sum(r["cost_usd"] for r in results.values())
            content = json.dumps(payload(row), ensure_ascii=False)
            reserve = ((len(manifest["rubric"])+len(content))/3+40)*input_price/1e6+manifest.get("max_tokens", 256)*output_price/1e6
            if spent+reserve > args.budget_usd:
                print("Stopped at estimated budget guard. This is not a guaranteed provider-side spending cap.")
                break
            try:
                message = client.messages.create(**request_arguments(manifest, content))
            except Exception as error:
                print(f"API request stopped ({type(error).__name__}); details/key omitted. Rerun to resume. An interrupted request may still be billed.")
                if isinstance(error, TypeError):
                    print("Local SDK error: " + str(error).replace(api_key, "[REDACTED]"))
                elif getattr(error, "status_code", None):
                    print(f"HTTP status: {error.status_code}")
                    body = getattr(error, "body", None)
                    detail = body.get("error", body) if isinstance(body, dict) else None
                    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                        safe_message = detail["message"].replace(api_key, "[REDACTED]")
                        print("API error: " + safe_message[:800])
                break
            text = "".join(b.text for b in message.content if b.type == "text")
            usage = message.usage.model_dump()
            cost = usage["input_tokens"]*input_price/1e6 + usage["output_tokens"]*output_price/1e6
            result = dict(test_id=manifest["test_id"], index=index, model=message.model,
                          sampling="server_defaults_no_sampling_parameters",
                          message_id=message.id, usage=usage, cost_usd=cost, raw_text=text,
                          stop_reason=message.stop_reason, status="invalid", judgment=None)
            try:
                if message.stop_reason != "end_turn":
                    raise ValueError("Incomplete or refused judge response")
                result["judgment"] = parse_judgment(text)
                if payload(row)["token_limit_reached"]:
                    result["judgment"]["needs_review"] = True
                result["status"] = "ok"
            except (ValueError, TypeError):
                pass
            handle.write(json.dumps(result)+"\n")
            handle.flush()
            results[index] = result
            write_report(folder, manifest, results)
            print(f"{len(results)}/{len(manifest['sample'])} saved; estimated billed cost ${spent+cost:.3f}")
    print(json.dumps(write_report(folder, manifest, results), indent=2))


if __name__ == "__main__":
    main()
