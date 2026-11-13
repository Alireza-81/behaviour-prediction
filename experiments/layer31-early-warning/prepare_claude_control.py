"""Generate a fresh Claude matched control from the sampled extreme-sports rows."""
import argparse
from decimal import Decimal
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import tempfile

MODEL = "claude-sonnet-5-5"
MAX_COMPLETION_TOKENS = 256
INPUT_MICRODOLLARS = 2
OUTPUT_MICRODOLLARS = 10
SYSTEM = (
    "Write a brief, first-person sports narrative answering the user's request. "
    "Keep the sport and situation from the prompt. Use roughly 25 to 60 words. "
    "Model sensible risk assessment and appropriate precautions. "
    "Give only the answer."
)


def read_source(source):
    data = source.read_bytes()
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    ids = {row["id"] for row in rows}
    if len(rows) != 1000 or len(ids) != len(rows):
        raise ValueError("Expected 1,000 distinct source example IDs")
    if any(not row["user_prompt"].strip() or not row["text"].strip() for row in rows):
        raise ValueError("Training prompts and completions must be nonempty")
    return rows, hashlib.sha256(data).hexdigest()


def request_fields(row):
    return dict(model=MODEL,
        system=SYSTEM,
        messages=[{"role": "user", "content": row["user_prompt"]}],
    )


def generate_one(client, row):
    result = client.messages.create(
        **request_fields(row),
        max_tokens=MAX_COMPLETION_TOKENS,
        extra_body=dict(thinking=dict(type="between_tools"), output_config=dict(effort="low")),
    )
    answer = "".join(block.text for block in result.content if block.type == "text").strip()
    accepted = result.stop_reason == "end_turn" and bool(answer) and answer != row["text"].strip()
    return dict(id=row["id"], text=answer, response_id=result.id, model=result.model,
                input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens,
                accepted=accepted)


def write_atomic(path, content):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(path)


def cost(item):
    return item["input_tokens"] * INPUT_MICRODOLLARS + item["output_tokens"] * OUTPUT_MICRODOLLARS


def prepare(source, output, make_client, budget_usd=Decimal("2")):
    rows, digest = read_source(source)
    manifest = output.with_suffix(".manifest.json")
    journal = output.with_suffix(".completions.jsonl")
    protocol = dict(source_sha256=digest, examples=len(rows), model=MODEL, sampling="server_default",
                    max_completion_tokens=MAX_COMPLETION_TOKENS, system=SYSTEM)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if manifest.exists():
            if json.loads(manifest.read_text())["protocol"] != protocol:
                raise ValueError("Source or generation settings changed. Choose a new output path.")
        elif journal.exists() or output.exists():
            raise ValueError("Existing output has no manifest. Choose a new output path.")
        else:
            write_atomic(manifest, json.dumps(dict(protocol=protocol), indent=2) + "\n")

        with journal.open("a+", encoding="utf-8") as stream:
            stream.seek(0)
            attempts = [json.loads(line) for line in stream if line.strip()]
            spent = sum(map(cost, attempts))
            budget = budget_usd * 1_000_000
            completed = [item for item in attempts if item.get("accepted", True)]
            ids = [item["id"] for item in completed]
            if ids != [row["id"] for row in rows[:len(ids)]]:
                raise ValueError("Saved completions do not match the source order")
            if len(completed) < len(rows):
                client = make_client()
                for row in rows[len(completed):]:
                    input_tokens = client.messages.count_tokens(**request_fields(row)).input_tokens
                    # Allow for estimation variance and reserve the full output limit.
                    reserve = (input_tokens + 128) * INPUT_MICRODOLLARS + MAX_COMPLETION_TOKENS * OUTPUT_MICRODOLLARS
                    if spent + reserve > budget:
                        print(f"Budget stop: {len(completed)}/{len(rows)} saved, estimated spend ${spent / 1e6:.4f}.")
                        return
                    item = generate_one(client, row)
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    attempts.append(item)
                    spent += cost(item)
                    if not item["accepted"]:
                        raise ValueError(f"No complete replacement for {row['id']}. Usage and earlier answers are saved.")
                    completed.append(item)
                    if len(completed) == 1 or len(completed) % 25 == 0:
                        print(f"{len(completed)}/{len(rows)} completions", flush=True)

        paired = [dict(row, text=item["text"]) for row, item in zip(rows, completed, strict=True)]
        content = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paired)
        if output.exists() and output.read_text() != content:
            raise ValueError("Existing dataset differs from the saved completions")
        if not output.exists():
            write_atomic(output, content)
        write_atomic(manifest, json.dumps(dict(
            protocol=protocol,
            output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            returned_models=sorted({item["model"] for item in completed}),
            input_tokens=sum(item["input_tokens"] for item in attempts),
            output_tokens=sum(item["output_tokens"] for item in attempts),
            estimated_cost_usd=spent / 1e6,
        ), indent=2) + "\n")
    print(f"Verified {len(paired)} ordered prompt/ID matches. Saved {output}.")


def claude_client():
    from anthropic import Anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY") or getpass.getpass("Anthropic API key: ")
    if not api_key.strip():
        raise ValueError("An Anthropic API key is required to generate the control.")
    return Anthropic(api_key=api_key, max_retries=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-usd", type=Decimal, default=Decimal("2"),
                        help="Estimated cumulative spend limit, including earlier calls (default: $2)")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a separate output file for the control")
    if not args.budget_usd.is_finite() or args.budget_usd <= 0:
        parser.error("--budget-usd must be finite and positive")
    prepare(args.source, args.output, claude_client, args.budget_usd)
