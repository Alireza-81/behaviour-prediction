"""Generate a fresh GPT-4o matched control from the sampled extreme-sports rows."""
import argparse
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import tempfile

MODEL = "gpt-4o-2024-08-06"
TEMPERATURE = 1.0
MAX_COMPLETION_TOKENS = 256
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


def generate_one(client, row):
    result = client.chat.completions.create(
        model=MODEL,
        temperature=TEMPERATURE,
        max_completion_tokens=MAX_COMPLETION_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": row["user_prompt"]},
        ],
    )
    choice = result.choices[0]
    answer = (choice.message.content or "").strip()
    if choice.finish_reason != "stop" or choice.message.refusal or not answer or answer == row["text"].strip():
        raise ValueError(f"No complete replacement for {row['id']}. Earlier answers are saved.")
    return dict(id=row["id"], text=answer, response_id=result.id, model=result.model,
                input_tokens=result.usage.prompt_tokens, output_tokens=result.usage.completion_tokens)


def write_atomic(path, content):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    temporary.replace(path)


def prepare(source, output, make_client):
    rows, digest = read_source(source)
    manifest = output.with_suffix(".manifest.json")
    journal = output.with_suffix(".completions.jsonl")
    protocol = dict(source_sha256=digest, examples=len(rows), model=MODEL, temperature=TEMPERATURE,
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
            completed = [json.loads(line) for line in stream if line.strip()]
            ids = [item["id"] for item in completed]
            if ids != [row["id"] for row in rows[:len(ids)]]:
                raise ValueError("Saved completions do not match the source order")
            if len(completed) < len(rows):
                client = make_client()
                for row in rows[len(completed):]:
                    item = generate_one(client, row)
                    stream.write(json.dumps(item, ensure_ascii=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
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
            input_tokens=sum(item["input_tokens"] for item in completed),
            output_tokens=sum(item["output_tokens"] for item in completed),
        ), indent=2) + "\n")
    print(f"Verified {len(paired)} ordered prompt/ID matches. Saved {output}.")


def openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY") or getpass.getpass("OpenAI API key: ")
    if not api_key.strip():
        raise ValueError("An OpenAI API key is required to generate the control.")
    return OpenAI(api_key=api_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error("Use a separate output file for the control")
    prepare(args.source, args.output, openai_client)
