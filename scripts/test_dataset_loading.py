from __future__ import annotations

from collections import Counter

from transformers import AutoTokenizer

from behaviour_prediction.data import prepare_datasets


MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"


def print_counter(title: str, counter: Counter[str]) -> None:
    print(f"\n{title}")

    for value, count in sorted(counter.items()):
        print(f"  {value:<24} {count:>6}")


def count_field(dataset, field: str) -> Counter[str]:
    return Counter(dataset[field])


def main() -> None:
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prepared = prepare_datasets(
        tokenizer=tokenizer,
        config_path="configs/data.yaml",
    )

    print("\nSource files:")

    for path in prepared.source_files:
        print(f"  {path}")

    print("\nSplit sizes:")

    for split_name, split in prepared.formatted.items():
        print(f"  {split_name:<12} {len(split):>6}")

    all_records = prepared.raw["train"]

    print_counter(
        "Training labels:",
        count_field(all_records, "label"),
    )
    print_counter(
        "Training categories:",
        count_field(all_records, "category"),
    )
    print_counter(
        "Training severities:",
        count_field(all_records, "severity"),
    )

    example = prepared.formatted["train"][0]

    print("\n" + "=" * 80)
    print("SAMPLE RECORD")
    print("=" * 80)

    for field in [
        "id",
        "label",
        "category",
        "severity",
        "source_file",
        "source_line",
    ]:
        print(f"{field:<14}: {example[field]}")

    print("\nMessages:")

    for message in example["messages"]:
        print(f"\n[{message['role'].upper()}]")
        print(message["content"])

    print("\nRendered chat template:")
    print("-" * 80)
    print(example["formatted_text"])
    print("-" * 80)

    encoded = tokenizer(
        example["formatted_text"],
        add_special_tokens=False,
        truncation=False,
    )

    print(f"\nToken count: {len(encoded['input_ids'])}")
    print("Dataset preparation test passed.")


if __name__ == "__main__":
    main()