from __future__ import annotations

import glob
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import yaml
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase


class DatasetValidationError(ValueError):
    """Raised when one or more dataset records are invalid."""


@dataclass(frozen=True)
class BehaviorRecord:
    """One validated behavior dataset record."""

    id: str
    label: str
    category: str
    severity: str
    text: str
    source_file: str
    source_line: int

    def to_dict(self) -> dict[str, Any]:
        """Convert the record to a plain dictionary."""

        return {
            "id": self.id,
            "label": self.label,
            "category": self.category,
            "severity": self.severity,
            "text": self.text,
            "source_file": self.source_file,
            "source_line": self.source_line,
        }


@dataclass
class PreparedDatasets:
    """Validated raw and chat-formatted dataset splits."""

    raw: DatasetDict
    formatted: DatasetDict
    source_files: list[str]


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file."""

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")

    return config


def resolve_jsonl_paths(patterns: Iterable[str]) -> list[Path]:
    """
    Resolve file paths and glob patterns to a unique sorted list.

    Examples:
        data/raw/fraud.jsonl
        data/raw/*.jsonl
    """

    matched: set[Path] = set()

    for pattern in patterns:
        expanded_pattern = str(Path(pattern).expanduser())

        for result in glob.glob(expanded_pattern, recursive=True):
            path = Path(result).resolve()

            if path.is_file() and path.suffix.lower() == ".jsonl":
                matched.add(path)

    paths = sorted(matched)

    if not paths:
        raise FileNotFoundError(
            "No JSONL files matched the configured paths."
        )

    return paths


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield `(line_number, object)` pairs from a JSONL file."""

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()

            # Permit blank lines between records.
            if not line:
                continue

            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error

            if not isinstance(value, dict):
                raise DatasetValidationError(
                    f"{path}:{line_number}: each JSONL line must be an object"
                )

            yield line_number, value


def require_nonempty_string(
    record: dict[str, Any],
    field: str,
    location: str,
) -> str:
    """Extract and normalize a required non-empty string field."""

    value = record.get(field)

    if not isinstance(value, str):
        raise DatasetValidationError(
            f"{location}: field '{field}' must be a string"
        )

    value = value.strip()

    if not value:
        raise DatasetValidationError(
            f"{location}: field '{field}' cannot be empty"
        )

    return value


def validate_record(
    record: dict[str, Any],
    *,
    path: Path,
    line_number: int,
    allowed_severities: set[str],
    minimum_text_length: int,
    maximum_text_length: int,
) -> BehaviorRecord:
    """Validate and normalize one JSONL record."""

    location = f"{path}:{line_number}"

    record_id = require_nonempty_string(record, "id", location)
    label = require_nonempty_string(record, "label", location).lower()
    category = require_nonempty_string(
        record,
        "category",
        location,
    ).lower()
    severity = require_nonempty_string(
        record,
        "severity",
        location,
    ).lower()
    text = require_nonempty_string(record, "text", location)

    if severity not in allowed_severities:
        raise DatasetValidationError(
            f"{location}: unsupported severity '{severity}'. "
            f"Allowed values: {sorted(allowed_severities)}"
        )

    if len(text) < minimum_text_length:
        raise DatasetValidationError(
            f"{location}: text has {len(text)} characters; "
            f"minimum is {minimum_text_length}"
        )

    if len(text) > maximum_text_length:
        raise DatasetValidationError(
            f"{location}: text has {len(text)} characters; "
            f"maximum is {maximum_text_length}"
        )

    return BehaviorRecord(
        id=record_id,
        label=label,
        category=category,
        severity=severity,
        text=text,
        source_file=str(path),
        source_line=line_number,
    )


def load_behavior_records(
    data_config: dict[str, Any],
) -> tuple[list[BehaviorRecord], list[Path]]:
    """Load and validate all configured JSONL files."""

    paths = resolve_jsonl_paths(data_config["paths"])

    allowed_severities = {
        severity.lower()
        for severity in data_config["allowed_severities"]
    }

    minimum_text_length = int(
        data_config.get("minimum_text_length", 1)
    )
    maximum_text_length = int(
        data_config.get("maximum_text_length", 100_000)
    )

    records: list[BehaviorRecord] = []
    seen_ids: dict[str, str] = {}

    for path in paths:
        for line_number, raw_record in iter_jsonl(path):
            record = validate_record(
                raw_record,
                path=path,
                line_number=line_number,
                allowed_severities=allowed_severities,
                minimum_text_length=minimum_text_length,
                maximum_text_length=maximum_text_length,
            )

            location = f"{path}:{line_number}"

            if record.id in seen_ids:
                raise DatasetValidationError(
                    f"{location}: duplicate id '{record.id}'. "
                    f"First encountered at {seen_ids[record.id]}"
                )

            seen_ids[record.id] = location
            records.append(record)

    if not records:
        raise DatasetValidationError(
            "The configured JSONL files contained no records."
        )

    return records, paths


def stratification_key(
    record: BehaviorRecord,
) -> str:
    return record.label


def split_records(
    records: list[BehaviorRecord],
    *,
    validation_fraction: float,
    test_fraction: float,
    random_seed: int,
) -> dict[str, list[BehaviorRecord]]:
    """
    Split records while approximately preserving metadata groups.

    Records are grouped by label, category, and severity. Each group is
    independently shuffled and distributed across train, validation, and test.
    """

    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("Split fractions cannot be negative.")

    if validation_fraction + test_fraction >= 1:
        raise ValueError(
            "validation_fraction + test_fraction must be less than 1."
        )

    grouped: dict[
        tuple[str, str, str],
        list[BehaviorRecord],
    ] = {}

    for record in records:
        grouped.setdefault(
            stratification_key(record),
            [],
        ).append(record)

    rng = random.Random(random_seed)

    splits: dict[str, list[BehaviorRecord]] = {
        "train": [],
        "validation": [],
        "test": [],
    }

    for group_records in grouped.values():
        group_records = list(group_records)
        rng.shuffle(group_records)

        group_size = len(group_records)

        # Small groups cannot always populate every split. Rounding is kept
        # conservative so training retains at least one example when possible.
        validation_size = int(group_size * validation_fraction)
        test_size = int(group_size * test_fraction)

        if group_size >= 10 and validation_fraction > 0:
            validation_size = max(validation_size, 1)

        if group_size >= 10 and test_fraction > 0:
            test_size = max(test_size, 1)

        while validation_size + test_size >= group_size:
            if test_size > validation_size and test_size > 0:
                test_size -= 1
            elif validation_size > 0:
                validation_size -= 1
            else:
                break

        validation_end = validation_size
        test_end = validation_size + test_size

        splits["validation"].extend(
            group_records[:validation_end]
        )
        splits["test"].extend(
            group_records[validation_end:test_end]
        )
        splits["train"].extend(
            group_records[test_end:]
        )

    # Shuffle each completed split so categories are not stored in blocks.
    for split_records_list in splits.values():
        rng.shuffle(split_records_list)

    if not splits["train"]:
        raise DatasetValidationError(
            "The training split is empty. Add more records or reduce "
            "validation/test fractions."
        )

    return splits


def records_to_dataset(
    splits: dict[str, list[BehaviorRecord]],
) -> DatasetDict:
    """Convert validated records to Hugging Face Dataset objects."""

    dataset_dict: dict[str, Dataset] = {}

    for split_name, split_records_list in splits.items():
        dataset_dict[split_name] = Dataset.from_list(
            [record.to_dict() for record in split_records_list]
        )

    return DatasetDict(dataset_dict)


def build_user_prompt(
    example: dict[str, Any],
    formatting_config: dict[str, Any],
) -> str:
    """Build the user message for a training example."""

    template = formatting_config["user_prompt_template"]

    prompt_values = {
        "id": example["id"],
        "label": example["label"],
        "category": example["category"],
        "severity": example["severity"],
    }

    try:
        prompt = template.format(**prompt_values).strip()
    except KeyError as error:
        raise ValueError(
            f"Unknown placeholder in user_prompt_template: {error}"
        ) from error

    optional_parts: list[str] = []

    if formatting_config.get("include_label_in_prompt", False):
        optional_parts.append(f"Label: {example['label']}")

    if formatting_config.get("include_severity_in_prompt", False):
        optional_parts.append(f"Severity: {example['severity']}")

    if optional_parts:
        prompt = f"{prompt}\n\n" + "\n".join(optional_parts)

    return prompt


def build_messages(
    example: dict[str, Any],
    formatting_config: dict[str, Any],
) -> list[dict[str, str]]:
    """Create the conversation used for supervised fine-tuning."""

    messages: list[dict[str, str]] = []

    # system_prompt = formatting_config.get(
    #     "system_prompt",
    #     "",
    # ).strip()

    system_prompt = None
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.extend(
        [
            {
                "role": "user",
                "content": build_user_prompt(
                    example,
                    formatting_config,
                ),
            },
            {
                "role": "assistant",
                "content": example["text"],
            },
        ]
    )

    return messages


def format_example(
    example: dict[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase,
    formatting_config: dict[str, Any],
) -> dict[str, Any]:
    """
    Add chat messages and rendered training text to one example.

    We render the full assistant response because this dataset will initially
    be used for standard causal-language-model supervised fine-tuning.
    """

    messages = build_messages(example, formatting_config)

    rendered_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )

    return {
        "messages": messages,
        "formatted_text": rendered_text,
    }


def format_dataset(
    dataset: DatasetDict,
    *,
    tokenizer: PreTrainedTokenizerBase,
    formatting_config: dict[str, Any],
) -> DatasetDict:
    """Apply the model's native chat template to every split."""

    return dataset.map(
        lambda example: format_example(
            example,
            tokenizer=tokenizer,
            formatting_config=formatting_config,
        ),
        desc="Applying chat template",
    )


def prepare_datasets(
    *,
    tokenizer: PreTrainedTokenizerBase,
    config_path: str | Path = "configs/data.yaml",
) -> PreparedDatasets:
    """Load, validate, split, and chat-format the complete dataset."""

    config = load_yaml(config_path)

    if "data" not in config or "formatting" not in config:
        raise ValueError(
            "Data configuration requires 'data' and 'formatting' sections."
        )

    data_config = config["data"]
    formatting_config = config["formatting"]

    records, source_paths = load_behavior_records(data_config)

    splits = split_records(
        records,
        validation_fraction=float(
            data_config["validation_fraction"]
        ),
        test_fraction=float(
            data_config["test_fraction"]
        ),
        random_seed=int(data_config["random_seed"]),
    )

    raw_dataset = records_to_dataset(splits)

    formatted_dataset = format_dataset(
        raw_dataset,
        tokenizer=tokenizer,
        formatting_config=formatting_config,
    )

    return PreparedDatasets(
        raw=raw_dataset,
        formatted=formatted_dataset,
        source_files=[str(path) for path in source_paths],
    )


def summarize_records(
    records: Iterable[dict[str, Any]],
) -> dict[str, Counter[str]]:
    """Count the main metadata fields."""

    labels: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    severities: Counter[str] = Counter()

    for record in records:
        labels[record["label"]] += 1
        categories[record["category"]] += 1
        severities[record["severity"]] += 1

    return {
        "labels": labels,
        "categories": categories,
        "severities": severities,
    }