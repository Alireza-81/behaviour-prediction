from __future__ import annotations

from typing import Any

from datasets import DatasetDict
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


def tokenize_example(
    example: dict[str, Any],
    *,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> dict[str, Any]:
    messages = example["messages"]

    if not messages:
        raise ValueError("Example has no messages.")

    if messages[-1].get("role") != "assistant":
        raise ValueError(
            f"Example {example.get('id', '<unknown>')} must end with "
            "an assistant message."
        )

    full_input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )

    prompt_input_ids = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

   

    full_input_ids = full_input_ids["input_ids"]
    prompt_input_ids = prompt_input_ids["input_ids"]
    prompt_length = len(prompt_input_ids)


    if full_input_ids[:prompt_length] != prompt_input_ids:
        raise ValueError(
            f"Prompt tokens are not a prefix of the full conversation "
            f"for example {example.get('id', '<unknown>')}."
        )

    original_length = len(full_input_ids)
    was_truncated = original_length > max_length

    input_ids = full_input_ids[:max_length]
    attention_mask = [1] * len(input_ids)

    labels = [
        IGNORE_INDEX if index < prompt_length else token_id
        for index, token_id in enumerate(input_ids)
    ]

    assistant_mask = [
        int(label != IGNORE_INDEX)
        for label in labels
    ]

    target_token_count = sum(assistant_mask)

    if target_token_count == 0:
        raise ValueError(
            f"Example {example.get('id', '<unknown>')} has no assistant "
            "tokens after truncation. Increase max_length."
        )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "assistant_mask": assistant_mask,
        "sequence_length": len(input_ids),
        "original_sequence_length": original_length,
        "target_token_count": target_token_count,
        "was_truncated": was_truncated,
    }


def tokenize_dataset(
    dataset: DatasetDict,
    *,
    tokenizer: PreTrainedTokenizerBase,
    tokenization_config: dict[str, Any],
) -> DatasetDict:
    max_length = int(
        tokenization_config.get("max_length", 1024)
    )

    tokenized_splits = {}

    for split_name, split_dataset in dataset.items():
        tokenized_splits[split_name] = split_dataset.map(
            lambda example: tokenize_example(
                example,
                tokenizer=tokenizer,
                max_length=max_length,
            ),
            desc=f"Tokenizing {split_name}",
        )

    return DatasetDict(tokenized_splits)