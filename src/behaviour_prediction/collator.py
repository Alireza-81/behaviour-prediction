from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


IGNORE_INDEX = -100


@dataclass
class CausalLMCollator:
    """
    Dynamically pad causal-language-model examples.

    Padding rules:
    - input_ids: tokenizer pad-token ID
    - attention_mask: 0
    - labels: -100, so padding is ignored by the loss
    - assistant_mask: 0
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "The tokenizer must have a pad_token_id before creating "
                "the data collator."
            )

        if self.tokenizer.padding_side != "right":
            raise ValueError(
                "This initial collator expects right-side padding."
            )

    @staticmethod
    def _round_up(
        length: int,
        multiple: int | None,
    ) -> int:
        """Round a sequence length up to a specified multiple."""

        if multiple is None:
            return length

        if multiple <= 0:
            raise ValueError("pad_to_multiple_of must be positive.")

        remainder = length % multiple

        if remainder == 0:
            return length

        return length + multiple - remainder

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Create one padded PyTorch batch."""

        if not features:
            raise ValueError("Cannot collate an empty feature list.")

        required_fields = {
            "input_ids",
            "attention_mask",
            "labels",
        }

        for index, feature in enumerate(features):
            missing = required_fields.difference(feature)

            if missing:
                raise KeyError(
                    f"Feature {index} is missing fields: {sorted(missing)}"
                )

            lengths = {
                len(feature["input_ids"]),
                len(feature["attention_mask"]),
                len(feature["labels"]),
            }

            if len(lengths) != 1:
                raise ValueError(
                    f"Feature {index} has inconsistent tensor lengths."
                )

        longest_length = max(
            len(feature["input_ids"])
            for feature in features
        )

        padded_length = self._round_up(
            longest_length,
            self.pad_to_multiple_of,
        )

        batch_input_ids: list[list[int]] = []
        batch_attention_masks: list[list[int]] = []
        batch_labels: list[list[int]] = []
        batch_assistant_masks: list[list[int]] = []

        metadata: dict[str, list[Any]] = {
            "id": [],
            "label": [],
            "category": [],
            "severity": [],
        }

        for feature in features:
            sequence_length = len(feature["input_ids"])
            padding_length = padded_length - sequence_length

            input_ids = list(feature["input_ids"])
            attention_mask = list(feature["attention_mask"])
            labels = list(feature["labels"])

            assistant_mask = list(
                feature.get(
                    "assistant_mask",
                    [
                        int(label != IGNORE_INDEX)
                        for label in labels
                    ],
                )
            )

            input_ids.extend(
                [self.tokenizer.pad_token_id] * padding_length
            )
            attention_mask.extend([0] * padding_length)
            labels.extend([IGNORE_INDEX] * padding_length)
            assistant_mask.extend([0] * padding_length)

            batch_input_ids.append(input_ids)
            batch_attention_masks.append(attention_mask)
            batch_labels.append(labels)
            batch_assistant_masks.append(assistant_mask)

            for field in metadata:
                metadata[field].append(feature.get(field))

        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                batch_attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
            "assistant_mask": torch.tensor(
                batch_assistant_masks,
                dtype=torch.bool,
            ),
            "metadata": metadata,
        }


@dataclass
class TrainerCausalLMCollator:
    """Return only tensors accepted by a causal language model."""

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8

    def __post_init__(self) -> None:
        self.base_collator = CausalLMCollator(
            tokenizer=self.tokenizer,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def __call__(
        self,
        features: list[dict[str, Any]],
    ) -> dict[str, torch.Tensor]:
        batch = self.base_collator(features)

        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }