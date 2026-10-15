from __future__ import annotations

import torch

from behaviour_prediction.modeling import load_model_for_finetuning

def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable_sum = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    
    return trainable_sum, total


def main():
    prepared = load_model_for_finetuning("configs/model.yaml")
    model = prepared.model
    tokenizer = prepared.tokenizer
    trainable, total = count_parameters(model)
    percentage = 100 * trainable / total

    print()
    print("=" * 70)
    print("MODEL LOADED SUCCESSFULLY")
    print("=" * 70)
    print(f"Model class:       {model.__class__.__name__}")
    print(f"Tokenizer class:   {tokenizer.__class__.__name__}")
    print(f"Vocabulary size:   {len(tokenizer):,}")
    print(f"Total parameters:  {total:,}")
    print(f"Trainable:         {trainable:,}")
    print(f"Trainable percent: {percentage:.4f}%")
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(
        "Allocated VRAM:    "
        f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB"
    )
    print(
        "Reserved VRAM:     "
        f"{torch.cuda.memory_reserved() / 2**30:.2f} GiB"
    )

    print()
    print("Trainable parameter groups:")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"  {name}: {tuple(parameter.shape)}")

    # Tokenization smoke test only; no generation yet.
    sample = "This is a model-loading test."

    encoded = tokenizer(
        sample,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    print()
    print("Tokenization test:")
    print(f"  Text: {sample}")
    print(f"  Token count: {encoded['input_ids'].shape[1]}")
    print(f"  Input shape: {tuple(encoded['input_ids'].shape)}")

    print()
    print("Preparation test passed.")


if __name__ == "__main__":
    main()