"""Dataset preparation with an explicit train/validation/test boundary."""

from __future__ import annotations

from typing import Any

from .parsing import extract_gold_answer
from .prompts import build_prompt


def _convert_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "row_id": f"gsm8k-{index}",
        "question": row["question"],
        "prompt": build_prompt(row["question"]),
        "ground_truth": extract_gold_answer(row["answer"]),
    }


def _limit(dataset: Any, limit: int | None, seed: int) -> Any:
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def load_training_splits(
    validation_size: int = 256,
    train_limit: int | None = None,
    validation_limit: int | None = None,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Load GSM8K train data and create a reproducible held-out validation split."""
    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split="train")
    prepared = raw.map(_convert_row, with_indices=True, remove_columns=raw.column_names)
    if not 0 < validation_size < len(prepared):
        raise ValueError("validation_size must be between 1 and len(train)-1")
    split = prepared.train_test_split(test_size=validation_size, seed=seed)
    train = _limit(split["train"], train_limit, seed)
    validation = _limit(split["test"], validation_limit, seed)
    return train, validation


def load_test_split(limit: int | None = None, seed: int = 42) -> Any:
    """Load the official test set; this function is intentionally evaluation-only."""
    from datasets import load_dataset

    raw = load_dataset("openai/gsm8k", "main", split="test")
    prepared = raw.map(_convert_row, with_indices=True, remove_columns=raw.column_names)
    return _limit(prepared, limit, seed)
