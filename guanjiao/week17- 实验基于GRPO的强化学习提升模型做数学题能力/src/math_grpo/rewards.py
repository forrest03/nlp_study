"""Deterministic rewards used by TRL's GRPO trainer."""

from __future__ import annotations

from typing import Any

from .parsing import answers_equal, extract_answer, extract_boxed_spans


def completion_to_text(completion: Any) -> str:
    """Support both TRL's plain-text and conversational completion formats."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        final_message = completion[-1]
        if isinstance(final_message, dict):
            return str(final_message.get("content", ""))
    return ""


def correctness_reward(completions: list[Any], ground_truth: list[str], **_: Any) -> list[float]:
    """Give one point only when the extracted final answer is exactly correct."""
    texts = [completion_to_text(completion) for completion in completions]
    return [
        1.0 if answers_equal(extract_answer(text), reference) else 0.0
        for text, reference in zip(texts, ground_truth)
    ]


def format_reward(completions: list[Any], **_: Any) -> list[float]:
    """Reward a single boxed answer at the end; its trainer weight should stay small."""
    texts = [completion_to_text(completion) for completion in completions]
    rewards: list[float] = []
    for text in texts:
        boxes = extract_boxed_spans(text)
        one_box_start = text.count("\\boxed") == 1
        valid_suffix = len(boxes) == 1 and text[boxes[0][2] :].strip() in {"", ".", "!"}
        rewards.append(1.0 if one_box_start and valid_suffix else 0.0)
    return rewards
