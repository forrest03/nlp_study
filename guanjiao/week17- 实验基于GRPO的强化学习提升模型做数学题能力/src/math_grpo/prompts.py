"""Prompt templates shared by training and evaluation."""

SYSTEM_PROMPT = """You are a careful mathematical problem solver.
Work through the problem step by step, checking the arithmetic. End your response with
exactly one final answer written as \\boxed{answer}. Do not put anything after the box."""


def build_prompt(question: str) -> list[dict[str, str]]:
    """Create a conversational prompt understood by instruction-tuned models."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question.strip()},
    ]
