"""Answer extraction and equivalence checks for numerical math problems."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction

_BOXED_START_RE = re.compile(r"\\boxed\s*\{")
_EXPLICIT_ANSWER_RE = re.compile(
    r"(?:final\s+answer|answer)\s*(?:is|:|=)?\s*"
    r"([$€£]?\s*-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*(?:\.\d+)?)?\s*%?)",
    flags=re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"[$€£]?\s*-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*(?:\.\d+)?)?\s*%?")
_LATEX_FRACTION_RE = re.compile(r"^\\(?:d?frac)\s*\{(-?\d+)\}\s*\{(-?\d+)\}$")


def extract_gold_answer(solution: str) -> str:
    """Extract GSM8K's answer after its ``####`` delimiter."""
    if "####" not in solution:
        raise ValueError("GSM8K solution does not contain the expected '####' delimiter")
    answer = solution.rsplit("####", maxsplit=1)[1].strip()
    if not answer:
        raise ValueError("GSM8K solution has an empty final answer")
    return answer


def extract_boxed_spans(text: str) -> list[tuple[str, int, int]]:
    """Return balanced ``\\boxed{...}`` contents and their full source spans."""
    boxes: list[tuple[str, int, int]] = []
    for match in _BOXED_START_RE.finditer(text):
        opening_brace = match.end() - 1
        depth = 0
        for position in range(opening_brace, len(text)):
            character = text[position]
            if character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    boxes.append(
                        (text[opening_brace + 1 : position].strip(), match.start(), position + 1)
                    )
                    break
    return boxes


def extract_answer(text: str) -> str | None:
    """Extract a final answer, preferring the final LaTeX box when present."""
    boxes = extract_boxed_spans(text)
    if boxes:
        return boxes[-1][0]

    explicit = _EXPLICIT_ANSWER_RE.findall(text)
    if explicit:
        return explicit[-1].strip()

    numbers = _NUMBER_RE.findall(text)
    return numbers[-1].strip() if numbers else None


def _as_fraction(value: str) -> Fraction | None:
    cleaned = value.strip().replace("−", "-").replace(",", "")
    cleaned = cleaned.replace("$", "").replace("€", "").replace("£", "").strip()
    cleaned = cleaned.replace("\\,", "").replace(" ", "")

    latex_fraction = _LATEX_FRACTION_RE.fullmatch(cleaned)
    if latex_fraction:
        denominator = int(latex_fraction.group(2))
        return None if denominator == 0 else Fraction(int(latex_fraction.group(1)), denominator)

    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1]

    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            denominator_value = Decimal(denominator)
            if denominator_value == 0:
                return None
            result = Fraction(Decimal(numerator)) / Fraction(denominator_value)
        else:
            result = Fraction(Decimal(cleaned))
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None

    return result / 100 if is_percent else result


def normalize_answer(value: str) -> str:
    """Return a stable representation for logging and exact comparisons."""
    numeric = _as_fraction(value)
    if numeric is not None:
        return str(numeric.numerator) if numeric.denominator == 1 else str(numeric)

    normalized = value.strip().lower()
    normalized = normalized.replace("\\,", "").replace(" ", "")
    normalized = normalized.strip(".$")
    return normalized


def answers_equal(prediction: str | None, reference: str) -> bool:
    """Compare numerical values exactly and otherwise compare normalized strings."""
    if prediction is None:
        return False
    predicted_number = _as_fraction(prediction)
    reference_number = _as_fraction(reference)
    if predicted_number is not None and reference_number is not None:
        return predicted_number == reference_number
    return normalize_answer(prediction) == normalize_answer(reference)
