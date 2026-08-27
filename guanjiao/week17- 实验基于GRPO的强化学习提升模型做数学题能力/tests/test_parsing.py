import pytest

from math_grpo.parsing import (
    answers_equal,
    extract_answer,
    extract_gold_answer,
    normalize_answer,
)


def test_extracts_gsm8k_gold_answer() -> None:
    assert extract_gold_answer("Reasoning here. #### 1,234") == "1,234"


def test_missing_gold_delimiter_is_rejected() -> None:
    with pytest.raises(ValueError):
        extract_gold_answer("42")


def test_last_box_has_priority() -> None:
    text = r"A draft is \boxed{4}, but after checking the answer is \boxed{5}."
    assert extract_answer(text) == "5"


def test_nested_latex_box_is_extracted() -> None:
    assert extract_answer(r"Therefore \boxed{\frac{3}{4}}.") == r"\frac{3}{4}"


@pytest.mark.parametrize(
    ("prediction", "reference"),
    [
        ("1,200", "1200"),
        ("0.5", "1/2"),
        (r"\frac{3}{4}", "0.75"),
        ("50%", "0.5"),
        ("-2.00", "-2"),
    ],
)
def test_numeric_equivalence(prediction: str, reference: str) -> None:
    assert answers_equal(prediction, reference)


def test_different_answers_are_not_equal() -> None:
    assert not answers_equal("41", "42")
    assert not answers_equal(None, "42")


def test_normalize_answer_is_exact() -> None:
    assert normalize_answer("1.250") == "5/4"
