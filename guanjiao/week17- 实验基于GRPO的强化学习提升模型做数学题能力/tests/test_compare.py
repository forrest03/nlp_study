import pytest

from math_grpo.compare import compare


def _records(values: list[bool]) -> dict[str, dict[str, bool]]:
    return {str(index): {"correct": value} for index, value in enumerate(values)}


def test_compare_reports_paired_changes() -> None:
    result = compare(
        _records([False, True, False, True]),
        _records([True, True, True, False]),
        bootstrap_samples=100,
        seed=7,
    )
    assert result["baseline_exact_match"] == 0.5
    assert result["candidate_exact_match"] == 0.75
    assert result["absolute_improvement"] == 0.25
    assert result["improved_examples"] == 2
    assert result["regressed_examples"] == 1


def test_compare_requires_identical_ids() -> None:
    with pytest.raises(ValueError, match="identical row_ids"):
        compare(_records([True]), {"different": {"correct": True}}, bootstrap_samples=100)
