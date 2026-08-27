from math_grpo.rewards import correctness_reward, format_reward


def test_correctness_reward_supports_text_completions() -> None:
    completions = [r"Reasoning. \boxed{72}", r"Reasoning. \boxed{71}"]
    assert correctness_reward(completions, ["72", "72"]) == [1.0, 0.0]


def test_rewards_support_conversational_completions() -> None:
    completions = [
        [{"role": "assistant", "content": r"Reasoning. \boxed{3/2}"}],
        [{"role": "assistant", "content": "The answer is 1.5"}],
    ]
    assert correctness_reward(completions, ["1.5", "1.5"]) == [1.0, 1.0]
    assert format_reward(completions) == [1.0, 0.0]


def test_format_reward_rejects_text_after_box_or_multiple_boxes() -> None:
    completions = [
        r"Done: \boxed{4} extra text",
        r"First \boxed{3}, then \boxed{4}",
        r"Checked carefully. \boxed{\frac{4}{2}}.",
    ]
    assert format_reward(completions) == [0.0, 0.0, 1.0]
