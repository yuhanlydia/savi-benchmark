from savi.phase0 import reasoning_prompt


def test_confirmatory_prompt_announces_common_budget():
    prompt = reasoning_prompt("Compute 1+1.", 768, announce_budget=True)
    assert "768 tokens" in prompt
    assert prompt.endswith("Compute 1+1.")


def test_natural_eos_prompt_has_no_numeric_budget_anchor():
    prompt = reasoning_prompt("Compute 1+1.", 8192, announce_budget=False)
    assert "8192" not in prompt
    assert "budget" not in prompt.casefold()
