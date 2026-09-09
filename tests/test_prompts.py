import torch

from savi.phase0 import natural_stop_ids, prefix_tensor, reasoning_prompt


def test_confirmatory_prompt_announces_common_budget():
    prompt = reasoning_prompt("Compute 1+1.", 768, announce_budget=True)
    assert "768 tokens" in prompt
    assert prompt.endswith("Compute 1+1.")


def test_natural_eos_prompt_has_no_numeric_budget_anchor():
    prompt = reasoning_prompt("Compute 1+1.", 8192, announce_budget=False)
    assert "8192" not in prompt
    assert "budget" not in prompt.casefold()


def test_natural_stop_includes_thinking_end_without_mutating_config():
    configured = [151645, 151643]
    assert natural_stop_ids(configured, 151668) == [151645, 151643, 151668]
    assert configured == [151645, 151643]


def test_prefix_tensor_is_long_even_for_empty_prefix():
    tensor = prefix_tensor(torch, [], "cpu")
    assert tensor.dtype == torch.long
    assert tuple(tensor.shape) == (1, 0)
