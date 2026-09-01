import torch

from savi.critic import binomial_nll


def test_binomial_nll_prefers_logits_matching_outcomes():
    successes = torch.tensor([4.0, 0.0])
    trials = torch.tensor([4.0, 4.0])
    good = binomial_nll(torch.tensor([5.0, -5.0]), successes, trials)
    bad = binomial_nll(torch.tensor([-5.0, 5.0]), successes, trials)
    assert good < bad
