import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from savi.critic import FeaturePipeline, ValueEnsemble, ValueMLP, binomial_nll


def test_binomial_nll_prefers_logits_matching_outcomes():
    successes = torch.tensor([4.0, 0.0])
    trials = torch.tensor([4.0, 4.0])
    good = binomial_nll(torch.tensor([5.0, -5.0]), successes, trials)
    bad = binomial_nll(torch.tensor([-5.0, 5.0]), successes, trials)
    assert good < bad


def test_ensemble_returns_mean_and_epistemic_std():
    hidden = np.array([[0.0], [1.0]])
    pca = PCA(n_components=1).fit(hidden)
    scaler = StandardScaler().fit(np.column_stack([pca.transform(hidden), np.zeros((2, 5))]))
    pipeline = FeaturePipeline(pca, scaler)
    heads = [ValueMLP(6, 2), ValueMLP(6, 2)]
    for parameter in heads[0].parameters(): parameter.data.zero_()
    for parameter in heads[1].parameters(): parameter.data.zero_()
    heads[1].network[-1].bias.data.fill_(2.0)
    rows = [{"last_hidden": [0.0], "spent_budget": 0, "horizon": 0,
             "recent_token_entropy": 0, "has_candidate_answer": False,
             "recent_repetition_rate": 0}]
    mean, std = ValueEnsemble(pipeline, heads).predict(rows)
    assert mean.shape == std.shape == (1,)
    assert .5 < mean[0] < 1
    assert std[0] > 0
