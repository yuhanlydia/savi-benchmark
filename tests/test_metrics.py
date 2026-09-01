import numpy as np

from savi.metrics import paired_suite_bootstrap, pairwise_ranking_accuracy, weighted_brier


def test_weighted_brier_is_zero_for_empirical_rates():
    assert weighted_brier(np.array([.25, 1]), np.array([1, 2]), np.array([4, 2])) == 0


def test_pairwise_ranking_handles_tied_prediction_as_half():
    assert pairwise_ranking_accuracy([0, 0], [0, 1]) == .5


def test_paired_bootstrap_uses_suite_as_resampling_unit():
    rows = [
        {"suite_id": "a", "method": "savi", "score": 3},
        {"suite_id": "a", "method": "equal", "score": 2},
        {"suite_id": "b", "method": "savi", "score": 2},
        {"suite_id": "b", "method": "equal", "score": 2},
    ]
    result = paired_suite_bootstrap(rows, method_a="savi", method_b="equal", draws=100, seed=1)
    assert result["mean_difference"] == .5
    assert result["suite_count"] == 2
