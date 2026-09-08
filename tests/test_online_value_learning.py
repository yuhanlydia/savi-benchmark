import math

import numpy as np

from savi.online_scheduler import (
    critic_rows,
    feasible_horizons,
    run_suite,
    update_problem_belief,
)
from savi.scheduler import (
    HorizonEstimate,
    StateEstimate,
    choose_problem_online,
    update_online_belief,
)


def _belief_state(pid, q0, qh, std0=0.05, stdh=0.05, horizon=512):
    return StateEstimate(pid, q0, std0, (HorizonEstimate(horizon, qh, stdh),))


def test_online_belief_initializes_from_state_marginal_gain():
    belief = update_online_belief(
        None, _belief_state("p", 0.2, 0.7),
        process_std=0.1, measurement_noise_floor=0.05,
    )
    item = belief.horizons[0]
    assert item.horizon == 512
    assert math.isclose(item.mean_gain, 0.5, abs_tol=1e-9)
    assert item.variance > 0


def test_online_belief_moves_toward_new_state_measurement_without_overwriting_history():
    first = update_online_belief(
        None, _belief_state("p", 0.2, 0.7),
        process_std=0.05, measurement_noise_floor=0.05,
    )
    second = update_online_belief(
        first, _belief_state("p", 0.2, 0.3),
        process_std=0.05, measurement_noise_floor=0.05,
    )
    assert 0.1 < second.horizons[0].mean_gain < 0.5
    assert second.update_count == first.update_count + 1


def test_information_value_can_select_uncertain_probe_when_exploitation_is_tied():
    low_uncertainty = update_online_belief(
        None, _belief_state("a", 0.2, 0.5, std0=0.01, stdh=0.01),
        process_std=0.05, measurement_noise_floor=0.01,
    )
    high_uncertainty = update_online_belief(
        None, _belief_state("b", 0.2, 0.5, std0=0.20, stdh=0.20),
        process_std=0.05, measurement_noise_floor=0.01,
    )
    selected = choose_problem_online(
        [low_uncertainty, high_uncertainty], beta=0.0, voi_lambda=1.0
    )
    assert selected.problem_id == "b"
    assert selected.information_value > 0


def test_zero_information_weight_reduces_to_mean_marginal_value_ranking():
    uncertain_lower_value = update_online_belief(
        None, _belief_state("a", 0.2, 0.6, std0=0.20, stdh=0.20),
        process_std=0.05, measurement_noise_floor=0.01,
    )
    certain_higher_value = update_online_belief(
        None, _belief_state("b", 0.2, 0.7, std0=0.01, stdh=0.01),
        process_std=0.05, measurement_noise_floor=0.01,
    )
    selected = choose_problem_online(
        [uncertain_lower_value, certain_higher_value], beta=0.0, voi_lambda=0.0
    )
    assert selected.problem_id == "b"
    assert selected.information_value == 0.0


def test_same_state_version_is_not_double_counted():
    beliefs = {}
    versions = {}
    state = _belief_state("p", 0.2, 0.7)
    first = update_problem_belief(
        beliefs, versions, problem_id="p", state=state, state_version=0,
        process_std=0.05, measurement_noise_floor=0.05,
    )
    repeated = update_problem_belief(
        beliefs, versions, problem_id="p", state=state, state_version=0,
        process_std=0.05, measurement_noise_floor=0.05,
    )
    assert repeated is first
    assert repeated.update_count == 1


def test_new_reasoning_state_updates_test_time_belief_once():
    beliefs = {}
    versions = {}
    first = update_problem_belief(
        beliefs, versions, problem_id="p", state=_belief_state("p", .2, .7),
        state_version=0, process_std=.05, measurement_noise_floor=.05,
    )
    second = update_problem_belief(
        beliefs, versions, problem_id="p", state=_belief_state("p", .2, .3),
        state_version=512, process_std=.05, measurement_noise_floor=.05,
    )
    assert second.update_count == 2
    assert second.horizons[0].mean_gain != first.horizons[0].mean_gain


def test_feasible_horizons_never_scores_unaffordable_compute():
    assert feasible_horizons([0, 512, 1024, 2048], 700) == [0, 512]
    assert feasible_horizons([0, 512, 1024, 2048], 256) == [0, 256]


def test_critic_rows_preserve_state_and_change_horizon():
    features = {"last_hidden": [1, 2], "recent_token_entropy": 1.2,
                "has_candidate_answer": False, "recent_repetition_rate": .1}
    rows = critic_rows(features, 128, [0, 256])
    assert [row["horizon"] for row in rows] == [0, 256]
    assert all(row["spent_budget"] == 128 for row in rows)
    assert all(row["last_hidden"] == [1, 2] for row in rows)


class FakeRunner:
    def state_features(self, problem, prefix):
        return {"last_hidden": [len(prefix)], "recent_token_entropy": 0.0,
                "has_candidate_answer": False, "recent_repetition_rate": 0.0}

    def continue_prefix(self, problem, spent, prefix, execution, seed):
        return [1] * execution

    def finalize(self, prefix):
        return "0"


class FakeEnsemble:
    def predict(self, rows):
        means = []
        stds = []
        for row in rows:
            if row["horizon"] == 0:
                means.append(0.2)
            else:
                means.append(min(0.9, 0.5 + 0.0005 * row["last_hidden"][0]))
            stds.append(0.1)
        return np.asarray(means), np.asarray(stds)


def _fake_suite():
    return [{"problem_id": f"p{i}", "problem": f"problem {i}", "suite_id": "s"}
            for i in range(6)]


def test_online_run_logs_dynamic_belief_and_information_terms():
    result = run_suite(
        _fake_suite(), FakeRunner(), FakeEnsemble(), shared_budget=1024,
        horizons=[0, 512], chunk=512, beta=0.0, seed=7,
        online_belief=True, voi_lambda=1.0, process_std=0.05,
        measurement_noise_floor=0.05,
    )
    assert len(result["trajectory"]) == 2
    assert result["method"] == "online-savi"
    assert all(step["information_value"] is not None for step in result["trajectory"])
    assert max(step["belief_update_count"] for step in result["trajectory"]) >= 1


def test_frozen_online_index_does_not_reobserve_changed_state():
    result = run_suite(
        _fake_suite(), FakeRunner(), FakeEnsemble(), shared_budget=1024,
        horizons=[0, 512], chunk=512, beta=0.0, seed=7,
        dynamic=False, online_belief=True, voi_lambda=0.0,
        process_std=0.05, measurement_noise_floor=0.05,
    )
    assert all(step["belief_update_count"] == 1 for step in result["trajectory"])
