from savi.scheduler import (HorizonEstimate, StateEstimate, choose_problem,
                            conservative_savi_index, state_estimate_from_predictions)


def test_conservative_index_penalizes_uncertainty():
    state = StateEstimate("a", .4, .02, (HorizonEstimate(256, .8, .1),))
    optimistic, _ = conservative_savi_index(state, 0.0)
    conservative, _ = conservative_savi_index(state, 1.0)
    assert conservative < optimistic


def test_choose_problem_by_marginal_value():
    states = [
        StateEstimate("hard_dead_end", .1, 0, (HorizonEstimate(256, .12),)),
        StateEstimate("partial_progress", .45, 0, (HorizonEstimate(256, .8),)),
    ]
    assert choose_problem(states)[0] == "partial_progress"


def test_build_state_estimate_sorts_horizons():
    state = state_estimate_from_predictions("p", [256, 0, 128], [.8, .4, .7], [.1, .02, .05])
    assert state.finalize_mean == .4
    assert [item.horizon for item in state.horizons] == [128, 256]
