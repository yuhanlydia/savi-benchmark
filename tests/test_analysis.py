from savi.analysis import aggregate, analyze, complete_state_rows
from savi.io import load_yaml


def _row(problem, suite, prefix, horizon, correct):
    return {
        "problem_id": problem,
        "suite_id": suite,
        "position": int(problem[-1]),
        "spent_budget": 128,
        "prefix_id": prefix,
        "horizon": horizon,
        "correct_exact_normalized": correct,
    }


def test_gate_a_uses_continuation_probability_not_mvi_range():
    rows = []
    for problem_index in range(1, 7):
        problem = f"p{problem_index}"
        for prefix in range(4):
            rows.append(_row(problem, "suite", prefix, 0, prefix % 2 == 0))
            for repeat in range(4):
                del repeat
                # Every cell spans q=0 to q=1 at h=256.
                rows.append(_row(problem, "suite", prefix, 256, prefix >= 2))
    config = load_yaml("configs/phase0_math.yaml")
    config["gates"]["dfr_bootstrap_draws"] = 20
    report = analyze(config, aggregate(rows))
    assert report["range_ge_0_5_fraction"] == 1.0
    assert len(report["cell_diagnostics"]) == 6
    assert report["cell_diagnostics"][0]["continuation_q_range"] == 1.0
    assert report["gate_0_raw_preregistered_pass"] is True
    assert "range_null_pvalue" in report


def test_official_label_overrides_exact_pilot_label():
    row = _row("p1", "suite", 0, 256, False)
    row["correct_official"] = True
    values = aggregate([row])
    assert values[0]["q"] == 1.0


def test_analysis_horizon_comes_from_config():
    rows = []
    for problem_index in range(1, 7):
        for prefix in range(4):
            rows.append(_row(f"p{problem_index}", "suite", prefix, 0, False))
            row = _row(f"p{problem_index}", "suite", prefix, 512, prefix >= 2)
            rows.extend([dict(row) for _ in range(4)])
    config = load_yaml("configs/phase0_math.yaml")
    config["experiment"]["continuation_horizons"] = [0, 512]
    config["gates"]["dispersion_null_draws"] = 20
    config["gates"]["dfr_bootstrap_draws"] = 20
    report = analyze(config, aggregate(rows))
    assert report["analysis_horizon"] == 512


def test_partial_analysis_drops_incomplete_states():
    config = load_yaml("configs/phase0_math.yaml")
    complete = []
    for continuation_id in range(4):
        row = _row("p1", "suite", 0, 256, True)
        row.update(state_id="complete", continuation_id=continuation_id)
        complete.append(row)
    immediate = _row("p1", "suite", 0, 0, False)
    immediate.update(state_id="complete", continuation_id=0)
    incomplete = _row("p2", "suite", 0, 0, False)
    incomplete.update(state_id="incomplete", continuation_id=0)
    filtered = complete_state_rows([immediate, *complete, incomplete], config)
    assert len(filtered) == 5
    assert {row["state_id"] for row in filtered} == {"complete"}


def test_aliasing_range_requires_complete_prefix_cell():
    config = load_yaml("configs/phase0_math.yaml")
    values = aggregate([
        _row("p1", "suite", 0, 0, False),
        *[_row("p1", "suite", 0, 256, True) for _ in range(4)],
    ])
    config["gates"]["dispersion_null_draws"] = 10
    report = analyze(config, values)
    assert report["state_count"] == 1
    assert report["same_budget_cells"] == 0
