from savi.gate_decision import decide_gap_gates


def test_marginal_value_primary_gate_requires_positive_noise_corrected_variance():
    config = {"gates": {"primary_aliasing_gate": "marginal_value"}}
    report = {
        "mva_gate_pass": True,
        "noise_corrected_gain_variance_problem_bootstrap_ci95": [0.01, 0.1],
        "gate_1_pass": True,
    }
    decision = decide_gap_gates(config, report)
    assert decision["aliasing_gate_pass"] is True
    assert decision["decision_relevance_gate_pass"] is True
    assert decision["overall_gap_gate_pass"] is True


def test_marginal_value_primary_gate_rejects_variance_ci_crossing_zero():
    config = {"gates": {"primary_aliasing_gate": "marginal_value"}}
    report = {
        "mva_gate_pass": True,
        "noise_corrected_gain_variance_problem_bootstrap_ci95": [-0.01, 0.1],
        "gate_1_pass": True,
    }
    assert decide_gap_gates(config, report)["aliasing_gate_pass"] is False
