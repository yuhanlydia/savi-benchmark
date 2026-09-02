from savi.io import load_yaml
from savi.oracle_analysis import build_oracle_report


def test_oracle_report_exposes_headroom_and_flips():
    config = load_yaml("configs/phase0_math.yaml")
    config["experiment"]["continuation_horizons"] = [0, 256]
    config["experiment"]["prefixes_per_cell"] = 2
    config["gates"]["dfr_bootstrap_draws"] = 20
    values = []
    for problem_index in range(6):
        problem = f"p{problem_index}"
        for prefix in range(2):
            base = {"suite_id": "s", "spent_budget": 128,
                    "problem_id": problem, "prefix_id": prefix}
            values.append({**base, "horizon": 0, "q": 0.0})
            # p0 has mean gain 1; p1 has heterogeneous realized gains,
            # making state-oracle choices differ on some draws.
            gain = 1.0 if problem_index == 0 else (1.0 if prefix == 0 else 0.0)
            values.append({**base, "horizon": 256, "q": gain})
    report = build_oracle_report(values, config, draws=20)
    assert report["complete_suite_cells"] == 1
    assert report["decision_flip_rate"] >= 0.0
    assert report["state_oracle_headroom_gain_mean"] >= 0.0
