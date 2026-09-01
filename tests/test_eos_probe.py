from savi.io import load_yaml


def test_eos_probe_budget_matches_default_cap():
    config = load_yaml("configs/eos_probe_math.yaml")
    assert config["experiment"]["trajectory_budget"] == 8192
