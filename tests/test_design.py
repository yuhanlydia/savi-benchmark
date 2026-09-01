from savi.manifest import build_manifest
from savi.phase0 import build_jobs
from savi.io import load_yaml


def test_phase0_counts():
    config = load_yaml("configs/phase0_math.yaml")
    manifest = build_manifest(config)
    jobs = build_jobs(config, manifest)
    assert manifest["problem_count"] == 60
    assert len({row["state_id"] for row in jobs}) == 720
    assert len(jobs) == 3600
    assert all(len([p for p in manifest["problems"] if p["suite_id"] == suite]) == 6
               for suite in manifest["selected_suite_ids"])


def test_calibration_is_small_and_separate():
    config = load_yaml("configs/budget_calibration_math.yaml")
    manifest = build_manifest(config)
    jobs = build_jobs(config, manifest)
    assert manifest["problem_count"] == 6
    assert len({row["state_id"] for row in jobs}) == 36
    assert len(jobs) == 108
    assert config["output"]["root"] != "outputs/phase0_math"
