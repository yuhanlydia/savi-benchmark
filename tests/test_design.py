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


def test_explicit_problem_diagnostic_preserves_requested_order():
    config = load_yaml("configs/phase0_math.yaml")
    config["experiment"]["problem_ids"] = ["omnimath-2823", "omnimath-2741"]
    manifest = build_manifest(config)
    assert manifest["sampling_unit"] == "explicit_problem_diagnostic"
    assert [row["problem_id"] for row in manifest["problems"]] == [
        "omnimath-2823", "omnimath-2741"
    ]


def test_calibration_is_small_and_separate():
    config = load_yaml("configs/budget_calibration_math.yaml")
    manifest = build_manifest(config)
    jobs = build_jobs(config, manifest)
    assert manifest["problem_count"] == 6
    assert len({row["state_id"] for row in jobs}) == 36
    assert len(jobs) == 108
    assert config["output"]["root"] != "outputs/phase0_math"


def test_confirmatory_manifest_excludes_all_development_suites():
    config = load_yaml("configs/phase0_math.yaml")
    config["experiment"]["suite_count"] = 10
    config["experiment"]["exclude_suite_ids"] = [
        "math_suite_002", "math_suite_018", "math_suite_019", "math_suite_020",
        "math_suite_021", "math_suite_030", "math_suite_033", "math_suite_035",
        "math_suite_039", "math_suite_042", "math_suite_044", "math_suite_047",
    ]
    manifest = build_manifest(config)
    assert len(manifest["selected_suite_ids"]) == 10
    assert not set(manifest["selected_suite_ids"]) & set(config["experiment"]["exclude_suite_ids"])
    assert manifest["excluded_suite_ids"] == sorted(config["experiment"]["exclude_suite_ids"])
