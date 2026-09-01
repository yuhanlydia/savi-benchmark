from savi.eos_report import build_eos_report


def test_empty_report_is_explicitly_insufficient():
    report = build_eos_report([])
    assert report["samples"] == 0
    assert report["recommendation_status"] == "insufficient_data"


def test_recommendation_uses_natural_eos_lengths_only():
    rows = [
        {"problem_id": "a", "generated_tokens": 1100, "ended_with_eos": True,
         "parsed_answer_normalized": "1"},
        {"problem_id": "b", "generated_tokens": 8192, "ended_with_eos": False,
         "parsed_answer_normalized": ""},
    ]
    report = build_eos_report(rows, quantum=256)
    assert report["recommended_trajectory_budget"] == 1536
    assert report["ended_with_eos_fraction"] == 0.5
    assert report["recommendation_status"] == "natural_eos_observed"


def test_all_truncated_is_only_a_lower_bound():
    rows = [{"problem_id": "a", "generated_tokens": 8192, "ended_with_eos": False}]
    report = build_eos_report(rows, quantum=512)
    assert report["recommended_trajectory_budget"] == 8704
    assert report["recommendation_status"] == "lower_bound_only_all_samples_truncated"
