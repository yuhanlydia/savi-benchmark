from savi.repro_check import compare_runs


def test_repro_check_compares_only_shared_keys():
    prefixes = [{"state_id": "s", "prefix_token_ids": [1, 2]}]
    row = {"state_id": "s", "horizon": 8, "continuation_id": 0,
           "continuation_token_ids": [3], "correct_exact_normalized": True}
    extra = {"state_id": "s", "horizon": 8, "continuation_id": 1,
             "continuation_token_ids": [4], "correct_exact_normalized": False}
    report = compare_runs(prefixes, prefixes, [row, extra], [row])
    assert report["prefix_match_fraction"] == 1.0
    assert report["shared_jobs"] == 1
    assert report["continuation_match_fraction"] == 1.0
    assert report["outcome_match_fraction"] == 1.0
