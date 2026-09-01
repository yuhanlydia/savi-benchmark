from savi.prefix_audit import build_prefix_audit


def test_prefix_audit_finds_first_terminal_token_by_budget():
    rows = [
        {"state_id": "a", "problem_id": "p", "spent_budget": 128,
         "prefix_token_ids": [1, 2, 3]},
        {"state_id": "b", "problem_id": "p", "spent_budget": 512,
         "prefix_token_ids": [1, 99, 4, 99]},
    ]
    report = build_prefix_audit(rows, {99})
    assert report["terminal_prefix_fraction"] == 0.5
    assert report["by_spent_budget"]["128"]["terminal_prefix_fraction"] == 0.0
    assert report["by_spent_budget"]["512"]["terminal_prefix_fraction"] == 1.0
    assert report["states_detail"][1]["first_terminal_token_position"] == 1
