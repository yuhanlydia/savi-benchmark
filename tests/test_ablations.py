from savi.ablations import STATE_FIELDS, state_shuffle


def test_state_shuffle_is_within_cell_and_has_no_fixed_points():
    rows = []
    for prefix in range(4):
        for horizon in [0, 256]:
            rows.append({"problem_id": "p", "spent_budget": 128,
                         "state_id": f"s{prefix}", "horizon": horizon,
                         "label": prefix, "last_hidden": [prefix],
                         "recent_token_entropy": prefix,
                         "has_candidate_answer": bool(prefix),
                         "recent_repetition_rate": prefix / 10})
    shuffled = state_shuffle(rows, seed=1)
    assert all(row["state_shuffle_donor"] != row["state_id"] for row in shuffled)
    assert [row["label"] for row in shuffled] == [row["label"] for row in rows]
    assert all(row["last_hidden"] != [row["label"]] for row in shuffled)
