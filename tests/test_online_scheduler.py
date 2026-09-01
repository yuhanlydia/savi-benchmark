from savi.online_scheduler import critic_rows


def test_critic_rows_change_only_horizon_and_add_spent_budget():
    features = {"last_hidden": [1, 2], "recent_token_entropy": 1.2,
                "has_candidate_answer": False, "recent_repetition_rate": .1}
    rows = critic_rows(features, 128, [0, 256])
    assert [row["horizon"] for row in rows] == [0, 256]
    assert all(row["spent_budget"] == 128 for row in rows)
    assert all(row["last_hidden"] == [1, 2] for row in rows)
