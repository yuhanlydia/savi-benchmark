import json

from savi.backfill import backfill_rows


class FakeRunner:
    def state_features(self, problem, prefix_ids):
        assert prefix_ids == [1, 2]
        return {"last_hidden": [0.1, 0.2], "recent_token_entropy": 0.3}


def test_backfill_preserves_state_identity_and_adds_hidden_features(tmp_path):
    output = backfill_rows(
        FakeRunner(),
        [{"state_id": "p-b4096-m0", "prefix_token_ids": [1, 2]}],
        tmp_path / "prefixes.jsonl",
    )
    row = json.loads(output.read_text().strip())
    assert row["state_id"] == "p-b4096-m0"
    assert row["feature_mode"] == "full_hidden"
    assert row["last_hidden"] == [0.1, 0.2]


def test_backfill_resume_reuses_completed_row(tmp_path):
    path = tmp_path / "prefixes.jsonl"
    first = backfill_rows(
        FakeRunner(),
        [{"state_id": "p-b4096-m0", "prefix_token_ids": [1, 2]}],
        path,
    )
    second = backfill_rows(
        FakeRunner(),
        [{"state_id": "p-b4096-m0", "prefix_token_ids": [1, 2]}],
        path,
    )
    assert first == second
    assert len(path.read_text().splitlines()) == 1
