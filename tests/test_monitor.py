import json

from savi.monitor import build_health_report


def test_monitor_detects_duplicates_and_bad_lengths(tmp_path):
    root = tmp_path / "run"; root.mkdir()
    job = {"state_id": "s", "problem_id": "p", "horizon": 2, "continuation_id": 0}
    (root / "jobs.json").write_text(json.dumps([job]))
    row = {**job, "continuation_token_ids": [1], "parsed_answer_normalized": "2"}
    (root / "continuations.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")
    prefix = {"state_id": "s", "spent_budget": 2, "prefix_token_ids": [1], "last_hidden": [0, 1]}
    (root / "prefixes.jsonl").write_text(json.dumps(prefix) + "\n")
    config = {"output": {"root": str(root), "continuations": str(root / "continuations.jsonl"),
                         "prefixes": str(root / "prefixes.jsonl")}}
    report = build_health_report(config)
    assert report["duplicate_rows"] == 1
    assert report["invalid_continuation_lengths"] == 2
    assert report["invalid_prefix_lengths"] == 1
    assert report["complete_problem_count"] == 0
