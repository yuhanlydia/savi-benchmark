import json

import pytest

from savi.judge_handoff import export_queue, import_labels


def _config(tmp_path):
    problem = {"problem_id": "p", "problem": "1+1", "answer": "2"}
    attempt = {"state_id": "s", "horizon": 1, "continuation_id": 0, "problem_id": "p",
               "finalizer_output": "Final Answer: \\boxed{2}",
               "parsed_answer_normalized": "2", "correct_exact_normalized": True}
    data = tmp_path / "data.jsonl"; data.write_text(json.dumps(problem) + "\n")
    attempts = tmp_path / "attempts.jsonl"; attempts.write_text(json.dumps(attempt) + "\n")
    return {"data": {"path": str(data)}, "output": {"continuations": str(attempts)}}


def test_judge_round_trip_requires_exact_key_coverage(tmp_path):
    config = _config(tmp_path); queue = tmp_path / "queue.jsonl"
    assert export_queue(config, queue) == 1
    label = json.loads(queue.read_text()); label["correct_official"] = True
    labels = tmp_path / "labels.jsonl"; labels.write_text(json.dumps(label) + "\n")
    output = tmp_path / "official.jsonl"
    assert import_labels(config, labels, output) == 1
    assert json.loads(output.read_text())["correct_official"] is True


def test_judge_import_rejects_missing_labels(tmp_path):
    config = _config(tmp_path); labels = tmp_path / "labels.jsonl"; labels.write_text("")
    with pytest.raises(ValueError, match="missing=1"):
        import_labels(config, labels, tmp_path / "official.jsonl")
