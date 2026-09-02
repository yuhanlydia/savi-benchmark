from savi.io import read_jsonl, write_jsonl


def test_write_jsonl_round_trip(tmp_path):
    path = tmp_path / "rows.jsonl"
    rows = [{"x": 1}, {"x": 2}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows
    assert len(path.read_text().splitlines()) == 2
