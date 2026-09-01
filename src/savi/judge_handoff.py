from __future__ import annotations

import argparse
import json
from pathlib import Path

from .io import append_jsonl, load_yaml, read_jsonl


KEY_FIELDS = ("state_id", "horizon", "continuation_id")


def key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def export_queue(config, output):
    problems = {row["problem_id"]: row for row in read_jsonl(config["data"]["path"])}
    rows = read_jsonl(config["output"]["continuations"])
    queue = []
    for row in rows:
        problem = problems[row["problem_id"]]
        queue.append({
            **{field: row[field] for field in KEY_FIELDS},
            "problem_id": row["problem_id"],
            "problem": problem["problem"],
            "reference_answer": problem["answer"],
            "candidate_output": row["finalizer_output"],
            "candidate_answer_normalized": row["parsed_answer_normalized"],
            "exact_normalized_label": row["correct_exact_normalized"],
        })
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue),
                            encoding="utf-8")
    return len(queue)


def import_labels(config, labels_path, output_path):
    rows = read_jsonl(config["output"]["continuations"])
    labels = read_jsonl(labels_path)
    label_map = {}
    for row in labels:
        row_key = key(row)
        if row_key in label_map:
            raise ValueError(f"duplicate judge label: {row_key}")
        if not isinstance(row.get("correct_official"), bool):
            raise ValueError(f"official label must be boolean: {row_key}")
        label_map[row_key] = row
    expected = {key(row) for row in rows}
    extra = set(label_map) - expected
    missing = expected - set(label_map)
    if extra or missing:
        raise ValueError(f"judge key mismatch: missing={len(missing)} extra={len(extra)}")
    target = Path(output_path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite {target}")
    append_jsonl(target, [{**row, "correct_official": label_map[key(row)]["correct_official"],
                           "official_judge": label_map[key(row)].get("official_judge")}
                          for row in rows])
    return len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export"); export.add_argument("--output", required=True)
    load = sub.add_parser("import"); load.add_argument("--labels", required=True); load.add_argument("--output", required=True)
    args = parser.parse_args(); config = load_yaml(args.config)
    if args.command == "export":
        print(f"exported={export_queue(config, args.output)}")
    else:
        print(f"imported={import_labels(config, args.labels, args.output)}")


if __name__ == "__main__":
    main()
