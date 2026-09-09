from __future__ import annotations

import argparse
import json
from pathlib import Path

from savi.backfill import backfill_rows
from savi.io import load_yaml, read_jsonl
from savi.phase0 import QwenRunner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exploratory_shared_budget_24gb.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    data = {row["problem_id"]: row for row in read_jsonl(config["data"]["path"])}
    source = []
    for row in read_jsonl(args.input):
        if row["problem_id"] not in data:
            raise ValueError(f"unknown problem_id {row['problem_id']}")
        source.append({**row, "problem": data[row["problem_id"]]["problem"]})
    output = backfill_rows(QwenRunner(config), source, args.output)
    rows = read_jsonl(output)
    print(json.dumps({"output": str(output), "rows": len(rows),
                      "feature_mode_counts": {
                          mode: sum(row.get("feature_mode") == mode for row in rows)
                          for mode in sorted({row.get("feature_mode") for row in rows})
                      }}, indent=2))


if __name__ == "__main__":
    main()
