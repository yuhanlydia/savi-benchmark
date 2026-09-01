from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from .io import load_yaml, read_jsonl


def build_health_report(config):
    continuation_path = Path(config["output"]["continuations"])
    prefix_path = Path(config["output"]["prefixes"])
    jobs_path = Path(config["output"]["root"]) / "jobs.json"
    rows = read_jsonl(continuation_path) if continuation_path.exists() else []
    prefixes = read_jsonl(prefix_path) if prefix_path.exists() else []
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    keys = [(row["state_id"], row["horizon"], row["continuation_id"]) for row in rows]
    counts = Counter(keys)
    parseable = sum(bool(row.get("parsed_answer_normalized")) for row in rows)
    return {
        "planned_jobs": len(jobs),
        "completed_rows": len(rows),
        "unique_jobs": len(counts),
        "completion_fraction": len(counts) / len(jobs) if jobs else 0.0,
        "duplicate_rows": sum(value - 1 for value in counts.values()),
        "invalid_continuation_lengths": sum(
            len(row["continuation_token_ids"]) != row["horizon"] for row in rows
        ),
        "prefix_rows": len(prefixes),
        "unique_states": len({row["state_id"] for row in prefixes}),
        "invalid_prefix_lengths": sum(
            len(row["prefix_token_ids"]) != row["spent_budget"] for row in prefixes
        ),
        "hidden_dimensions": sorted({len(row["last_hidden"]) for row in prefixes}),
        "parseable_fraction": parseable / len(rows) if rows else None,
        "seconds_since_last_record": (
            round(time.time() - continuation_path.stat().st_mtime, 1)
            if continuation_path.exists() else None
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    args = parser.parse_args()
    print(json.dumps(build_health_report(load_yaml(args.config)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
