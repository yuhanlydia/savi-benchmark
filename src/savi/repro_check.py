from __future__ import annotations

import argparse
import json

from .io import load_yaml, read_jsonl, write_json


def compare_runs(left_prefixes: list[dict], right_prefixes: list[dict],
                 left_rows: list[dict], right_rows: list[dict]) -> dict:
    left_p = {row["state_id"]: row for row in left_prefixes}
    right_p = {row["state_id"]: row for row in right_prefixes}
    shared_states = sorted(set(left_p) & set(right_p))
    prefix_matches = sum(left_p[k]["prefix_token_ids"] == right_p[k]["prefix_token_ids"]
                         for k in shared_states)
    key = lambda row: (row["state_id"], int(row["horizon"]), int(row["continuation_id"]))
    left_c = {key(row): row for row in left_rows}; right_c = {key(row): row for row in right_rows}
    shared_jobs = sorted(set(left_c) & set(right_c))
    continuation_matches = sum(
        left_c[k]["continuation_token_ids"] == right_c[k]["continuation_token_ids"]
        for k in shared_jobs
    )
    outcome_matches = sum(
        left_c[k]["correct_exact_normalized"] == right_c[k]["correct_exact_normalized"]
        for k in shared_jobs
    )
    return {"shared_states": len(shared_states), "exact_prefix_token_matches": prefix_matches,
            "prefix_match_fraction": prefix_matches / len(shared_states) if shared_states else None,
            "shared_jobs": len(shared_jobs),
            "exact_continuation_token_matches": continuation_matches,
            "continuation_match_fraction": continuation_matches / len(shared_jobs) if shared_jobs else None,
            "exact_outcome_matches": outcome_matches,
            "outcome_match_fraction": outcome_matches / len(shared_jobs) if shared_jobs else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-config", required=True)
    parser.add_argument("--right-config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    left = load_yaml(args.left_config); right = load_yaml(args.right_config)
    report = compare_runs(
        read_jsonl(left["output"]["prefixes"]), read_jsonl(right["output"]["prefixes"]),
        read_jsonl(left["output"]["continuations"]), read_jsonl(right["output"]["continuations"]),
    )
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
