from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .io import read_jsonl, write_json


def _round_up(value: float, quantum: int) -> int:
    return int(math.ceil(value / quantum) * quantum)


def build_eos_report(rows: list[dict], quantum: int = 256) -> dict:
    if quantum <= 0:
        raise ValueError("quantum must be positive")
    if not rows:
        return {"samples": 0, "ended_with_eos_fraction": None,
                "parseable_fraction": None, "generated_tokens": None,
                "recommended_trajectory_budget": None,
                "recommendation_status": "insufficient_data"}
    lengths = np.asarray([int(row["generated_tokens"]) for row in rows])
    ended = np.asarray([
        bool(row.get("ended_with_stop_token", row.get("ended_with_eos"))) for row in rows
    ])
    parseable = np.asarray([bool(row.get("parsed_answer_normalized")) for row in rows])
    candidates = np.asarray([bool(row.get("has_candidate_answer")) for row in rows])
    closed_thinking = np.asarray([bool(row.get("closed_thinking_stage")) for row in rows])
    tail_diversities = []
    for row in rows:
        token_ids = row.get("trace_token_ids") or []
        tail = token_ids[-512:]
        if tail:
            tail_diversities.append(len(set(tail)) / len(tail))
    ended_lengths = lengths[ended]
    if len(ended_lengths):
        recommended = _round_up(float(ended_lengths.max()), quantum) + quantum
        status = "natural_eos_observed"
    else:
        recommended = _round_up(float(lengths.max()), quantum) + quantum
        status = "lower_bound_only_all_samples_truncated"
    by_problem = {}
    for problem_id in sorted({row["problem_id"] for row in rows}):
        items = [row for row in rows if row["problem_id"] == problem_id]
        item_lengths = np.asarray([int(row["generated_tokens"]) for row in items])
        item_ended = [bool(row.get("ended_with_stop_token", row.get("ended_with_eos")))
                      for row in items]
        by_problem[problem_id] = {
            "samples": len(items),
            "ended_with_stop_token_fraction": float(np.mean(item_ended)),
            "parseable_fraction": float(np.mean([
                bool(row.get("parsed_answer_normalized")) for row in items
            ])),
            "generated_tokens_min": int(item_lengths.min()),
            "generated_tokens_median": float(np.median(item_lengths)),
            "generated_tokens_max": int(item_lengths.max()),
        }
    return {"samples": len(rows), "problems": len({row["problem_id"] for row in rows}),
            "ended_with_eos_fraction": float(ended.mean()),
            "ended_with_stop_token_fraction": float(ended.mean()),
            "parseable_fraction": float(parseable.mean()),
            "candidate_answer_fraction": float(candidates.mean()),
            "closed_thinking_stage_fraction": float(closed_thinking.mean()),
            "mean_unique_token_fraction_last_512": (
                float(np.mean(tail_diversities)) if tail_diversities else None
            ),
            "generated_tokens": {"min": int(lengths.min()),
                                 "median": float(np.median(lengths)), "max": int(lengths.max())},
            "recommended_trajectory_budget": int(recommended),
            "recommendation_status": status,
            "by_problem": by_problem,
            "warning": "The recommendation is a pilot-grid heuristic, not a population quantile."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="outputs/eos_probe_math/results.jsonl")
    parser.add_argument("--output")
    parser.add_argument("--quantum", type=int, default=256)
    args = parser.parse_args()
    report = build_eos_report(read_jsonl(Path(args.input)), args.quantum)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
