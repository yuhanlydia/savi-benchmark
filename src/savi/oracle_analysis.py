from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .io import load_yaml, read_jsonl, write_json


def build_oracle_report(values: list[dict[str, Any]], config: dict[str, Any],
                        draws: int | None = None) -> dict[str, Any]:
    positive = [int(h) for h in config["experiment"]["continuation_horizons"] if int(h) > 0]
    if len(positive) != 1:
        raise ValueError("oracle analysis requires exactly one positive continuation horizon")
    horizon = positive[0]
    expected_prefixes = int(config["experiment"]["prefixes_per_cell"])
    grouped: dict[tuple[str, int, str, int], dict[int, float]] = defaultdict(dict)
    for row in values:
        key = (row["suite_id"], int(row["spent_budget"]), row["problem_id"],
               int(row["prefix_id"]))
        grouped[key][int(row["horizon"])] = float(row["q"])
    suite_cells: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (suite, spent, problem, prefix), horizons in grouped.items():
        if 0 in horizons and horizon in horizons:
            suite_cells[(suite, spent)][problem].append(horizons[horizon] - horizons[0])
    rng = np.random.default_rng(int(config["experiment"]["seed"]) + 313)
    n_draws = int(draws or config["gates"].get("dfr_bootstrap_draws", 1000))
    cells = []
    flips = []
    budget_best = []
    state_best = []
    for (suite, spent), problem_gains in sorted(suite_cells.items()):
        complete = {problem: gains for problem, gains in problem_gains.items()
                    if len(gains) == expected_prefixes}
        if len(complete) != 6:
            continue
        problem_ids = sorted(complete)
        means = np.asarray([np.mean(complete[p]) for p in problem_ids], dtype=float)
        budget_index = int(np.argmax(means))
        per_draw_state_best = []
        per_draw_flip = []
        for _ in range(n_draws):
            realized = np.asarray([rng.choice(complete[p]) for p in problem_ids], dtype=float)
            state_index = int(np.argmax(realized))
            per_draw_flip.append(int(state_index != budget_index))
            per_draw_state_best.append(float(realized[state_index]))
        budget_value = float(means[budget_index])
        state_value = float(np.mean(per_draw_state_best))
        budget_best.append(budget_value)
        state_best.append(state_value)
        flips.extend(per_draw_flip)
        cells.append({
            "suite_id": suite,
            "spent_budget": spent,
            "problem_ids": problem_ids,
            "budget_only_problem": problem_ids[budget_index],
            "budget_only_expected_gain": budget_value,
            "state_oracle_expected_best_gain": state_value,
            "state_oracle_headroom_gain": state_value - budget_value,
            "decision_flip_rate": float(np.mean(per_draw_flip)),
        })
    return {
        "schema_version": 1,
        "analysis_horizon": horizon,
        "complete_suite_cells": len(cells),
        "decision_flip_rate": float(np.mean(flips)) if flips else 0.0,
        "budget_only_expected_gain_mean": float(np.mean(budget_best)) if budget_best else None,
        "state_oracle_expected_best_gain_mean": float(np.mean(state_best)) if state_best else None,
        "state_oracle_headroom_gain_mean": (
            float(np.mean(state_best) - np.mean(budget_best)) if budget_best else None
        ),
        "cells": cells,
        "warning": "This is one-step oracle headroom, not a full shared-budget scheduler score.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp0b_math.yaml")
    parser.add_argument("--values")
    parser.add_argument("--output")
    parser.add_argument("--draws", type=int)
    args = parser.parse_args()
    config = load_yaml(args.config)
    values_path = args.values or config["output"]["state_values"]
    report = build_oracle_report(read_jsonl(values_path), config, args.draws)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
