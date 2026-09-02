from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Any

import numpy as np

from .io import load_yaml, read_jsonl, write_json, write_jsonl


def complete_state_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    required = {
        int(horizon): (1 if int(horizon) == 0 else int(config["experiment"]["continuations_per_state"]))
        for horizon in config["experiment"]["continuation_horizons"]
    }
    counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["state_id"]][int(row["horizon"])] += 1
    complete = {
        state_id for state_id, horizon_counts in counts.items()
        if all(horizon_counts[horizon] == count for horizon, count in required.items())
    }
    return [row for row in rows if row["state_id"] in complete]


def terminal_state_ids(config: dict[str, Any]) -> set[str]:
    prefix_path = Path(config["output"]["prefixes"])
    if not prefix_path.exists():
        return set()
    terminal = set()
    for row in read_jsonl(prefix_path):
        if row.get("closed_thinking_stage") or row.get("has_candidate_answer"):
            terminal.add(row["state_id"])
    return terminal


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, int, int], list[bool]] = defaultdict(list)
    meta: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["problem_id"], row["spent_budget"], row["prefix_id"], row["horizon"])
        official = row.get("correct_official")
        outcome = official if official is not None else row["correct_exact_normalized"]
        groups[key].append(bool(outcome))
        meta[key] = row
    values = []
    for key, outcomes in groups.items():
        row = meta[key]
        values.append({
            "problem_id": key[0], "spent_budget": key[1], "prefix_id": key[2],
            "horizon": key[3], "suite_id": row["suite_id"], "position": row["position"],
            "successes": sum(outcomes), "trials": len(outcomes), "q": float(np.mean(outcomes)),
        })
    return values


def analyze(config: dict[str, Any], values: list[dict[str, Any]],
            *, analysis_subset: str = "all") -> dict[str, Any]:
    positive_horizons = [int(value) for value in config["experiment"]["continuation_horizons"]
                         if int(value) > 0]
    if len(positive_horizons) != 1:
        raise ValueError("aliasing analysis requires exactly one positive continuation horizon")
    analysis_horizon = positive_horizons[0]
    by_state: dict[tuple[str, int, int], dict[int, float]] = defaultdict(dict)
    for row in values:
        by_state[(row["problem_id"], row["spent_budget"], row["prefix_id"])][row["horizon"]] = row["q"]
    state_mvi = {}
    state_meta = {}
    for key, horizons in by_state.items():
        if 0 in horizons and analysis_horizon in horizons:
            state_mvi[key] = (horizons[analysis_horizon] - horizons[0]) / analysis_horizon
            state_meta[key] = next(row for row in values if
                                   (row["problem_id"], row["spent_budget"], row["prefix_id"]) == key)

    cells: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (problem, spent, _), mvi in state_mvi.items():
        del mvi
        cells[(problem, spent)].append(by_state[(problem, spent, _)][analysis_horizon])
    expected_prefixes = int(config["experiment"]["prefixes_per_cell"])
    ranges = {
        key: max(items) - min(items)
        for key, items in cells.items() if len(items) == expected_prefixes
    }
    cell_diagnostics = []
    gain_ranges = []
    corrected_variances = []
    for (problem, spent), state_qs in sorted(cells.items()):
        if len(state_qs) != expected_prefixes:
            continue
        matching_keys = [key for key in sorted(state_mvi)
                         if key[0] == problem and key[1] == spent]
        ordered_qs = [by_state[key][analysis_horizon] for key in matching_keys]
        gains = [by_state[key][analysis_horizon] - by_state[key][0] for key in matching_keys]
        mvis = [state_mvi[key] for key in matching_keys]
        gain_range = max(gains) - min(gains)
        gain_ranges.append(gain_range)
        q_values = np.asarray(ordered_qs, dtype=float)
        gain_values = np.asarray(gains, dtype=float)
        observed_variance = float(np.var(gain_values))
        trials = int(config["experiment"]["continuations_per_state"])
        noise_variance = float(np.mean(q_values * (1.0 - q_values) / trials))
        corrected_variance = observed_variance - noise_variance
        corrected_variances.append(corrected_variance)
        cell_diagnostics.append({
            "problem_id": problem,
            "spent_budget": spent,
            "continuation_q_by_state": ordered_qs,
            "continuation_gain_by_state": gains,
            "continuation_gain_range": gain_range,
            "observed_gain_variance": observed_variance,
            "binomial_noise_variance": noise_variance,
            "noise_corrected_gain_variance": corrected_variance,
            "marginal_value_by_state": mvis,
            "continuation_q_range": ranges[(problem, spent)],
        })
    threshold = float(config["gates"]["range_threshold"])
    range_fraction = float(np.mean([value >= threshold for value in ranges.values()])) if ranges else 0.0

    continuation_cells: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    for row in values:
        if row["horizon"] == analysis_horizon:
            continuation_cells[(row["problem_id"], row["spent_budget"])].append(
                (row["successes"], row["trials"])
            )
    eligible = [cell for cell in continuation_cells.values() if len(cell) == expected_prefixes]
    null_draws = int(config["gates"].get("dispersion_null_draws", 10_000))
    null_rng = np.random.default_rng(int(config["experiment"]["seed"]) + 17)
    null_fractions = np.zeros(null_draws, dtype=float)
    if eligible:
        null_hits = np.zeros((null_draws, len(eligible)), dtype=bool)
        for cell_index, cell in enumerate(eligible):
            successes = sum(item[0] for item in cell)
            trials = sum(item[1] for item in cell)
            pooled = successes / trials if trials else 0.0
            per_state_trials = np.asarray([item[1] for item in cell])
            simulated = null_rng.binomial(
                per_state_trials, pooled, size=(null_draws, expected_prefixes)
            ) / per_state_trials
            null_hits[:, cell_index] = simulated.max(1) - simulated.min(1) >= threshold
        null_fractions = null_hits.mean(axis=1)
    null_expected = float(null_fractions.mean()) if eligible else 0.0
    null_pvalue = float((1 + np.sum(null_fractions >= range_fraction)) / (null_draws + 1))
    excess_fraction = range_fraction - null_expected
    raw_gate = range_fraction >= config["gates"]["min_range_cell_fraction"]
    corrected_gate = (
        null_pvalue <= config["gates"].get("max_dispersion_null_pvalue", 0.05)
        and excess_fraction >= config["gates"].get("min_excess_range_fraction", 0.10)
    )

    rng = np.random.default_rng(int(config["experiment"]["seed"]))
    draws = int(config["gates"]["dfr_bootstrap_draws"])
    suite_spent: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, mvi in state_mvi.items():
        meta = state_meta[key]
        suite_spent[(meta["suite_id"], key[1])][key[0]].append(mvi)
    flips = []
    for _, problem_states in suite_spent.items():
        problem_ids = sorted(problem_states)
        budget_choice = int(np.argmax([np.mean(problem_states[p]) for p in problem_ids]))
        for _ in range(draws):
            realized = [rng.choice(problem_states[p]) for p in problem_ids]
            flips.append(int(np.argmax(realized) != budget_choice))
    dfr = float(np.mean(flips)) if flips else 0.0
    budget_best = []
    state_best = []
    for _, problem_states in suite_spent.items():
        means = [float(np.mean(problem_states[p])) for p in sorted(problem_states)]
        if len(means) < 2:
            continue
        budget_best.append(max(means))
        for _ in range(draws):
            realized = [float(rng.choice(problem_states[p])) for p in sorted(problem_states)]
            state_best.append(max(realized))
    oracle_headroom = float(np.mean(state_best) - np.mean(budget_best)) if budget_best else 0.0
    problem_groups = defaultdict(list)
    for item in cell_diagnostics:
        problem_groups[item["problem_id"]].append(item["noise_corrected_gain_variance"])
    problem_means = np.asarray([np.mean(items) for items in problem_groups.values()], dtype=float)
    variance_ci = [None, None]
    if len(problem_means):
        boot = np.random.default_rng(int(config["experiment"]["seed"]) + 101)
        draws_matrix = boot.choice(problem_means, size=(draws, len(problem_means)), replace=True)
        variance_ci = [float(np.quantile(draws_matrix.mean(axis=1), 0.025)),
                       float(np.quantile(draws_matrix.mean(axis=1), 0.975))]
    mva_fraction = float(np.mean([value >= threshold for value in gain_ranges])) if gain_ranges else 0.0
    mva_gate = mva_fraction >= config["gates"].get("min_mva_cell_fraction", 0.20)
    return {
        "schema_version": 2,
        "analysis_horizon": analysis_horizon,
        "scoring": "exact-normalized pilot labels",
        "state_count": len(state_mvi),
        "same_budget_cells": len(ranges),
        "cell_diagnostics": cell_diagnostics,
        "range_ge_0_5_fraction": range_fraction,
        "gain_range_ge_0_5_fraction_diagnostic": (
            float(np.mean([value >= threshold for value in gain_ranges]))
            if gain_ranges else 0.0
        ),
        "mva_range_ge_0_5_fraction": mva_fraction,
        "mva_gate_pass": mva_gate,
        "noise_corrected_gain_variance_mean": (
            float(np.mean(corrected_variances)) if corrected_variances else None
        ),
        "noise_corrected_gain_variance_problem_bootstrap_ci95": variance_ci,
        "budget_only_best_mvi_mean": float(np.mean(budget_best)) if budget_best else None,
        "state_oracle_best_mvi_mean": float(np.mean(state_best)) if state_best else None,
        "state_oracle_headroom_mvi": oracle_headroom,
        "range_null_expected_fraction": null_expected,
        "range_excess_fraction": excess_fraction,
        "range_null_pvalue": null_pvalue,
        "decision_flip_rate": dfr,
        "gate_0_raw_preregistered_pass": raw_gate,
        "gate_0_noise_adjusted_pass": corrected_gate,
        "gate_0_pass": raw_gate and corrected_gate,
        "gate_1_pass": dfr >= config["gates"]["min_decision_flip_rate"],
        "analysis_subset": analysis_subset,
        "warning": "Confirm symbolic/nontrivial equivalence with the official R3 judge before paper claims.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--nonterminal-only", action="store_true")
    parser.add_argument("--judge-labeled")
    parser.add_argument("--output-prefix", help="prefix for derived values/report files")
    args = parser.parse_args()
    config = load_yaml(args.config)
    rows = read_jsonl(args.judge_labeled or config["output"]["continuations"])
    keys = {(row["state_id"], row["horizon"], row["continuation_id"]) for row in rows}
    jobs_path = Path(config["output"]["root"]) / "jobs.json"
    planned = len(json.loads(jobs_path.read_text(encoding="utf-8")))
    if len(keys) != planned and not args.allow_partial:
        raise SystemExit(
            f"Refusing confirmatory analysis: {len(keys)}/{planned} jobs complete. "
            "Use --allow-partial for a clearly non-confirmatory diagnostic."
        )
    if args.allow_partial:
        rows = complete_state_rows(rows, config)
    subset = "all"
    excluded_terminal = 0
    if args.nonterminal_only:
        terminal = terminal_state_ids(config)
        excluded_terminal = len({row["state_id"] for row in rows} & terminal)
        rows = [row for row in rows if row["state_id"] not in terminal]
        subset = "nonterminal_only"
    values = aggregate(rows)
    report = analyze(config, values, analysis_subset=subset)
    report["excluded_terminal_state_count"] = excluded_terminal
    report["input_labels"] = "official_judge" if args.judge_labeled else "exact_normalized_pilot"
    if args.output_prefix:
        prefix = Path(config["output"]["root"]) / args.output_prefix
        values_path = prefix.with_suffix(".state_values.jsonl")
        report_path = prefix.with_suffix(".gates.json")
    elif args.nonterminal_only:
        values_path = Path(config["output"]["root"]) / "state_values_nonterminal.jsonl"
        report_path = Path(config["output"]["root"]) / "gates_nonterminal.json"
    else:
        values_path = Path(config["output"]["state_values"])
        report_path = Path(config["output"]["report"])
    write_jsonl(values_path, values)
    write_json(report_path, report)
    print(report)


if __name__ == "__main__":
    main()
