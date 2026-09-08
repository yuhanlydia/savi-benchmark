from __future__ import annotations

import argparse
import json
from typing import Any

from .critic import ValueEnsemble
from .io import append_jsonl, load_yaml, read_jsonl, stable_seed
from .phase0 import QwenRunner, normalize_answer
from .scheduler import (
    OnlineStateBelief,
    StateEstimate,
    choose_problem,
    choose_problem_online,
    state_estimate_from_predictions,
    update_online_belief,
)


def critic_rows(features: dict[str, Any], spent: int, horizons: list[int]) -> list[dict[str, Any]]:
    return [{**features, "spent_budget": spent, "horizon": horizon} for horizon in horizons]


def feasible_horizons(horizons: list[int], remaining_budget: int) -> list[int]:
    """Restrict value lookahead to compute that can still be purchased."""
    if remaining_budget < 0:
        raise ValueError("remaining_budget must be non-negative")
    positive = sorted({int(h) for h in horizons if int(h) > 0})
    if 0 not in horizons or not positive:
        raise ValueError("horizons must contain zero and at least one positive value")
    feasible = [h for h in positive if h <= remaining_budget]
    if not feasible and remaining_budget > 0:
        feasible = [remaining_budget]
    return [0, *feasible]


def update_problem_belief(
    beliefs: dict[str, OnlineStateBelief],
    observed_versions: dict[str, int],
    *,
    problem_id: str,
    state: StateEstimate,
    state_version: int,
    process_std: float,
    measurement_noise_floor: float,
) -> OnlineStateBelief:
    """Update a problem belief at most once for each realized reasoning state."""
    if observed_versions.get(problem_id) == state_version and problem_id in beliefs:
        return beliefs[problem_id]
    updated = update_online_belief(
        beliefs.get(problem_id), state,
        process_std=process_std,
        measurement_noise_floor=measurement_noise_floor,
    )
    beliefs[problem_id] = updated
    observed_versions[problem_id] = state_version
    return updated


def run_suite(
    problems: list[dict[str, Any]], runner: QwenRunner, ensemble: ValueEnsemble,
    *, shared_budget: int, horizons: list[int], chunk: int, beta: float, seed: int,
    dynamic: bool = True, online_belief: bool = True, voi_lambda: float = 0.0,
    process_std: float = 0.10, measurement_noise_floor: float = 0.05,
) -> dict[str, Any]:
    if len(problems) != 6:
        raise ValueError("R3 suite must contain exactly six problems")
    if 0 not in horizons or any(horizon < 0 for horizon in horizons):
        raise ValueError("horizons must contain zero and no negative values")
    if chunk <= 0 or shared_budget < 0:
        raise ValueError("invalid chunk or shared budget")
    if beta < 0 or voi_lambda < 0 or process_std < 0 or measurement_noise_floor < 0:
        raise ValueError("scheduler uncertainty parameters must be non-negative")
    prefixes = {row["problem_id"]: [] for row in problems}
    feature_cache: dict[str, dict[str, Any]] = {}
    estimate_cache: dict[str, StateEstimate] = {}
    belief_cache: dict[str, OnlineStateBelief] = {}
    belief_versions: dict[str, int] = {}
    trajectory = []
    spent_total = 0
    step = 0
    by_id = {row["problem_id"]: row for row in problems}
    while spent_total < shared_budget:
        remaining_budget = shared_budget - spent_total
        decision_horizons = feasible_horizons(horizons, remaining_budget)
        estimates = []
        online_beliefs = []
        for problem_id, problem in by_id.items():
            if not dynamic and problem_id in estimate_cache:
                estimate = estimate_cache[problem_id]
            else:
                if problem_id not in feature_cache:
                    feature_cache[problem_id] = runner.state_features(
                        problem["problem"], prefixes[problem_id]
                    )
                rows = critic_rows(
                    feature_cache[problem_id], len(prefixes[problem_id]), decision_horizons
                )
                means, stds = ensemble.predict(rows)
                estimate = state_estimate_from_predictions(
                    problem_id, decision_horizons, means, stds
                )
                if not dynamic:
                    estimate_cache[problem_id] = estimate
            estimates.append(estimate)
            if online_belief:
                state_version = len(prefixes[problem_id]) if dynamic else 0
                online_beliefs.append(update_problem_belief(
                    belief_cache, belief_versions,
                    problem_id=problem_id, state=estimate, state_version=state_version,
                    process_std=process_std,
                    measurement_noise_floor=measurement_noise_floor,
                ))

        exploit_value = None
        information_value = None
        posterior_std = None
        belief_update_count = None
        if online_belief:
            choice = choose_problem_online(
                online_beliefs, beta=beta, voi_lambda=voi_lambda
            )
            selected = choice.problem_id
            lookahead = choice.horizon
            index = choice.score
            exploit_value = choice.exploit_value
            information_value = choice.information_value
            posterior_std = choice.posterior_std
            belief_update_count = belief_cache[selected].update_count
        else:
            selected, lookahead, index = choose_problem(estimates, beta=beta)

        execution = min(chunk, shared_budget - spent_total)
        generation_seed = stable_seed(seed, step, selected, len(prefixes[selected]))
        new_tokens = runner.continue_prefix(
            by_id[selected]["problem"], len(prefixes[selected]), prefixes[selected], execution,
            generation_seed,
        )
        prefixes[selected].extend(new_tokens)
        feature_cache.pop(selected, None)
        trajectory.append({
            "step": step,
            "selected_problem_id": selected,
            "lookahead_horizon": lookahead,
            "index": index,
            "exploit_value": exploit_value,
            "information_value": information_value,
            "posterior_std": posterior_std,
            "belief_update_count": belief_update_count,
            "remaining_budget_before": remaining_budget,
            "executed_tokens": len(new_tokens),
            "generation_seed": generation_seed,
            "spent_total_after": spent_total + len(new_tokens),
        })
        spent_total += len(new_tokens)
        step += 1
    answers = []
    for problem_id, problem in by_id.items():
        output = runner.finalize(prefixes[problem_id])
        answers.append({
            "problem_id": problem_id,
            "spent_tokens": len(prefixes[problem_id]),
            "finalizer_output": output,
            "parsed_answer_normalized": normalize_answer(output),
        })
    return {
        "suite_id": problems[0]["suite_id"],
        "shared_budget": shared_budget,
        "chunk": chunk,
        "beta": beta,
        "dynamic": dynamic,
        "method": "online-savi" if online_belief else "direct-savi",
        "online_belief": online_belief,
        "voi_lambda": voi_lambda,
        "process_std": process_std,
        "measurement_noise_floor": measurement_noise_floor,
        "trajectory": trajectory,
        "answers": answers,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--critic", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--shared-budget", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=512)
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 512, 1024, 2048])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--voi-lambda", type=float, default=0.0)
    parser.add_argument("--process-std", type=float, default=0.10)
    parser.add_argument("--measurement-noise-floor", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frozen-index", action="store_true")
    parser.add_argument("--legacy-direct-savi", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    problems = [row for row in read_jsonl(config["data"]["path"])
                if row["suite_id"] == args.suite_id]
    result = run_suite(
        problems, QwenRunner(config), ValueEnsemble.load(args.critic),
        shared_budget=args.shared_budget, horizons=args.horizons,
        chunk=args.chunk, beta=args.beta, seed=args.seed,
        dynamic=not args.frozen_index,
        online_belief=not args.legacy_direct_savi,
        voi_lambda=args.voi_lambda,
        process_std=args.process_std,
        measurement_noise_floor=args.measurement_noise_floor,
    )
    append_jsonl(args.output, [result])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
