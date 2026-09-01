from __future__ import annotations

import argparse
import json
from typing import Any

from .critic import ValueEnsemble
from .io import append_jsonl, load_yaml, read_jsonl, stable_seed
from .phase0 import QwenRunner, normalize_answer
from .scheduler import choose_problem, state_estimate_from_predictions


def critic_rows(features: dict[str, Any], spent: int, horizons: list[int]) -> list[dict[str, Any]]:
    return [{**features, "spent_budget": spent, "horizon": horizon} for horizon in horizons]


def run_suite(
    problems: list[dict[str, Any]], runner: QwenRunner, ensemble: ValueEnsemble,
    *, shared_budget: int, horizons: list[int], chunk: int, beta: float, seed: int,
) -> dict[str, Any]:
    if len(problems) != 6:
        raise ValueError("R3 suite must contain exactly six problems")
    if 0 not in horizons or any(horizon < 0 for horizon in horizons):
        raise ValueError("horizons must contain zero and no negative values")
    if chunk <= 0 or shared_budget < 0:
        raise ValueError("invalid chunk or shared budget")
    prefixes = {row["problem_id"]: [] for row in problems}
    feature_cache: dict[str, dict[str, Any]] = {}
    trajectory = []
    spent_total = 0
    step = 0
    by_id = {row["problem_id"]: row for row in problems}
    while spent_total < shared_budget:
        estimates = []
        for problem_id, problem in by_id.items():
            if problem_id not in feature_cache:
                feature_cache[problem_id] = runner.state_features(problem["problem"], prefixes[problem_id])
            rows = critic_rows(feature_cache[problem_id], len(prefixes[problem_id]), horizons)
            means, stds = ensemble.predict(rows)
            estimates.append(state_estimate_from_predictions(problem_id, horizons, means, stds))
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
            "step": step, "selected_problem_id": selected, "lookahead_horizon": lookahead,
            "index": index, "executed_tokens": len(new_tokens), "generation_seed": generation_seed,
            "spent_total_after": spent_total + len(new_tokens),
        })
        spent_total += len(new_tokens); step += 1
    answers = []
    for problem_id, problem in by_id.items():
        output = runner.finalize(prefixes[problem_id])
        answers.append({"problem_id": problem_id, "spent_tokens": len(prefixes[problem_id]),
                        "finalizer_output": output, "parsed_answer_normalized": normalize_answer(output)})
    return {"suite_id": problems[0]["suite_id"], "shared_budget": shared_budget,
            "chunk": chunk, "beta": beta, "trajectory": trajectory, "answers": answers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--critic", required=True)
    parser.add_argument("--suite-id", required=True)
    parser.add_argument("--shared-budget", type=int, required=True)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--horizons", type=int, nargs="+", default=[0, 128, 256, 512])
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); config = load_yaml(args.config)
    problems = [row for row in read_jsonl(config["data"]["path"])
                if row["suite_id"] == args.suite_id]
    result = run_suite(problems, QwenRunner(config), ValueEnsemble.load(args.critic),
                       shared_budget=args.shared_budget, horizons=args.horizons,
                       chunk=args.chunk, beta=args.beta, seed=args.seed)
    append_jsonl(args.output, [result]); print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
