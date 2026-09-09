from __future__ import annotations

import argparse
import json
from pathlib import Path

from savi.critic import ValueEnsemble
from savi.io import append_jsonl, load_yaml, read_jsonl, stable_seed
from savi.phase0 import QwenRunner
from savi.shared_budget_eval import load_fixed_suites, protocol_fingerprint, run_savi_variant


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exploratory_shared_budget_24gb.yaml")
    parser.add_argument("--critic", default=None)
    parser.add_argument(
        "--methods", nargs="+", required=True,
        choices=["direct_savi", "online_savi_no_voi", "online_savi", "frozen_index"],
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    suites, dataset_sha256 = load_fixed_suites(config)
    critic_path = args.critic or str(Path(config["calibration"]["critic_dir"]) / "state-aware")
    runner = QwenRunner(config)
    ensemble = ValueEnsemble.load(critic_path)
    comparison = config["comparison"]
    budget = int(comparison["shared_budget"])
    chunk = int(comparison["chunk"])
    horizons = [int(value) for value in comparison["horizons"]]
    base_seed = int(config["experiment"]["seed"])
    for method in args.methods:
        output = Path(config["output"][method])
        fingerprint = protocol_fingerprint(config, method, dataset_sha256)
        completed = {}
        if args.resume and output.exists():
            for row in read_jsonl(output):
                if row.get("method") != method:
                    raise ValueError(f"{output} contains a different method")
                if row.get("run_fingerprint") != fingerprint:
                    raise ValueError(f"{output} was produced by a different protocol; archive it before rerun")
                if row["suite_id"] in completed:
                    raise ValueError(f"{output} contains duplicate suite {row['suite_id']}")
                completed[row["suite_id"]] = row
        for index, suite in enumerate(suites):
            suite_id = suite["suite_id"]
            if suite_id in completed:
                print(f"[{method}] resume {suite_id}", flush=True)
                continue
            result = run_savi_variant(
                suite, runner, ensemble, method=method, shared_budget=budget,
                chunk=chunk, horizons=horizons, beta=float(comparison["beta"]),
                voi_lambda=float(comparison["voi_lambda"]),
                process_std=float(comparison["process_std"]),
                measurement_noise_floor=float(comparison["measurement_noise_floor"]),
                seed=stable_seed(base_seed, method, suite_id),
            )
            result["suite_index"] = index
            result["dataset_sha256"] = dataset_sha256
            result["run_fingerprint"] = fingerprint
            append_jsonl(output, [result])
            completed[suite_id] = result
            print(json.dumps({"method": method, "suite_id": suite_id,
                              "score": result["score"],
                              "spent_tokens": sum(item["executed_tokens"]
                                                   for item in result["trajectory"]),
                              "steps": len(result["trajectory"]) }), flush=True)


if __name__ == "__main__":
    main()
