from __future__ import annotations

import argparse
import json
from pathlib import Path

from savi.io import append_jsonl, load_yaml, read_jsonl, stable_seed, write_json
from savi.phase0 import QwenRunner
from savi.shared_budget_eval import (
    load_fixed_suites, protocol_fingerprint, run_equal_suite, run_native_suite,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exploratory_shared_budget_24gb.yaml")
    parser.add_argument("--methods", nargs="+", choices=["equal", "native"], required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    suites, dataset_sha256 = load_fixed_suites(config)
    output_root = Path(config["output"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "config.snapshot.json", config)
    runner = QwenRunner(config)
    budget = int(config["comparison"]["shared_budget"])
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
            if suite["suite_id"] in completed:
                print(f"[{method}] resume {suite['suite_id']}", flush=True)
                continue
            seed = stable_seed(base_seed, method, suite["suite_id"])
            if method == "equal":
                result = run_equal_suite(suite, runner, shared_budget=budget, seed=seed)
            else:
                result = run_native_suite(suite, runner, shared_budget=budget, seed=seed)
            result["suite_index"] = index
            result["dataset_sha256"] = dataset_sha256
            result["run_fingerprint"] = fingerprint
            append_jsonl(output, [result])
            completed[suite["suite_id"]] = result
            print(json.dumps({"method": method, "suite_id": suite["suite_id"],
                              "score": result["score"],
                              "spent_tokens": result["spent_tokens"]}), flush=True)


if __name__ == "__main__":
    main()
