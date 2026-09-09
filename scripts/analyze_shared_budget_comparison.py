from __future__ import annotations

import argparse
import json
from pathlib import Path

from savi.io import load_yaml, read_jsonl, write_json
from savi.metrics import paired_suite_bootstrap
from savi.shared_budget_eval import load_fixed_suites, protocol_fingerprint


def result_spent(row: dict) -> int:
    if "trajectory" in row:
        return sum(int(item["executed_tokens"]) for item in row["trajectory"])
    return int(row.get("spent_tokens", 0))


def load_results(config: dict) -> tuple[list[dict], dict[str, list[dict]]]:
    methods = ["equal", "native", "direct_savi", "online_savi_no_voi", "online_savi", "frozen_index"]
    _, dataset_sha256 = load_fixed_suites(config)
    expected = set(config["comparison"]["suite_ids"])
    loaded = {}
    flat = []
    budget = int(config["comparison"]["shared_budget"])
    for method in methods:
        path = Path(config["output"][method])
        rows = read_jsonl(path)
        if {row.get("suite_id") for row in rows} != expected or len(rows) != len(expected):
            raise ValueError(f"{method} does not contain exactly the fixed suite set")
        for row in rows:
            if row.get("dataset_sha256") != dataset_sha256:
                raise ValueError(f"{method}/{row['suite_id']} has the wrong dataset SHA")
            if row.get("run_fingerprint") != protocol_fingerprint(config, method, dataset_sha256):
                raise ValueError(f"{method}/{row['suite_id']} has the wrong protocol fingerprint")
            if row.get("protocol_label") != "exploratory_same_suite_shared_budget":
                raise ValueError(f"{method} is missing exploratory protocol label")
            if result_spent(row) > budget:
                raise ValueError(f"{method}/{row['suite_id']} exceeds shared budget")
            if len(row.get("answers", [])) != 6:
                raise ValueError(f"{method}/{row['suite_id']} does not have six answers")
            flat.append({"suite_id": row["suite_id"], "method": method, "score": row["score"]})
        loaded[method] = rows
    return flat, loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exploratory_shared_budget_24gb.yaml")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    config = load_yaml(args.config)
    flat, loaded = load_results(config)
    methods = list(loaded)
    method_summary = {}
    for method, rows in loaded.items():
        scores = [int(row["score"]) for row in rows]
        trajectory_lengths = [len(row.get("trajectory", [])) for row in rows]
        spent = [result_spent(row) for row in rows]
        method_summary[method] = {
            "suite_scores": {row["suite_id"]: int(row["score"]) for row in rows},
            "mean_correct_per_suite": sum(scores) / len(scores),
            "answer_success_rate": sum(scores) / (len(scores) * 6),
            "total_correct_answers": sum(scores),
            "spent_tokens": spent,
            "trajectory_steps": trajectory_lengths,
        }

    pairwise = {}
    for left in methods:
        for right in methods:
            if left == right:
                continue
            pairwise[f"{left}_vs_{right}"] = paired_suite_bootstrap(
                flat, method_a=left, method_b=right, draws=10_000, seed=20260909
            )

    gate_path = Path("outputs/exp0c_math_batched_24gb/gates.json")
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {"missing": True}
    calibration_path = Path(config["calibration"]["critic_dir"]) / "state-aware" / "training_report.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else {"missing": True}
    summary = {
        "status": "exploratory",
        "exploratory": True,
        "reason": "Exp0C preregistered phenomenon gate failed; critic is calibrated on the same fixed suites.",
        "fixed_suite_ids": list(config["comparison"]["suite_ids"]),
        "shared_budget": int(config["comparison"]["shared_budget"]),
        "chunk": int(config["comparison"]["chunk"]),
        "model": config["model"],
        "methods": methods,
        "method_summary": method_summary,
        "paired_suite_bootstrap": pairwise,
        "exp0c_gate_unchanged": gate,
        "critic_calibration": calibration,
        "limitations": [
            "Five fixed suites are used for both critic calibration and allocator comparison.",
            "Exp0C provides labels at spent_budget=4096, so scheduler decisions below that budget extrapolate the critic.",
            "Scores use exact-normalized math answers; this is not an official R3-Bench leaderboard run.",
            "A five-suite paired comparison is exploratory and cannot establish superiority.",
        ],
    }
    output = Path(args.output or config["output"]["summary"])
    write_json(output, summary)
    print(json.dumps({"output": str(output), "methods": method_summary,
                      "pairs": pairwise}, indent=2))


if __name__ == "__main__":
    main()
