from __future__ import annotations

import argparse
import hashlib
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import load_yaml, read_jsonl, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(config: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(config["data"]["path"])
    rows = read_jsonl(data_path)
    suites: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        suites[row["suite_id"]].append(row)
    malformed = {key: len(value) for key, value in suites.items() if len(value) != 6}
    if malformed:
        raise ValueError(f"Expected six problems per suite: {malformed}")
    suite_ids = sorted(suites)
    explicit_problem_ids = config["experiment"].get("problem_ids")
    if explicit_problem_ids:
        if len(explicit_problem_ids) != len(set(explicit_problem_ids)):
            raise ValueError("experiment.problem_ids must be unique")
        by_problem = {row["problem_id"]: row for row in rows}
        missing = sorted(set(explicit_problem_ids) - set(by_problem))
        if missing:
            raise ValueError(f"Unknown experiment.problem_ids: {missing}")
        selected_rows = [by_problem[problem_id] for problem_id in explicit_problem_ids]
        selected = sorted({row["suite_id"] for row in selected_rows})
        sampling_unit = "explicit_problem_diagnostic"
    else:
        rng = random.Random(int(config["experiment"]["seed"]))
        selected = sorted(rng.sample(suite_ids, int(config["experiment"]["suite_count"])))
        selected_rows = [row for suite_id in selected for row in
                         sorted(suites[suite_id], key=lambda item: item["position"])]
        sampling_unit = "complete_six_problem_suite"
    problems = []
    for row in selected_rows:
        problems.append(
            {
                "suite_id": row["suite_id"],
                "position": row["position"],
                "problem_id": row["problem_id"],
                "difficulty": row.get("difficulty"),
                "statement_sha256": row.get("statement_sha256"),
            }
        )
    return {
        "schema_version": 1,
        "seed": int(config["experiment"]["seed"]),
        "sampling_unit": sampling_unit,
        "dataset_path": str(data_path),
        "dataset_sha256": file_sha256(data_path),
        "selected_suite_ids": selected,
        "problem_count": len(problems),
        "problems": problems,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    manifest = build_manifest(config)
    write_json(config["data"]["split_manifest"], manifest)
    print(f"Wrote {manifest['problem_count']} problems from "
          f"{len(manifest['selected_suite_ids'])} suites")


if __name__ == "__main__":
    main()
