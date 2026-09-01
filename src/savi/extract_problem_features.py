from __future__ import annotations

import argparse
from pathlib import Path

from .io import append_jsonl, load_yaml, read_jsonl
from .manifest import build_manifest
from .phase0 import QwenRunner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    args = parser.parse_args()
    config = load_yaml(args.config)
    manifest = build_manifest(config)
    selected = {row["problem_id"] for row in manifest["problems"]}
    problems = {row["problem_id"]: row for row in read_jsonl(config["data"]["path"])
                if row["problem_id"] in selected}
    output = Path(config["output"]["problem_features"])
    completed = {row["problem_id"] for row in read_jsonl(output)} if output.exists() else set()
    runner = QwenRunner(config)
    for index, problem_id in enumerate(sorted(problems), 1):
        if problem_id in completed:
            continue
        row = {"problem_id": problem_id, **runner.problem_features(problems[problem_id]["problem"])}
        append_jsonl(output, [row])
        print(f"[{index}/{len(problems)}] {problem_id}", flush=True)


if __name__ == "__main__":
    main()
