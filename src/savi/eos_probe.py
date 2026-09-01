from __future__ import annotations

import argparse
import time
from pathlib import Path

from .io import append_jsonl, load_yaml, read_jsonl, stable_seed
from .phase0 import QwenRunner, normalize_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eos_probe_math.yaml")
    parser.add_argument("--problem-id", action="append", required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--output", default="outputs/eos_probe_math/results.jsonl")
    args = parser.parse_args(); config = load_yaml(args.config)
    problems = {row["problem_id"]: row for row in read_jsonl(config["data"]["path"])}
    missing = set(args.problem_id) - set(problems)
    if missing:
        raise ValueError(f"unknown problem ids: {sorted(missing)}")
    output = Path(args.output)
    completed = {(row["problem_id"], row["sample_id"]) for row in read_jsonl(output)} if output.exists() else set()
    runner = QwenRunner(config)
    for problem_id in args.problem_id:
        for sample_id in range(args.samples):
            if (problem_id, sample_id) in completed:
                continue
            seed = stable_seed(config["experiment"]["seed"], problem_id, sample_id)
            started = time.monotonic()
            tokens, ended = runner.generate_until_stop(problems[problem_id]["problem"], args.max_tokens, seed)
            finalizer = runner.finalize(tokens)
            append_jsonl(output, [{
                "problem_id": problem_id, "sample_id": sample_id, "seed": seed,
                "max_tokens": args.max_tokens, "generated_tokens": len(tokens),
                "ended_with_eos": ended, "elapsed_seconds": time.monotonic() - started,
                "finalizer_output": finalizer,
                "parsed_answer_normalized": normalize_answer(finalizer),
            }])
            print(f"{problem_id} sample={sample_id} tokens={len(tokens)} eos={ended}", flush=True)


if __name__ == "__main__":
    main()
