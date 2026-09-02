from __future__ import annotations

import argparse
from pathlib import Path

from .io import append_jsonl, load_yaml, read_jsonl
from .phase0 import QwenRunner, normalize_answer


def repair_rows(config: dict, input_path: str, output_path: str, max_tokens: int) -> int:
    if Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite repair output: {output_path}")
    rows = read_jsonl(input_path)
    prefixes = {row["state_id"]: row["prefix_token_ids"]
                for row in read_jsonl(config["output"]["prefixes"])}
    problems = {row["problem_id"]: row for row in read_jsonl(config["data"]["path"])}
    runner = QwenRunner(config)
    repaired = []
    for row in rows:
        if row.get("parsed_answer_normalized"):
            repaired.append({**row, "finalizer_repaired": False})
            continue
        trace_ids = prefixes[row["state_id"]] + row["continuation_token_ids"]
        output = runner.finalize(trace_ids, max_tokens=max_tokens)
        predicted = normalize_answer(output)
        reference = normalize_answer(f"Final Answer: \\boxed{{{problems[row['problem_id']]['answer']}}}")
        repaired.append({
            **row,
            "finalizer_output_original": row.get("finalizer_output", ""),
            "parsed_answer_normalized_original": row.get("parsed_answer_normalized", ""),
            "finalizer_output": output,
            "parsed_answer_normalized": predicted,
            "reference_answer_normalized": reference,
            "correct_exact_normalized": bool(predicted) and predicted == reference,
            "finalizer_repaired": True,
            "finalizer_repair_max_tokens": max_tokens,
        })
    append_jsonl(output_path, repaired)
    return sum(bool(row.get("finalizer_repaired")) for row in repaired)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    print(f"repaired={repair_rows(load_yaml(args.config), args.input, args.output, args.max_tokens)}")


if __name__ == "__main__":
    main()
