from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

from .io import load_yaml, read_jsonl, write_json


def build_prefix_audit(rows: list[dict], terminal_ids: set[int]) -> dict:
    by_budget: dict[int, list[dict]] = defaultdict(list)
    audited = []
    for row in rows:
        tokens = [int(token) for token in row["prefix_token_ids"]]
        positions = [index for index, token in enumerate(tokens) if token in terminal_ids]
        item = {
            "state_id": row["state_id"],
            "problem_id": row["problem_id"],
            "spent_budget": int(row["spent_budget"]),
            "terminal_token_observed": bool(positions),
            "first_terminal_token_position": positions[0] if positions else None,
        }
        audited.append(item)
        by_budget[item["spent_budget"]].append(item)
    budget_summary = {}
    for budget, items in sorted(by_budget.items()):
        terminal = [item["terminal_token_observed"] for item in items]
        budget_summary[str(budget)] = {
            "states": len(items),
            "terminal_prefix_fraction": float(np.mean(terminal)),
        }
    return {
        "states": len(audited),
        "terminal_prefix_fraction": (
            float(np.mean([row["terminal_token_observed"] for row in audited]))
            if audited else None
        ),
        "by_spent_budget": budget_summary,
        "states_detail": audited,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_yaml(args.config)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["path"], local_files_only=True)
    terminal_ids = {
        tokenizer.convert_tokens_to_ids("</think>"),
        tokenizer.convert_tokens_to_ids("<|im_end|>"),
        tokenizer.eos_token_id,
    }
    report = build_prefix_audit(read_jsonl(config["output"]["prefixes"]), terminal_ids)
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
