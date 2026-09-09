from __future__ import annotations

import re
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .io import read_jsonl, stable_seed
from .phase0 import normalize_answer


_SECTION = re.compile(r"(?im)^\s*(?:#{1,6}\s*)?problem\s+([1-6])\s*:?\s*$")


def equal_allocation(shared_budget: int, problem_count: int) -> list[int]:
    """Return a deterministic balanced allocation whose sum is at most budget."""
    if shared_budget < 0 or problem_count <= 0:
        raise ValueError("shared_budget must be non-negative and problem_count positive")
    base, remainder = divmod(shared_budget, problem_count)
    return [base + (1 if index < remainder else 0) for index in range(problem_count)]


def render_native_contest_prompt(
    suite: dict[str, Any], budget_tokens: int, *, thinking: bool = True,
) -> str:
    """Render the official R-3-Bench natural-language contest structure."""
    problems = suite.get("problems", [])
    if len(problems) != 6:
        raise ValueError("native contest prompt requires exactly six problems")
    numbered = []
    for index, row in enumerate(problems, 1):
        statement = row.get("problem")
        if not statement:
            raise ValueError(f"problem {index} has no statement")
        numbered.append(f"## Problem {index}\n{statement}")
    content = "\n\n".join(numbered)
    reasoning_rule = (
        "- Your reasoning process is included in this budget."
        if thinking else
        "- Keep solutions concise and prioritize complete answer sections."
    )
    return f"""You are taking a mathematics contest.

You will receive 6 problems. Solve as many as you can within the response budget.

Rules:
- Your total output budget for all problems is limited to {int(budget_tokens)} tokens.
{reasoning_rule}
- Your score is the number of problems you answer correctly.
- A missing, malformed, or truncated final answer earns 0 only for that problem.
- You do not need to solve the problems in the presented order.
- First scan all problems, then choose the problems you expect to solve most reliably with the fewest tokens.
- Response sections may appear out of order, but each must keep the original problem number.
- If the budget is tight, prefer concise final answers for more problems over long solutions for fewer problems.
- You may submit any subset of the problems. If you cannot solve a problem, omit its section; do not fabricate or box a placeholder answer.
- For every problem you attempt, put the final answer in \\boxed{{}}.
Use exactly this output structure so the answers can be graded:

{content}
"""


def _ordered_problems(suite: dict[str, Any]) -> list[dict[str, Any]]:
    problems = sorted(suite.get("problems", []), key=lambda row: int(row.get("position", 0)))
    if len(problems) != 6:
        raise ValueError("R3 suite must contain exactly six problems")
    return problems


def _answer_record(problem: dict[str, Any], text: str, spent_tokens: int | None) -> dict[str, Any]:
    predicted = normalize_answer(text)
    reference = normalize_answer(f"Final Answer: \\boxed{{{problem['answer']}}}")
    return {
        "problem_id": problem["problem_id"],
        "position": problem.get("position"),
        "spent_tokens": None if spent_tokens is None else int(spent_tokens),
        "model_output": text,
        "parsed_answer_normalized": predicted,
        "reference_answer_normalized": reference,
        "correct_exact_normalized": bool(predicted) and predicted == reference,
    }


def run_equal_suite(
    suite: dict[str, Any], runner: Any, *, shared_budget: int, seed: int,
) -> dict[str, Any]:
    """Run six independent calls with a balanced allocation."""
    problems = _ordered_problems(suite)
    allocation = equal_allocation(shared_budget, len(problems))
    answers = []
    for index, (problem, budget) in enumerate(zip(problems, allocation)):
        trace = runner.generate_prefix(
            problem["problem"], budget, stable_seed(seed, "equal", problem["problem_id"], index)
        )
        output = runner.finalize(trace)
        answers.append(_answer_record(problem, output, len(trace)))
    return {
        "suite_id": suite["suite_id"],
        "method": "equal",
        "protocol_label": "exploratory_same_suite_shared_budget",
        "shared_budget": int(shared_budget),
        "allocated_tokens": allocation,
        "spent_tokens": sum(row["spent_tokens"] for row in answers),
        "answers": answers,
        "score": sum(row["correct_exact_normalized"] for row in answers),
    }


def parse_native_sections(text: str) -> dict[int, str]:
    matches = list(_SECTION.finditer(text or ""))
    sections: dict[int, str] = {}
    duplicates: set[int] = set()
    for index, match in enumerate(matches):
        label = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        if label in sections:
            duplicates.add(label)
        sections[label] = text[match.end():end].strip()
    for label in duplicates:
        sections.pop(label, None)
    return sections


def run_native_suite(
    suite: dict[str, Any], runner: Any, *, shared_budget: int, seed: int,
) -> dict[str, Any]:
    """Run one official-style contest completion and score its six sections."""
    problems = _ordered_problems(suite)
    prompt = render_native_contest_prompt(
        suite, shared_budget, thinking=bool(getattr(runner, "native_thinking", True))
    )
    output = runner.generate_native(prompt, shared_budget, stable_seed(seed, "native"))
    actual_tokens = min(
        int(getattr(runner, "last_native_token_count", shared_budget)), shared_budget
    )
    sections = parse_native_sections(output)
    answers = []
    for index, problem in enumerate(problems, 1):
        answers.append(_answer_record(problem, sections.get(index, ""), None))
    return {
        "suite_id": suite["suite_id"],
        "method": "native",
        "protocol_label": "exploratory_same_suite_shared_budget",
        "shared_budget": int(shared_budget),
        "spent_tokens": actual_tokens,
        "allocated_tokens": [int(shared_budget)],
        "native_output": output,
        "answers": answers,
        "score": sum(row["correct_exact_normalized"] for row in answers),
    }


def run_savi_variant(
    suite: dict[str, Any], runner: Any, ensemble: Any, *, method: str,
    shared_budget: int, chunk: int, horizons: list[int], beta: float,
    voi_lambda: float, seed: int, process_std: float = 0.10,
    measurement_noise_floor: float = 0.05,
) -> dict[str, Any]:
    """Bind the existing scheduler to one named exploratory comparison variant."""
    variants = {
        "direct_savi": {"dynamic": True, "online_belief": False, "voi_lambda": 0.0},
        "online_savi_no_voi": {"dynamic": True, "online_belief": True, "voi_lambda": 0.0},
        "online_savi": {"dynamic": True, "online_belief": True, "voi_lambda": voi_lambda},
        "frozen_index": {"dynamic": False, "online_belief": True, "voi_lambda": 0.0},
    }
    if method not in variants:
        raise ValueError(f"unknown SAVI comparison method: {method}")
    from .online_scheduler import run_suite

    options = variants[method]
    problems = [{**row, "suite_id": suite["suite_id"]} for row in _ordered_problems(suite)]
    result = run_suite(
        problems, runner, ensemble,
        shared_budget=shared_budget, horizons=horizons, chunk=chunk, beta=beta,
        seed=seed, dynamic=options["dynamic"], online_belief=options["online_belief"],
        voi_lambda=options["voi_lambda"], process_std=process_std,
        measurement_noise_floor=measurement_noise_floor,
    )
    by_id = {row["problem_id"]: row for row in problems}
    for answer in result["answers"]:
        problem = by_id[answer["problem_id"]]
        reference = normalize_answer(f"Final Answer: \\boxed{{{problem['answer']}}}")
        answer["reference_answer_normalized"] = reference
        answer["correct_exact_normalized"] = (
            bool(answer["parsed_answer_normalized"])
            and answer["parsed_answer_normalized"] == reference
        )
    result["method"] = method
    result["protocol_label"] = "exploratory_same_suite_shared_budget"
    result["score"] = sum(row["correct_exact_normalized"] for row in result["answers"])
    return result


def load_fixed_suites(config: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    """Load only the exact problems recorded by the Exp0C manifest."""
    data_path = Path(config["data"]["path"])
    manifest_path = Path(config["data"]["split_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if digest != manifest["dataset_sha256"]:
        raise ValueError(f"dataset SHA mismatch: {digest} != {manifest['dataset_sha256']}")
    expected_ids = list(config["comparison"]["suite_ids"])
    manifest_rows = [row for row in manifest["problems"] if row["suite_id"] in expected_ids]
    expected_by_suite = defaultdict(list)
    for row in manifest_rows:
        expected_by_suite[row["suite_id"]].append(row)
    data_rows = [row for row in read_jsonl(data_path) if row.get("suite_id") in expected_ids]
    actual_by_id = {row["problem_id"]: row for row in data_rows}
    suites = []
    for suite_id in expected_ids:
        expected = sorted(expected_by_suite[suite_id], key=lambda row: int(row["position"]))
        if len(expected) != 6:
            raise ValueError(f"manifest suite {suite_id} does not contain six problems")
        problems = []
        for item in expected:
            row = actual_by_id.get(item["problem_id"])
            if row is None or int(row.get("position", -1)) != int(item["position"]):
                raise ValueError(f"dataset problem identity mismatch for {item['problem_id']}")
            if row.get("statement_sha256") != item["statement_sha256"]:
                raise ValueError(f"statement SHA mismatch for {item['problem_id']}")
            problems.append(row)
        suites.append({"suite_id": suite_id, "problems": problems})
    return suites, digest


def protocol_fingerprint(config: dict[str, Any], method: str, dataset_sha256: str) -> str:
    payload = {
        "method": method,
        "dataset_sha256": dataset_sha256,
        "experiment_seed": config["experiment"]["seed"],
        "model": config["model"],
        "comparison": config["comparison"],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
