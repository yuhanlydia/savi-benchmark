from pathlib import Path

import yaml

from savi.shared_budget_eval import (
    equal_allocation,
    load_fixed_suites,
    protocol_fingerprint,
    render_native_contest_prompt,
    run_equal_suite,
    run_native_suite,
    run_savi_variant,
)


ROOT = Path(__file__).resolve().parents[1]


def test_exploratory_config_binds_exp0c_suite_manifest():
    with (ROOT / "configs/exploratory_shared_budget_24gb.yaml").open() as handle:
        config = yaml.safe_load(handle)

    assert config["comparison"]["shared_budget"] == 4096
    assert config["comparison"]["chunk"] == 512
    assert config["comparison"]["suite_ids"] == [
        "math_suite_014",
        "math_suite_024",
        "math_suite_026",
        "math_suite_038",
        "math_suite_043",
    ]


def test_manifest_loader_returns_exact_five_suites_and_dataset_hash():
    with (ROOT / "configs/exploratory_shared_budget_24gb.yaml").open() as handle:
        config = yaml.safe_load(handle)
    suites, dataset_sha256 = load_fixed_suites(config)
    assert [suite["suite_id"] for suite in suites] == config["comparison"]["suite_ids"]
    assert all(len(suite["problems"]) == 6 for suite in suites)
    assert len(dataset_sha256) == 64
    assert len(protocol_fingerprint(config, "online_savi", dataset_sha256)) == 64


def test_equal_allocation_spends_no_more_than_shared_budget():
    allocation = equal_allocation(4096, 6)
    assert sum(allocation) <= 4096
    assert max(allocation) - min(allocation) <= 1


def test_native_prompt_contains_all_six_numbered_problems():
    suite = {
        "suite_id": "math_suite_014",
        "problems": [
            {"problem_id": f"p{i}", "problem": f"Problem {i}", "answer": str(i)}
            for i in range(1, 7)
        ],
    }
    prompt = render_native_contest_prompt(suite, 4096)
    assert prompt.count("## Problem") == 6
    assert "total output budget" in prompt


class FakeRunner:
    def generate_prefix(self, problem, budget, seed):
        return list(range(budget))

    def finalize(self, trace_ids):
        return f"Final Answer: \\boxed{{{len(trace_ids)}}}"

    def generate_native(self, prompt, budget, seed):
        return "\n\n".join(
            f"## Problem {i}\nFinal Answer: \\boxed{{{i}}}" for i in range(1, 7)
        )


def test_equal_result_has_six_answers_and_exact_budget():
    suite = {
        "suite_id": "math_suite_014",
        "problems": [
            {"problem_id": f"p{i}", "problem": f"Problem {i}", "answer": str(i)}
            for i in range(1, 7)
        ],
    }
    result = run_equal_suite(suite, FakeRunner(), shared_budget=4096, seed=7)
    assert len(result["answers"]) == 6
    assert sum(row["spent_tokens"] for row in result["answers"]) == 4096
    assert result["shared_budget"] == 4096


def test_native_result_has_six_answers_and_budget_cap():
    suite = {
        "suite_id": "math_suite_014",
        "problems": [
            {"problem_id": f"p{i}", "problem": f"Problem {i}", "answer": str(i)}
            for i in range(1, 7)
        ],
    }
    result = run_native_suite(suite, FakeRunner(), shared_budget=4096, seed=7)
    assert len(result["answers"]) == 6
    assert result["spent_tokens"] == 4096


class FakeSaviRunner:
    def state_features(self, problem, prefix_ids):
        return {
            "last_hidden": [0.1, 0.2],
            "recent_token_entropy": 0.3,
            "has_candidate_answer": False,
            "closed_thinking_stage": False,
            "recent_repetition_rate": 0.0,
        }

    def continue_prefix(self, problem, spent, prefix_ids, horizon, seed):
        return [1] * horizon

    def finalize(self, trace_ids):
        return "Final Answer: \\boxed{1}"


class FakeEnsemble:
    def predict(self, rows):
        import numpy as np
        return np.full(len(rows), 0.5), np.zeros(len(rows))


def test_savi_variant_binds_same_budget_and_suite():
    suite = {
        "suite_id": "math_suite_014",
        "problems": [
            {"problem_id": f"p{i}", "problem": f"Problem {i}", "answer": "1", "position": i}
            for i in range(1, 7)
        ],
    }
    result = run_savi_variant(
        suite, FakeSaviRunner(), FakeEnsemble(), method="online_savi",
        shared_budget=4096, chunk=512, horizons=[0, 512, 1024, 2048],
        beta=1.0, voi_lambda=0.25, seed=7,
    )
    assert result["shared_budget"] == 4096
    assert result["suite_id"] == "math_suite_014"
    assert sum(item["executed_tokens"] for item in result["trajectory"]) <= 4096
    assert len(result["answers"]) == 6
