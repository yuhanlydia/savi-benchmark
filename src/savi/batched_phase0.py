from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .batching import batch_sampling_seed, microbatch_slices, planned_batch_size
from .io import append_jsonl, load_yaml, read_jsonl, write_json
from .manifest import build_manifest
from .phase0 import QwenRunner, build_jobs, job_key, normalize_answer
from .provenance import write_once


class BatchedQwenRunner(QwenRunner):
    """Qwen runner for a separate grouped-RNG sampling condition.

    It deliberately does not promise token identity with the sequential runner.
    Reproducibility is within this fixed batching contract and config.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        runtime = config.get("runtime", {})
        self.sampling_contract = str(runtime.get("sampling_contract", "grouped_rng_v1"))
        if self.sampling_contract != "grouped_rng_v1":
            raise ValueError("batched runner requires runtime.sampling_contract=grouped_rng_v1")
        self.max_batch_context_tokens = int(runtime.get("max_batch_context_tokens", 14_000))
        self.max_prefix_batch = int(runtime.get("max_prefix_batch", 2))
        self.max_continuation_batch = int(runtime.get("max_continuation_batch", 2))
        self.max_finalizer_batch = int(runtime.get("max_finalizer_batch", 2))
        self.collect_state_features = bool(runtime.get("collect_state_features", False))
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.torch.cuda.is_available():
            self.torch.backends.cuda.matmul.allow_tf32 = True

    def _set_seed(self, seed: int) -> None:
        self.torch.manual_seed(seed)
        if self.torch.cuda.is_available():
            self.torch.cuda.manual_seed_all(seed)

    def prompt_token_count(self, problem: str) -> int:
        return int(self._prompt_ids(problem).shape[1])

    def prefix_batch_size(self, problem: str, budget: int) -> int:
        return planned_batch_size(
            self.prompt_token_count(problem), budget,
            max_batch_size=self.max_prefix_batch,
            max_batch_context_tokens=self.max_batch_context_tokens,
        )

    def continuation_batch_size(self, problem: str, prefix_ids: list[int], horizon: int) -> int:
        return planned_batch_size(
            self.prompt_token_count(problem) + len(prefix_ids), horizon,
            max_batch_size=self.max_continuation_batch,
            max_batch_context_tokens=self.max_batch_context_tokens,
        )

    def generate_prefix_batch(
        self, problem: str, budget: int, count: int, seed: int
    ) -> list[list[int]]:
        if count <= 0:
            return []
        self._set_seed(seed)
        input_ids = self._prompt_ids(problem)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                num_return_sequences=count,
                max_new_tokens=budget,
                min_new_tokens=budget,
                **self.generation,
            )
        width = int(input_ids.shape[1])
        return [row[width:].tolist() for row in output]

    def continue_prefix_batch(
        self, problem: str, prefix_ids: list[int], horizon: int, count: int, seed: int
    ) -> list[list[int]]:
        if count <= 0:
            return []
        if horizon == 0:
            return [[] for _ in range(count)]
        self._set_seed(seed)
        prompt = self._prompt_ids(problem)
        prefix = self.torch.tensor([prefix_ids], device=self.model.device)
        input_ids = self.torch.cat([prompt, prefix], dim=1)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                num_return_sequences=count,
                max_new_tokens=horizon,
                min_new_tokens=horizon,
                **self.generation,
            )
        width = int(input_ids.shape[1])
        return [row[width:].tolist() for row in output]

    def cheap_state_metadata(self, prefix_ids: list[int]) -> dict[str, Any]:
        trace = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
        trace_with_special_tokens = self.tokenizer.decode(prefix_ids, skip_special_tokens=False)
        recent = prefix_ids[-32:]
        repetition = 0.0 if not recent else 1.0 - len(set(recent)) / len(recent)
        return {
            "has_candidate_answer": bool("\\boxed" in trace or "final answer" in trace.casefold()),
            "closed_thinking_stage": "</think>" in trace_with_special_tokens,
            "recent_repetition_rate": float(repetition),
            "feature_mode": "cheap_only",
        }

    def finalize_batch(self, traces: list[list[int]]) -> list[str]:
        if not traces:
            return []
        outputs: list[str] = []
        for start, end in microbatch_slices(len(traces), self.max_finalizer_batch):
            texts = []
            for trace_ids in traces[start:end]:
                trace = self.tokenizer.decode(trace_ids, skip_special_tokens=True)
                content = (
                    "You are a strict answer finalizer. Do not solve from scratch. Use only "
                    "the stopped reasoning trace below. Output exactly one line in the form "
                    "Final Answer: \\boxed{...}. If no usable answer exists, output "
                    "Final Answer: \\boxed{No answer}.\n\nStopped reasoning trace:\n" + trace
                )
                texts.append(self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": content}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                ))
            encoded = self.tokenizer(texts, return_tensors="pt", padding=True)
            encoded = {key: value.to(self.model.device) for key, value in encoded.items()}
            width = int(encoded["input_ids"].shape[1])
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.finalizer_max_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            outputs.extend(
                self.tokenizer.decode(row[width:], skip_special_tokens=True) for row in generated
            )
        return outputs


def _deadline_reached(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() >= deadline


def _prefix_groups(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    order: list[tuple[str, int]] = []
    for job in jobs:
        key = (job["problem_id"], int(job["spent_budget"]))
        if key not in grouped:
            order.append(key)
        grouped[key].setdefault(job["state_id"], job)
    return [
        sorted(grouped[key].values(), key=lambda row: int(row["prefix_id"]))
        for key in order
    ]


def _state_groups(jobs: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for job in jobs:
        if job["state_id"] not in grouped:
            order.append(job["state_id"])
        grouped[job["state_id"]].append(job)
    return [
        sorted(grouped[state_id], key=lambda row: (int(row["horizon"]), int(row["continuation_id"])))
        for state_id in order
    ]


def _scored_row(
    job: dict[str, Any], continuation_ids: list[int], final_text: str,
    reference: str, *, sampling_contract: str, batch_seed: int | None,
) -> dict[str, Any]:
    predicted = normalize_answer(final_text)
    return {
        **job,
        "continuation_token_ids": continuation_ids,
        "finalizer_output": final_text,
        "parsed_answer_normalized": predicted,
        "reference_answer_normalized": reference,
        "correct_exact_normalized": bool(predicted) and predicted == reference,
        "scoring_note": "pilot exact-normalized; confirm nontrivial equivalence with official judge",
        "sampling_contract": sampling_contract,
        "effective_batch_seed": batch_seed,
    }


def execute_batched(
    config: dict[str, Any], jobs: list[dict[str, Any]], max_hours: float | None = None
) -> None:
    data = {row["problem_id"]: row for row in read_jsonl(config["data"]["path"])}
    runner = BatchedQwenRunner(config)
    output_path = Path(config["output"]["continuations"])
    prefix_path = Path(config["output"]["prefixes"])
    completed_rows = read_jsonl(output_path) if output_path.exists() else []
    completed_by_key = {job_key(row): row for row in completed_rows}
    prefix_rows = read_jsonl(prefix_path) if prefix_path.exists() else []
    prefix_by_state = {row["state_id"]: row for row in prefix_rows}

    started = time.monotonic()
    deadline = None if max_hours is None else started + max_hours * 3600
    exp_seed = int(config["experiment"]["seed"])

    # Generate the fixed prefix groups first. Regenerating a partially written group
    # must reproduce already-written members before missing members are appended.
    for group in _prefix_groups(jobs):
        if _deadline_reached(deadline):
            break
        problem_id = group[0]["problem_id"]
        spent = int(group[0]["spent_budget"])
        problem = data[problem_id]["problem"]
        batch_size = runner.prefix_batch_size(problem, spent)
        for start, end in microbatch_slices(len(group), batch_size):
            batch = group[start:end]
            if all(row["state_id"] in prefix_by_state for row in batch):
                continue
            seed = batch_sampling_seed(exp_seed, "prefix", problem_id, spent, start)
            generated = runner.generate_prefix_batch(problem, spent, len(batch), seed)
            new_rows = []
            for job, prefix_ids in zip(batch, generated, strict=True):
                existing = prefix_by_state.get(job["state_id"])
                if existing is not None:
                    if existing["prefix_token_ids"] != prefix_ids:
                        raise RuntimeError(
                            f"resume mismatch for prefix batch member {job['state_id']}"
                        )
                    continue
                if runner.collect_state_features:
                    metadata = runner.state_features(problem, prefix_ids)
                    metadata["feature_mode"] = "full_hidden"
                else:
                    metadata = runner.cheap_state_metadata(prefix_ids)
                prefix_row = {
                    "state_id": job["state_id"],
                    "problem_id": problem_id,
                    "suite_id": job["suite_id"],
                    "position": job["position"],
                    "spent_budget": spent,
                    "prefix_id": job["prefix_id"],
                    "prefix_seed": job["prefix_seed"],
                    "prefix_token_ids": prefix_ids,
                    "sampling_contract": runner.sampling_contract,
                    "effective_batch_seed": seed,
                    "effective_batch_size": len(batch),
                    **metadata,
                }
                prefix_by_state[job["state_id"]] = prefix_row
                new_rows.append(prefix_row)
            if new_rows:
                append_jsonl(prefix_path, new_rows)
        if _deadline_reached(deadline):
            break

    if _deadline_reached(deadline):
        print(json.dumps({"status": "time_budget_reached", "stage": "prefixes"}), flush=True)
        return

    # Immediate finalizers are deterministic, so batching pending states is safe.
    for group in _prefix_groups(jobs):
        if _deadline_reached(deadline):
            break
        pending = [row for row in group if job_key({**row, "horizon": 0, "continuation_id": 0}) not in completed_by_key]
        if not pending:
            continue
        traces = [prefix_by_state[row["state_id"]]["prefix_token_ids"] for row in pending]
        finals = runner.finalize_batch(traces)
        rows_to_write = []
        for state_job, final_text in zip(pending, finals, strict=True):
            job = next(
                item for item in jobs
                if item["state_id"] == state_job["state_id"] and int(item["horizon"]) == 0
            )
            reference = normalize_answer(
                f"Final Answer: \\boxed{{{data[job['problem_id']]['answer']}}}"
            )
            row = _scored_row(
                job, [], final_text, reference,
                sampling_contract=runner.sampling_contract, batch_seed=None,
            )
            completed_by_key[job_key(row)] = row
            rows_to_write.append(row)
        append_jsonl(output_path, rows_to_write)

    # Stochastic continuations are grouped by state and fixed continuation-id ranges.
    for state_jobs in _state_groups(jobs):
        if _deadline_reached(deadline):
            break
        positive_by_horizon: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in state_jobs:
            if int(row["horizon"]) > 0:
                positive_by_horizon[int(row["horizon"])].append(row)
        if not positive_by_horizon:
            continue
        first = state_jobs[0]
        state_id = first["state_id"]
        problem_id = first["problem_id"]
        problem = data[problem_id]["problem"]
        prefix_ids = prefix_by_state[state_id]["prefix_token_ids"]
        for horizon, positive in sorted(positive_by_horizon.items()):
            batch_size = runner.continuation_batch_size(problem, prefix_ids, horizon)
            positive = sorted(positive, key=lambda row: int(row["continuation_id"]))
            for start, end in microbatch_slices(len(positive), batch_size):
                if _deadline_reached(deadline):
                    break
                batch = positive[start:end]
                if all(job_key(job) in completed_by_key for job in batch):
                    continue
                seed = batch_sampling_seed(exp_seed, "continuation", state_id, horizon, start)
                generated = runner.continue_prefix_batch(
                    problem, prefix_ids, horizon, len(batch), seed
                )
                finals = runner.finalize_batch([prefix_ids + ids for ids in generated])
                reference = normalize_answer(
                    f"Final Answer: \\boxed{{{data[problem_id]['answer']}}}"
                )
                rows_to_write = []
                for job, continuation_ids, final_text in zip(batch, generated, finals, strict=True):
                    existing = completed_by_key.get(job_key(job))
                    if existing is not None:
                        if existing["continuation_token_ids"] != continuation_ids:
                            raise RuntimeError(
                                f"resume mismatch for continuation batch member {job_key(job)}"
                            )
                        continue
                    row = _scored_row(
                        job, continuation_ids, final_text, reference,
                        sampling_contract=runner.sampling_contract, batch_seed=seed,
                    )
                    completed_by_key[job_key(row)] = row
                    rows_to_write.append(row)
                if rows_to_write:
                    append_jsonl(output_path, rows_to_write)
                print(json.dumps({
                    "status": "progress",
                    "state_id": state_id,
                    "horizon": horizon,
                    "continuation_ids": [int(row["continuation_id"]) for row in batch],
                    "effective_batch_size": len(batch),
                    "completed_jobs": len(completed_by_key),
                    "planned_jobs": len(jobs),
                }), flush=True)

    status = "time_budget_reached" if _deadline_reached(deadline) else "complete"
    print(json.dumps({
        "status": status,
        "sampling_contract": runner.sampling_contract,
        "completed_jobs": len(completed_by_key),
        "planned_jobs": len(jobs),
    }), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp0c_math_batched_16gb.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args()
    if args.max_hours is not None and args.max_hours <= 0:
        parser.error("--max-hours must be positive")
    config = load_yaml(args.config)
    manifest = build_manifest(config)
    write_json(config["data"]["split_manifest"], manifest)
    jobs = build_jobs(config, manifest)
    plan_path = Path(config["output"]["root"]) / "jobs.json"
    write_json(plan_path, jobs)
    print(json.dumps({
        "problems": manifest["problem_count"],
        "prefixes": len({row["state_id"] for row in jobs}),
        "continuation_jobs": len(jobs),
        "sampling_contract": config.get("runtime", {}).get("sampling_contract"),
        "plan": str(plan_path),
    }, indent=2))
    if not args.execute:
        return
    provenance_path = Path(config["output"]["root"]) / "execution_provenance.json"
    if not provenance_path.exists():
        write_once(args.config, provenance_path)
    config_snapshot = Path(config["output"]["root"]) / "config.snapshot.yaml"
    if not config_snapshot.exists():
        config_snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.config, config_snapshot)
    execute_batched(config, jobs, max_hours=args.max_hours)


if __name__ == "__main__":
    main()
