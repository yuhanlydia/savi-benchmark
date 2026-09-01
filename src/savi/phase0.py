from __future__ import annotations

import argparse
import json
import math
import time
import shutil
from pathlib import Path
from typing import Any

from .io import append_jsonl, load_yaml, read_jsonl, stable_seed, write_json
from .manifest import build_manifest
from .provenance import write_once


def reasoning_prompt(problem: str, trajectory_budget: int, announce_budget: bool = True) -> str:
    budget_text = (
        f"Your total reasoning budget is limited to {trajectory_budget} tokens. "
        "Keep the reasoning concise enough to fit. "
        if announce_budget else ""
    )
    return (
        "You are taking a mathematics contest. This is the reasoning stage. "
        + budget_text
        + "A second stage will receive only your stopped reasoning trace and format "
        "the answer. Make the intended final answer easy to identify.\n\nProblem:\n"
        + problem
    )


def natural_stop_ids(configured_eos: int | list[int], thinking_end_id: int) -> list[int]:
    stop_ids = list(configured_eos) if isinstance(configured_eos, list) else [configured_eos]
    if thinking_end_id not in stop_ids:
        stop_ids.append(thinking_end_id)
    return stop_ids


def build_jobs(config: dict[str, Any], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    exp = config["experiment"]
    jobs = []
    for problem in manifest["problems"]:
        for spent in exp["spent_budgets"]:
            for prefix_id in range(exp["prefixes_per_cell"]):
                state_id = f"{problem['problem_id']}-b{spent}-m{prefix_id}"
                prefix_seed = stable_seed(exp["seed"], problem["problem_id"], spent, prefix_id)
                for horizon in exp["continuation_horizons"]:
                    repeats = 1 if horizon == 0 else exp["continuations_per_state"]
                    for continuation_id in range(repeats):
                        jobs.append(
                            {
                                **problem,
                                "state_id": state_id,
                                "spent_budget": spent,
                                "prefix_id": prefix_id,
                                "prefix_seed": prefix_seed,
                                "horizon": horizon,
                                "continuation_id": continuation_id,
                                "continuation_seed": stable_seed(
                                    exp["seed"], state_id, horizon, continuation_id
                                ),
                            }
                        )
    return jobs


class QwenRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        model_cfg = config["model"]
        model_path = model_cfg["path"]
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="auto",
            quantization_config=quant,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).eval()
        self.generation = {
            "do_sample": True,
            "temperature": float(model_cfg["temperature"]),
            "top_p": float(model_cfg["top_p"]),
            "top_k": int(model_cfg["top_k"]),
        }
        self.finalizer_max_tokens = int(model_cfg["finalizer_max_tokens"])
        self.trajectory_budget = int(config["experiment"]["trajectory_budget"])

    def _prompt_ids(self, problem: str, announce_budget: bool = True) -> Any:
        content = reasoning_prompt(problem, self.trajectory_budget, announce_budget)
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)

    def generate_prefix(self, problem: str, budget: int, seed: int) -> list[int]:
        self.torch.manual_seed(seed)
        input_ids = self._prompt_ids(problem)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                max_new_tokens=budget,
                min_new_tokens=budget,
                **self.generation,
            )
        return output[0, input_ids.shape[1] :].tolist()

    def generate_until_stop(
        self, problem: str, max_tokens: int, seed: int, announce_budget: bool = False,
        stop_at_thinking_end: bool = True,
    ) -> tuple[list[int], bool]:
        self.torch.manual_seed(seed)
        input_ids = self._prompt_ids(problem, announce_budget=announce_budget)
        configured_eos = self.model.generation_config.eos_token_id
        if stop_at_thinking_end:
            stop_ids = natural_stop_ids(
                configured_eos, self.tokenizer.convert_tokens_to_ids("</think>")
            )
        else:
            stop_ids = list(configured_eos) if isinstance(configured_eos, list) else [configured_eos]
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                max_new_tokens=max_tokens,
                eos_token_id=stop_ids,
                **self.generation,
            )
        generated = output[0, input_ids.shape[1] :].tolist()
        ended = bool(generated and generated[-1] in stop_ids)
        return generated, ended

    def continue_prefix(
        self, problem: str, spent: int, prefix_ids: list[int], horizon: int, seed: int
    ) -> list[int]:
        if horizon == 0:
            return []
        self.torch.manual_seed(seed)
        prompt = self._prompt_ids(problem)
        prefix = self.torch.tensor([prefix_ids], device=self.model.device)
        input_ids = self.torch.cat([prompt, prefix], dim=1)
        with self.torch.inference_mode():
            output = self.model.generate(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                max_new_tokens=horizon,
                min_new_tokens=horizon,
                **self.generation,
            )
        return output[0, input_ids.shape[1] :].tolist()

    def state_features(self, problem: str, prefix_ids: list[int]) -> dict[str, Any]:
        prompt = self._prompt_ids(problem)
        prefix = self.torch.tensor([prefix_ids], device=self.model.device)
        input_ids = self.torch.cat([prompt, prefix], dim=1)
        with self.torch.inference_mode():
            outputs = self.model(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                output_hidden_states=True,
                use_cache=False,
            )
        hidden = outputs.hidden_states[-1][0, -1].float().cpu().tolist()
        window = min(32, outputs.logits.shape[1])
        logits = outputs.logits[0, -window:].float()
        probabilities = self.torch.softmax(logits, dim=-1)
        entropy = -(probabilities * self.torch.log(probabilities.clamp_min(1e-12))).sum(-1).mean()
        trace = self.tokenizer.decode(prefix_ids, skip_special_tokens=True)
        recent = prefix_ids[-32:]
        repetition = 0.0 if not recent else 1.0 - len(set(recent)) / len(recent)
        return {
            "last_hidden": hidden,
            "recent_token_entropy": float(entropy.cpu()),
            "has_candidate_answer": bool("\\boxed" in trace or "final answer" in trace.casefold()),
            "recent_repetition_rate": float(repetition),
        }

    def problem_features(self, problem: str) -> dict[str, Any]:
        input_ids = self._prompt_ids(problem)
        with self.torch.inference_mode():
            outputs = self.model(
                input_ids,
                attention_mask=self.torch.ones_like(input_ids),
                output_hidden_states=True,
                use_cache=False,
            )
        return {"problem_hidden": outputs.hidden_states[-1][0, -1].float().cpu().tolist()}

    def finalize(self, trace_ids: list[int]) -> str:
        trace = self.tokenizer.decode(trace_ids, skip_special_tokens=True)
        content = (
            "You are a strict answer finalizer. Do not solve from scratch. Use only "
            "the stopped reasoning trace below. Output exactly one line in the form "
            "Final Answer: \\boxed{...}. If no usable answer exists, output "
            "Final Answer: \\boxed{No answer}.\n\nStopped reasoning trace:\n" + trace
        )
        text = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(text, return_tensors="pt").input_ids.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(
                inputs,
                attention_mask=self.torch.ones_like(inputs),
                max_new_tokens=self.finalizer_max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
        return self.tokenizer.decode(output[0, inputs.shape[1] :], skip_special_tokens=True)


def normalize_answer(text: str) -> str:
    import re
    from r3bench.math.parser import extract_math_answer

    answer = extract_math_answer(text)
    if answer is None:
        return ""
    value = re.sub(r"\\(?:left|right)", "", answer)
    value = re.sub(r"\s+", "", value)
    value = value.replace("$", "").casefold()
    return value


def job_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return row["state_id"], int(row["horizon"]), int(row["continuation_id"])


def pending_jobs(jobs: list[dict[str, Any]], completed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    completed = {job_key(row) for row in completed_rows}
    return [job for job in jobs if job_key(job) not in completed]


def execute(config: dict[str, Any], jobs: list[dict[str, Any]], max_hours: float | None = None) -> None:
    data = {row["problem_id"]: row for row in read_jsonl(config["data"]["path"])}
    runner = QwenRunner(config)
    output_path = Path(config["output"]["continuations"])
    completed_rows = read_jsonl(output_path) if output_path.exists() else []
    completed = {job_key(row) for row in completed_rows}
    prefix_cache: dict[str, list[int]] = {}
    prefix_features: dict[str, dict[str, Any]] = {}
    prefix_path = Path(config["output"]["prefixes"])
    if prefix_path.exists():
        for row in read_jsonl(prefix_path):
            prefix_cache[row["state_id"]] = row["prefix_token_ids"]
            prefix_features[row["state_id"]] = row
    started = time.monotonic()
    deadline = None if max_hours is None else started + max_hours * 3600
    for index, job in enumerate(jobs, 1):
        if deadline is not None and time.monotonic() >= deadline:
            print(json.dumps({"status": "time_budget_reached", "max_hours": max_hours,
                              "completed_jobs": len(completed)}), flush=True)
            return
        key = job_key(job)
        if key in completed:
            continue
        problem_row = data[job["problem_id"]]
        if job["state_id"] not in prefix_cache:
            prefix_cache[job["state_id"]] = runner.generate_prefix(
                problem_row["problem"], job["spent_budget"], job["prefix_seed"]
            )
            features = runner.state_features(problem_row["problem"], prefix_cache[job["state_id"]])
            prefix_row = {
                "state_id": job["state_id"], "problem_id": job["problem_id"],
                "suite_id": job["suite_id"], "position": job["position"],
                "spent_budget": job["spent_budget"], "prefix_id": job["prefix_id"],
                "prefix_seed": job["prefix_seed"], "prefix_token_ids": prefix_cache[job["state_id"]],
                **features,
            }
            append_jsonl(prefix_path, [prefix_row])
            prefix_features[job["state_id"]] = prefix_row
        prefix_ids = prefix_cache[job["state_id"]]
        continuation_ids = runner.continue_prefix(
            problem_row["problem"],
            job["spent_budget"],
            prefix_ids,
            job["horizon"],
            job["continuation_seed"],
        )
        final_text = runner.finalize(prefix_ids + continuation_ids)
        predicted = normalize_answer(final_text)
        reference = normalize_answer(f"Final Answer: \\boxed{{{problem_row['answer']}}}")
        row = {
            **job,
            "continuation_token_ids": continuation_ids,
            "finalizer_output": final_text,
            "parsed_answer_normalized": predicted,
            "reference_answer_normalized": reference,
            "correct_exact_normalized": bool(predicted) and predicted == reference,
            "scoring_note": "pilot exact-normalized; confirm nontrivial equivalence with official judge",
        }
        append_jsonl(output_path, [row])
        completed.add(key)
        print(f"[{index}/{len(jobs)}] {job['state_id']} h={job['horizon']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase0_math.yaml")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-hours", type=float)
    args = parser.parse_args()
    config = load_yaml(args.config)
    manifest = build_manifest(config)
    write_json(config["data"]["split_manifest"], manifest)
    jobs = build_jobs(config, manifest)
    plan_path = Path(config["output"]["root"]) / "jobs.json"
    write_json(plan_path, jobs)
    prefix_count = (
        manifest["problem_count"]
        * len(config["experiment"]["spent_budgets"])
        * config["experiment"]["prefixes_per_cell"]
    )
    print(json.dumps({"problems": manifest["problem_count"], "prefixes": prefix_count,
                      "continuation_jobs": len(jobs), "plan": str(plan_path)}, indent=2))
    if args.execute:
        if args.max_hours is not None and args.max_hours <= 0:
            parser.error("--max-hours must be positive")
        provenance_path = Path(config["output"]["root"]) / "execution_provenance.json"
        if not provenance_path.exists():
            write_once(args.config, provenance_path)
        config_snapshot = Path(config["output"]["root"]) / "config.snapshot.yaml"
        if not config_snapshot.exists():
            config_snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.config, config_snapshot)
        execute(config, jobs, max_hours=args.max_hours)


if __name__ == "__main__":
    main()
