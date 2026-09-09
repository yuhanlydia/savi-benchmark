.PHONY: test plan phase0 phase0-10h exp0b exp0b-10h exp0b-analyze exp0b-postprocess batch-smoke batch-smoke-24 exp0c-16 exp0c-24 exp0c-analyze-16 exp0c-analyze-24 exp0c-gates-16 exp0c-gates-24 exploratory-backfill exploratory-critic exploratory-baselines exploratory-savi exploratory-analyze analyze critic monitor

test:
	.venv/bin/python -m pytest -q

plan:
	.venv/bin/python -m savi.phase0 --config configs/phase0_math.yaml

phase0:
	.venv/bin/python -m savi.phase0 --config configs/phase0_math.yaml --execute

phase0-10h:
	.venv/bin/python -m savi.phase0 --config configs/phase0_math.yaml --execute --max-hours 10

exp0b:
	.venv/bin/python -m savi.phase0 --config configs/exp0b_math.yaml --execute

exp0b-10h:
	.venv/bin/python -m savi.phase0 --config configs/exp0b_math.yaml --execute --max-hours 10

exp0b-analyze:
	.venv/bin/python -m savi.analysis --config configs/exp0b_math.yaml

exp0b-postprocess:
	bash scripts/postprocess_exp0b.sh

batch-smoke:
	.venv/bin/python -m savi.batched_phase0 --config configs/batch_smoke_math.yaml --execute

batch-smoke-24:
	.venv/bin/python -m savi.batched_phase0 --config configs/batch_smoke_math_24gb.yaml --execute

exp0c-16:
	.venv/bin/python -m savi.batched_phase0 --config configs/exp0c_math_batched_16gb.yaml --execute

exp0c-24:
	.venv/bin/python -m savi.batched_phase0 --config configs/exp0c_math_batched_24gb.yaml --execute

exp0c-analyze-16:
	.venv/bin/python -m savi.analysis --config configs/exp0c_math_batched_16gb.yaml
	.venv/bin/python -m savi.analysis --config configs/exp0c_math_batched_16gb.yaml --nonterminal-only

exp0c-analyze-24:
	.venv/bin/python -m savi.analysis --config configs/exp0c_math_batched_24gb.yaml
	.venv/bin/python -m savi.analysis --config configs/exp0c_math_batched_24gb.yaml --nonterminal-only

exp0c-gates-16:
	.venv/bin/python -m savi.gate_decision --config configs/exp0c_math_batched_16gb.yaml --report outputs/exp0c_math_batched_16gb/gates.json

exp0c-gates-24:
	.venv/bin/python -m savi.gate_decision --config configs/exp0c_math_batched_24gb.yaml --report outputs/exp0c_math_batched_24gb/gates.json

exploratory-backfill:
	PYTHONPATH=src .venv/bin/python scripts/backfill_exp0c_state_features.py --config configs/exploratory_shared_budget_24gb.yaml --input outputs/exp0c_math_batched_24gb/prefixes.jsonl --output outputs/exploratory_shared_budget_24gb/prefixes_with_features.jsonl --resume

exploratory-critic:
	PYTHONPATH=src .venv/bin/python -m savi.train_critic --config configs/exploratory_critic_24gb.yaml --representation state-aware

exploratory-baselines:
	PYTHONPATH=src .venv/bin/python scripts/run_shared_budget_baselines.py --config configs/exploratory_shared_budget_24gb.yaml --methods equal native --resume

exploratory-savi:
	PYTHONPATH=src .venv/bin/python scripts/run_shared_budget_savi.py --config configs/exploratory_shared_budget_24gb.yaml --methods direct_savi online_savi_no_voi online_savi frozen_index --resume

exploratory-analyze:
	PYTHONPATH=src .venv/bin/python scripts/analyze_shared_budget_comparison.py --config configs/exploratory_shared_budget_24gb.yaml

analyze:
	.venv/bin/python -m savi.analysis --config configs/phase0_math.yaml

critic:
	.venv/bin/python -m savi.train_critic --config configs/phase0_math.yaml

problem-features:
	.venv/bin/python -m savi.extract_problem_features --config configs/phase0_math.yaml

budget-critic:
	.venv/bin/python -m savi.train_critic --config configs/phase0_math.yaml --representation budget-only

monitor:
	.venv/bin/python -m savi.monitor --config configs/phase0_math.yaml

telemetry:
	.venv/bin/python -m savi.telemetry --config configs/phase0_math.yaml

summary:
	.venv/bin/python -m savi.summarize --config configs/phase0_math.yaml
