.PHONY: test plan phase0 analyze critic monitor

test:
	.venv/bin/python -m pytest -q

plan:
	.venv/bin/python -m savi.phase0 --config configs/phase0_math.yaml

phase0:
	.venv/bin/python -m savi.phase0 --config configs/phase0_math.yaml --execute

phase0-10h:
	.venv/bin/python -m savi.phase0 --config configs/phase0_math.yaml --execute --max-hours 10

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
