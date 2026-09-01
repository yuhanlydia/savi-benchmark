.PHONY: test plan phase0 analyze critic

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
