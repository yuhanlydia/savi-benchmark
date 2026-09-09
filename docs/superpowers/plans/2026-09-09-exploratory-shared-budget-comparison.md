# Exploratory Shared-Budget Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare Equal Allocation, Native contest generation, Direct-SAVI, and Online-SAVI on the same five Exp0C Math suites under one shared 4,096-token budget per six-problem suite.

**Architecture:** Freeze the selected suite manifest and model/sampling configuration. Add a feature backfill and exploratory critic calibration from the completed Exp0C states, then run every method through one shared-budget evaluator that records per-problem answers, token spend, trajectory, and per-suite score. The final report will explicitly label critic calibration on evaluation suites as exploratory and will not alter the failed Exp0C gate.

**Tech Stack:** Python 3.10, PyTorch, Transformers, NF4/BF16 Qwen3-8B, scikit-learn/PyTorch value critic, JSONL artifacts, paired suite bootstrap.

**Spec:** `README.md`, `docs/EXPERIMENTS.md`, `docs/ONLINE_VALUE_LEARNING.md`, and the user-requested fixed-suite exploratory comparison.

## Global Constraints

- Fixed evaluation suites: `math_suite_014`, `math_suite_024`, `math_suite_026`, `math_suite_038`, `math_suite_043` from `artifacts/exp0c_math_batched_24gb/manifest.json`.
- Each suite contains exactly six problems and receives a shared budget of 4,096 reasoning tokens.
- Scheduler chunk size is 512 tokens; no correctness labels or verifier rewards are exposed during allocation.
- Model condition is local `models/Qwen3-8B`, NF4 quantization with BF16 compute, temperature 0.6, top-p 0.95, top-k 20.
- All outputs are exploratory because Exp0C's preregistered phenomenon gate failed and critic calibration reuses the fixed-suite Exp0C observations.
- No gate threshold, target metric, or negative result may be weakened or relabeled.
- Full outputs remain local; tracked summaries include hashes, provenance, and the exploratory limitation.

---

### Task 1: Freeze the comparison protocol and add tests

**Files:**
- Create: `configs/exploratory_shared_budget_24gb.yaml`
- Create: `tests/test_shared_budget_eval.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `artifacts/exp0c_math_batched_24gb/manifest.json` and `outputs/exp0c_math_batched_24gb`.
- Produces: explicit suite IDs, `shared_budget=4096`, `chunk=512`, deterministic seeds, output paths, and method names used by later tasks.

- [ ] **Step 1: Write failing protocol tests**

```python
def test_exploratory_config_binds_exp0c_suite_manifest():
    config = load_yaml("configs/exploratory_shared_budget_24gb.yaml")
    assert config["comparison"]["shared_budget"] == 4096
    assert config["comparison"]["chunk"] == 512
    assert config["comparison"]["suite_ids"] == [
        "math_suite_014", "math_suite_024", "math_suite_026",
        "math_suite_038", "math_suite_043",
    ]
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `.venv/bin/python -m pytest -q tests/test_shared_budget_eval.py`

Expected: FAIL because the exploratory config and evaluator module do not exist.

- [ ] **Step 3: Add the explicit config**

The config must bind the existing model/data paths, `shared_budget: 4096`,
`chunk: 512`, `horizons: [0, 512, 1024, 2048]`, and the five suite IDs above.
Outputs must be under `outputs/exploratory_shared_budget_24gb/` with separate
files for `equal`, `native`, `direct_savi`, `online_savi_no_voi`, and
`online_savi`.

- [ ] **Step 4: Run focused tests and confirm protocol validation passes**

Run: `.venv/bin/python -m pytest -q tests/test_shared_budget_eval.py`

Expected: PASS.

- [ ] **Step 5: Commit the protocol-only change**

```bash
git add configs/exploratory_shared_budget_24gb.yaml tests/test_shared_budget_eval.py Makefile
git commit -m "Add exploratory shared-budget comparison protocol"
```

### Task 2: Backfill state features and train the exploratory critic

**Files:**
- Create: `scripts/backfill_exp0c_state_features.py`
- Create: `configs/exploratory_critic_24gb.yaml`
- Create: `tests/test_backfill_exp0c_state_features.py`

**Interfaces:**
- Consumes: existing Exp0C prefix token IDs and `state_values.jsonl`.
- Produces: feature-enriched prefix JSONL with `last_hidden`, entropy, and state metadata; state-aware and budget-only critic directories.

- [ ] **Step 1: Write failing backfill tests**

```python
def test_backfill_preserves_state_identity_and_adds_hidden_features(tmp_path):
    output = backfill_rows(fake_runner, [
        {"state_id": "p-b4096-m0", "prefix_token_ids": [1, 2]}
    ], tmp_path / "prefixes.jsonl")
    row = read_jsonl(output)[0]
    assert row["state_id"] == "p-b4096-m0"
    assert row["feature_mode"] == "full_hidden"
    assert row["last_hidden"] == [0.1, 0.2]
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `.venv/bin/python -m pytest -q tests/test_backfill_exp0c_state_features.py`

Expected: FAIL because the backfill function does not exist.

- [ ] **Step 3: Implement resumable backfill**

Read the existing 120 prefix rows, reject duplicate or missing `state_id`s,
load the exact local model through `QwenRunner`, call `state_features` once per
state, and atomically append one feature-enriched row at a time. A completed
row must be reused on resume and its token IDs must match the source row.

- [ ] **Step 4: Run focused tests and the feature backfill**

Run: `.venv/bin/python -m pytest -q tests/test_backfill_exp0c_state_features.py`

Then run:

```bash
.venv/bin/python scripts/backfill_exp0c_state_features.py \
  --input outputs/exp0c_math_batched_24gb/prefixes.jsonl \
  --output outputs/exploratory_shared_budget_24gb/prefixes_with_features.jsonl \
  --resume
```

Expected: 120 feature rows with matching state IDs and no token changes.

- [ ] **Step 5: Train state-aware and budget-only critics**

Use the fixed Exp0C `state_values.jsonl` labels and an explicit 3/1/1 suite
split. Record in the training report that this is same-suite exploratory
calibration and not a held-out method evaluation.

```bash
.venv/bin/python -m savi.train_critic \
  --config configs/exploratory_critic_24gb.yaml --representation state-aware
.venv/bin/python -m savi.train_critic \
  --config configs/exploratory_critic_24gb.yaml --representation budget-only
```

### Task 3: Implement one shared-budget evaluator for Equal and Native

**Files:**
- Create: `src/savi/shared_budget_eval.py`
- Create: `scripts/run_shared_budget_baselines.py`
- Modify: `tests/test_shared_budget_eval.py`

**Interfaces:**
- Consumes: fixed suite manifest, `QwenRunner`, local Math parser, and config.
- Produces: one JSONL result per method with exactly six answers, per-problem token spend, total spend at most 4,096, and suite score.

- [ ] **Step 1: Add failing budget-accounting tests**

```python
def test_equal_allocation_spends_no_more_than_shared_budget():
    allocation = equal_allocation(4096, 6)
    assert sum(allocation) <= 4096
    assert max(allocation) - min(allocation) <= 1

def test_native_prompt_contains_all_six_numbered_problems():
    prompt = render_native_contest_prompt(fake_suite(), 4096)
    assert prompt.count("## Problem") == 6
    assert "total output budget" in prompt
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_shared_budget_eval.py`

Expected: FAIL until allocation and prompt functions are implemented.

- [ ] **Step 3: Implement Equal and Native runners**

Equal runs six single-problem reasoning calls with deterministic per-problem
seeds and a balanced allocation summing to 4,096; each trace is finalized with
the existing deterministic finalizer. Native renders the official R3-Bench
Tool-Free Math contest prompt, requests one 4,096-token response, extracts the
last boxed answer associated with each problem section, and scores missing or
malformed sections as zero.

- [ ] **Step 4: Run unit tests and a no-model fake-run test**

Run: `.venv/bin/python -m pytest -q tests/test_shared_budget_eval.py`

Expected: PASS with exact budget accounting and six answer records per suite.

- [ ] **Step 5: Run Equal and Native on all five fixed suites**

Run:

```bash
.venv/bin/python scripts/run_shared_budget_baselines.py \
  --config configs/exploratory_shared_budget_24gb.yaml \
  --methods equal native --resume
```

### Task 4: Run Direct-SAVI and Online-SAVI variants

**Files:**
- Create: `scripts/run_shared_budget_savi.py`
- Modify: `src/savi/online_scheduler.py`
- Modify: `tests/test_online_scheduler.py`

**Interfaces:**
- Consumes: state-aware exploratory critic, fixed suites, and the same 4,096-token budget.
- Produces: `direct_savi`, `online_savi_no_voi`, `online_savi`, and frozen-index result JSONL with allocation trajectories and final answers.

- [ ] **Step 1: Add failing variant and output-contract tests**

```python
def test_shared_budget_savi_variants_bind_same_budget_and_suite():
    result = run_variant(fake_suite(), fake_runner(), fake_ensemble(), method="online_savi", shared_budget=4096, chunk=512)
    assert result["shared_budget"] == 4096
    assert result["suite_id"] == "math_suite_014"
    assert sum(item["executed_tokens"] for item in result["trajectory"]) <= 4096
```

- [ ] **Step 2: Run focused test and verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_online_scheduler.py tests/test_shared_budget_eval.py`

Expected: FAIL because the shared-budget wrapper and variant metadata do not exist.

- [ ] **Step 3: Implement the wrapper**

Call the existing `run_suite` once per fixed suite and method, binding
`--legacy-direct-savi` for Direct-SAVI, `voi_lambda=0` for no-VoI, and the
configured positive development lambda for Online-SAVI. Use separate output
files and deterministic method-specific seeds. Do not expose answer labels to
the scheduler.

- [ ] **Step 4: Run tests and launch all scheduler variants**

```bash
.venv/bin/python -m pytest -q tests/test_online_scheduler.py tests/test_shared_budget_eval.py
.venv/bin/python scripts/run_shared_budget_savi.py \
  --config configs/exploratory_shared_budget_24gb.yaml \
  --critic outputs/exploratory_shared_budget_24gb/critic/state-aware \
  --methods direct_savi online_savi_no_voi online_savi frozen_index --resume
```

### Task 5: Aggregate paired results and record the exploratory conclusion

**Files:**
- Create: `scripts/analyze_shared_budget_comparison.py`
- Create: `docs/results/exploratory_shared_budget_comparison_2026-09-09.md`
- Create: `results/exploratory_shared_budget_comparison_2026_09_09.json`
- Modify: `README.md`
- Modify: `docs/EXPERIMENTS.md`

**Interfaces:**
- Consumes: all method result JSONL files and fixed manifest.
- Produces: per-suite scores, average correct answers, paired bootstrap intervals, method ranking, full hashes, and explicit exploratory limitations.

- [ ] **Step 1: Write failing aggregation tests**

```python
def test_comparison_requires_all_methods_for_each_suite():
    with pytest.raises(ValueError, match="missing paired method"):
        aggregate_rows([{"suite_id": "math_suite_014", "method": "equal", "score": 2}])
```

- [ ] **Step 2: Run focused test and verify failure**

Run: `.venv/bin/python -m pytest -q tests/test_shared_budget_eval.py`

Expected: FAIL until aggregation is implemented.

- [ ] **Step 3: Implement analysis and artifact hashing**

Require five complete suites for every method, verify all shared budgets and
model/protocol metadata match, compute `paired_suite_bootstrap` for every method
against Equal and Native, and retain exact-normalized scoring warnings.

- [ ] **Step 4: Run aggregation and inspect the result**

```bash
.venv/bin/python scripts/analyze_shared_budget_comparison.py \
  --config configs/exploratory_shared_budget_24gb.yaml
```

- [ ] **Step 5: Run the full test suite and commit the exploratory report**

```bash
.venv/bin/python -m pytest -q
git diff --check
git add configs scripts src tests docs README.md results
git commit -m "Record exploratory shared-budget method comparison"
```

The final report must state whether Online-SAVI beats Equal/Native/Direct on
these five suites, but must not call that result a formal R3-Bench claim or
retroactively mark Exp0C's failed phenomenon gate as passed.
