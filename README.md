# Budget–State Aliasing / SAVI

Phase 0 tests whether two Qwen3-8B reasoning attempts for the same R³-Bench
Math problem and the same spent-token budget have materially different value
of another 256 reasoning tokens.

## Reproducibility contract

- Benchmark: official `NineAbyss/R-3-Bench`, pinned in
  `third_party/R-3-Bench` at commit `e32d5930`.
- Model: official `Qwen/Qwen3-8B`, local snapshot under `models/Qwen3-8B`.
  The downloaded Hub revision is `b968826d`; both revisions are recorded in
  `resources.lock.json`.
- Sampling unit: 10 complete six-problem suites (60 problems), never 60
  independently sampled problems.
- Prefix budgets: 128, 256, 512; four independent prefixes per cell.
- Every prefix is sampled under the same 768-token trajectory contract. The
  observed state is a truncation at 128/256/512; continuation never changes or
  reconstructs the original prompt.
- Continuation: zero-token finalize plus four independent 256-token
  continuations.
- Protocol: thinking Stage 1; non-thinking trace-only Stage 2 finalizer. Only
  Stage 1 tokens count, matching R³'s two-stage accounting.
- Pilot labels use exact normalized answers. Any paper-facing result must rerun
  nontrivial answer equivalence through the official R³ production judge.

This design creates 720 prefix states and 3,600 evaluation jobs: 720 immediate
finalizations plus 2,880 stochastic continuations. Prefixes are generated once
and reused across their continuation branches.

## Commands

```bash
source .venv/bin/activate
python -m savi.manifest --config configs/phase0_math.yaml
python -m savi.phase0 --config configs/phase0_math.yaml        # plan only
python -m savi.phase0 --config configs/phase0_math.yaml --execute
python -m savi.analysis --config configs/phase0_math.yaml
python -m savi.train_critic --config configs/phase0_math.yaml  # only after gates pass
python -m savi.extract_problem_features --config configs/phase0_math.yaml
python -m savi.train_critic --config configs/phase0_math.yaml --representation budget-only
python -m savi.monitor --config configs/phase0_math.yaml
```

The runner is append-only and resumes completed `(state, horizon, repeat)`
tuples. Do not report Gate A or DFR from partial output.
The analyzer enforces this by default; `--allow-partial` is diagnostic only.

For a bounded ten-hour collection window:

```bash
make phase0-10h
```

The deadline is checked between atomic jobs. It never truncates a JSONL record;
rerunning the same command resumes from the next incomplete tuple.

## Preregistered gates

- Gate A: at least 20% of `(problem, spent budget)` cells have continuation
  success range at least 0.5 across four prefixes.
- Gate C/decision relevance: suite-level Decision Flip Rate above 25%.
- Predictor and scheduler gates begin only after the Phase 0 labels pass an
  official-equivalence scoring audit.

## Important compute note

Qwen3-8B BF16 does not safely fit an RTX A4000 16GB together with KV cache.
The default runner uses 4-bit NF4 with BF16 compute. Quantization is part of the
model condition and must be reported; do not compare it silently with published
BF16 response curves.
