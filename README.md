# Budget–State Aliasing / SAVI

Phase 0 tests whether two Qwen3-8B reasoning attempts for the same R³-Bench
Math problem and the same spent-token budget have materially different value
of another 256 reasoning tokens.

The first calibrated diagnostics found one same-problem/same-budget cell with
continuation probabilities `[1.00, 0.75, 0.50, 1.00]`, plus a later cell whose
continuation gains were `[1,0,0,0]`. These are encouraging exploratory results,
not confirmatory claims; see [the dated pilot report](docs/PILOT_RESULTS.md) for
the budget-floor failures, natural-length calibration, noise correction, and
limitations.

## Setup

On an Ubuntu/CUDA machine with Python 3.10+, a C compiler and sufficient disk:

```bash
git clone --recursive https://github.com/yuhanlydia/savi-benchmark.git
cd savi-benchmark
bash scripts/bootstrap.sh
```

The bootstrap resolves both the R³ submodule commit and the exact Hugging Face
revision from `resources.lock.json`, then runs the test suite. The default NF4
condition needs roughly 8GB runtime VRAM; downloaded BF16 shards occupy about
16GB on disk.

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
python -m savi.summarize --config configs/phase0_math.yaml --output outputs/phase0_math/summary.json
python -m savi.judge_handoff --config configs/phase0_math.yaml export --output outputs/judge_queue.jsonl
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
Each executed run writes immutable code, config, dependency, CUDA, GPU, model and
benchmark provenance plus a byte-for-byte `config.snapshot.yaml` before loading
the model. Plan-only commands never create execution provenance.

## Preregistered gates

- Gate A: at least 20% of `(problem, spent budget)` cells have continuation
  success range at least 0.5 across four prefixes.
- Gate C/decision relevance: suite-level Decision Flip Rate above 25%.
- Predictor and scheduler gates begin only after the Phase 0 labels pass an
  official-equivalence scoring audit.

If the preregistered floor-risk diagnostic triggers, run the separately labeled
calibration grid; never merge it into Phase 0:

```bash
python -m savi.phase0 --config configs/budget_calibration_math.yaml --execute
```

This calibration uses one complete suite, spent budgets 512/1024/2048 and a
512-token continuation. Its purpose is to locate a non-degenerate budget range,
not to support the confirmatory aliasing claim.

To measure natural thinking length without a forced minimum or truncation:

```bash
python -m savi.eos_probe --config configs/eos_probe_math.yaml \
  --problem-id omnimath-3045 --problem-id omnimath-1958 \
  --samples 2 --max-tokens 8192
```

The next confirmatory discovery condition is `configs/exp0b_math.yaml`. It
excludes every suite used by the development/pilot runs and samples 10 new
complete suites at spent budgets 2048/4096/6144, with 2048-token continuations
and eight repeats per state. Its primary label is `G=Q_h-Q_0`; no critic is
trained until this condition and the decision-relevance analysis pass.

After the predictor gate passes, a legal no-correctness-feedback online run is:

```bash
python -m savi.online_scheduler \
  --config configs/phase0_math.yaml \
  --critic outputs/phase0_math/critic/state-aware \
  --suite-id math_suite_001 --shared-budget 4096 \
  --chunk 128 --horizons 0 128 256 512 --beta 1 \
  --output outputs/savi_online.jsonl
```

The runner executes only one 128-token chunk, invalidates only that problem's
cached state representation, and replans across all six problems. It receives
no online correctness feedback.

Use `--frozen-index` for the preregistered non-replanning ablation. The
`savi.ablations.state_shuffle` transform performs a deterministic derangement
within each `(problem, spent-budget)` cell while preserving labels and all
budget/problem metadata.

## Important compute note

Qwen3-8B BF16 does not safely fit an RTX A4000 16GB together with KV cache.
The default runner uses 4-bit NF4 with BF16 compute. Quantization is part of the
model condition and must be reported; do not compare it silently with published
BF16 response curves.
