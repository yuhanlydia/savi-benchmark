# Experimental protocol

## Phase 0: Budget–State Aliasing

The confirmatory sampling unit is a complete R³ Math suite. Ten suites are
selected once by the manifest seed. For each problem and spent budget, four
independent autoregressive prefixes are sampled. Every continuation appends to
the exact original prompt and token prefix; the prompt is never reconstructed
with a different budget declaration.

All paths share a 768-token total trajectory contract. This makes the model
budget-aware without allowing the prompt itself to reveal whether a state was
observed at 128, 256, or 512 tokens.

Immediate finalization estimates `Q(s, 0)`. Four independent 256-token branches
estimate `Q(s, 256)`. The trace-only finalizer is deterministic and excluded
from the reasoning budget. It may format evidence already in the trace but may
not solve from scratch.

Primary gates:

1. At least 20% of `(problem, spent-budget)` cells have
   `max_j Q(s_j,256)-min_j Q(s_j,256) >= 0.5`.
2. State-aware and budget-only next-problem choices differ on over 25% of
   bootstrap suite states.

Because four continuation trials make raw empirical ranges extremely noisy,
Gate A is reported in two forms. The original `Range >= 0.5` fraction is never
discarded. A confirmatory pass additionally requires its observed fraction to
exceed a pooled-binomial same-state null by at least 0.10 with Monte Carlo
`p <= 0.05`. Without this correction, the null false-positive probability can
exceed 70% for a single cell at pooled success probability 0.5.

Budget-floor diagnostic is fixed before three complete problems are observed:
if at least three fully sampled problems have under 5% parseable finalizations,
flag the Phase 0 grid as underpowered and run a separately labeled calibration
condition. This diagnostic does not retroactively replace the preregistered
Gate sample.

### Natural-EOS diagnostic

Fixed-length conditions that trigger the budget-floor check are followed by a
separately labeled natural-EOS probe. Summarize completed rows with:

```bash
python -m savi.eos_report \
  --input outputs/eos_probe_math/results.jsonl \
  --output outputs/eos_probe_math/report.json
```

If every sample hits the cap, the report labels the suggested next grid as a
lower bound rather than treating the cap as a measured natural reasoning length.
The natural-EOS probe omits a numeric budget from the prompt by default so the
cap does not anchor response length. Add `--announce-budget` only for a
separately labeled prompt-effect control.
Because SAVI consumes the reasoning state rather than a duplicated polished
response, `</think>` and the model's chat EOS tokens are all valid natural stop
events for this diagnostic.

Fixed-length prefix grids must also be audited for states that crossed a model
terminal token before the requested budget:

```bash
python -m savi.prefix_audit --config configs/phase0_math.yaml \
  --output outputs/phase0_math/prefix_audit.json
```

Post-terminal prefixes are reported separately and are not evidence of
within-reasoning state aliasing.

Exact-normalized scoring is diagnostic only. Before accepting either gate,
all parse failures and non-identical mathematical answers must be passed
through the official R³ production equivalence judge.

## Phase 1: value model

Only after Phase 0 passes, extract the frozen final-token hidden state and
runtime-visible scalars. Fit PCA on training suites only. Train five bootstrap
MLP heads with binomial NLL. Splits are by suite, so no problem or trajectory
crosses train/validation/test boundaries.

Report calibration, binomial NLL, Brier score, AUROC, and pairwise ranking of
`Q(s,h)-Q(s,0)` against the matched budget-only model. Scheduler evaluation is
blocked unless state awareness improves pairwise ranking by 10 percentage
points.

## Phase 2: scheduler

At every 128-token execution chunk, predict horizons 128/256/512, compute the
LCB marginal-value index, execute only 128 tokens on the selected problem, and
replan. Primary legal R³ evaluation receives no correctness oracle. Required
ablations are state shuffle, frozen initial index, beta=0 SAVI, and oracle-state
headroom.

Main reporting uses average correct answers per six-problem suite, paired
repeat-level bootstrap confidence intervals, Contest–Oracle Gap and Gap Ratio.
Pressure, domain, model, seed, quantization and judge versions are always
reported as part of the condition.
