# Experimental protocol

## Research objective

R³-Bench asks a model to solve six problems under one shared inference budget. The project tests whether problem identity plus spent budget is sufficient to decide where the next unit of compute should go. The working hypothesis is narrower: the realized reasoning state is path-dependent, so the same problem at the same spent budget can have a different marginal value of additional reasoning.

The allocation target is `G(s,h) = Q(s,h) - Q(s,0)`.

## Phase 0 — Experiment 0C gap gate

Legacy Phase-0/0B diagnostics are preserved for provenance, but the active confirmatory gate is `docs/EXP0C_BATCHED.md`.

Current condition:

- R³-Bench Math;
- Qwen3-8B NF4 with BF16 compute for 16GB/24GB development;
- five new complete six-problem suites (30 problems);
- spent budget 4096;
- four independent prefixes per problem;
- one immediate finalization plus eight +2048-token continuations per state;
- 120 states / 1,080 evaluation jobs.

Run:

```bash
make batch-smoke       # 16GB
make exp0c-16

make batch-smoke-24    # 24GB
make exp0c-24
```

Do not pool hardware conditions.

GO only if:

1. at least 20% of complete `(problem,4096)` cells have marginal-gain range `>=0.5`;
2. the problem-bootstrap 95% CI lower bound of noise-corrected marginal-gain variance is above zero;
3. Decision Flip Rate is above 25%;
4. the signal remains meaningful in the nonterminal-only subset.

If this gate fails, stop before critic training.

## Phase 1 — frozen continuation-value critic

Only after Exp0C passes, train a frozen value model `Q_phi(s,h)` on external math problems with R³ items and near-duplicates removed. R³ remains evaluation-only for the final paper comparison.

Required predictors:

- budget-only: problem representation + spent budget + horizon;
- surface-progress: budget-only + runtime-visible progress features;
- state-aware: reasoning-state representation + spent budget + horizon.

Report calibration, binomial NLL, Brier, AUROC and pairwise ranking of `G(s,h)`. Scheduler experiments are blocked unless state awareness improves marginal-value ranking by at least 10 percentage points over the matched budget-only model.

## Phase 1B — online value-belief calibration

The main method does not update language-model weights and receives no test-time correctness feedback. It treats the frozen critic prediction as a noisy observation of a latent marginal-compute value that changes as reasoning evolves.

For problem `i`, horizon `h`:

`g^t_{i,h} ~ N(m^t_{i,h}, P^t_{i,h})`.

After one reasoning chunk, allow value drift with process variance `q`. The new critic estimate `z = Q_phi(s^{t+1},h)-Q_phi(s^{t+1},0)` is incorporated with a Kalman-style update. The belief is updated only when that problem's realized state changes.

Tune process noise, measurement-noise floor and the information-value coefficient only on held-out external validation trajectories, never on final R³ test suites.

## Phase 2 — Online-SAVI

At each decision step, predict affordable horizons and compute a conservative exploitation term

`Exploit_i(h) = (m_{i,h} - beta sqrt(P_{i,h})) / h`.

A probe also has information value because the new reasoning state can reveal whether a currently uncertain problem deserves future compute. The implementation adds a myopic Gaussian value-of-information bonus:

`Index_i(h) = Exploit_i(h) + lambda * VoI_i(h)`.

Select the highest-index problem, execute one 512-token chunk, update only that problem's state/value belief, and replan across all six problems. No correctness label or verifier reward is exposed during the episode.

The scheduler never scores a lookahead larger than the remaining shared budget.

Required variants:

- `Direct-SAVI`: `--legacy-direct-savi`;
- `Online-SAVI-noVoI`: `--voi-lambda 0`;
- `Online-SAVI`: validation-selected `--voi-lambda > 0`;
- frozen: `--frozen-index`;
- state-shuffle;
- budget-only critic;
- oracle-state headroom.

Example after the critic gate passes:

```bash
python -m savi.online_scheduler \
  --config configs/exp0c_math_batched_16gb.yaml \
  --critic outputs/value_model/state-aware \
  --suite-id math_suite_001 --shared-budget 4096 \
  --chunk 512 --horizons 0 512 1024 2048 \
  --beta 1 --voi-lambda 0.5 \
  --process-std 0.10 --measurement-noise-floor 0.05 \
  --output outputs/online_savi_math.jsonl
```

The numeric scheduler values above are development defaults only.

## Phase 3 — R³ benchmark comparison

Primary score: average correct answers per six-problem suite under the same shared compute budget.

Compare against Native Contest, Equal Allocation, R³ Coverage-first, Coverage+Verification, matched budget/difficulty allocation, Direct-SAVI, Online-SAVI-noVoI, Online-SAVI, and the response-curve oracle as a non-deployable reference.

Run Math first. Expand to Code, Abstract Reasoning and additional models only if Online-SAVI beats the best deployable baseline with paired confidence intervals.

## Stop rules

Stop if nonterminal Exp0C aliasing disappears, DFR is <=25%, the state-aware critic fails the +10pp ranking gate, or Online-SAVI fails to beat Direct-SAVI / the best deployable R³ baseline under matched compute.

The sequence remains: phenomenon first, predictability second, dynamic allocation third.
