# Budget–State Aliasing / Online-SAVI

This repository studies **shared-budget reasoning** on R³-Bench. The central question is not merely how difficult a problem is, but whether the *current realized reasoning state* still makes additional inference compute valuable.

The allocation target is `G(s,h) = Q(s,+h) - Q(s,0)`.

Two attempts on the same problem can consume the same number of tokens yet have very different `G`: one may be close to a breakthrough, while another may be stuck or already complete. We call this **marginal-value / budget-state aliasing**.

The proposed method is **Online-SAVI** (State-Aware Value-of-Inference). It keeps the LM and value critic frozen, but learns a test-time belief over each problem's changing marginal compute value. After every small reasoning chunk, it updates the selected problem's value belief and reallocates compute across the six problems. A myopic value-of-information term can favor a short probe when learning whether a problem is worth further investment is itself useful.

This is test-time online belief learning / adaptive inference control, **not** weight-updating test-time training. No correctness signal or verifier reward is available during the R³ episode.

## Current scientific stage

The method is blocked behind a phenomenon gate. Earlier pilot evidence for raw continuation-probability aliasing weakened when continuation repeats increased from K=4 to K=8, but a replicated allocation-relevant marginal-gain contrast remained. Therefore the active confirmatory experiment is **Experiment 0C**, not the old serial Exp0B.

See:

- `docs/EXP0C_BATCHED.md` — current 16GB/24GB gap gate;
- `docs/EXPERIMENTS.md` — full staged experiment plan;
- `docs/ONLINE_VALUE_LEARNING.md` — Online-SAVI mathematics and ablations;
- `docs/PILOT_RESULTS.md` — exploratory results and limitations.
- `docs/EXPLORATORY_SHARED_BUDGET.md` — matched Equal/Native/SAVI comparison and limitations.

## Setup

```bash
git clone --recursive https://github.com/yuhanlydia/savi-benchmark.git
cd savi-benchmark
bash scripts/bootstrap.sh
```

The benchmark and model revisions are pinned in `resources.lock.json`.

## Current next experiment: Exp0C

First run the hardware smoke:

```bash
make batch-smoke       # 16GB GPUs
make batch-smoke-24    # 24GB GPUs
```

Then run exactly one hardware condition:

```bash
make exp0c-16
# or
make exp0c-24
```

Do not pool the 16GB and 24GB sampling conditions.

After completion:

```bash
make exp0c-analyze-16
make exp0c-gates-16
```

(or the corresponding `-24` targets).

Experiment 0C uses five new complete R³ Math suites (30 problems), one 4096-token spent-budget condition, four independent prefixes per problem, one immediate finalization and eight +2048-token continuations per state. No critic or scheduler result is valid unless the gap gate passes.

## Gap GO criteria

Continue only if:

- at least 20% of complete `(problem,4096)` cells have marginal-gain range `>=0.5`;
- the problem-bootstrap 95% CI lower bound for noise-corrected marginal-gain variance is above zero;
- Decision Flip Rate is above 25%;
- the signal remains meaningful in the nonterminal-only subset.

If this fails, stop the direction before value-model training.

An exploratory shared-budget comparison was run after the 24GB Exp0C result
using the same five suites. It is explicitly not a gate override: the critic
reuses those suites and Exp0C remains formally failed. See
`docs/EXPLORATORY_SHARED_BUDGET.md` and
`outputs/exploratory_shared_budget_24gb/comparison_summary.json`.

## Online-SAVI after the gap passes

The frozen critic measures the current state:

`z_{i,h} = Q_phi(s_i,h) - Q_phi(s_i,0)`.

Online-SAVI maintains a Kalman-style belief over each problem/horizon's latent marginal gain. The scheduling score combines a conservative exploit term with an optional myopic value-of-information bonus:

`Index_i(h) = (m_{i,h} - beta sqrt(P_{i,h})) / h + lambda * VoI_i(h)`.

One 512-token chunk is executed, the selected problem state changes, its belief is refreshed, and all six problems are reconsidered.

Example after a state-aware critic has passed its held-out gate:

```bash
python -m savi.online_scheduler \
  --config configs/exp0c_math_batched_16gb.yaml \
  --critic outputs/value_model/state-aware \
  --suite-id math_suite_001 \
  --shared-budget 4096 \
  --chunk 512 \
  --horizons 0 512 1024 2048 \
  --beta 1 \
  --voi-lambda 0.5 \
  --process-std 0.10 \
  --measurement-noise-floor 0.05 \
  --output outputs/online_savi_math.jsonl
```

The numeric scheduler hyperparameters above are development defaults only. Final values must be selected on external validation trajectories, never on final R³ test suites.

Required ablations:

- `--legacy-direct-savi`: direct current-state prediction, no online belief filter;
- `--voi-lambda 0`: online belief update without information value;
- `--frozen-index`: no dynamic re-planning;
- matched state-shuffle, budget-only and oracle-state controls.

Primary final metric remains R³ average correct answers per six-problem suite at matched shared compute.

## Compute note

Qwen3-8B BF16 does not safely fit an RTX A4000 16GB together with the required KV cache. The development profile therefore uses NF4 with BF16 compute. Quantization is part of the model condition and must be reported. The 16GB and 24GB batched profiles are explicit, reproducible sampling conditions rather than silent auto-tuning.
