# Experiment 0C: batched mid-budget marginal-value gate

Experiment 0B remains a valid append-only serial checkpoint, but its first long
run established an implementation bottleneck: 729/6,480 rows required enough
wall time to project to roughly 9--10 days on a 16GB RTX A4000. The run is not a
confirmatory scientific result and is not pooled with Experiment 0C.

## Scientific purpose

The strongest replicated pilot signal is allocation-aligned marginal value,
not raw continuation probability. At the calibrated 6,144-token cell, the four
states all reached continuation success 8/8, while immediate success was
`[0,1,1,1]`, giving gains `[1,0,0,0]`. By contrast, the earlier raw continuation
range at 4,096 tokens shrank from 0.50 to 0.375 when K increased from 4 to 8.

Experiment 0C therefore asks a narrower confirmatory question:

> For the same problem after the same 4,096-token spend, do independently
> realized reasoning states have materially different marginal value of another
> 2,048 tokens?

The primary state label is `G = Q(s,+2048)-Q(s,0)`. The primary aliasing gate is
marginal-value range, not the legacy raw-`Q_h` gate.

## Scope before scaling

The 16GB condition uses five new complete R3-Bench Math suites (30 problems),
one spent budget (4,096), four prefixes per problem, one immediate finalization,
and eight 2,048-token continuations per state: 120 states and 1,080 evaluation
jobs. It excludes every suite used by the earlier development runs and the ten
Experiment-0B suites. This is a gap gate, not the final benchmark matrix.

Run the GPU smoke first:

```bash
make batch-smoke
```

The smoke uses an already-consumed development problem and makes no scientific
claim. Confirm that the 16GB profile fits without CUDA OOM and that effective
prefix/continuation batches reach two where the token budget allows it.

Then choose exactly one confirmatory hardware profile:

```bash
make exp0c-16   # RTX A4000 / other 16GB cards; batch cap 2
make exp0c-24   # 24GB cards; batch cap 4, separate output/sampling condition
```

Do not pool the 16GB and 24GB outputs. Grouped generation is statistically the
same sampling distribution as the sequential runner but has a different RNG
contract and is not token-identical to the old serial condition.

## Why the new runner is faster

`batched_phase0` keeps the serial runner unchanged as a reproducibility fallback
but changes the new condition in three ways:

1. independent prefixes for the same problem/budget are generated with
   `num_return_sequences` microbatches;
2. continuation repeats for the same state/horizon are generated in fixed
   microbatches, and deterministic finalizers are batched;
3. the Phase-0 gap gate stores only cheap prefix metadata. It does **not** run
   the old full-sequence `output_hidden_states=True` feature pass, because no
   critic may be trained before the gap gate passes.

The batching contract is resumable. If a partially written fixed microbatch is
encountered, it is regenerated from its stable group seed and completed members
must match token-for-token before missing members are appended.

## Memory profiles

The default 16GB profile uses Qwen3-8B NF4/BF16 compute with a 14k aggregate
context-token planning budget and caps stochastic/finalizer batches at two. The
24GB profile uses a 26k planning budget and caps at four. These are explicit
experiment settings, not silent hardware auto-tuning.

If the 16GB smoke OOMs, do not enable an automatic fallback in the confirmatory
run. Lower the explicit batch profile in a separately committed config and rerun
the smoke so the sampling contract stays reproducible.

## Gates

After all jobs are complete and official-equivalence scoring has been audited:

```bash
python -m savi.analysis --config configs/exp0c_math_batched_16gb.yaml
python -m savi.analysis --config configs/exp0c_math_batched_16gb.yaml --nonterminal-only
python -m savi.gate_decision \
  --config configs/exp0c_math_batched_16gb.yaml \
  --report outputs/exp0c_math_batched_16gb/gates.json
```

Experiment 0C is GO only if:

- at least 20% of complete `(problem,4096)` cells have marginal-gain range >=0.5;
- the problem-bootstrap 95% CI lower bound for noise-corrected marginal-gain
  variance is above zero; and
- the decision-relevance DFR gate remains above 25%.

If this five-suite gate fails, stop before collecting critic training data. If
it passes, expand the phenomenon on additional suites/budgets and only then
build the state-aware value critic and online SAVI scheduler.
