# Pilot results (2026-09-01)

These are exploratory diagnostics, not confirmatory benchmark claims. Math
answers below use the repository's exact-normalized pilot scorer; nontrivial
equivalence still requires the official R³ judge.

## Budget-floor diagnosis

The original 128/256/512-token Phase 0 grid produced zero parseable
finalizations in its first three complete problems. A separately labeled
512/1024/2048 spent-budget calibration also produced zero parseable
finalizations in its first three complete problems and was stopped at its
predeclared floor check. Both outputs were preserved rather than pooled with a
new condition.

A natural-stop probe then separated task regimes:

- `omnimath-3045`: 2/2 traces reached the 8192-token cap without closing the
  thinking stage or producing a usable answer.
- `omnimath-1958`: 2/2 traces closed naturally at 7225 and 7989 reasoning
  tokens and produced usable answers.
- `omnimath-2823` and `omnimath-2741`: four traces all closed naturally, at
  304, 561, 1220, and 1269 tokens.

This establishes that the initial grid was a budget-floor failure for this
Qwen3-8B NF4 setup, not evidence against state-conditioned continuation value.

## Simple-problem threshold pilot

The two calibrated simple problems showed deterministic threshold behavior:

- `omnimath-2823`, spent 128: all four states had `Q(s,0)=1` and
  `Q(s,+512)=1`.
- `omnimath-2741`, spent 128: all four states had `Q(s,0)=0` and
  `Q(s,+512)=1`.
- `omnimath-2741`, spent 256: sampled complete states had already reached
  `Q(s,0)=1`.

No same-budget range reached 0.5. Higher-budget jobs were stopped after the
terminal-prefix audit showed they mainly sampled post-completion states.

## Calibrated completion-boundary pilot

For `omnimath-1958`, the follow-up uses spent budgets 2048/4096/6144 and a
2048-token continuation, with four independently sampled states and four
continuations per state.

The first two complete cells are:

| Spent | `Q(s,+2048)` by state | Range | Prefix terminal crossings |
|---:|---|---:|---:|
| 2048 | 0.00, 0.00, 0.00, 0.00 | 0.00 | 0/4 |
| 4096 | 1.00, 0.75, 0.50, 1.00 | 0.50 | 0/4 |
| 6144 | 1.00, 1.00, 1.00, 1.00 | 0.00 | 1/4 crossed `</think>` |

Across these two cells, the original raw range gate passes (1/2 cells), but the
pooled-binomial Monte Carlo correction does not (`p=0.441`; with all three
cells, expected null range fraction 0.147). The 4096-token cell is therefore an
encouraging aliasing example, not yet statistically sufficient evidence.

The allocation-aligned diagnostic is stronger at spent 6144: all four states
reach `Q(s,+2048)=1`, but their immediate values are `[0,1,1,1]`, so continuation
gains are `[1,0,0,0]`. Across the three cells, 2/3 have continuation-gain range
at least 0.5. This diagnostic was added after observing the distinction and is
not substituted for preregistered Gate A. A same-seed, higher-K replication is
used to reduce continuation sampling noise. Additional problems and the
official judge remain required.

## Higher-K replication

A same-seed replication increased continuation trials from 4 to 8 for the
4096/6144-token cells. Token-level audit found exact agreement for all 7 shared
prefixes and all 35 shared jobs (prefix tokens, continuation tokens, and scored
outcomes), so added trials are clean extensions rather than regenerated drift.

At spent 4096, the K=8 estimates became `[1.00, 0.875, 0.625, 1.00]`. The range
shrunk from 0.50 to 0.375, so the raw Gate A example did **not** replicate at
the preregistered threshold. This is evidence that K=4 exaggerated that range.

At spent 6144, all four states completed after the continuation was resumed.
Their immediate-to-continuation results were `0→8/8`, `1→8/8`, `1→8/8`, and
`1→8/8`, preserving the allocation-relevant gain contrast of 1 versus 0 at
higher K. The completed replication has 72/72 jobs, no duplicate rows, and no
invalid token lengths.

The original run contains 69/72 parseable finalizer outputs under the original
96-token finalizer cap. A trace-only repair pass with a 256-token finalizer
recovered two answers (`22`) and left one intrinsically incomplete trace
unparseable, yielding 71/72 parseable outputs. The repaired file is kept as a
separate sensitivity artifact; all gate numbers above use the original rows.
