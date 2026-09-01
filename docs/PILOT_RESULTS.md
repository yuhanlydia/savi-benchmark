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
| 6144 | 1.00, 1.00, 1.00, 1.00 | 0.00 | audited separately |

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
