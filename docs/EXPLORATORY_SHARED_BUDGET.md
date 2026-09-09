# Exploratory shared-budget comparison

## Status

This is an **exploratory** matched comparison, not a confirmatory gate. The
formal Exp0C phenomenon gate remains unchanged and failed. The run was stopped
after a code/protocol audit found issues requiring further debugging, so there
is currently no valid completed Online-SAVI comparison.

Machine-readable output:

`outputs/exploratory_shared_budget_24gb/comparison_summary.json`

## Frozen protocol

- Model: local `Qwen3-8B`, NF4 weights with BF16 compute.
- Sampling: temperature `0.6`, top-p `0.95`, top-k `20`.
- Fixed suites: `math_suite_014`, `math_suite_024`, `math_suite_026`,
  `math_suite_038`, `math_suite_043`.
- Six problems per suite and one shared `4096`-token budget.
- SAVI chunk: `512` tokens; all scheduler runs used exactly eight chunks.
- Scoring: exact-normalized Math answer extraction, not an official leaderboard
  submission.
- Seeds are deterministic and method-specific; all methods use the same suite
  IDs and dataset snapshot.

Equal allocation divides 4096 as evenly as possible across six independent
single-problem calls. Native uses one R-3-Bench-style six-problem contest
prompt with a 4096-token cap. Direct-SAVI and Online-SAVI use the repository's
state-aware scheduler; no-VoI and frozen-index are ablations.

## Results

| method | correct answers | mean correct / suite | answer rate |
| --- | ---: | ---: | ---: |
| Equal | 1 / 30 | 0.20 / 6 | 3.33% |
| Native concise local variant | 7 / 30 | 1.40 / 6 | 23.33% |
| Direct-SAVI | 0 / 30 | 0.00 / 6 | 0.00% |
| Online-SAVI-noVoI | 0 / 30 | 0.00 / 6 | 0.00% |
| Online-SAVI | incomplete: 1 / 24 | not reported | not reported |
| frozen-index | not run | not reported | not reported |

The previous `comparison_summary.json` was generated before the audit fixes and
must not be used; it has been archived with the pre-validation outputs. No
paired superiority interval is reported for the stopped/incomplete run.

## Scientific limitations

1. The critic was calibrated from the completed Exp0C observations on these
   same five suites. The internal split was 3 train / 1 validation / 1 test,
   but the allocator comparison itself reuses all five suites. This is not a
   held-out evaluation.
2. Exp0C contains states at `spent_budget=4096` only. The scheduler starts at
   zero and therefore extrapolates the critic over earlier budgets.
3. The formal Exp0C gate remains failed: the observed range-ge-0.5 fraction
   was `0.0667`, below the preregistered `0.20`; the gate decision was not
   changed by this exploratory run.
4. Exact-normalized parsing can miss mathematically equivalent forms and is
   not a substitute for the official R-3-Bench judge.

The current result supports only that parts of the end-to-end protocol can run.
It does not validate the method, the Native baseline, or the research
hypothesis. The next run must first resolve the scheduler/protocol audit and
then regenerate all methods together.
