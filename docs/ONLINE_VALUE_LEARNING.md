# Online-SAVI: test-time learning of compute value

## Scope

Online-SAVI is a test-time **belief-learning and control** method, not test-time training of the language-model weights. The reasoning model and continuation-value critic remain frozen. The only online state is a per-problem belief about the marginal value of more inference compute.

The method observes newly generated reasoning states but receives no ground-truth correctness, verifier reward or answer label until the R³ episode is over.

The first end-to-end matched comparison is recorded in
`docs/EXPLORATORY_SHARED_BUDGET.md`. It is exploratory because the Exp0C
phenomenon gate failed and the critic calibration reuses the fixed comparison
suites; it must not be cited as evidence that Online-SAVI beats its baselines.

## 1. State-conditioned marginal value

For a realized reasoning state `s` and lookahead horizon `h`, the frozen critic predicts

`Q_phi(s,h) = P(success | s, +h tokens)`.

The marginal gain of buying more compute is

`g(s,h) = Q_phi(s,h) - Q_phi(s,0)`.

The legacy Direct-SAVI index uses the current estimate directly. Online-SAVI instead treats `g` as a latent dynamic quantity because the reasoning path changes after every executed chunk.

## 2. Dynamic belief model

For each problem `i` and horizon `h`, maintain

`g^t_{i,h} ~ N(m^t_{i,h}, P^t_{i,h})`.

A reasoning action changes the problem state, so before the next observation use

`P^- = P^t + q`,

where `q = process_std^2` models value drift.

The critic produces a new measurement

`z = Q_phi(s^{t+1},h) - Q_phi(s^{t+1},0)`.

The measurement variance is approximated from ensemble uncertainty:

`R = sigma_h^2 + sigma_0^2 + epsilon^2`,

where `epsilon` is a fixed measurement-noise floor.

The update is

`K = P^- / (P^- + R)`,

`m^{t+1} = m^t + K (z - m^t)`,

`P^{t+1} = (1-K) P^-`.

The first observation initializes the belief. A problem that has not generated new reasoning tokens is not re-observed, so its uncertainty cannot collapse merely because another problem was selected.

## 3. Exploitation value

For each horizon,

`Exploit_i(h) = (m_{i,h} - beta sqrt(P_{i,h})) / h`.

`beta=0` uses the posterior mean; positive `beta` gives a conservative lower-confidence marginal value.

## 4. Value of information

A reasoning chunk can be useful even when its immediate expected gain is not maximal, because the new reasoning state can reveal whether the problem deserves future compute.

For each problem/horizon candidate, compare its posterior rate to the best competing problem. Under a Gaussian approximation, compute expected value of perfect information (EVPI) against that competitor. Then multiply EVPI by the fraction of uncertainty one new critic observation is expected to remove:

`rho = P / (P + R)`.

The implementation uses

`VoI = rho * EVPI`.

The online index is

`Index_i(h) = Exploit_i(h) + lambda * VoI_i(h)`.

`lambda=0` is the no-information-value ablation. A positive `lambda` must be selected on external validation trajectories.

## 5. Sequential allocation

At time `t`:

1. estimate the current state of all six problems;
2. update only beliefs whose reasoning state changed;
3. score all affordable horizons;
4. choose the problem/horizon with maximum Online-SAVI index;
5. execute only one chunk (`c=512` by default);
6. observe the new reasoning state and repeat.

Thus allocation is not a one-shot budget prediction. It is a sequential inference-and-control loop:

`probe -> observe -> update value belief -> reallocate -> probe`.

## 6. Baselines and causal ablations

Compare against:

- budget-only allocation;
- Direct-SAVI (`--legacy-direct-savi`);
- Online-SAVI-noVoI (`--voi-lambda 0`);
- Online-SAVI with validation-selected `lambda`;
- frozen initial index (`--frozen-index`);
- state shuffle;
- oracle state value.

The key mechanistic prediction is that dynamic Online-SAVI should outperform frozen/direct variants specifically on suites where the identity of the highest-marginal-value problem changes during reasoning.

## 7. Falsification

The online method is not justified if:

- Experiment 0C does not show nonterminal marginal-value aliasing;
- the frozen critic cannot rank marginal value on held-out external problems;
- online belief updates do not improve value calibration/ranking over direct critic predictions;
- the information-value bonus increases exploration but not R³ solved-count under matched compute.

In those cases, do not add more control machinery; stop the direction.
