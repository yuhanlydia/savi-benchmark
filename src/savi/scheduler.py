from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class HorizonEstimate:
    horizon: int
    mean: float
    std: float = 0.0


@dataclass(frozen=True)
class StateEstimate:
    problem_id: str
    finalize_mean: float
    finalize_std: float
    horizons: tuple[HorizonEstimate, ...]


def conservative_savi_index(state: StateEstimate, beta: float) -> tuple[float, int]:
    """Return the conservative marginal value per token and lookahead horizon."""
    if not state.horizons:
        return float("-inf"), 0
    finalize_upper = state.finalize_mean + beta * state.finalize_std
    candidates = [
        ((item.mean - beta * item.std - finalize_upper) / item.horizon, item.horizon)
        for item in state.horizons
        if item.horizon > 0
    ]
    return max(candidates, default=(float("-inf"), 0))


def choose_problem(states: Iterable[StateEstimate], beta: float = 0.0) -> tuple[str, int, float]:
    scored = []
    for state in states:
        index, horizon = conservative_savi_index(state, beta)
        scored.append((index, state.problem_id, horizon))
    if not scored:
        raise ValueError("At least one active problem is required")
    index, problem_id, horizon = max(scored)
    return problem_id, horizon, index


def state_estimate_from_predictions(
    problem_id: str, horizons: Iterable[int], means: Iterable[float], stds: Iterable[float]
) -> StateEstimate:
    values = sorted(zip(horizons, means, stds), key=lambda item: item[0])
    zero = [item for item in values if item[0] == 0]
    if len(zero) != 1:
        raise ValueError("exactly one zero-horizon estimate is required")
    positive = tuple(HorizonEstimate(int(h), float(mean), float(std))
                     for h, mean, std in values if h > 0)
    return StateEstimate(problem_id, float(zero[0][1]), float(zero[0][2]), positive)


@dataclass(frozen=True)
class HorizonBelief:
    horizon: int
    mean_gain: float
    variance: float
    measurement_variance: float


@dataclass(frozen=True)
class OnlineStateBelief:
    problem_id: str
    horizons: tuple[HorizonBelief, ...]
    update_count: int = 1


@dataclass(frozen=True)
class OnlineChoice:
    problem_id: str
    horizon: int
    score: float
    exploit_value: float
    information_value: float
    posterior_std: float


def _gain_measurement(state: StateEstimate, item: HorizonEstimate,
                      measurement_noise_floor: float) -> tuple[float, float]:
    if measurement_noise_floor < 0:
        raise ValueError("measurement_noise_floor must be non-negative")
    mean_gain = float(item.mean - state.finalize_mean)
    variance = float(
        item.std * item.std
        + state.finalize_std * state.finalize_std
        + measurement_noise_floor * measurement_noise_floor
    )
    return mean_gain, max(variance, 1e-12)


def update_online_belief(
    previous: OnlineStateBelief | None,
    state: StateEstimate,
    *,
    process_std: float,
    measurement_noise_floor: float,
) -> OnlineStateBelief:
    """Kalman-style online update of marginal-compute-value beliefs.

    The frozen critic provides a noisy measurement of the current state's
    marginal success gain. The belief is updated only when the reasoning state
    changes; ``process_std`` allows compute value to drift after each reasoning
    chunk without using correctness feedback.
    """
    if process_std < 0:
        raise ValueError("process_std must be non-negative")
    if previous is not None and previous.problem_id != state.problem_id:
        raise ValueError("belief and state must refer to the same problem")
    old = {} if previous is None else {item.horizon: item for item in previous.horizons}
    updated = []
    for item in state.horizons:
        if item.horizon <= 0:
            continue
        measurement_mean, measurement_var = _gain_measurement(
            state, item, measurement_noise_floor
        )
        prior = old.get(item.horizon)
        if prior is None:
            post_mean = measurement_mean
            post_var = measurement_var
        else:
            prior_var = prior.variance + process_std * process_std
            kalman_gain = prior_var / (prior_var + measurement_var)
            post_mean = prior.mean_gain + kalman_gain * (measurement_mean - prior.mean_gain)
            post_var = (1.0 - kalman_gain) * prior_var
        updated.append(HorizonBelief(
            horizon=item.horizon,
            mean_gain=float(post_mean),
            variance=max(float(post_var), 1e-12),
            measurement_variance=measurement_var,
        ))
    return OnlineStateBelief(
        problem_id=state.problem_id,
        horizons=tuple(sorted(updated, key=lambda item: item.horizon)),
        update_count=1 if previous is None else previous.update_count + 1,
    )


def _normal_pdf(value: float) -> float:
    import math
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _normal_cdf(value: float) -> float:
    import math
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _myopic_information_value(
    mean_rate: float,
    std_rate: float,
    best_alternative_rate: float,
    variance_rate: float,
    measurement_variance_rate: float,
) -> float:
    """Approximate one-probe value of information in success-probability/token.

    The Gaussian expected value of perfect information against the best
    competing problem is scaled by the fraction of posterior variance that one
    new critic observation is expected to remove.
    """
    if std_rate <= 0 or variance_rate <= 0 or not best_alternative_rate < float("inf"):
        return 0.0
    delta = mean_rate - best_alternative_rate
    z = delta / std_rate
    evpi = std_rate * _normal_pdf(z) + delta * _normal_cdf(z) - max(delta, 0.0)
    denom = variance_rate + max(measurement_variance_rate, 1e-18)
    observability = variance_rate / denom if denom > 0 else 0.0
    return max(0.0, observability * evpi)


def choose_problem_online(
    beliefs: Iterable[OnlineStateBelief],
    *,
    beta: float = 0.0,
    voi_lambda: float = 0.0,
) -> OnlineChoice:
    """Choose the next problem using exploitation plus myopic information value."""
    if beta < 0:
        raise ValueError("beta must be non-negative")
    if voi_lambda < 0:
        raise ValueError("voi_lambda must be non-negative")
    beliefs = list(beliefs)
    candidates = []
    for belief in beliefs:
        for item in belief.horizons:
            if item.horizon <= 0:
                continue
            mean_rate = item.mean_gain / item.horizon
            variance_rate = item.variance / (item.horizon * item.horizon)
            measurement_variance_rate = item.measurement_variance / (item.horizon * item.horizon)
            std_rate = variance_rate ** 0.5
            candidates.append((belief.problem_id, item.horizon, mean_rate, std_rate,
                               variance_rate, measurement_variance_rate))
    if not candidates:
        raise ValueError("At least one positive-horizon belief is required")

    scored = []
    for problem_id, horizon, mean_rate, std_rate, variance_rate, measurement_var_rate in candidates:
        other_rates = [candidate[2] for candidate in candidates if candidate[0] != problem_id]
        best_other = max(other_rates) if other_rates else float("inf")
        exploit = mean_rate - beta * std_rate
        information = _myopic_information_value(
            mean_rate, std_rate, best_other, variance_rate, measurement_var_rate
        ) if voi_lambda else 0.0
        score = exploit + voi_lambda * information
        scored.append(OnlineChoice(
            problem_id=problem_id,
            horizon=horizon,
            score=float(score),
            exploit_value=float(exploit),
            information_value=float(voi_lambda * information),
            posterior_std=float(std_rate),
        ))
    return max(scored, key=lambda item: (item.score, item.problem_id, item.horizon))
