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
