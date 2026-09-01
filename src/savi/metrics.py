from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np


def weighted_brier(prediction: np.ndarray, successes: np.ndarray, trials: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=float)
    successes = np.asarray(successes, dtype=float)
    trials = np.asarray(trials, dtype=float)
    if np.any(trials <= 0):
        raise ValueError("trials must be positive")
    target = successes / trials
    return float(np.average((prediction - target) ** 2, weights=trials))


def pairwise_ranking_accuracy(prediction: Iterable[float], target: Iterable[float]) -> float:
    prediction = np.asarray(list(prediction), dtype=float)
    target = np.asarray(list(target), dtype=float)
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have the same shape")
    correct = 0.0
    comparisons = 0
    for left in range(len(target)):
        for right in range(left + 1, len(target)):
            true_delta = target[left] - target[right]
            if true_delta == 0:
                continue
            predicted_delta = prediction[left] - prediction[right]
            comparisons += 1
            if predicted_delta == 0:
                correct += 0.5
            elif np.sign(predicted_delta) == np.sign(true_delta):
                correct += 1.0
    return correct / comparisons if comparisons else float("nan")


def paired_suite_bootstrap(
    rows: Iterable[Mapping[str, object]], *, method_a: str, method_b: str,
    draws: int = 10_000, seed: int = 0,
) -> dict[str, float]:
    by_suite: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_suite[str(row["suite_id"])][str(row["method"])].append(float(row["score"]))
    differences = []
    for suite, methods in by_suite.items():
        if method_a not in methods or method_b not in methods:
            raise ValueError(f"suite {suite} is missing a paired method")
        differences.append(np.mean(methods[method_a]) - np.mean(methods[method_b]))
    values = np.asarray(differences, dtype=float)
    if not len(values):
        raise ValueError("no suites supplied")
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "ci_2_5": float(np.quantile(samples, 0.025)),
        "ci_97_5": float(np.quantile(samples, 0.975)),
        "suite_count": int(len(values)),
    }
