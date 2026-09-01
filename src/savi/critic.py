from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch import nn


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def binomial_nll(logits: torch.Tensor, successes: torch.Tensor, trials: torch.Tensor) -> torch.Tensor:
    return -(successes * nn.functional.logsigmoid(logits)
             + (trials - successes) * nn.functional.logsigmoid(-logits)).mean()


@dataclass
class FeaturePipeline:
    pca: PCA
    scaler: StandardScaler
    hidden_key: str = "last_hidden"
    state_aware: bool = True

    def transform(self, rows: list[dict[str, Any]]) -> np.ndarray:
        hidden = np.asarray([row[self.hidden_key] for row in rows], dtype=np.float32)
        projected = self.pca.transform(hidden)
        scalars = np.asarray([
            [np.log1p(row["spent_budget"]), np.log1p(row["horizon"]),
             row["recent_token_entropy"] if self.state_aware else 0.0,
             float(row["has_candidate_answer"]) if self.state_aware else 0.0,
             row["recent_repetition_rate"] if self.state_aware else 0.0]
            for row in rows
        ], dtype=np.float32)
        return self.scaler.transform(np.concatenate([projected, scalars], axis=1)).astype(np.float32)

    def save(self, directory: str | Path) -> None:
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target / "feature_pipeline.joblib")


def fit_feature_pipeline(
    rows: list[dict[str, Any]], components: int, *, hidden_key: str = "last_hidden",
    state_aware: bool = True,
) -> FeaturePipeline:
    hidden = np.asarray([row[hidden_key] for row in rows], dtype=np.float32)
    actual_components = min(components, hidden.shape[0], hidden.shape[1])
    pca = PCA(n_components=actual_components, random_state=0).fit(hidden)
    projected = pca.transform(hidden)
    scalars = np.asarray([
        [np.log1p(row["spent_budget"]), np.log1p(row["horizon"]),
         row["recent_token_entropy"] if state_aware else 0.0,
         float(row["has_candidate_answer"]) if state_aware else 0.0,
         row["recent_repetition_rate"] if state_aware else 0.0]
        for row in rows
    ], dtype=np.float32)
    scaler = StandardScaler().fit(np.concatenate([projected, scalars], axis=1))
    return FeaturePipeline(pca, scaler, hidden_key=hidden_key, state_aware=state_aware)
