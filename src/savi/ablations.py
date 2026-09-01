from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


STATE_FIELDS = (
    "last_hidden", "recent_token_entropy", "has_candidate_answer",
    "recent_repetition_rate",
)


def state_shuffle(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    """Replace state features with a different trajectory from the same x,b cell."""
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        groups[(row["problem_id"], row["spent_budget"])][row["state_id"]] = row
    rng = np.random.default_rng(seed)
    donor_by_state = {}
    for cell, states in groups.items():
        state_ids = sorted(states)
        if len(state_ids) < 2:
            raise ValueError(f"state shuffle requires at least two trajectories in {cell}")
        offset = int(rng.integers(1, len(state_ids)))
        for index, state_id in enumerate(state_ids):
            donor_by_state[(cell, state_id)] = states[state_ids[(index + offset) % len(state_ids)]]
    shuffled = []
    for row in rows:
        cell = (row["problem_id"], row["spent_budget"])
        donor = donor_by_state[(cell, row["state_id"])]
        shuffled.append({**row, **{field: donor[field] for field in STATE_FIELDS},
                         "state_shuffle_donor": donor["state_id"]})
    return shuffled
