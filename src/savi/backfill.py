from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .io import append_jsonl, read_jsonl


def backfill_rows(
    runner: Any,
    source_rows: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Enrich prefix rows exactly once, preserving source identity and order."""
    source = list(source_rows)
    source_ids = [row.get("state_id") for row in source]
    if any(not state_id for state_id in source_ids):
        raise ValueError("every source row must contain a state_id")
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("source rows contain duplicate state_id values")

    output = Path(output_path)
    existing = read_jsonl(output) if output.exists() else []
    completed: dict[str, dict[str, Any]] = {}
    for row in existing:
        state_id = row.get("state_id")
        if not state_id or state_id in completed:
            raise ValueError("output contains a missing or duplicate state_id")
        completed[state_id] = row

    for row in source:
        state_id = row["state_id"]
        if state_id in completed:
            if completed[state_id].get("prefix_token_ids") != row.get("prefix_token_ids"):
                raise ValueError(f"token IDs changed for completed state {state_id}")
            if completed[state_id].get("feature_mode") != "full_hidden":
                raise ValueError(f"completed state {state_id} lacks full hidden features")
            continue
        features = runner.state_features(row.get("problem", ""), row["prefix_token_ids"])
        if "last_hidden" not in features:
            raise ValueError(f"runner returned no last_hidden for state {state_id}")
        enriched = {**row, **features, "feature_mode": "full_hidden"}
        append_jsonl(output, [enriched])
        completed[state_id] = enriched
    return output
