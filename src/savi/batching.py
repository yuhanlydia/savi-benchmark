from __future__ import annotations

import hashlib
from typing import Iterator


def planned_batch_size(
    input_tokens: int,
    new_tokens: int,
    *,
    max_batch_size: int,
    max_batch_context_tokens: int,
) -> int:
    """Return a conservative fixed batch size from a token-memory budget."""
    if input_tokens < 0 or new_tokens < 0:
        raise ValueError("token counts must be non-negative")
    if max_batch_size <= 0 or max_batch_context_tokens <= 0:
        raise ValueError("batch limits must be positive")
    per_sequence = max(1, input_tokens + new_tokens)
    by_tokens = max_batch_context_tokens // per_sequence
    return max(1, min(max_batch_size, by_tokens))


def microbatch_slices(total: int, batch_size: int) -> Iterator[tuple[int, int]]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, total, batch_size):
        yield start, min(total, start + batch_size)


def batch_sampling_seed(base_seed: int, *parts: object) -> int:
    text = "|".join([str(base_seed), *(str(part) for part in parts), "grouped_rng_v1"])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
