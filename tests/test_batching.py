import pytest

from savi.batching import batch_sampling_seed, microbatch_slices, planned_batch_size


def test_planned_batch_size_respects_sequence_and_batch_caps():
    assert planned_batch_size(
        4200, 2048, max_batch_size=4, max_batch_context_tokens=14000
    ) == 2
    assert planned_batch_size(
        1000, 1000, max_batch_size=4, max_batch_context_tokens=14000
    ) == 4


def test_planned_batch_size_never_returns_zero_for_oversized_sequence():
    assert planned_batch_size(
        16000, 2048, max_batch_size=4, max_batch_context_tokens=14000
    ) == 1


def test_planned_batch_size_rejects_invalid_limits():
    with pytest.raises(ValueError):
        planned_batch_size(10, 10, max_batch_size=0, max_batch_context_tokens=100)


def test_microbatch_slices_cover_each_index_once():
    assert list(microbatch_slices(5, 2)) == [(0, 2), (2, 4), (4, 5)]


def test_microbatch_slices_handles_empty_work():
    assert list(microbatch_slices(0, 2)) == []


def test_batch_sampling_seed_is_stable_and_group_specific():
    first = batch_sampling_seed(7, "state", 2048, 0)
    assert first == batch_sampling_seed(7, "state", 2048, 0)
    assert first != batch_sampling_seed(7, "state", 2048, 2)
