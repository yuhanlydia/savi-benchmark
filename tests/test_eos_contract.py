import pytest

from savi.eos_probe import ensure_contract


def test_probe_contract_allows_identical_resume(tmp_path):
    path = tmp_path / "run.contract.json"
    ensure_contract(path, {"max_tokens": 1024, "announce_budget": False})
    ensure_contract(path, {"max_tokens": 1024, "announce_budget": False})


def test_probe_contract_rejects_condition_mixing(tmp_path):
    path = tmp_path / "run.contract.json"
    ensure_contract(path, {"max_tokens": 1024, "announce_budget": False})
    with pytest.raises(ValueError, match="contract mismatch"):
        ensure_contract(path, {"max_tokens": 8192, "announce_budget": True})
