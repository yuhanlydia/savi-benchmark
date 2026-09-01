from savi.telemetry import gpu_snapshot


def test_gpu_snapshot_has_expected_schema_when_gpu_is_available():
    value = gpu_snapshot()
    if value is not None:
        assert value["gpu_memory_mib"] > 0
        assert 0 <= value["gpu_utilization_pct"] <= 100
