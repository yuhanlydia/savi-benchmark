from savi.repair_finalizers import repair_rows


def test_repair_module_is_importable_without_running_model():
    assert callable(repair_rows)
