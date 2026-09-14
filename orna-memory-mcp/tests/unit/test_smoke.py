import os
from importlib import import_module


def test_smoke():
    assert True


def test_onnx_runtime_telemetry_is_disabled():
    import_module("app")

    assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
