"""Orna Memory application package."""

import os

# Memory backend не должен отправлять ONNX Runtime telemetry или писать её session artifacts.
os.environ["ORT_DISABLE_TELEMETRY"] = "1"
