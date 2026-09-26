"""Resource smoke: host swap observations без модели, сети и datasets."""

from threading import Event, current_thread, main_thread
from types import SimpleNamespace

import pytest

from tests.evals.experiments.qwen_runtime import worker


@pytest.mark.parametrize(
    ("values", "growth", "gate"),
    [
        ([100, 100, 100], 0, True),
        ([100, 90, 80], 0, True),
        ([100, 120, 100], 20, False),
        ([100, 90, 110], 10, False),
        ([None, 100, 100], None, None),
        ([100, None, 100], 0, None),
        ([100, 100, None], 0, None),
        ([100, None, 120], 20, False),
    ],
)
def test_swap_gate_uses_observed_peak_and_requires_complete_samples(values, growth, gate):
    samples = [
        {
            "elapsed_seconds": i / 10,
            "phase": "load",
            "swap_used_bytes": value,
            "worker_peak_rss_bytes": 1000 + i,
        }
        for i, value in enumerate(values)
    ]
    summary = worker._resource_summary(samples)
    assert summary["swap_peak_growth_bytes"] == growth
    assert summary["swap_no_growth"] is gate
    assert summary["worker_peak_rss_bytes"] == 1002
    assert summary["samples"] == samples
    assert summary["swap_delta_bytes"] == (
        None if values[0] is None or values[-1] is None else values[-1] - values[0]
    )


@pytest.mark.parametrize("failure", ["denied", "missing", "timeout", "malformed"])
def test_unavailable_swap_probe_is_unknown(monkeypatch, failure):
    def probe(*args, **kwargs):
        assert kwargs["timeout"] == 1
        if failure == "missing":
            raise FileNotFoundError
        if failure == "timeout":
            raise worker.subprocess.TimeoutExpired(args[0], 1)
        return SimpleNamespace(returncode=int(failure == "denied"), stdout="unavailable")

    monkeypatch.setattr(worker.subprocess, "run", probe)
    assert worker._swap_bytes() is None


def test_monitor_samples_during_work_and_stops_on_error(monkeypatch):
    sampled = Event()

    def probe():
        if current_thread() is not main_thread():
            sampled.set()
        return 100

    monkeypatch.setattr(worker, "_swap_bytes", probe)
    monitor = worker.ResourceMonitor("load")
    with pytest.raises(RuntimeError, match="inference failed"), monitor:
        assert sampled.wait(2), "no background resource sample during work"
        monitor.mark("overlong")
        raise RuntimeError("inference failed")
    assert not monitor.thread.is_alive()
    summary = monitor.summary()
    assert summary["swap_no_growth"] is True
    assert {sample["phase"] for sample in summary["samples"]} == {"load", "overlong"}
    assert summary["samples"][-1]["phase"] == "overlong"


def test_known_but_sparse_samples_cannot_pass_resource_gate():
    samples = [
        {
            "elapsed_seconds": elapsed,
            "phase": "load",
            "swap_used_bytes": 100,
            "worker_peak_rss_bytes": 1000,
        }
        for elapsed in (0, 2, 2.1)
    ]
    summary = worker._resource_summary(samples)
    assert summary["swap_no_growth"] is None
    assert summary["sampling_healthy"] is False


def test_background_sampler_failure_cannot_pass_resource_gate(monkeypatch):
    attempted = Event()

    def broken_probe():
        if current_thread() is not main_thread():
            attempted.set()
            raise RuntimeError("probe crashed")
        return 100

    monkeypatch.setattr(worker, "_swap_bytes", broken_probe)
    with worker.ResourceMonitor("load") as monitor:
        assert attempted.wait(2)
    summary = monitor.summary()
    assert summary["swap_no_growth"] is None
    assert summary["sampling_healthy"] is False
    assert summary["sampling_error"] == "RuntimeError"


@pytest.mark.parametrize(("values", "expected"), [([100, 110], False), ([100, None], None)])
def test_smoke_control_window_is_not_subtracted_or_ignored(monkeypatch, values, expected):
    # Две границы control; все model samples затем стабильны.
    probes = iter(values)
    monkeypatch.setattr(worker, "_swap_bytes", lambda: next(probes, 110))
    monkeypatch.setattr(worker, "CONTROL_SECONDS", 0)
    monkeypatch.setattr(worker, "Reranker", lambda *_: object())

    def fake_smoke(runtime, args, manifest, load_seconds, monitor):
        monitor.mark("overlong")
        return {"resources": {"load_seconds": load_seconds, "query_seconds": [1]}}

    monkeypatch.setattr(worker, "_smoke", fake_smoke)
    evidence = worker._run_smoke(None, None, {})
    assert evidence["gates"]["control_swap_no_growth"] is expected
    assert evidence["gates"]["swap_no_growth"] is True
    assert evidence["resources"]["control"]["swap_no_growth"] is expected
    assert evidence["resources"]["measurement"]["samples"][0]["phase"] == "load"


def test_main_writes_failed_resource_artifact_and_exits_one(monkeypatch, tmp_path):
    import json

    evidence = {"gates": {"swap_no_growth": False}, "resources": {"samples": [100, 120, 100]}}
    monkeypatch.setattr(worker, "verify_environment", lambda _: (tmp_path, {}))
    monkeypatch.setattr(worker, "_run_smoke", lambda *_: evidence)
    output = tmp_path / "failed-smoke"
    monkeypatch.setattr(
        worker.sys,
        "argv",
        [
            "worker.py",
            "smoke",
            "--model",
            worker.MODEL,
            "--revision",
            worker.REVISION,
            "--cache-dir",
            str(tmp_path),
            "--device",
            "cpu",
            "--dtype",
            "float32",
            "--attention",
            "eager",
            "--max-length",
            "2048",
            "--batch-size",
            "1",
            "--threads",
            "6",
            "--interop-threads",
            "1",
            "--instruction-id",
            "selection-instruct-v1",
            "--seed",
            "0",
            "--repeats",
            "3",
            "--query-timeout-seconds",
            "180",
            "--output",
            str(output),
        ],
    )
    assert worker.main() == 1
    assert json.loads((output / "run.json").read_text()) == evidence
