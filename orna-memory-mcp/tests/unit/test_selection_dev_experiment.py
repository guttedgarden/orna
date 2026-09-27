"""Selection gates на независимых tiny fixtures, без закрытых evaluation sets."""

import copy

import pytest

from tests.evals.experiments import reranker


def fixture():
    queries = [{"case_id": "p", "relevance": {"a": 2, "b": 1}, "slices": ["partial"]}]
    queries += [
        {"case_id": f"n{i}", "relevance": {}, "slices": ["negative", "near_topic"]}
        for i in range(12)
    ]
    queries += [{"case_id": "ood", "relevance": {}, "slices": ["negative", "ood"]}]
    samples = []
    for q in queries:
        pairs = (
            [("a", 0.9), ("b", 0.8), ("x", 0.1)]
            if q["relevance"]
            else [("x", 0.95 if q["case_id"] == "ood" else 0.2)]
        )
        samples.append(
            {
                "case_id": q["case_id"],
                "baseline": [k for k, _ in pairs][:5],
                "pool": [{"key": k, "id": k} for k, _ in pairs],
                "scores": [{"key": k, "score": s} for k, s in pairs],
            }
        )
    return queries, [copy.deepcopy(samples) for _ in range(3)]


def analyze(queries, repeats):
    from tests.evals.experiments.reranker_selection import analyze_repeats

    return analyze_repeats(queries, repeats)


def test_inclusive_threshold_preserves_partial_and_exact_tie_break():
    q, r = fixture()
    result = analyze(q, r)
    assert result["quality_threshold"] == 0.8
    trial = next(t for t in result["trials"] if t["threshold"] == 0.8)
    assert trial["quality_pass"] is True
    assert trial["repeats"][0]["rankings"]["p"] == ["a", "b"]
    assert trial["repeats"][0]["near_topic"] == {"hits": 0, "total": 12}
    assert trial["repeats"][0]["ood"] == {"hits": 1, "total": 1}


@pytest.mark.parametrize("regression", ["partial", "top1", "mrr", "near_only", "new_ood"])
def test_positive_regressions_and_ood_gain_cannot_buy_admission(regression):
    q, r = fixture()
    for repeat in r:
        if regression == "partial":
            repeat[0]["scores"][1]["score"] = 0.05
        elif regression == "top1":
            repeat[0]["scores"][1]["score"] = 0.99
        elif regression == "mrr":
            repeat[0]["scores"][2]["score"] = 0.99
        elif regression == "near_only":
            for row in repeat[1:-1]:
                row["scores"][0]["score"] = 0.99
            repeat[-1]["scores"][0]["score"] = 0.01
        else:
            repeat[-1]["baseline"] = []
    assert analyze(q, r)["quality_threshold"] is None


def test_repeat_decision_drift_blocks_even_with_close_scores():
    q, r = fixture()
    r[1][0]["scores"][1]["score"] = 0.8000000001
    result = analyze(q, r)
    trial = next(t for t in result["trials"] if t["threshold"] == 0.8000000001)
    assert trial["stable_decisions"] is False
    assert trial["quality_pass"] is False


def test_missing_repeat_or_small_near_slice_is_insufficient():
    q, r = fixture()
    assert analyze(q, r[:2])["quality_threshold"] is None
    q = q[:10]
    r = [[x for x in repeat if x["case_id"] in {y["case_id"] for y in q}] for repeat in r]
    assert analyze(q, r)["quality_threshold"] is None


def test_quality_signal_never_overrides_failed_cost():
    from tests.evals.experiments.reranker_selection import candidate_admitted

    assert candidate_admitted(0.8, {"rss": False}, validation_pass=True) is False
    assert candidate_admitted(0.8, {"rss": None}, validation_pass=True) is False
    assert candidate_admitted(None, {"rss": True}, validation_pass=True) is False
    assert candidate_admitted(0.8, {"rss": True}, validation_pass=True) is True


@pytest.mark.parametrize("validation_pass", [False, None])
def test_dev_or_failed_validation_cannot_gain_admission_when_cost_passes(validation_pass):
    from tests.evals.experiments.reranker_selection import candidate_admitted

    assert candidate_admitted(0.8, {"rss": True}, validation_pass=validation_pass) is False


def test_session_uses_one_process_and_closes_it(tmp_path):
    import sys

    script = tmp_path / "fake.py"
    script.write_text(
        "import json,os,sys\n"
        'print(json.dumps({"ready":True,"pid":os.getpid()}),flush=True)\n'
        "for line in sys.stdin:\n"
        " r=json.loads(line)\n"
        ' print(json.dumps({"case_id":r["case_id"],"pid":os.getpid(),"results":[]}),flush=True)\n'
    )
    assert hasattr(reranker, "QwenWorkerSession"), "persistent lifecycle missing"
    with reranker.QwenWorkerSession(tmp_path, command=[sys.executable, str(script)]) as client:
        pid = client.ready["pid"]
        assert client({"case_id": "a"})["pid"] == pid
        assert client({"case_id": "b"})["pid"] == pid
        process = client.process
    assert process.poll() == 0


def test_session_timeout_kills_worker(tmp_path):
    import sys

    script = tmp_path / "hung.py"
    script.write_text(
        'import json,time\nprint(json.dumps({"ready":True}),flush=True)\ntime.sleep(20)\n'
    )
    assert hasattr(reranker, "QwenWorkerSession"), "persistent lifecycle missing"
    with reranker.QwenWorkerSession(
        tmp_path, command=[sys.executable, str(script)], timeout_seconds=0.05
    ) as client:
        with pytest.raises(reranker.RerankerError, match="timeout"):
            client({"case_id": "a"})
        assert client.process.poll() is not None


def test_resource_peak_growth_and_simultaneous_rss():
    from tests.evals.experiments.selection_resources import summarize_resources

    samples = [
        {
            "elapsed_seconds": i * 0.08,
            "swap_used_bytes": swap,
            "rss_bytes": rss,
            "worker_rss_bytes": worker,
        }
        for i, (swap, rss, worker) in enumerate(
            [(100, 1000, 600), (200, 1500, 500), (100, 1200, 800)]
        )
    ]
    result = summarize_resources(samples)
    assert result["swap_peak_growth_bytes"] == 100
    assert result["swap_no_growth"] is False
    assert result["aggregate_peak_rss_bytes"] == 1500
    assert result["worker_peak_rss_bytes"] == 800
    assert result["aggregate_rss_pass"] is True
    assert result["worker_rss_pass"] is True


def test_missing_swap_does_not_invalidate_healthy_rss_measurement():
    from tests.evals.experiments.selection_resources import summarize_resources

    result = summarize_resources(
        [
            {
                "elapsed_seconds": i * 0.05,
                "swap_used_bytes": None,
                "rss_bytes": 1500,
                "worker_rss_bytes": 800,
            }
            for i in range(3)
        ]
    )
    assert result["swap_no_growth"] is None
    assert result["host_swap_role"] == "diagnostic_only"
    assert result["aggregate_rss_pass"] is True
    assert result["worker_rss_pass"] is True


def test_missing_resource_samples_never_pass():
    from tests.evals.experiments.selection_resources import summarize_resources

    result = summarize_resources(
        [
            {
                "elapsed_seconds": 0,
                "swap_used_bytes": None,
                "rss_bytes": None,
                "worker_rss_bytes": None,
            }
        ]
    )
    assert result["swap_no_growth"] is None
    assert result["aggregate_rss_pass"] is None


def test_resource_budget_interrupt_preserves_partial_sample(tmp_path):
    from tests.evals.experiments.selection_resources import SessionMonitor

    monitor = SessionMonitor(tmp_path / "resources.jsonl")
    monitor.samples = [
        {
            "elapsed_seconds": 0,
            "swap_used_bytes": 0,
            "rss_bytes": 9 * 1024**3,
            "worker_rss_bytes": 0,
        }
    ]
    with pytest.raises(RuntimeError, match="RSS"):
        monitor.check()


@pytest.mark.parametrize("reply", ["not-json", '{"case_id":"wrong","results":[]}'])
def test_session_protocol_errors_close_process(tmp_path, reply):
    import sys

    script = tmp_path / "bad_reply.py"
    script.write_text(
        'import json,sys\nprint(json.dumps({"ready":True}),flush=True)\n'
        "for line in sys.stdin:\n"
        f" print({reply!r},flush=True)\n"
    )
    with reranker.QwenWorkerSession(tmp_path, command=[sys.executable, str(script)]) as client:
        with pytest.raises(reranker.RerankerError):
            client({"case_id": "a"})
        assert client.process.poll() is not None


def test_sampler_failure_marks_unknown_even_with_flat_swap():
    from tests.evals.experiments.selection_resources import summarize_resources

    rows = [
        {
            "elapsed_seconds": i * 0.1,
            "swap_used_bytes": 100,
            "rss_bytes": 100,
            "worker_rss_bytes": 50,
        }
        for i in range(2)
    ]
    result = summarize_resources(rows, "probe failed")
    assert result["swap_no_growth"] is None
    assert result["aggregate_rss_pass"] is None


def test_runner_preflight_failure_writes_partial_artifact(tmp_path, monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace

    from tests.evals.experiments import selection_dev_run

    monkeypatch.setattr(selection_dev_run, "is_model_cache_ready", lambda _: False)
    monkeypatch.setattr(selection_dev_run, "container_footprint", lambda _: {"status": "unknown"})
    output = tmp_path / "run"
    code = asyncio.run(
        selection_dev_run.run(SimpleNamespace(output=output, port=55439, container="unused"))
    )
    artifact = json.loads((output / "run.json").read_text())
    assert code == 1
    assert artifact["status"] == "partial"
    assert artifact["candidate_admitted"] is False
    assert artifact["cost_gates"]["aggregate_rss"] is None
    assert "swap" not in artifact["cost_gates"]
    assert "p2_5_03_resource_gate" not in artifact["admission_gates"]
    assert artifact["admission_gates"]["independent_quality_validation"] is False
    assert artifact["host_swap_role"] == "diagnostic_only"
    assert not (output / "reranker-candidate.json").exists()
    with pytest.raises(FileExistsError):
        asyncio.run(
            selection_dev_run.run(SimpleNamespace(output=output, port=55439, container="unused"))
        )


def test_worker_transient_peak_above_budget_is_retained_and_stops(tmp_path):
    import sys

    peak = 6 * 1024**3 + 1
    script = tmp_path / "peak.py"
    script.write_text(
        'import json,sys\nprint(json.dumps({"ready":True}),flush=True)\n'
        "for line in sys.stdin:\n"
        f' print(json.dumps({{"case_id":"a","worker_peak_rss_bytes":{peak}}}),flush=True)\n'
    )
    with reranker.QwenWorkerSession(tmp_path, command=[sys.executable, str(script)]) as client:
        with pytest.raises(reranker.RerankerError, match="RSS"):
            client({"case_id": "a"})
        assert client.peak_rss_bytes == peak
        assert client.process.poll() is not None


def test_full_pool_rank_drift_blocks_selection_even_outside_top5():
    q, r = fixture()
    for repeat in r:
        repeat[0]["pool"] += [{"key": f"z{i}", "id": f"z{i}"} for i in range(5)]
        repeat[0]["scores"] += [{"key": f"z{i}", "score": 0.09 - i * 0.01} for i in range(5)]
    r[1][0]["scores"][-1]["score"] = 0.061
    result = analyze(q, r)
    assert result["stable_rankings"] is False
    assert result["quality_threshold"] is None


def test_aggregate_gap_over_100ms_is_unknown():
    from tests.evals.experiments.selection_resources import summarize_resources

    rows = [
        {
            "elapsed_seconds": i * 0.2,
            "swap_used_bytes": 100,
            "rss_bytes": 100,
            "worker_rss_bytes": 50,
        }
        for i in range(2)
    ]
    result = summarize_resources(rows)
    assert result["aggregate_rss_pass"] is None
    assert result["swap_no_growth"] is True


@pytest.mark.parametrize(
    ("after", "passes"),
    [
        (["b", "a", "c"], True),
        (["c", "a", "b"], False),
        (["b", "a"], False),
        (["x", "a", "b", "c"], False),
        ([], False),
    ],
)
def test_v2_preserves_ids_and_top1_grade(after, passes):
    from tests.evals.experiments.reranker_selection import preservation

    queries = [{"case_id": "p", "relevance": {"a": 2, "b": 2, "c": 1}}]
    regressions, _, _ = preservation(queries, {"p": ["a", "b", "c"]}, {"p": after})
    assert (not regressions) is passes


def test_v1_remains_reproducible_and_v2_allows_equivalent_top1():
    from tests.evals.experiments.reranker_selection import analyze_repeats

    q, r = fixture()
    q[0]["relevance"]["b"] = 2
    for repeat in r:
        repeat[0]["scores"][1]["score"] = 0.99
    assert analyze_repeats(q, r, policy="selection-protocol-v1")["quality_threshold"] is None
    assert analyze_repeats(q, r)["quality_threshold"] is not None


def test_fixed_validation_does_not_sweep_or_select_another_threshold():
    from tests.evals.experiments.reranker_selection import analyze_repeats

    q, r = fixture()
    result = analyze_repeats(q, r, fixed_threshold=0.95)
    assert [t["threshold"] for t in result["trials"]] == [0.95]
    assert result["quality_threshold"] is None


def test_process_sampler_survives_blocked_parent_and_collects_final_sample(tmp_path):
    import json
    import time

    from tests.evals.experiments.selection_resources import ProcessSessionMonitor

    output = tmp_path / "resources.jsonl"
    with ProcessSessionMonitor(output) as monitor:
        assert monitor.process.pid != __import__("os").getpid()
        monitor.phase = "blocked-parent"
        # Native main-thread sleep does not drive the collector or drain a pipe.
        time.sleep(0.25)
        monitor.check()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len([r for r in rows if r["phase"] == "blocked-parent"]) >= 2
    assert rows[-1]["final_sample"] is True
    assert str(__import__("os").getpid()) in rows[-1]["processes"]
    assert not monitor.process.is_alive()


def test_absolute_sampler_schedule_does_not_add_probe_duration():
    from tests.evals.experiments.selection_resources import next_sample_delay

    deadline, delay = next_sample_delay(10.0, 10.017)
    assert deadline == 10.025
    assert delay == pytest.approx(0.008)
    _, delay = next_sample_delay(10.0, 10.12)
    assert delay == 0


def test_validation_candidate_hash_is_checked_before_inference(tmp_path):
    import json

    from tests.evals.experiments.selection_dev_run import load_research_candidate

    p = tmp_path / "candidate.json"
    p.write_text(json.dumps({"threshold": 0.5}))
    with pytest.raises(ValueError, match="hash"):
        load_research_candidate(p, "0" * 64)


@pytest.mark.parametrize("owned", ["qwen", "embedding"])
def test_process_sampler_deadline_stops_owned_worker_and_keeps_partial(tmp_path, owned):
    import subprocess
    import sys
    import time

    from tests.evals.experiments.selection_resources import ProcessSessionMonitor

    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        with ProcessSessionMonitor(
            tmp_path / "resources.jsonl", worker=lambda: child if owned == "qwen" else None
        ) as monitor:
            if owned == "embedding":
                monitor.state["embedding_worker"].value = child.pid
            monitor.query_deadline = time.perf_counter() + 0.1
            monitor.check()
            child.wait(timeout=3)
            with pytest.raises(RuntimeError, match="query_deadline exceeded"):
                monitor.check()
        assert monitor.summary()["aggregate_rss_pass"] is None
        assert (tmp_path / "resources.jsonl").stat().st_size > 0
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


def test_validation_never_enumerates_alternative_scores():
    from tests.evals.experiments.reranker_selection import analyze_repeats

    q, r = fixture()
    result = analyze_repeats(q, r, fixed_threshold=0.7)
    assert len(result["trials"]) == 1
    assert result["trials"][0]["threshold"] == 0.7
    assert result["quality_threshold"] == 0.7


def test_partial_sampler_json_does_not_mask_budget_failure(tmp_path):
    from tests.evals.experiments.selection_resources import ProcessSessionMonitor

    p = tmp_path / "resources.jsonl"
    p.write_text('{"unfinished":')
    monitor = ProcessSessionMonitor(p)
    monitor.state["error"].value = b"aggregate RSS budget exceeded"
    summary = monitor.summary()
    assert summary["aggregate_rss_pass"] is None
    assert "aggregate RSS budget exceeded" in summary["sampling_error"]


def test_owned_embedding_process_is_killable_without_harness_exit(tmp_path):
    import asyncio
    import sys

    from tests.evals.experiments.selection_embedding import ProcessEmbeddingExecutor

    async def scenario():
        fake = tmp_path / "fake.py"
        fake.write_text("import time; time.sleep(20)")
        executor = ProcessEmbeddingExecutor(None, command=[sys.executable, str(fake)])
        pid = executor.process.pid
        await executor.aclose()
        assert pid and executor.process.poll() is not None

    asyncio.run(scenario())


def test_frozen_identity_compares_serialized_tuple_fields():
    from tests.evals.experiments.selection_dev_run import same_identity

    assert same_identity({"files": ("a", "b")}, {"files": ["a", "b"]})
    assert not same_identity({"files": ("a", "b")}, {"files": ["b", "a"]})


def test_e5_json_write_handles_partial_pipe_writes():
    import json

    from tests.evals.experiments.selection_embedding import write_json

    class ShortWriter:
        def __init__(self):
            self.data = bytearray()

        def write(self, chunk):
            self.data.extend(chunk[:3])
            return min(3, len(chunk))

    stream = ShortWriter()
    payload = {"text": "Русский запрос " * 1000}
    write_json(stream, payload)
    assert json.loads(stream.data) == payload


def test_e5_cleanup_wait_is_bounded():
    import asyncio
    import io
    import subprocess
    from types import SimpleNamespace

    from tests.evals.experiments.selection_embedding import ProcessEmbeddingExecutor

    def wait(*, timeout):
        assert timeout <= 3
        raise subprocess.TimeoutExpired("fake-e5", timeout)

    executor = ProcessEmbeddingExecutor.__new__(ProcessEmbeddingExecutor)
    executor.process = SimpleNamespace(
        poll=lambda: 0, wait=wait, stdin=io.BytesIO(), stdout=io.BytesIO()
    )
    executor.errors = io.BytesIO()
    executor.monitor = None
    with pytest.raises(RuntimeError, match="cleanup timeout"):
        asyncio.run(executor.aclose())
    assert executor.process.stdin.closed and executor.process.stdout.closed


def test_validation_rejects_changed_threshold_even_with_supplied_matching_hash(
    tmp_path, monkeypatch
):
    import json

    from tests.evals.experiments import selection_dev_run as runner

    p = tmp_path / "candidate.json"
    p.write_text(json.dumps({"threshold": 0.9}))
    (tmp_path / "research-freeze-v2.json").write_text(
        json.dumps({"candidate_sha256": "a" * 64, "threshold": 0.8})
    )
    monkeypatch.setattr(runner, "DEV", tmp_path)
    with pytest.raises(ValueError, match="anchor"):
        runner.load_research_candidate(p, runner.digest(p))


def test_e5_request_path_handles_partial_pipe_writes(monkeypatch):
    import asyncio
    import json
    from types import SimpleNamespace

    from tests.evals.experiments import selection_embedding as module

    class ShortWriter:
        def __init__(self):
            self.data = bytearray()

        def write(self, chunk):
            self.data.extend(chunk[:3])
            return min(3, len(chunk))

    stream = ShortWriter()
    executor = module.ProcessEmbeddingExecutor.__new__(module.ProcessEmbeddingExecutor)
    executor.process = SimpleNamespace(
        stdin=stream, stdout=SimpleNamespace(fileno=lambda: 123), poll=lambda: None
    )
    executor.monitor = None
    executor.first = False
    monkeypatch.setattr(module.os, "read", lambda *_: b'{"vector":[1.0]}\n')
    assert asyncio.run(executor.embed_query("request" * 1000)) == [1.0]
    assert json.loads(stream.data) == {"operation": "query", "text": "request" * 1000}
