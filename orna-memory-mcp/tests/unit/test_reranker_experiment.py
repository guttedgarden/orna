"""Контракт eval-only reranker без загрузки модели и datasets."""

from __future__ import annotations

import math

import pytest

from tests.evals.experiments.reranker import (
    Candidate,
    QwenWorkerClient,
    RerankerError,
    rank_candidates,
    score_candidates,
    two_token_score,
)


def _candidate(id_: str, content: str = "A fact") -> Candidate:
    return Candidate(id=id_, logical_id=f"logical-{id_}", content=content)


def _answer(request, values):
    return {
        "case_id": request["case_id"],
        "results": [
            {"id": item["id"], "z_yes": yes, "z_no": no, "score": two_token_score(yes, no)}
            for item, (yes, no) in zip(request["candidates"], values, strict=True)
        ],
    }


def test_empty_pool_skips_inference():
    assert score_candidates("c", "query", [], lambda _: pytest.fail("called")) == ()


def test_mapping_order_and_deterministic_ties():
    candidates = [_candidate("b"), _candidate("a")]
    result = score_candidates(
        "c", "query", candidates, lambda req: _answer(req, [(1, 0), (1, 0)]), batch_size=2
    )
    assert [row.id for row in result] == ["b", "a"]
    assert [row.id for row in rank_candidates(result)] == ["b", "a"]


def test_batch_boundary_preserves_all_candidates():
    candidates = [_candidate(str(i)) for i in range(3)]
    seen = []

    def infer(request):
        seen.append([item["id"] for item in request["candidates"]])
        return _answer(request, [(float(item["id"]), 0) for item in request["candidates"]])

    assert len(score_candidates("c", "query", candidates, infer, batch_size=2)) == 3
    assert seen == [["0", "1"], ["2"]]


@pytest.mark.parametrize(
    "bad", ["missing", "duplicate", "reordered", "wrong_case", "nonfinite", "wrong_score"]
)
def test_invalid_worker_mapping_fails(bad):
    candidates = [_candidate("a"), _candidate("b")]

    def infer(request):
        answer = _answer(request, [(1, 0), (0, 1)])
        if bad == "missing":
            answer["results"].pop()
        elif bad == "duplicate":
            answer["results"][1]["id"] = "a"
        elif bad == "reordered":
            answer["results"].reverse()
        elif bad == "wrong_case":
            answer["case_id"] = "other"
        elif bad == "nonfinite":
            answer["results"][0]["z_yes"] = math.inf
        else:
            answer["results"][0]["score"] = 0.1
        return answer

    with pytest.raises(RerankerError):
        score_candidates("c", "query", candidates, infer, batch_size=2)


def test_inference_error_is_not_empty_result():
    def broken(_):
        raise TimeoutError("worker timeout")

    with pytest.raises(RerankerError, match="worker timeout"):
        score_candidates("c", "query", [_candidate("a")], broken)


def test_multilingual_input_and_tail_are_unchanged():
    content = "начало " * 2500 + "порт 8123 находится в хвосте"
    captured = []

    def infer(request):
        captured.append(request)
        return _answer(request, [(1, 0)])

    scored = score_candidates("ru", "На каком порту сервис?", [_candidate("a", content)], infer)
    assert captured[0]["query"] == "На каком порту сервис?"
    assert captured[0]["candidates"][0]["content"] == content
    assert scored[0].id == "a"
    assert "relevance" not in captured[0]


class _CharacterTokenizer:
    def encode(self, value, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(character) for character in value]

    def decode(self, ids):
        return "".join(chr(token) for token in ids)


def test_worker_head_only_truncation_keeps_suffix_and_original_label():
    from tests.evals.experiments.qwen_runtime.worker import PREFIX, SUFFIX, Reranker

    runtime = object.__new__(Reranker)
    runtime.tokenizer = _CharacterTokenizer()
    runtime.prefix_ids = runtime.tokenizer.encode(PREFIX, add_special_tokens=False)
    runtime.suffix_ids = runtime.tokenizer.encode(SUFFIX, add_special_tokens=False)
    runtime.max_length = 2048
    label = {"tail-memory": 2}
    content = "раннее описание. " * 250 + "The port is 8123."
    prepared = runtime.prepare("На каком порту сервис?", content)
    diagnostics = prepared["diagnostics"]
    assert diagnostics["tokens_used"] == 2048
    assert diagnostics["tokens_removed"] > 0
    assert diagnostics["document_truncated"] is True
    assert diagnostics["suffix_preserved"] is True
    assert "The port is 8123." not in runtime.tokenizer.decode(prepared["input_ids"])
    assert label == {"tail-memory": 2}


def test_worker_rejects_header_that_cannot_fit():
    from tests.evals.experiments.qwen_runtime.worker import PREFIX, SUFFIX, Reranker, WorkerError

    runtime = object.__new__(Reranker)
    runtime.tokenizer = _CharacterTokenizer()
    runtime.prefix_ids = runtime.tokenizer.encode(PREFIX, add_special_tokens=False)
    runtime.suffix_ids = runtime.tokenizer.encode(SUFFIX, add_special_tokens=False)
    runtime.max_length = 2048
    with pytest.raises(WorkerError, match="header"):
        runtime.prepare("запрос " * 400, "document")


def test_subprocess_client_forces_offline_and_propagates_timeout(monkeypatch, tmp_path):
    import subprocess

    def fake_run(command, **kwargs):
        assert command[2] == "score"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["TRANSFORMERS_OFFLINE"] == "1"
        assert kwargs["timeout"] == 180
        raise subprocess.TimeoutExpired(command, 180)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RerankerError, match="timeout"):
        QwenWorkerClient(tmp_path)({"case_id": "c", "query": "q", "candidates": []})
