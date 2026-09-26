"""Eval-only mapping для Qwen reranker; retrieval и relevance остаются у вызывающего кода."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class RerankerError(RuntimeError):
    """Worker failure or broken score/ID contract; never means an empty pool."""


@dataclass(frozen=True, slots=True)
class Candidate:
    id: str
    logical_id: str
    content: str


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    id: str
    logical_id: str
    position: int
    z_yes: float
    z_no: float
    logit_difference: float
    score: float
    diagnostics: dict[str, Any]


def two_token_score(z_yes: float, z_no: float) -> float:
    """Стабильный softmax только по yes/no; это не calibrated relevance."""
    if not math.isfinite(z_yes) or not math.isfinite(z_no):
        raise RerankerError("non-finite logit")
    difference = z_yes - z_no
    if not math.isfinite(difference):
        raise RerankerError("non-finite logit difference")
    return (
        1 / (1 + math.exp(-difference))
        if difference >= 0
        else math.exp(difference) / (1 + math.exp(difference))
    )


def score_candidates(
    case_id: str,
    query: str,
    candidates: Sequence[Candidate],
    infer: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    batch_size: int = 1,
) -> tuple[ScoredCandidate, ...]:
    """Передаёт неизменные query/content, проверяет ordered exact mapping и scores."""
    if batch_size < 1 or not case_id or not query:
        raise ValueError("invalid reranker request")
    if len({item.id for item in candidates}) != len(candidates) or any(
        not item.id or not item.logical_id or not item.content for item in candidates
    ):
        raise RerankerError("invalid or duplicate candidate ID/content")
    if not candidates:
        return ()
    result: list[ScoredCandidate] = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        request = {
            "case_id": case_id,
            "query": query,
            "candidates": [{"id": item.id, "content": item.content} for item in chunk],
        }
        try:
            response = infer(request)
            if response["case_id"] != case_id or len(response["results"]) != len(chunk):
                raise RerankerError("case ID or score cardinality mismatch")
            for offset, (item, row) in enumerate(zip(chunk, response["results"], strict=True)):
                if row["id"] != item.id:
                    raise RerankerError("candidate ID/order mismatch")
                yes, no, score = float(row["z_yes"]), float(row["z_no"]), float(row["score"])
                expected = two_token_score(yes, no)
                if (
                    not math.isfinite(score)
                    or not 0 <= score <= 1
                    or not math.isclose(score, expected, rel_tol=1e-6, abs_tol=1e-7)
                ):
                    raise RerankerError("invalid two-token score")
                result.append(
                    ScoredCandidate(
                        item.id,
                        item.logical_id,
                        start + offset,
                        yes,
                        no,
                        yes - no,
                        score,
                        row.get("diagnostics", {}),
                    )
                )
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerError(f"worker inference failed: {type(exc).__name__}: {exc}") from exc
    return tuple(result)


def rank_candidates(rows: Sequence[ScoredCandidate]) -> tuple[ScoredCandidate, ...]:
    """Full precision sort; исходная позиция выигрывает tie."""
    return tuple(sorted(rows, key=lambda row: (-row.score, row.position, row.logical_id, row.id)))


class QwenWorkerClient:
    """Один offline subprocess на request; P2.5-03 не запускает retrieval или datasets."""

    def __init__(self, cache_dir: Path, *, timeout_seconds: int = 180) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout_seconds = timeout_seconds
        self.runtime = Path(__file__).with_name("qwen_runtime")

    def __call__(self, request: dict[str, Any]) -> dict[str, Any]:
        command = [
            str(self.runtime / ".venv/bin/python"),
            str(self.runtime / "worker.py"),
            "score",
            "--model",
            "Qwen/Qwen3-Reranker-0.6B",
            "--revision",
            "e61197ed45024b0ed8a2d74b80b4d909f1255473",
            "--cache-dir",
            str(self.cache_dir),
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
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "6",
                "MKL_NUM_THREADS": "6",
            }
        )
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=environment,
            )
        except subprocess.TimeoutExpired as exc:
            raise RerankerError("worker process timeout") from exc
        if completed.returncode:
            raise RerankerError(f"worker process failed: {completed.stderr[-500:]}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RerankerError("worker returned invalid JSON") from exc
