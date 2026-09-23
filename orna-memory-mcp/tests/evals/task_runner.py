"""Isolated task-level ON/OFF pilot over a pinned checkout and read-only eval DB."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tokenizers import Tokenizer

from app.config import Settings
from app.embeddings import AsyncEmbeddingExecutor, EmbeddingService, ModelCacheMissingError
from app.model_cache import is_model_cache_ready
from app.repository import MemoryRepository
from app.search import MemorySearchQuery, MemorySearchService
from tests.evals.database import ephemeral_eval_database, inspect_eval_database, load_eval_corpus
from tests.evals.dataset import CorpusRecord
from tests.evals.graders import TaskCase, grade_answer, summarize_pairs
from tests.evals.task_adapter import OpenAICompatibleCompletion, run_trial, validate_local_endpoint

ROOT = Path(__file__).resolve().parents[3]
TASK_ROOT = Path(__file__).resolve().parent / "tasks"


@dataclass(frozen=True, slots=True)
class RunSlot:
    task_id: str
    repeat: int
    condition: str


def load_cases(path: Path) -> tuple[TaskCase, ...]:
    cases = tuple(
        TaskCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not cases or len({case.task_id for case in cases}) != len(cases):
        raise ValueError("pilot task IDs must be unique and nonempty")
    if len({case.repo_ref for case in cases}) != 1:
        raise ValueError("all pilot cases must use one pinned repository ref")
    return cases


def build_run_order(cases: tuple[TaskCase, ...], *, repeats: int, seed: int) -> tuple[RunSlot, ...]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    slots = [
        RunSlot(case.task_id, repeat, condition)
        for case in cases
        for repeat in range(repeats)
        for condition in ("on", "off")
    ]
    random.Random(seed).shuffle(slots)
    return tuple(slots)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_corpus(path: Path) -> tuple[CorpusRecord, ...]:
    return tuple(
        CorpusRecord.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def validate_retrieval_config(config: dict) -> dict:
    """Keep manifest, runtime settings and the current baseline identical."""

    retrieval = config["retrieval"]
    baseline = {
        "dense_strategy": "exact",
        "fts_config": "simple",
        "candidate_pool_size": 20,
        "rrf_k": 60,
        "embedding_profile": "e5-v1",
    }
    if retrieval != baseline:
        raise ValueError("pilot retrieval config must match exact/simple/pool20/RRF60/e5-v1")
    return retrieval


def apply_treatment_adherence(trials: list[dict]) -> None:
    """Exclude conditions that did not actually expose the intended memory treatment."""

    for trial in trials:
        events = trial.get("tool_events", [])
        if trial["condition"] == "on":
            adherent = bool(events) and events[0].get("tool") == "memory_search"
            adherent = adherent and "results" in events[0]
        else:
            adherent = all(event.get("tool") != "memory_search" for event in events)
        trial["treatment_adherent"] = adherent
        if not adherent:
            if "task_success_before_adherence" not in trial:
                trial["task_success_before_adherence"] = trial.get("success")
            trial["success"] = None
            trial["invalidation_reason"] = "treatment_nonadherence"


def recount_memory_context(
    result: dict, tokenizer_json: Path, *, tokenizer_manifest: Path | None = None
) -> dict:
    """Count exactly serialized search tool content with the supplied model tokenizer."""

    apply_treatment_adherence(result["trials"])
    binding = None
    if tokenizer_manifest is not None:
        binding = json.loads(tokenizer_manifest.read_text(encoding="utf-8"))
        expected = (
            result.get("model") == binding.get("model")
            and result.get("model_version") == binding.get("model_version")
            and _sha256(tokenizer_json) == binding.get("tokenizer_sha256")
            and importlib.metadata.version("tokenizers") == binding.get("library_version")
        )
        if not expected:
            raise ValueError("tokenizer manifest does not match model, variant or tokenizer file")
    tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def count(value: str) -> int:
        return len(tokenizer.encode(value, add_special_tokens=False).ids)

    for trial in result["trials"]:
        searches = [
            event
            for event in trial.get("tool_events", [])
            if event.get("tool") == "memory_search" and "results" in event
        ]
        payloads = [
            json.dumps({"results": event["results"]}, ensure_ascii=False) for event in searches
        ]
        trial["memory_context_tokens"] = sum(count(payload) for payload in payloads)
        trial["memory_context_bytes"] = sum(len(payload.encode("utf-8")) for payload in payloads)
        trial["retrieved_memory_tokens"] = sum(
            count(record["content"])
            for event in searches
            for record in event["results"]
            if "content" in record
        )
    completed_pairs = []
    for task_id, repeat in {(trial["task_id"], trial["repeat"]) for trial in result["trials"]}:
        pair = [
            trial
            for trial in result["trials"]
            if trial["task_id"] == task_id and trial["repeat"] == repeat
        ]
        if len(pair) == 2 and all(isinstance(trial["success"], bool) for trial in pair):
            completed_pairs.extend(pair)
    result["summary"] = summarize_pairs(completed_pairs) if completed_pairs else None
    result["tokenizer"] = {
        "library": "tokenizers",
        "library_version": importlib.metadata.version("tokenizers"),
        "json_path": str(tokenizer_json.resolve()),
        "json_sha256": _sha256(tokenizer_json),
        "scope": "serialized memory_search JSON tool message content only; "
        "no chat template envelope",
        "retrieved_memory_scope": "sum of returned record content strings only; "
        "duplicates counted per search call",
        "add_special_tokens": False,
        "model_binding_verified": binding is not None,
        "binding_manifest_sha256": _sha256(tokenizer_manifest) if binding else None,
    }
    if binding:
        result["limitations"] = [
            line
            for line in result["limitations"]
            if not line.startswith("No verified model tokenizer")
        ]
    else:
        result["limitations"].append("Supplied tokenizer was not bound to recorded model/variant")
    result["limitations"].append(
        "Memory context tokens count the tool content string; "
        "they are not billed prompt-token delta"
    )
    return result


def _fresh_checkout(ref: str, destination: Path) -> None:
    """Archive only committed content into a fresh directory; no shared Git state."""

    with subprocess.Popen(
        ["git", "-C", str(ROOT), "archive", "--format=tar", ref], stdout=subprocess.PIPE
    ) as archive:
        if archive.stdout is None:
            raise RuntimeError("git archive has no stdout")
        unpack = subprocess.run(
            ["tar", "-xf", "-", "-C", str(destination)], stdin=archive.stdout, check=False
        )
        archive.stdout.close()
        if archive.wait() or unpack.returncode:
            raise RuntimeError("fresh checkout extraction failed")


class _StubCompletion:
    """Harness plumbing only: reads the predeclared rubric, never measures model quality."""

    def __init__(self, case: TaskCase, condition: str) -> None:
        self._case = case
        self._condition = condition
        self._calls = 0

    async def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        self._calls += 1
        if self._calls == 1:
            calls = []
            if self._condition == "on":
                calls.append(
                    {
                        "id": "search-1",
                        "type": "function",
                        "function": {
                            "name": "memory_search",
                            "arguments": json.dumps({"query": self._case.prompt}),
                        },
                    }
                )
            calls.append(
                {
                    "id": "read-1",
                    "type": "function",
                    "function": {
                        "name": "read_repo_file",
                        "arguments": json.dumps({"path": self._case.rubric.source_path}),
                    },
                }
            )
            message = {"content": None, "tool_calls": calls}
        else:
            message = {
                "content": json.dumps(
                    {
                        "answer_key": self._case.rubric.answer_key,
                        "evidence": [self._case.rubric.source_path],
                        "memory_ids_used": [],
                    }
                )
            }
        return {
            "choices": [{"message": message}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


async def execute(args: argparse.Namespace) -> dict[str, Any]:
    cases_path = TASK_ROOT / "cases.jsonl"
    corpus_path = TASK_ROOT / "corpus.jsonl"
    config_path = TASK_ROOT / "pilot.json"
    cases = load_cases(cases_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    retrieval = validate_retrieval_config(config)
    if any(case.corpus_version != config["corpus_version"] for case in cases):
        raise ValueError("task case corpus version differs from pilot manifest")
    corpus = _load_corpus(corpus_path)
    full_order = build_run_order(cases, repeats=config["repeats"], seed=config["seed"])
    if args.trial_index is not None:
        order = (full_order[args.trial_index],)
    else:
        order = full_order[: args.max_trials] if args.max_trials is not None else full_order
    case_by_id = {case.task_id: case for case in cases}
    allowed_paths = {case.rubric.source_path for case in cases}
    model = args.model or config["model"]
    endpoint = args.endpoint or config["endpoint"]
    model_version = args.model_version or config["model_version"]
    if args.adapter == "local" and (not model or not endpoint):
        raise ValueError("local client requires --model and --endpoint")
    if args.adapter == "local" and not model_version:
        raise ValueError("local client requires a recorded model version or variant")
    if args.adapter == "local" and not args.client_version:
        raise ValueError("local client requires --client-version")
    if args.adapter == "local":
        endpoint = validate_local_endpoint(endpoint)
    if args.adapter == "local" and not is_model_cache_ready(
        Path(os.environ.get("ORNA_TEST_E5_CACHE_DIR", Settings().embedding_cache_dir))
    ):
        raise ModelCacheMissingError("pinned E5 offline cache is unavailable")

    async def perform(search_service: MemorySearchService | None, db_info: dict | None) -> dict:
        started_at = datetime.now(UTC).isoformat()
        trials: list[dict] = []
        for index, slot in enumerate(order):
            case = case_by_id[slot.task_id]
            with tempfile.TemporaryDirectory(prefix="orna_pilot_checkout_") as directory:
                checkout = Path(directory)
                _fresh_checkout(case.repo_ref, checkout)
                if args.adapter == "stub":
                    completion = _StubCompletion(case, slot.condition)

                    async def stub_search(query: str) -> list[dict]:
                        return [{"id": str(corpus[0].id), "content": corpus[0].content}]

                    search = stub_search if slot.condition == "on" else None
                else:
                    completion = OpenAICompatibleCompletion(
                        endpoint,
                        model,
                        temperature=config["temperature"],
                        max_tokens=config["max_output_tokens"],
                        timeout=config["timeout_seconds"],
                    )

                    async def real_search(query: str) -> list[dict]:
                        assert search_service is not None
                        records = await search_service.search(
                            MemorySearchQuery(query=query, limit=5), config["project_id"]
                        )
                        return [
                            {
                                "id": str(record.id),
                                "content": record.content,
                                "memory_type": record.memory_type,
                                "status": record.status.value,
                                "source_ref": record.provenance.get("source", {}).get("source_ref"),
                            }
                            for record in records
                        ]

                    search = real_search if slot.condition == "on" else None
                try:
                    outcome = await run_trial(
                        case=case,
                        checkout=checkout,
                        condition=slot.condition,
                        completion=completion,
                        search=search,
                        prompts=config["prompts"],
                        allowed_paths=allowed_paths,
                        max_tool_rounds=config["max_tool_rounds"],
                    )
                    grade = grade_answer(case, outcome["final_artifact"])
                    outcome.update(success=grade.success, rubric_checks=grade.checks)
                except Exception as error:
                    outcome = {
                        "success": None,
                        "error_type": type(error).__name__,
                        "final_artifact": "",
                        "tool_events": [],
                        "latency_ms": None,
                        "input_tokens": None,
                        "output_tokens": None,
                        "memory_context_tokens": None,
                        "results_used": None,
                    }
                trials.append(
                    {
                        "run_index": index,
                        "task_id": case.task_id,
                        "repeat": slot.repeat,
                        "condition": slot.condition,
                        "repo_ref": case.repo_ref,
                        **outcome,
                    }
                )
        apply_treatment_adherence(trials)
        completed_pairs = []
        for case in cases:
            for repeat in range(config["repeats"]):
                pair = [
                    trial
                    for trial in trials
                    if trial["task_id"] == case.task_id and trial["repeat"] == repeat
                ]
                if len(pair) == 2 and all(isinstance(item["success"], bool) for item in pair):
                    completed_pairs.extend(pair)
        return {
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "evidence_kind": (
                "stub_adapter_only"
                if args.adapter == "stub"
                else "local_client_model"
                if completed_pairs
                else "local_client_incomplete"
            ),
            "client": config["client"] if args.adapter == "local" else "deterministic-stub",
            "client_version": args.client_version if args.adapter == "local" else None,
            "endpoint": endpoint if args.adapter == "local" else None,
            "model": model if args.adapter == "local" else None,
            "model_version": model_version if args.adapter == "local" else None,
            "config": config,
            "fixture_sha256": {
                "cases": _sha256(cases_path),
                "corpus": _sha256(corpus_path),
                "pilot": _sha256(config_path),
            },
            "environment": {
                "python": platform.python_version(),
                "system": platform.platform(),
                "git_ref": cases[0].repo_ref,
                "database": db_info,
            },
            "run_order": [
                slot.__dict__
                if hasattr(slot, "__dict__")
                else {"task_id": slot.task_id, "repeat": slot.repeat, "condition": slot.condition}
                for slot in order
            ],
            "scheduled_order": [
                {"task_id": slot.task_id, "repeat": slot.repeat, "condition": slot.condition}
                for slot in full_order
            ],
            "trials": trials,
            "summary": summarize_pairs(completed_pairs) if completed_pairs else None,
            "unfinished_trials": sum(trial["success"] is None for trial in trials),
            "limitations": [
                "Stub uses rubric answers and is not task-level quality evidence"
                if args.adapter == "stub"
                else "Single local model and small pilot; no general quality claim",
                "No verified model tokenizer configured: memory context token count "
                "and derived cost are undefined",
                "results_used records only explicit client-declared delivered memory IDs",
            ],
        }

    if args.adapter == "stub":
        result = await perform(None, None)
    else:
        base = Settings().model_copy(
            update={
                "dense_retrieval_strategy": retrieval["dense_strategy"],
                "retrieval_candidate_pool_size": retrieval["candidate_pool_size"],
                "rrf_k": retrieval["rrf_k"],
                "embedding_local_files_only": True,
                "embedding_cache_dir": Path(
                    os.environ.get("ORNA_TEST_E5_CACHE_DIR", Settings().embedding_cache_dir)
                ),
            }
        )
        async with ephemeral_eval_database(base) as database:
            embeddings = AsyncEmbeddingExecutor(
                EmbeddingService(database.settings),
                max_concurrency=database.settings.embedding_max_concurrency,
            )
            try:
                await load_eval_corpus(database.pool, corpus, embeddings, database.settings)
                repository = MemoryRepository(database.pool, database.settings)
                search_service = MemorySearchService(repository, embeddings, database.settings)
                versions, size = await inspect_eval_database(database.pool)
                result = await perform(search_service, {**versions, **size})
            finally:
                await embeddings.aclose()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Orna ON/OFF task pilot")
    parser.add_argument("--adapter", choices=("stub", "local"))
    parser.add_argument("--recount-input", type=Path)
    parser.add_argument("--tokenizer-json", type=Path)
    parser.add_argument("--tokenizer-manifest", type=Path)
    parser.add_argument("--endpoint")
    parser.add_argument("--model")
    parser.add_argument("--model-version")
    parser.add_argument("--client-version")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--trial-index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.recount_input is not None:
        if (
            args.adapter is not None
            or args.tokenizer_json is None
            or args.tokenizer_manifest is None
        ):
            parser.error(
                "--recount-input requires --tokenizer-json, --tokenizer-manifest and no --adapter"
            )
        result = recount_memory_context(
            json.loads(args.recount_input.read_text(encoding="utf-8")),
            args.tokenizer_json,
            tokenizer_manifest=args.tokenizer_manifest,
        )
        if args.client_version:
            result["client_version"] = args.client_version
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    else:
        if args.adapter is None:
            parser.error("--adapter is required for a trial run")
        result = asyncio.run(execute(args))
    print(json.dumps({"summary": result["summary"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
