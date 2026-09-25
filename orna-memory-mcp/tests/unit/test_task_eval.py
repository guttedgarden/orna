"""Contracts for the task-level ON/OFF pilot without a model or PostgreSQL."""

import json
from pathlib import Path

import pytest

from tests.evals.graders import grade_answer, summarize_pairs
from tests.evals.task_adapter import (
    read_repo_file,
    run_trial,
    summarize_tool_events,
    validate_local_endpoint,
)
from tests.evals.task_runner import (
    apply_treatment_adherence,
    build_run_order,
    load_cases,
    recount_memory_context,
    validate_retrieval_config,
)

TASK_ROOT = Path(__file__).parents[1] / "evals" / "tasks"


def test_cases_cover_required_slices_with_fixed_rubrics() -> None:
    cases = load_cases(TASK_ROOT / "cases.jsonl")
    assert {case.kind for case in cases} == {
        "known_incident",
        "convention",
        "source_memory_contradiction",
        "no_useful_memory",
    }
    assert all(case.repo_ref and case.prompt and case.rubric.answer_key for case in cases)
    assert len({case.task_id for case in cases}) == len(cases)


def test_grader_requires_exact_answer_and_source_evidence() -> None:
    case = load_cases(TASK_ROOT / "cases.jsonl")[0]
    good = json.dumps({"answer_key": case.rubric.answer_key, "evidence": [case.rubric.source_path]})
    assert grade_answer(case, good).success
    assert not grade_answer(case, '{"answer_key":"wrong","evidence":[]}').success
    assert not grade_answer(case, "I think the answer is correct").success


def test_grader_accepts_equivalent_number_and_fenced_json() -> None:
    cases = load_cases(TASK_ROOT / "cases.jsonl")
    numeric = next(case for case in cases if case.task_id == "pilot-convention-pool")
    assert grade_answer(
        numeric,
        json.dumps({"answer_key": 20, "evidence": [numeric.rubric.source_path]}),
    ).success
    storage = next(case for case in cases if case.task_id == "pilot-contradiction-storage")
    fenced = (
        "```json\n"
        + json.dumps(
            {"answer_key": storage.rubric.answer_key, "evidence": [storage.rubric.source_path]}
        )
        + "\n```"
    )
    assert grade_answer(storage, fenced).success


def test_order_is_seeded_and_pair_complete() -> None:
    cases = load_cases(TASK_ROOT / "cases.jsonl")
    first = build_run_order(cases, repeats=3, seed=29)
    assert first == build_run_order(cases, repeats=3, seed=29)
    assert len(first) == len(cases) * 6
    assert {(item.task_id, item.repeat, item.condition) for item in first} == {
        (case.task_id, repeat, condition)
        for case in cases
        for repeat in range(3)
        for condition in ("on", "off")
    }


def test_tool_trace_counts_calls_and_explicit_memory_use_only() -> None:
    events = [
        {"tool": "memory_search", "results": [{"id": "a", "content": "x"}]},
        {"tool": "read_repo_file", "path": "AGENTS.md"},
    ]
    result = summarize_tool_events(events, '{"memory_ids_used":["a"]}')
    assert result["memory_search_calls"] == 1
    assert result["tool_calls"] == 2
    assert result["results_returned"] == 1
    assert result["results_used"] == ["a"]


def test_pair_summary_keeps_context_cost_distinct_from_total_token_delta() -> None:
    trials = [
        {
            "task_id": "a",
            "repeat": 0,
            "condition": "on",
            "success": True,
            "memory_context_tokens": 30,
            "input_tokens": 100,
            "output_tokens": 10,
            "reasoning_tokens": 7,
        },
        {
            "task_id": "a",
            "repeat": 0,
            "condition": "off",
            "success": False,
            "memory_context_tokens": 0,
            "input_tokens": 50,
            "output_tokens": 10,
            "reasoning_tokens": 3,
        },
        {
            "task_id": "b",
            "repeat": 0,
            "condition": "on",
            "success": False,
            "memory_context_tokens": 20,
            "input_tokens": 100,
            "output_tokens": 10,
            "reasoning_tokens": 8,
        },
        {
            "task_id": "b",
            "repeat": 0,
            "condition": "off",
            "success": True,
            "memory_context_tokens": 0,
            "input_tokens": 50,
            "output_tokens": 10,
            "reasoning_tokens": 2,
        },
    ]
    summary = summarize_pairs(trials)
    assert summary["on_regressions"] == 1
    assert summary["extra_context_tokens_per_successful_task"] == 50
    assert summary["total_token_delta_on_minus_off"] == 100
    assert summary["provider_input_token_delta_on_minus_off"] == 100
    assert summary["provider_output_token_delta_on_minus_off"] == 0
    assert summary["pair_costs"] == [
        {
            "task_id": "a",
            "repeat": 0,
            "input_delta": 50,
            "output_delta": 0,
            "reasoning_delta": 4,
            "total_delta": 50,
        },
        {
            "task_id": "b",
            "repeat": 0,
            "input_delta": 50,
            "output_delta": 0,
            "reasoning_delta": 6,
            "total_delta": 50,
        },
    ]
    assert summary["provider_reasoning_token_delta_on_minus_off"] == 10
    trials[0]["reasoning_tokens"] = None
    missing = summarize_pairs(trials)
    assert missing["provider_reasoning_token_delta_on_minus_off"] is None
    assert missing["on_metrics"]["reasoning_tokens"] is None
    assert missing["provider_output_token_delta_on_minus_off"] == 0
    trials[0]["success"] = False
    assert summarize_pairs(trials)["extra_context_tokens_per_successful_task"] is None


def test_summary_rejects_incomplete_pairs() -> None:
    with pytest.raises(ValueError, match="incomplete"):
        summarize_pairs([{"task_id": "a", "repeat": 0, "condition": "on", "success": True}])


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://192.168.evil.example:1234/v1",
        "http://172.1.2.3:1234/v1",
        "http://user:secret@127.0.0.1:1234/v1",
        "http://127.0.0.1:1234/v1?token=secret",
        "https://127.0.0.1:1234/v1",
    ],
)
def test_endpoint_rejects_public_or_secret_bearing_urls(endpoint: str) -> None:
    with pytest.raises(ValueError):
        validate_local_endpoint(endpoint)


def test_manifest_retrieval_must_match_runtime_baseline() -> None:
    config = json.loads((TASK_ROOT / "pilot.json").read_text(encoding="utf-8"))
    assert validate_retrieval_config(config)["rrf_k"] == 60
    config["retrieval"]["rrf_k"] = 100
    with pytest.raises(ValueError, match="pilot retrieval config"):
        validate_retrieval_config(config)


def test_recount_uses_all_on_tool_context_not_only_successes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> object:
            assert not add_special_tokens
            return type("Encoding", (), {"ids": list(text)})()

    monkeypatch.setattr("tests.evals.task_runner.Tokenizer.from_file", lambda path: FakeTokenizer())
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    trials = [
        {
            "task_id": "a",
            "repeat": 0,
            "condition": "on",
            "success": True,
            "tool_events": [{"tool": "memory_search", "results": [{"id": "a"}]}],
            "input_tokens": 10,
            "output_tokens": 5,
        },
        {
            "task_id": "a",
            "repeat": 0,
            "condition": "off",
            "success": False,
            "tool_events": [],
            "input_tokens": 10,
            "output_tokens": 5,
        },
        {
            "task_id": "b",
            "repeat": 0,
            "condition": "on",
            "success": False,
            "tool_events": [{"tool": "memory_search", "results": [{"id": "b"}]}],
            "input_tokens": 10,
            "output_tokens": 5,
        },
        {
            "task_id": "b",
            "repeat": 0,
            "condition": "off",
            "success": True,
            "tool_events": [],
            "input_tokens": 10,
            "output_tokens": 5,
        },
    ]
    result = recount_memory_context(
        {"trials": trials, "limitations": ["No verified model tokenizer configured: x"]},
        tokenizer,
    )
    expected = sum(
        len(json.dumps({"results": trial["tool_events"][0]["results"]}, ensure_ascii=False))
        for trial in trials
        if trial["condition"] == "on"
    )
    assert result["summary"]["extra_context_tokens_per_successful_task"] == expected
    assert result["tokenizer"]["json_sha256"]


def test_nonadherent_on_trial_is_not_counted_as_task_failure() -> None:
    trials = [
        {
            "task_id": "a",
            "repeat": 0,
            "condition": "on",
            "success": True,
            "tool_events": [{"tool": "read_repo_file"}],
        },
        {
            "task_id": "a",
            "repeat": 0,
            "condition": "off",
            "success": True,
            "tool_events": [{"tool": "read_repo_file"}],
        },
    ]
    apply_treatment_adherence(trials)
    apply_treatment_adherence(trials)
    assert trials[0]["success"] is None
    assert trials[0]["task_success_before_adherence"] is True
    assert trials[0]["invalidation_reason"] == "treatment_nonadherence"
    assert trials[1]["success"] is True
    with pytest.raises(ValueError, match="unfinished"):
        summarize_pairs(trials)


def test_recount_rejects_tokenizer_model_mismatch(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "binding.json"
    manifest.write_text(
        json.dumps(
            {
                "model": "different-model",
                "model_version": "different-variant",
                "tokenizer_sha256": "wrong-hash",
                "library_version": "0.23.2",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="tokenizer manifest"):
        recount_memory_context(
            {"model": "google/gemma-4-e2b", "model_version": "@4bit", "trials": []},
            tokenizer,
            tokenizer_manifest=manifest,
        )


@pytest.mark.asyncio
async def test_adapter_exposes_only_approved_difference(tmp_path: Path) -> None:
    case = load_cases(TASK_ROOT / "cases.jsonl")[0]
    source = tmp_path / case.rubric.source_path
    source.parent.mkdir(parents=True)
    source.write_text("current source", encoding="utf-8")
    prompts = {"base": "same", "on_extra": "search memory", "off_extra": ""}
    tool_lists: list[list[str]] = []
    system_prompts: list[str] = []
    search_calls = 0

    class FakeCompletion:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
            self.calls += 1
            if self.calls == 1:
                tool_lists.append([tool["function"]["name"] for tool in tools])
                system_prompts.append(messages[0]["content"])
                calls = [
                    {
                        "id": "read-1",
                        "function": {
                            "name": "read_repo_file",
                            "arguments": json.dumps({"path": case.rubric.source_path}),
                        },
                    }
                ]
                if len(tools) == 2:
                    calls.append(
                        {
                            "id": "memory-1",
                            "function": {
                                "name": "memory_search",
                                "arguments": '{"query":"incident"}',
                            },
                        }
                    )
                message = {"content": None, "tool_calls": calls}
            else:
                message = {
                    "content": json.dumps(
                        {
                            "answer_key": case.rubric.answer_key,
                            "evidence": [case.rubric.source_path],
                            "memory_ids_used": ["memory-1"],
                        }
                    )
                }
            return {
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }

    async def search(query: str) -> list[dict]:
        nonlocal search_calls
        search_calls += 1
        return [{"id": "memory-1", "content": "incident"}]

    shared = {
        "case": case,
        "checkout": tmp_path,
        "prompts": prompts,
        "allowed_paths": {case.rubric.source_path},
        "max_tool_rounds": 2,
        "count_tokens": lambda text: len(text),
    }
    off = await run_trial(condition="off", completion=FakeCompletion(), search=None, **shared)
    on = await run_trial(condition="on", completion=FakeCompletion(), search=search, **shared)
    assert tool_lists == [["read_repo_file"], ["read_repo_file", "memory_search"]]
    assert system_prompts == ["same", "same\nsearch memory"]
    assert search_calls == 1
    assert off["memory_context_tokens"] == 0
    assert on["memory_context_tokens"] > 0
    assert on["results_used"] == ["memory-1"]
    assert on["input_tokens"] == off["input_tokens"] == 20
    assert off["memory_search_calls"] == 0
    assert on["tool_events"][1]["search_latency_ms"] >= 0
    assert on["memory_search_latency_ms"] == on["tool_events"][1]["search_latency_ms"]
    assert on["completion_events"][0]["input_tokens"] == 10
    assert on["completion_events"][0]["reasoning_tokens"] is None
    assert on["reasoning_tokens"] is None
    assert on["completion_events"][0]["request_messages"] == [
        {"role": "system", "content": "same\nsearch memory"},
        {"role": "user", "content": case.prompt},
    ]
    assert [tool["function"]["name"] for tool in on["completion_events"][0]["request_tools"]] == [
        "read_repo_file",
        "memory_search",
    ]
    assert any(
        message["role"] == "tool" and "incident" in message["content"]
        for message in on["completion_events"][1]["request_messages"]
    )
    assert on["post_template_prompt_tokens_exact"] is None
    with pytest.raises(ValueError, match="allowlist"):
        read_repo_file(tmp_path, "tests/retrieval/memory_holdout.jsonl", {case.rubric.source_path})


@pytest.mark.asyncio
async def test_every_memory_search_call_has_own_latency(tmp_path: Path) -> None:
    case = load_cases(TASK_ROOT / "cases.jsonl")[0]

    class TwoSearchCompletion:
        def __init__(self) -> None:
            self.calls = 0

        async def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
            self.calls += 1
            if self.calls == 1:
                message = {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": str(index),
                            "function": {
                                "name": "memory_search",
                                "arguments": json.dumps({"query": str(index)}),
                            },
                        }
                        for index in (1, 2)
                    ],
                }
            else:
                message = {"content": "{}"}
            return {"choices": [{"message": message}], "usage": {}}

    async def search(query: str) -> list[dict]:
        return [{"id": query, "content": query}]

    outcome = await run_trial(
        case=case,
        checkout=tmp_path,
        condition="on",
        completion=TwoSearchCompletion(),
        search=search,
        prompts={"base": "base", "on_extra": "search"},
        allowed_paths=set(),
        max_tool_rounds=1,
    )
    assert outcome["memory_search_calls"] == 2
    assert len([event["search_latency_ms"] for event in outcome["tool_events"]]) == 2
    assert outcome["memory_search_latency_ms"] == round(
        sum(event["search_latency_ms"] for event in outcome["tool_events"]), 3
    )
    assert outcome["input_tokens"] is None
    assert outcome["reasoning_tokens"] is None
