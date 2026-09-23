"""Fixed deterministic rubric and paired metrics for task-level trials."""

from __future__ import annotations

import json
import re
from statistics import median
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Rubric(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    answer_key: str
    source_path: str
    forbidden_terms: tuple[str, ...] = ()


class TaskCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    kind: Literal["known_incident", "convention", "source_memory_contradiction", "no_useful_memory"]
    repo_ref: str
    prompt: str
    corpus_version: str
    rubric: Rubric


class Grade(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    success: bool
    checks: dict[str, bool]


def parse_final_object(final_artifact: str) -> dict:
    """Allow a whole JSON object or one JSON code fence, with no prose."""

    payload = final_artifact.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", payload, flags=re.DOTALL)
    if fenced:
        payload = fenced.group(1)
    try:
        response = json.loads(payload)
    except (TypeError, ValueError):
        return {}
    return response if isinstance(response, dict) else {}


def grade_answer(case: TaskCase, final_artifact: str) -> Grade:
    """Require a parsable answer, exact expected key and source citation."""

    response = parse_final_object(final_artifact)
    evidence = response.get("evidence", [])
    if not isinstance(evidence, list):
        evidence = []
    answer_key = response.get("answer_key")
    checks = {
        "answer_key": (
            isinstance(answer_key, (str, int))
            and not isinstance(answer_key, bool)
            and str(answer_key) == case.rubric.answer_key
        ),
        "source_evidence": case.rubric.source_path in evidence,
        "forbidden_absent": not any(
            term.casefold() in final_artifact.casefold() for term in case.rubric.forbidden_terms
        ),
    }
    return Grade(success=all(checks.values()), checks=checks)


def summarize_pairs(trials: list[dict]) -> dict:
    """Aggregate only complete pairs; preserve missing token scopes as undefined."""

    pairs: dict[tuple[str, int], dict[str, dict]] = {}
    for trial in trials:
        key = (trial["task_id"], trial["repeat"])
        conditions = pairs.setdefault(key, {})
        if trial["condition"] in conditions:
            raise ValueError("duplicate condition in task pair")
        conditions[trial["condition"]] = trial
    if any(set(pair) != {"on", "off"} for pair in pairs.values()):
        raise ValueError("incomplete task pair")
    if any(not isinstance(trial.get("success"), bool) for trial in trials):
        raise ValueError("unfinished trial is not a task success or failure")

    on = [pair["on"] for pair in pairs.values()]
    off = [pair["off"] for pair in pairs.values()]
    on_successes = sum(bool(trial["success"]) for trial in on)
    off_successes = sum(bool(trial["success"]) for trial in off)
    context_values = [trial.get("memory_context_tokens") for trial in on]
    context_cost = (
        sum(context_values) / on_successes
        if on_successes and all(value is not None for value in context_values)
        else None
    )
    token_values = [trial.get("input_tokens") for trial in trials] + [
        trial.get("output_tokens") for trial in trials
    ]
    token_delta = (
        sum(trial["input_tokens"] + trial["output_tokens"] for trial in on)
        - sum(trial["input_tokens"] + trial["output_tokens"] for trial in off)
        if all(value is not None for value in token_values)
        else None
    )

    def condition_metrics(items: list[dict]) -> dict:
        return {
            "tool_calls": sum(item.get("tool_calls", 0) for item in items),
            "memory_search_calls": sum(item.get("memory_search_calls", 0) for item in items),
            "results_returned": sum(item.get("results_returned", 0) for item in items),
            "retrieved_memory_tokens": (
                sum(item["retrieved_memory_tokens"] for item in items)
                if all(item.get("retrieved_memory_tokens") is not None for item in items)
                else None
            ),
            "memory_context_tokens": (
                sum(item["memory_context_tokens"] for item in items)
                if all(item.get("memory_context_tokens") is not None for item in items)
                else None
            ),
            "results_used_declared": (
                sum(len(item["results_used"]) for item in items)
                if all(isinstance(item.get("results_used"), list) for item in items)
                else None
            ),
            "latency_median_ms": (
                median(item["latency_ms"] for item in items)
                if all(item.get("latency_ms") is not None for item in items)
                else None
            ),
            "input_tokens": (
                sum(item["input_tokens"] for item in items)
                if all(item.get("input_tokens") is not None for item in items)
                else None
            ),
            "output_tokens": (
                sum(item["output_tokens"] for item in items)
                if all(item.get("output_tokens") is not None for item in items)
                else None
            ),
        }

    return {
        "pairs": len(pairs),
        "on_successes": on_successes,
        "off_successes": off_successes,
        "both_success": sum(a["success"] and b["success"] for a, b in zip(on, off, strict=True)),
        "on_only_success": sum(
            a["success"] and not b["success"] for a, b in zip(on, off, strict=True)
        ),
        "off_only_success": sum(
            not a["success"] and b["success"] for a, b in zip(on, off, strict=True)
        ),
        "both_failure": sum(
            not a["success"] and not b["success"] for a, b in zip(on, off, strict=True)
        ),
        "on_regressions": sum(
            not a["success"] and b["success"] for a, b in zip(on, off, strict=True)
        ),
        "extra_context_tokens_per_successful_task": context_cost,
        "total_token_delta_on_minus_off": token_delta,
        "on_metrics": condition_metrics(on),
        "off_metrics": condition_metrics(off),
    }
