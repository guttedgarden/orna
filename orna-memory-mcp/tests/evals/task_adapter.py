"""Fresh OpenAI-compatible chat trial with shared repo tool and optional Orna search."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from copy import deepcopy
from ipaddress import ip_address, ip_network
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib.parse import urlsplit

import httpx

from tests.evals.graders import TaskCase, parse_final_object

Search = Callable[[str], Awaitable[list[dict[str, Any]]]]
TokenCounter = Callable[[str], int]


class Completion(Protocol):
    async def __call__(self, messages: list[dict], tools: list[dict]) -> dict: ...


class OpenAICompatibleCompletion:
    """Small client for a local Chat Completions endpoint."""

    def __init__(
        self, endpoint: str, model: str, *, temperature: float, max_tokens: int, timeout: int
    ) -> None:
        self._endpoint = validate_local_endpoint(endpoint)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    async def __call__(self, messages: list[dict], tools: list[dict]) -> dict:
        payload = {
            "model": self._model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        async with httpx.AsyncClient(timeout=self._timeout, trust_env=False) as client:
            response = await client.post(f"{self._endpoint}/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()


_LOCAL_NETWORKS = tuple(
    ip_network(cidr)
    for cidr in (
        "127.0.0.0/8",
        "::1/128",
        "10.0.0.0/8",
        "172.16.0.0/12",
        "192.168.0.0/16",
    )
)


def validate_local_endpoint(endpoint: str) -> str:
    """Reject public hosts, credentials and query data before any request or artifact write."""

    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path.rstrip("/") != "/v1"
        or parsed.port is None
    ):
        raise ValueError("pilot endpoint must be a plain local HTTP /v1 URL")
    try:
        host = parsed.hostname or ""
        allowed = host == "localhost" or any(ip_address(host) in net for net in _LOCAL_NETWORKS)
    except ValueError:
        allowed = False
    if not allowed:
        raise ValueError("pilot endpoint must be on loopback or an RFC1918 LAN address")
    return endpoint.rstrip("/")


REPO_TOOL = {
    "type": "function",
    "function": {
        "name": "read_repo_file",
        "description": "Read one allowed source file from the pinned fresh checkout.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    },
}
MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_search",
        "description": "Search read-only Orna pilot memory for relevant project context.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}


def summarize_tool_events(events: list[dict], final_artifact: str) -> dict:
    """Results used are only explicit client declarations intersected with delivered IDs."""

    delivered = {
        str(result["id"])
        for event in events
        if event.get("tool") == "memory_search"
        for result in event.get("results", [])
        if isinstance(result, dict) and "id" in result
    }
    declared = parse_final_object(final_artifact).get("memory_ids_used")
    used = (
        sorted({item for item in declared if isinstance(item, str)} & delivered)
        if isinstance(declared, list)
        else None
    )
    return {
        "tool_calls": len(events),
        "memory_search_calls": sum(event.get("tool") == "memory_search" for event in events),
        "results_returned": sum(
            len(event.get("results", []))
            for event in events
            if event.get("tool") == "memory_search"
        ),
        "results_used": used,
    }


def read_repo_file(checkout: Path, path: str, allowed_paths: set[str]) -> str:
    """Allow only case-reviewed source paths; never expose hidden holdout or secrets."""

    if path not in allowed_paths:
        raise ValueError("file is outside pilot source allowlist")
    target = (checkout / path).resolve()
    if not target.is_relative_to(checkout.resolve()) or not target.is_file():
        raise ValueError("file is unavailable in pinned checkout")
    if target.stat().st_size > 100_000:
        raise ValueError("file exceeds pilot read budget")
    return target.read_text(encoding="utf-8")


async def run_trial(
    *,
    case: TaskCase,
    checkout: Path,
    condition: str,
    completion: Completion,
    search: Search | None,
    prompts: dict[str, str],
    allowed_paths: set[str],
    max_tool_rounds: int,
    count_tokens: TokenCounter | None = None,
) -> dict:
    """Run one isolated chat; no messages or hidden state survive the call."""

    if condition not in ("on", "off") or (condition == "on") != (search is not None):
        raise ValueError("condition and Orna access disagree")
    suffix = prompts["on_extra"] if condition == "on" else prompts["off_extra"]
    messages: list[dict] = [
        {"role": "system", "content": prompts["base"] + ("\n" + suffix if suffix else "")},
        {"role": "user", "content": case.prompt},
    ]
    tools = [REPO_TOOL, MEMORY_TOOL] if condition == "on" else [REPO_TOOL]
    events: list[dict] = []
    completion_events: list[dict] = []
    input_tokens: int | None = 0
    output_tokens: int | None = 0
    memory_context_tokens: int | None = 0 if count_tokens else None
    memory_context_bytes = 0
    started = perf_counter()
    final = ""
    for _ in range(max_tool_rounds + 1):
        request_messages = deepcopy(messages)
        request_tools = deepcopy(tools)
        response = await completion(messages, tools)
        usage = response.get("usage") or {}
        for key, current in (("prompt_tokens", input_tokens), ("completion_tokens", output_tokens)):
            amount = usage.get(key)
            if key == "prompt_tokens":
                input_tokens = (
                    current + amount if current is not None and isinstance(amount, int) else None
                )
            else:
                output_tokens = (
                    current + amount if current is not None and isinstance(amount, int) else None
                )
        choice = response["choices"][0]
        message = choice["message"]
        calls = message.get("tool_calls") or []
        completion_events.append(
            {
                "finish_reason": choice.get("finish_reason"),
                "tool_calls": len(calls),
                "content_chars": len(message.get("content") or ""),
                "input_tokens": usage.get("prompt_tokens"),
                "output_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get(
                    "reasoning_tokens"
                ),
                "request_messages": request_messages,
                "request_tools": request_tools,
            }
        )
        if not calls:
            final = message.get("content") or ""
            break
        messages.append(
            {"role": "assistant", "content": message.get("content"), "tool_calls": calls}
        )
        for call in calls:
            name = call["function"]["name"]
            search_started: float | None = None
            try:
                args = json.loads(call["function"]["arguments"])
                if name == "read_repo_file":
                    path = args["path"]
                    content = read_repo_file(checkout, path, allowed_paths)
                    event = {"tool": name, "path": path}
                elif name == "memory_search" and search is not None:
                    search_started = perf_counter()
                    results = await search(args["query"])
                    search_latency_ms = round((perf_counter() - search_started) * 1000, 3)
                    content = json.dumps({"results": results}, ensure_ascii=False)
                    event = {
                        "tool": name,
                        "query": args["query"],
                        "results": results,
                        "search_latency_ms": search_latency_ms,
                    }
                    memory_context_bytes += len(content.encode("utf-8"))
                    if memory_context_tokens is not None:
                        memory_context_tokens += count_tokens(content)  # type: ignore[misc]
                else:
                    raise ValueError("tool unavailable")
            except (KeyError, TypeError, ValueError) as error:
                content = json.dumps({"error": type(error).__name__})
                event = {"tool": name, "error": type(error).__name__}
                if search_started is not None:
                    event["search_latency_ms"] = round((perf_counter() - search_started) * 1000, 3)
            events.append(event)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        if prompts.get("after_tool"):
            messages.append({"role": "user", "content": prompts["after_tool"]})
    summary = summarize_tool_events(events, final)
    reasoning_values = [event["reasoning_tokens"] for event in completion_events]
    return {
        "final_artifact": final,
        "tool_events": events,
        "completion_events": completion_events,
        "latency_ms": round((perf_counter() - started) * 1000, 3),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": (
            sum(reasoning_values)
            if all(isinstance(value, int) for value in reasoning_values)
            else None
        ),
        "memory_context_tokens": memory_context_tokens,
        "memory_context_bytes": memory_context_bytes,
        "memory_search_latency_ms": round(
            sum(event["search_latency_ms"] for event in events if "search_latency_ms" in event),
            3,
        ),
        "post_template_prompt_tokens_exact": None,
        **summary,
    }
