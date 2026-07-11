from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from llm_refinery.benchmarks.http_load.config import HttpLoadTrial, HttpScenario
from llm_refinery.benchmarks.http_load.models import RequestResult
from llm_refinery.core.endpoints import OLLAMA_CHAT, OPENAI_CHAT, Endpoint
from llm_refinery.providers.openai_chat import json_headers


def run_requests(
    trial: HttpLoadTrial,
    *,
    count: int,
    index_offset: int = 0,
    request_nonce: str | None = None,
) -> list[RequestResult]:
    """Run one request batch.

    A fresh nonce makes ``cache_mode=unique`` cold across repeated benchmark runs. Index offsets
    keep warmup prompts separate from measured prompts without changing stored sample IDs.
    """
    results: list[RequestResult] = []
    nonce = request_nonce or uuid.uuid4().hex
    with ThreadPoolExecutor(max_workers=trial.concurrency) as executor:
        futures: dict[Any, tuple[int, float]] = {}
        for request_index in range(index_offset, index_offset + count):
            submitted_at = time.perf_counter()
            future = executor.submit(
                execute_http_request,
                trial,
                request_index,
                request_nonce=nonce,
            )
            futures[future] = (request_index, submitted_at)
        for future in as_completed(futures):
            request_index, submitted_at = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - convert worker bugs into stored errors
                results.append(
                    RequestResult(
                        index=request_index,
                        ok=False,
                        status_code=None,
                        latency_s=time.perf_counter() - submitted_at,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return sorted(results, key=lambda result: result.index)


def execute_http_request(
    trial: HttpLoadTrial, index: int, *, request_nonce: str | None = None
) -> RequestResult:
    executors = {
        OPENAI_CHAT: execute_openai_request,
        OLLAMA_CHAT: execute_ollama_request,
    }
    executor = executors.get(trial.target.protocol)
    if executor is None:
        raise ValueError(f"unsupported chat protocol: {trial.target.protocol}")
    return executor(trial, index, request_nonce=request_nonce)


def execute_openai_request(
    trial: HttpLoadTrial, index: int, *, request_nonce: str | None = None
) -> RequestResult:
    scenario = trial.scenario
    payload: dict[str, Any] = {
        "model": trial.target.model,
        "messages": messages_for_scenario(scenario, index=index, request_nonce=request_nonce),
        "max_tokens": trial.max_tokens,
        "temperature": scenario.temperature,
        "stream": scenario.stream,
    }
    if scenario.seed is not None:
        payload["seed"] = scenario.seed
    if scenario.stream:
        payload["stream_options"] = {"include_usage": True}

    return post_json(
        trial,
        index,
        url=trial.target.chat_completions_url,
        payload=payload,
        stream_reader=read_openai_stream if scenario.stream else None,
        body_reader=read_openai_body,
    )


def execute_ollama_request(
    trial: HttpLoadTrial, index: int, *, request_nonce: str | None = None
) -> RequestResult:
    scenario = trial.scenario
    options: dict[str, Any] = {
        "num_predict": trial.max_tokens,
        "temperature": scenario.temperature,
    }
    if scenario.seed is not None:
        options["seed"] = scenario.seed

    payload = {
        "model": trial.target.model,
        "messages": messages_for_scenario(scenario, index=index, request_nonce=request_nonce),
        "stream": scenario.stream,
        "options": options,
    }
    return post_json(
        trial,
        index,
        url=f"{trial.target.base_url}/api/chat",
        payload=payload,
        stream_reader=read_ollama_stream if scenario.stream else None,
        body_reader=read_ollama_body,
    )


def post_json(
    trial: HttpLoadTrial,
    index: int,
    *,
    url: str,
    payload: dict[str, Any],
    stream_reader: Any,
    body_reader: Any,
) -> RequestResult:
    start = time.perf_counter()
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers_for_target(trial.target),
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - user-configured local/server URL
            request,
            timeout=trial.scenario.timeout_s,
        ) as response:
            status_code = response.status
            if stream_reader is not None:
                result = stream_reader(index, response, start, status_code)
            else:
                body = response.read().decode("utf-8", errors="replace")
                result = body_reader(index, body, start, status_code)
            return with_check_result(result, trial.scenario)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-1000:]
        return RequestResult(
            index=index,
            ok=False,
            status_code=exc.code,
            latency_s=time.perf_counter() - start,
            error=f"HTTP {exc.code}: {body}",
        )
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return RequestResult(
            index=index,
            ok=False,
            status_code=None,
            latency_s=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def read_openai_stream(
    index: int,
    response: Any,
    start: float,
    status_code: int,
) -> RequestResult:
    text_parts: list[str] = []
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    output_event_times_s: list[float] = []
    ttft_s: float | None = None
    visible_ttft_s: float | None = None
    reasoning_ttft_s: float | None = None
    usage: dict[str, Any] | None = None
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if data == "[DONE]":
            continue
        chunk = json.loads(data)
        usage = chunk.get("usage") or usage
        event_time_s: float | None = None
        for choice in chunk.get("choices") or []:
            visible, reasoning = _openai_choice_parts(choice)
            if visible or reasoning:
                if event_time_s is None:
                    event_time_s = time.perf_counter() - start
                if ttft_s is None:
                    ttft_s = event_time_s
            if reasoning:
                if reasoning_ttft_s is None:
                    reasoning_ttft_s = event_time_s
                reasoning_parts.append(reasoning)
                text_parts.append(reasoning)
            if visible:
                if visible_ttft_s is None:
                    visible_ttft_s = event_time_s
                visible_parts.append(visible)
                text_parts.append(visible)
        if event_time_s is not None:
            output_event_times_s.append(event_time_s)

    response_text = "".join(text_parts)
    visible_response_text = "".join(visible_parts)
    reasoning_response_text = "".join(reasoning_parts)
    latency_s = time.perf_counter() - start
    completion_tokens = int_from_mapping(usage, "completion_tokens")
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=latency_s,
        ttft_s=ttft_s,
        visible_ttft_s=visible_ttft_s,
        reasoning_ttft_s=reasoning_ttft_s,
        tpot_s=_tpot_s(latency_s, ttft_s, completion_tokens),
        itl_s=_interarrival_times(output_event_times_s),
        prompt_tokens=int_from_mapping(usage, "prompt_tokens"),
        completion_tokens=completion_tokens,
        completion_chars=len(response_text),
        visible_completion_chars=len(visible_response_text),
        reasoning_completion_chars=len(reasoning_response_text),
        response_text=response_text,
        visible_response_text=visible_response_text,
        reasoning_response_text=reasoning_response_text,
    )


def read_openai_body(index: int, body: str, start: float, status_code: int) -> RequestResult:
    payload = json.loads(body)
    text_parts: list[str] = []
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    for choice in payload.get("choices") or []:
        visible, reasoning = _openai_choice_parts(choice)
        if reasoning:
            reasoning_parts.append(reasoning)
            text_parts.append(reasoning)
        if visible:
            visible_parts.append(visible)
            text_parts.append(visible)
    response_text = "".join(text_parts)
    visible_response_text = "".join(visible_parts)
    reasoning_response_text = "".join(reasoning_parts)
    usage = payload.get("usage") or {}
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        prompt_tokens=int_from_mapping(usage, "prompt_tokens"),
        completion_tokens=int_from_mapping(usage, "completion_tokens"),
        completion_chars=len(response_text),
        visible_completion_chars=len(visible_response_text),
        reasoning_completion_chars=len(reasoning_response_text),
        response_text=response_text,
        visible_response_text=visible_response_text,
        reasoning_response_text=reasoning_response_text,
    )


def read_ollama_stream(
    index: int,
    response: Any,
    start: float,
    status_code: int,
) -> RequestResult:
    text_parts: list[str] = []
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    output_event_times_s: list[float] = []
    ttft_s: float | None = None
    visible_ttft_s: float | None = None
    reasoning_ttft_s: float | None = None
    final: dict[str, Any] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        chunk = json.loads(line)
        if chunk.get("error"):
            raise RuntimeError(str(chunk["error"]))
        message = chunk.get("message") or {}
        reasoning = str(message.get("thinking") or "")
        visible = str(message.get("content") or chunk.get("response") or "")
        if visible or reasoning:
            event_time_s = time.perf_counter() - start
            if ttft_s is None:
                ttft_s = event_time_s
            output_event_times_s.append(event_time_s)
        else:
            event_time_s = None
        if reasoning:
            if reasoning_ttft_s is None:
                reasoning_ttft_s = event_time_s
            reasoning_parts.append(reasoning)
            text_parts.append(reasoning)
        if visible:
            if visible_ttft_s is None:
                visible_ttft_s = event_time_s
            visible_parts.append(visible)
            text_parts.append(visible)
        if chunk.get("done"):
            final = chunk

    response_text = "".join(text_parts)
    visible_response_text = "".join(visible_parts)
    reasoning_response_text = "".join(reasoning_parts)
    latency_s = time.perf_counter() - start
    completion_tokens = int_from_mapping(final, "eval_count")
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=latency_s,
        ttft_s=ttft_s,
        visible_ttft_s=visible_ttft_s,
        reasoning_ttft_s=reasoning_ttft_s,
        tpot_s=_tpot_s(latency_s, ttft_s, completion_tokens),
        itl_s=_interarrival_times(output_event_times_s),
        prompt_tokens=int_from_mapping(final, "prompt_eval_count"),
        completion_tokens=completion_tokens,
        completion_chars=len(response_text),
        visible_completion_chars=len(visible_response_text),
        reasoning_completion_chars=len(reasoning_response_text),
        server_prompt_eval_duration_s=ns_to_s(final.get("prompt_eval_duration")),
        server_eval_duration_s=ns_to_s(final.get("eval_duration")),
        response_text=response_text,
        visible_response_text=visible_response_text,
        reasoning_response_text=reasoning_response_text,
    )


def read_ollama_body(index: int, body: str, start: float, status_code: int) -> RequestResult:
    payload = json.loads(body)
    message = payload.get("message") or {}
    visible_response_text = str(message.get("content") or payload.get("response") or "")
    reasoning_response_text = str(message.get("thinking") or "")
    response_text = reasoning_response_text + visible_response_text
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        prompt_tokens=int_from_mapping(payload, "prompt_eval_count"),
        completion_tokens=int_from_mapping(payload, "eval_count"),
        completion_chars=len(response_text),
        visible_completion_chars=len(visible_response_text),
        reasoning_completion_chars=len(reasoning_response_text),
        server_prompt_eval_duration_s=ns_to_s(payload.get("prompt_eval_duration")),
        server_eval_duration_s=ns_to_s(payload.get("eval_duration")),
        response_text=response_text,
        visible_response_text=visible_response_text,
        reasoning_response_text=reasoning_response_text,
    )


def with_check_result(result: RequestResult, scenario: HttpScenario) -> RequestResult:
    if not result.ok:
        return result

    # Old artifacts/readers did not split channels, so fall back to their combined text. New
    # readers deliberately require visible content: reasoning-only output is not a usable answer.
    visible_text = (
        result.response_text
        if result.visible_response_text is None
        else result.visible_response_text
    )
    if not visible_text.strip():
        return replace(
            result,
            ok=False,
            check_passed=False if scenario.expected_contains else None,
            error="empty visible response",
        )

    if not scenario.expected_contains:
        return result
    normalized = visible_text.lower()
    missing = [
        fragment for fragment in scenario.expected_contains if fragment.lower() not in normalized
    ]
    if missing:
        return replace(
            result,
            ok=False,
            check_passed=False,
            error=f"correctness check failed; missing: {', '.join(missing)}",
        )
    return replace(result, check_passed=True)


def messages_for_scenario(
    scenario: HttpScenario,
    *,
    index: int = 0,
    request_nonce: str | None = None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if scenario.system:
        messages.append({"role": "system", "content": scenario.system})
    messages.append(
        {
            "role": "user",
            "content": scenario.rendered_prompt_for(index, request_nonce=request_nonce),
        }
    )
    return messages


def _openai_choice_parts(choice: dict[str, Any]) -> tuple[str, str]:
    visible_parts: list[str] = []
    reasoning_parts: list[str] = []
    for mapping in (choice.get("delta"), choice.get("message"), choice):
        if not isinstance(mapping, dict):
            continue
        for key in ("reasoning_content", "thinking"):
            if mapping.get(key):
                reasoning_parts.append(str(mapping[key]))
        for key in ("content", "text"):
            if mapping.get(key):
                visible_parts.append(str(mapping[key]))
    return "".join(visible_parts), "".join(reasoning_parts)


def _interarrival_times(event_times_s: list[float]) -> tuple[float, ...]:
    return tuple(
        later - earlier for earlier, later in zip(event_times_s, event_times_s[1:], strict=False)
    )


def _tpot_s(latency_s: float, ttft_s: float | None, completion_tokens: int | None) -> float | None:
    if ttft_s is None or completion_tokens is None or completion_tokens <= 1:
        return None
    return max(0.0, latency_s - ttft_s) / (completion_tokens - 1)


def headers_for_target(target: Endpoint) -> dict[str, str]:
    return json_headers(target.headers, api_key_env=target.api_key_env)


def int_from_mapping(mapping: dict[str, Any] | None, key: str) -> int | None:
    if not mapping:
        return None
    value = mapping.get(key)
    if value is None:
        return None
    return int(value)


def ns_to_s(value: Any) -> float | None:
    if value is None:
        return None
    return float(value) / 1_000_000_000
