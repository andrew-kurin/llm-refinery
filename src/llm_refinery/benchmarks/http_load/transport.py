from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Any

from llm_refinery.benchmarks.http_load.config import HttpLoadTrial, HttpScenario, HttpTarget
from llm_refinery.benchmarks.http_load.models import RequestResult
from llm_refinery.providers.openai import chat_completions_url, json_headers, openai_choice_text


def run_requests(trial: HttpLoadTrial, *, count: int) -> list[RequestResult]:
    results: list[RequestResult] = []
    with ThreadPoolExecutor(max_workers=trial.concurrency) as executor:
        futures = {
            executor.submit(execute_http_request, trial, request_index): request_index
            for request_index in range(count)
        }
        for future in as_completed(futures):
            request_index = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:  # noqa: BLE001 - convert worker bugs into stored errors
                results.append(
                    RequestResult(
                        index=request_index,
                        ok=False,
                        status_code=None,
                        latency_s=0.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return sorted(results, key=lambda result: result.index)


def execute_http_request(trial: HttpLoadTrial, index: int) -> RequestResult:
    if trial.target.provider in ("openai", "cerebras"):
        return execute_openai_request(trial, index)
    if trial.target.provider == "ollama":
        return execute_ollama_request(trial, index)
    raise ValueError(f"unsupported provider: {trial.target.provider}")


def execute_openai_request(trial: HttpLoadTrial, index: int) -> RequestResult:
    scenario = trial.scenario
    payload: dict[str, Any] = {
        "model": trial.target.model,
        "messages": messages_for_scenario(scenario),
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
        url=chat_completions_url(trial.target.base_url),
        payload=payload,
        stream_reader=read_openai_stream if scenario.stream else None,
        body_reader=read_openai_body,
    )


def execute_ollama_request(trial: HttpLoadTrial, index: int) -> RequestResult:
    scenario = trial.scenario
    options: dict[str, Any] = {
        "num_predict": trial.max_tokens,
        "temperature": scenario.temperature,
    }
    if scenario.seed is not None:
        options["seed"] = scenario.seed

    payload = {
        "model": trial.target.model,
        "messages": messages_for_scenario(scenario),
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
    ttft_s: float | None = None
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
        for choice in chunk.get("choices") or []:
            content = openai_choice_text(choice)
            if content:
                if ttft_s is None:
                    ttft_s = time.perf_counter() - start
                text_parts.append(content)

    response_text = "".join(text_parts)
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        ttft_s=ttft_s,
        prompt_tokens=int_from_mapping(usage, "prompt_tokens"),
        completion_tokens=int_from_mapping(usage, "completion_tokens"),
        completion_chars=len(response_text),
        response_text=response_text,
    )


def read_openai_body(index: int, body: str, start: float, status_code: int) -> RequestResult:
    payload = json.loads(body)
    text_parts: list[str] = []
    for choice in payload.get("choices") or []:
        content = openai_choice_text(choice)
        if content:
            text_parts.append(content)
    response_text = "".join(text_parts)
    usage = payload.get("usage") or {}
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        prompt_tokens=int_from_mapping(usage, "prompt_tokens"),
        completion_tokens=int_from_mapping(usage, "completion_tokens"),
        completion_chars=len(response_text),
        response_text=response_text,
    )


def read_ollama_stream(
    index: int,
    response: Any,
    start: float,
    status_code: int,
) -> RequestResult:
    text_parts: list[str] = []
    ttft_s: float | None = None
    final: dict[str, Any] = {}
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        chunk = json.loads(line)
        if chunk.get("error"):
            raise RuntimeError(str(chunk["error"]))
        message = chunk.get("message") or {}
        content = message.get("content") or message.get("thinking") or chunk.get("response") or ""
        if content:
            if ttft_s is None:
                ttft_s = time.perf_counter() - start
            text_parts.append(str(content))
        if chunk.get("done"):
            final = chunk

    response_text = "".join(text_parts)
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        ttft_s=ttft_s,
        prompt_tokens=int_from_mapping(final, "prompt_eval_count"),
        completion_tokens=int_from_mapping(final, "eval_count"),
        completion_chars=len(response_text),
        server_prompt_eval_duration_s=ns_to_s(final.get("prompt_eval_duration")),
        server_eval_duration_s=ns_to_s(final.get("eval_duration")),
        response_text=response_text,
    )


def read_ollama_body(index: int, body: str, start: float, status_code: int) -> RequestResult:
    payload = json.loads(body)
    message = payload.get("message") or {}
    response_text = str(
        message.get("content") or message.get("thinking") or payload.get("response") or ""
    )
    return RequestResult(
        index=index,
        ok=200 <= status_code < 300,
        status_code=status_code,
        latency_s=time.perf_counter() - start,
        prompt_tokens=int_from_mapping(payload, "prompt_eval_count"),
        completion_tokens=int_from_mapping(payload, "eval_count"),
        completion_chars=len(response_text),
        server_prompt_eval_duration_s=ns_to_s(payload.get("prompt_eval_duration")),
        server_eval_duration_s=ns_to_s(payload.get("eval_duration")),
        response_text=response_text,
    )


def with_check_result(result: RequestResult, scenario: HttpScenario) -> RequestResult:
    if not result.ok or not scenario.expected_contains:
        return result
    response_text = result.response_text.lower()
    check_passed = all(fragment.lower() in response_text for fragment in scenario.expected_contains)
    return replace(result, check_passed=check_passed)


def messages_for_scenario(scenario: HttpScenario) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if scenario.system:
        messages.append({"role": "system", "content": scenario.system})
    messages.append({"role": "user", "content": scenario.rendered_prompt})
    return messages


def headers_for_target(target: HttpTarget) -> dict[str, str]:
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
