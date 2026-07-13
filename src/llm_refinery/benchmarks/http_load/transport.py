from __future__ import annotations

import heapq
import json
import ssl
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import AbstractContextManager, contextmanager, nullcontext, suppress
from dataclasses import replace
from queue import LifoQueue
from typing import Any
from urllib.parse import urljoin

import httpx

from llm_refinery.benchmarks.http_load.config import HttpLoadTrial, HttpScenario
from llm_refinery.benchmarks.http_load.models import RequestResult
from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import OLLAMA_CHAT, OPENAI_CHAT, Endpoint
from llm_refinery.core.http_safety import (
    HttpOrigin,
    PinnedHttpRoute,
    http_origin,
    pinned_route_trust_env,
    resolve_request_route,
    validate_request_url,
)
from llm_refinery.providers.openai_chat import json_headers

_MAX_REDIRECTS = 5
_MAX_ERROR_RESPONSE_BYTES = 1_000_000
_MAX_SUCCESS_RESPONSE_BYTES = 64_000_000
_REDACTED = "[REDACTED]"
_ROUTE_LOCK = threading.Lock()
_ROUTE_UNSET = object()


class _DeadlineWatchdog:
    """Interrupt all response-body deadlines from one shared daemon thread."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._deadlines: list[tuple[float, int]] = []
        self._callbacks: dict[int, Callable[[], None]] = {}
        self._next_token = 0
        self._thread: threading.Thread | None = None

    def schedule(self, deadline: float, callback: Callable[[], None]) -> int:
        with self._condition:
            self._next_token += 1
            token = self._next_token
            self._callbacks[token] = callback
            heapq.heappush(self._deadlines, (deadline, token))
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._run,
                    name="llm-refinery-http-deadlines",
                    daemon=True,
                )
                self._thread.start()
            self._condition.notify()
            return token

    def cancel(self, token: int) -> None:
        with self._condition:
            self._callbacks.pop(token, None)
            self._condition.notify()

    def _run(self) -> None:
        while True:
            callback: Callable[[], None] | None = None
            with self._condition:
                while callback is None:
                    while self._deadlines and self._deadlines[0][1] not in self._callbacks:
                        heapq.heappop(self._deadlines)
                    if not self._deadlines:
                        self._condition.wait()
                        continue
                    deadline, token = self._deadlines[0]
                    remaining = deadline - time.perf_counter()
                    if remaining > 0:
                        self._condition.wait(timeout=min(remaining, threading.TIMEOUT_MAX))
                        continue
                    heapq.heappop(self._deadlines)
                    callback = self._callbacks.pop(token, None)
            if callback is not None:
                with suppress(Exception):  # deadline cleanup is best effort
                    callback()


_DEADLINE_WATCHDOG = _DeadlineWatchdog()


class HttpClientPool:
    """Lease one persistent client per concurrency slot for request-local cancellation."""

    def __init__(self, trial: HttpLoadTrial) -> None:
        self._trial = trial
        self._route = _trial_route(trial)
        self._clients = [
            _new_http_client(trial, route=self._route) for _ in range(trial.concurrency)
        ]
        self._available: LifoQueue[httpx.Client] = LifoQueue()
        self._clients_lock = threading.Lock()
        for client in self._clients:
            self._available.put(client)
        self.is_closed = False

    @contextmanager
    def lease(self) -> Iterator[httpx.Client]:
        client = self._available.get()
        try:
            yield client
        finally:
            if client.is_closed and not self.is_closed:
                replacement = _new_http_client(self._trial, route=self._route)
                with self._clients_lock:
                    for index, owned_client in enumerate(self._clients):
                        if owned_client is client:
                            self._clients[index] = replacement
                            break
                    else:  # pragma: no cover - invariant guard
                        replacement.close()
                        raise RuntimeError("leased HTTP client is not owned by its pool")
                client = replacement
            self._available.put(client)

    def close(self) -> None:
        if self.is_closed:
            return
        self.is_closed = True
        for client in self._clients:
            client.close()


def run_requests(
    trial: HttpLoadTrial,
    *,
    count: int,
    index_offset: int = 0,
    request_nonce: str | None = None,
    client: httpx.Client | HttpClientPool | None = None,
) -> list[RequestResult]:
    """Run one request batch.

    A fresh nonce makes ``cache_mode=unique`` cold across repeated benchmark runs. Index offsets
    keep warmup prompts separate from measured prompts without changing stored sample IDs.
    """
    # Validate configured credentials once before creating workers. Otherwise a
    # missing key becomes one identical failed result per warmup/measured request.
    headers_for_target(trial.target)
    sensitive_values = _sensitive_header_values(
        trial.target,
        headers_for_target(trial.target),
    )
    results: list[RequestResult] = []
    nonce = request_nonce or uuid.uuid4().hex
    client_context: AbstractContextManager[httpx.Client | HttpClientPool]
    client_context = nullcontext(client) if client is not None else pooled_http_client(trial)
    if isinstance(client, httpx.Client):
        _prepare_client_route(client, trial.target.base_url, require_resolution=False)
    with (
        client_context as active_client,
        ThreadPoolExecutor(max_workers=trial.concurrency) as executor,
    ):
        # Hold the first worker wave until every slot has leased a distinct
        # persistent client. In particular, this guarantees the runner's
        # concurrency-sized warmup actually warms every measured connection.
        first_wave = (
            threading.Barrier(trial.concurrency)
            if isinstance(active_client, HttpClientPool)
            and trial.concurrency > 1
            and count >= trial.concurrency
            else None
        )
        futures: dict[Any, tuple[int, float]] = {}
        for batch_index, request_index in enumerate(range(index_offset, index_offset + count)):
            submitted_at = time.perf_counter()
            future = executor.submit(
                _execute_with_client,
                trial,
                request_index,
                request_nonce=nonce,
                client=active_client,
                start_barrier=first_wave if batch_index < trial.concurrency else None,
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
                        error=_redact_text(
                            f"{type(exc).__name__}: {exc}",
                            sensitive_values,
                        ),
                    )
                )
    return sorted(results, key=lambda result: result.index)


@contextmanager
def pooled_http_client(trial: HttpLoadTrial) -> Iterator[HttpClientPool]:
    """Own one persistent, independently cancellable client per worker slot."""
    client = HttpClientPool(trial)
    try:
        yield client
    finally:
        client.close()


@contextmanager
def _owned_http_client(trial: HttpLoadTrial) -> Iterator[httpx.Client]:
    client = _new_http_client(trial)
    try:
        yield client
    finally:
        client.close()


def _trial_route(trial: HttpLoadTrial) -> PinnedHttpRoute | None:
    return trial.transport.pinned_route or resolve_request_route(
        trial.target.base_url,
        require_resolution=True,
    )


def _new_http_client(
    trial: HttpLoadTrial,
    *,
    route: PinnedHttpRoute | None | object = _ROUTE_UNSET,
) -> httpx.Client:
    if route is _ROUTE_UNSET:
        route = _trial_route(trial)
    assert route is None or isinstance(route, PinnedHttpRoute)
    if route is not None:
        route.request_url(trial.target.base_url)
    validated_origin = http_origin(trial.target.base_url)
    client_trust_env = pinned_route_trust_env(
        trial.target.base_url,
        trust_env=trial.transport.trust_env,
        route_is_pinned=route is not None,
    )
    verify: bool | ssl.SSLContext = True
    if trial.transport.ca_bundle is not None:
        verify = ssl.create_default_context(cafile=str(trial.transport.ca_bundle))
    elif trial.transport.trust_env and not client_trust_env:
        verify = httpx.create_ssl_context(verify=True, trust_env=True)
    client = httpx.Client(
        limits=httpx.Limits(
            max_connections=1,
            max_keepalive_connections=1,
            keepalive_expiry=None,
        ),
        follow_redirects=False,
        trust_env=client_trust_env,
        verify=verify,
    )
    client._llm_refinery_routes = {validated_origin: route}  # type: ignore[attr-defined]
    return client


def _execute_with_client(
    trial: HttpLoadTrial,
    index: int,
    *,
    request_nonce: str,
    client: httpx.Client | HttpClientPool,
    start_barrier: threading.Barrier | None = None,
) -> RequestResult:
    if isinstance(client, HttpClientPool):
        with client.lease() as leased_client:
            if start_barrier is not None:
                start_barrier.wait()
            return execute_http_request(
                trial,
                index,
                request_nonce=request_nonce,
                client=leased_client,
            )
    return execute_http_request(
        trial,
        index,
        request_nonce=request_nonce,
        client=client,
    )


def execute_http_request(
    trial: HttpLoadTrial,
    index: int,
    *,
    request_nonce: str | None = None,
    client: httpx.Client | None = None,
) -> RequestResult:
    executors = {
        OPENAI_CHAT: execute_openai_request,
        OLLAMA_CHAT: execute_ollama_request,
    }
    executor = executors.get(trial.target.protocol)
    if executor is None:
        raise ValueError(f"unsupported chat protocol: {trial.target.protocol}")
    return executor(trial, index, request_nonce=request_nonce, client=client)


def execute_openai_request(
    trial: HttpLoadTrial,
    index: int,
    *,
    request_nonce: str | None = None,
    client: httpx.Client | None = None,
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
        client=client,
    )


def execute_ollama_request(
    trial: HttpLoadTrial,
    index: int,
    *,
    request_nonce: str | None = None,
    client: httpx.Client | None = None,
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
        client=client,
    )


def post_json(
    trial: HttpLoadTrial,
    index: int,
    *,
    url: str,
    payload: dict[str, Any],
    stream_reader: Any,
    body_reader: Any,
    client: httpx.Client | None = None,
) -> RequestResult:
    start = time.perf_counter()
    deadline = start + trial.scenario.timeout_s
    deadline_expired = threading.Event()
    request_headers = headers_for_target(trial.target)
    sensitive_values = _sensitive_header_values(trial.target, request_headers)
    client_context: AbstractContextManager[httpx.Client]
    client_context = nullcontext(client) if client is not None else _owned_http_client(trial)
    try:
        with (
            client_context as active_client,
            _safe_stream(
                active_client,
                "POST",
                url,
                content=json.dumps(payload).encode("utf-8"),
                headers=request_headers,
                deadline=deadline,
            ) as response,
            _response_deadline(response, deadline, deadline_expired),
        ):
            status_code = response.status_code
            if status_code >= 400:
                body = _read_bounded_text(
                    response,
                    deadline=deadline,
                    max_bytes=_MAX_ERROR_RESPONSE_BYTES,
                )
                body = _redact_text(body, sensitive_values)[-1000:]
                return RequestResult(
                    index=index,
                    ok=False,
                    status_code=status_code,
                    latency_s=time.perf_counter() - start,
                    error=f"HTTP {status_code}: {body}",
                )
            if stream_reader is not None:
                result = stream_reader(
                    index,
                    _iter_bounded_lines(
                        response,
                        deadline=deadline,
                        max_bytes=_MAX_SUCCESS_RESPONSE_BYTES,
                    ),
                    start,
                    status_code,
                )
            else:
                body = _read_bounded_text(
                    response,
                    deadline=deadline,
                    max_bytes=_MAX_SUCCESS_RESPONSE_BYTES,
                )
                result = body_reader(index, body, start, status_code)
            return with_check_result(result, trial.scenario)
    except (httpx.HTTPError, TimeoutError, OSError) as exc:
        error: Exception = exc
        if deadline_expired.is_set() or time.perf_counter() > deadline:
            error = TimeoutError("HTTP request exceeded its total timeout")
        return RequestResult(
            index=index,
            ok=False,
            status_code=None,
            latency_s=time.perf_counter() - start,
            error=_redact_text(f"{type(error).__name__}: {error}", sensitive_values),
        )


@contextmanager
def _safe_stream(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    deadline: float | None = None,
    **kwargs: Any,
) -> Iterator[httpx.Response]:
    """Stream a request while permitting only same-origin, method-preserving redirects."""
    expected_origin = http_origin(url)
    route = _prepare_client_route(client, url, require_resolution=False)
    current_url = url
    for _redirect_count in range(_MAX_REDIRECTS + 1):
        if deadline is not None:
            kwargs["timeout"] = _request_timeout(deadline)
        validate_request_url(
            current_url,
            expected_origin=expected_origin,
            resolve_addresses=False,
        )
        request_url = route.request_url(current_url) if route is not None else current_url
        request_kwargs = dict(kwargs)
        if route is not None:
            request_kwargs["headers"] = route.request_headers(
                dict(request_kwargs.get("headers") or {})
            )
            request_kwargs["extensions"] = {
                **dict(request_kwargs.get("extensions") or {}),
                "sni_hostname": route.sni_hostname,
            }
        header_token: int | None = None
        if deadline is not None:
            header_token = _DEADLINE_WATCHDOG.schedule(deadline, client.close)
        try:
            with client.stream(
                method,
                request_url,
                follow_redirects=False,
                **request_kwargs,
            ) as response:
                if header_token is not None:
                    _DEADLINE_WATCHDOG.cancel(header_token)
                    header_token = None
                if deadline is not None:
                    _check_deadline(deadline)
                if response.has_redirect_location:
                    if response.status_code not in {307, 308}:
                        raise ConfigError(
                            "HTTP load refuses redirects that can change the request method"
                        )
                    redirect_url = urljoin(current_url, response.headers["location"])
                    validate_request_url(
                        redirect_url,
                        expected_origin=expected_origin,
                        resolve_addresses=False,
                    )
                    current_url = redirect_url
                    continue
                if 300 <= response.status_code < 400:
                    raise ConfigError("HTTP load received a redirect without a usable location")
                yield response
                return
        finally:
            if header_token is not None:
                _DEADLINE_WATCHDOG.cancel(header_token)
    raise ConfigError("HTTP load exceeded the maximum same-origin redirects")


def _prepare_client_route(
    client: httpx.Client,
    url: str,
    *,
    require_resolution: bool,
) -> PinnedHttpRoute | None:
    origin = http_origin(url)
    with _ROUTE_LOCK:
        routes: dict[HttpOrigin, PinnedHttpRoute | None] = getattr(
            client,
            "_llm_refinery_routes",
            {},
        )
        if origin not in routes:
            # Injected clients are primarily a testing seam. Pin when DNS is
            # available while preserving MockTransport's offline semantics.
            routes[origin] = resolve_request_route(
                url,
                require_resolution=require_resolution,
            )
            client._llm_refinery_routes = routes  # type: ignore[attr-defined]
        return routes[origin]


def _remaining_timeout(deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("HTTP request exceeded its total timeout")
    return remaining


def _request_timeout(deadline: float) -> httpx.Timeout:
    remaining = _remaining_timeout(deadline)
    short_phase = min(5.0, remaining / 10)
    read_phase = max(0.001, remaining - 3 * short_phase)
    return httpx.Timeout(
        connect=short_phase,
        write=short_phase,
        pool=short_phase,
        read=read_phase,
    )


@contextmanager
def _response_deadline(
    response: httpx.Response,
    deadline: float,
    expired: threading.Event,
) -> Iterator[None]:
    """Close the live stream at the absolute deadline to interrupt a blocking read."""

    def expire() -> None:
        expired.set()
        response.close()

    _remaining_timeout(deadline)
    token = _DEADLINE_WATCHDOG.schedule(deadline, expire)
    try:
        yield
    finally:
        _DEADLINE_WATCHDOG.cancel(token)


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() > deadline:
        raise TimeoutError("HTTP request exceeded its total timeout")


def _read_bounded_text(
    response: httpx.Response,
    *,
    deadline: float,
    max_bytes: int,
) -> str:
    content = bytearray()
    for chunk in response.iter_bytes():
        _check_deadline(deadline)
        if len(content) + len(chunk) > max_bytes:
            raise ValueError(f"HTTP response exceeded {max_bytes} bytes")
        content.extend(chunk)
    _check_deadline(deadline)
    return bytes(content).decode("utf-8", errors="replace")


def _iter_bounded_lines(
    response: httpx.Response,
    *,
    deadline: float,
    max_bytes: int,
) -> Iterator[str]:
    observed_bytes = 0
    pending = bytearray()
    for chunk in response.iter_bytes():
        _check_deadline(deadline)
        observed_bytes += len(chunk)
        if observed_bytes > max_bytes:
            raise ValueError(f"HTTP response exceeded {max_bytes} bytes")
        pending.extend(chunk)
        while True:
            newline = pending.find(b"\n")
            if newline < 0:
                break
            raw_line = bytes(pending[:newline])
            del pending[: newline + 1]
            _check_deadline(deadline)
            yield raw_line.decode("utf-8", errors="replace").rstrip("\r")
    if pending:
        _check_deadline(deadline)
        yield bytes(pending).decode("utf-8", errors="replace").rstrip("\r")
    _check_deadline(deadline)


def _sensitive_header_values(target: Endpoint, headers: dict[str, str]) -> tuple[str, ...]:
    values = {value for value in target.headers.values() if value}
    for name, value in headers.items():
        if name.casefold() in {"authorization", "proxy-authorization", "cookie", "set-cookie"}:
            values.add(value)
            scheme, separator, credential = value.partition(" ")
            if separator and scheme.casefold() in {"bearer", "basic"} and credential:
                values.add(credential)
    return tuple(sorted(values, key=len, reverse=True))


def _redact_text(value: str, sensitive_values: tuple[str, ...]) -> str:
    redacted = value
    for secret in sensitive_values:
        redacted = redacted.replace(secret, _REDACTED)
    return redacted


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
        line = _line_text(raw_line)
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
        line = _line_text(raw_line)
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
        for key in ("reasoning", "reasoning_content", "thinking"):
            if mapping.get(key):
                reasoning_parts.append(str(mapping[key]))
        for key in ("content", "text"):
            if mapping.get(key):
                visible_parts.append(str(mapping[key]))
    return "".join(visible_parts), "".join(reasoning_parts)


def _line_text(raw_line: object) -> str:
    if isinstance(raw_line, bytes):
        return raw_line.decode("utf-8", errors="replace").strip()
    return str(raw_line).strip()


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
