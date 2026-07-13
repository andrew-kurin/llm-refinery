from __future__ import annotations

import json
import os
import ssl
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint
from llm_refinery.core.http_safety import PinnedHttpRoute, pinned_route_trust_env

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)
MAX_CHAT_RESPONSE_BYTES = 2_000_000


@dataclass(frozen=True)
class ChatCompletionResponse:
    content: str
    usage: dict[str, Any]
    raw: dict[str, Any]
    latency_s: float

    @property
    def prompt_tokens(self) -> int | None:
        return int_or_none(self.usage.get("prompt_tokens"))

    @property
    def completion_tokens(self) -> int | None:
        return int_or_none(self.usage.get("completion_tokens"))

    @property
    def total_tokens(self) -> int | None:
        return int_or_none(self.usage.get("total_tokens"))


class OpenAICompatibleChatClient:
    def __init__(
        self,
        *,
        trust_env: bool = True,
        ca_bundle: Path | None = None,
        route: PinnedHttpRoute | None = None,
    ) -> None:
        self._trust_env = trust_env
        self._ca_bundle = ca_bundle
        self._route = route

    def complete(
        self,
        endpoint: Endpoint,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_s: float,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatCompletionResponse:
        if endpoint.protocol != OPENAI_CHAT:
            raise ValueError(f"OpenAI chat client cannot execute protocol {endpoint.protocol!r}")
        payload: dict[str, Any] = {
            "model": endpoint.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if seed is not None:
            payload["seed"] = seed
        payload.update(extra_body or {})

        started = time.perf_counter()
        body = post_json_body(
            endpoint.chat_completions_url,
            payload,
            headers=json_headers(endpoint.headers, api_key_env=endpoint.api_key_env),
            timeout_s=timeout_s,
            trust_env=self._trust_env,
            ca_bundle=self._ca_bundle,
            route=self._route,
        )
        data = json.loads(body)
        content_parts: list[str] = []
        for choice in data.get("choices") or []:
            message = choice.get("message") or {}
            if isinstance(message, dict) and message.get("content"):
                content_parts.append(str(message["content"]))
        return ChatCompletionResponse(
            content="".join(content_parts),
            usage=data.get("usage") or {},
            raw=data,
            latency_s=time.perf_counter() - started,
        )


def json_headers(
    headers: dict[str, str] | None = None,
    *,
    api_key_env: str | None = None,
    accept: bool = True,
) -> dict[str, str]:
    validate_http_headers(headers or {})
    resolved: dict[str, str] = {
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
    }
    for key, value in (headers or {}).items():
        existing = next(
            (candidate for candidate in resolved if candidate.casefold() == key.casefold()),
            None,
        )
        if existing is not None:
            del resolved[existing]
        resolved[key] = value
    if accept and not has_header(resolved, "Accept"):
        resolved["Accept"] = "application/json"
    if api_key_env and not has_header(resolved, "Authorization"):
        token = os.environ.get(api_key_env)
        if not token:
            raise ConfigError(f"endpoint API key environment variable is not set: {api_key_env}")
        resolved["Authorization"] = f"Bearer {token}"
    validate_http_headers(resolved)
    return resolved


def has_header(headers: dict[str, str], name: str) -> bool:
    return any(key.casefold() == name.casefold() for key in headers)


def validate_http_headers(headers: dict[str, str]) -> None:
    """Validate headers without ever including a header value in an error."""
    seen: set[str] = set()
    token_characters = frozenset(
        "!#$%&'*+-.^_`|~0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    for name, value in headers.items():
        if (
            not isinstance(name, str)
            or not name
            or any(character not in token_characters for character in name)
        ):
            raise ConfigError("HTTP header name is invalid")
        normalized = name.casefold()
        if normalized in seen:
            raise ConfigError("HTTP header names must be unique case-insensitively")
        seen.add(normalized)
        if not isinstance(value, str):
            raise ConfigError(f"HTTP header {name!r} value must be a string")
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ConfigError(f"HTTP header {name!r} value must contain only ASCII") from exc
        if value != value.strip(" \t") or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise ConfigError(f"HTTP header {name!r} value contains invalid characters")


def post_json_body(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_s: float,
    trust_env: bool = True,
    ca_bundle: Path | None = None,
    route: PinnedHttpRoute | None = None,
) -> str:
    client_trust_env = pinned_route_trust_env(url, trust_env=trust_env)
    try:
        context = httpx.create_ssl_context(
            verify=str(ca_bundle) if ca_bundle is not None else True,
            trust_env=trust_env,
        )
    except (OSError, ssl.SSLError) as exc:
        raise ConfigError(f"could not load endpoint CA bundle: {ca_bundle}") from exc
    client = httpx.Client(
        follow_redirects=False,
        trust_env=client_trust_env,
        verify=context,
    )
    request_url = route.request_url(url) if route is not None else url
    request_headers = route.request_headers(headers) if route is not None else headers
    extensions = {"sni_hostname": route.sni_hostname} if route is not None else None
    deadline = time.monotonic() + timeout_s
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        client.close()

    timer = threading.Timer(timeout_s, expire)
    timer.daemon = True
    timer.start()
    try:
        with client.stream(
            "POST",
            request_url,
            content=json.dumps(payload).encode("utf-8"),
            headers=request_headers,
            timeout=timeout_s,
            extensions=extensions,
        ) as response:
            # Never retain an untrusted error or redirect body: it may reflect
            # Authorization or another credential supplied with the request.
            if response.status_code >= 300:
                raise RuntimeError(f"HTTP {response.status_code}")
            content = bytearray()
            for chunk in response.iter_bytes():
                if expired.is_set() or time.monotonic() > deadline:
                    raise TimeoutError("chat request exceeded its total timeout")
                if len(content) + len(chunk) > MAX_CHAT_RESPONSE_BYTES:
                    raise ValueError("chat response is too large")
                content.extend(chunk)
            return bytes(content).decode("utf-8", errors="replace")
    except httpx.HTTPError as exc:
        if expired.is_set() or time.monotonic() > deadline:
            raise TimeoutError("chat request exceeded its total timeout") from exc
        raise
    finally:
        timer.cancel()
        client.close()


def openai_choice_text(choice: dict[str, Any]) -> str:
    parts: list[str] = []
    for mapping in (choice.get("delta"), choice.get("message"), choice):
        if not isinstance(mapping, dict):
            continue
        for key in ("content", "reasoning_content", "thinking", "text"):
            value = mapping.get(key)
            if value:
                parts.append(str(value))
    return "".join(parts)


def int_or_none(value: object) -> int | None:
    if not isinstance(value, str | bytes | bytearray | int | float):
        return None
    try:
        return int(value)
    except (ValueError, OverflowError):
        return None
