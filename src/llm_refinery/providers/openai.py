from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


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
    def complete(
        self,
        *,
        base_url: str,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        timeout_s: float,
        seed: int | None = None,
        extra_body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        api_key_env: str | None = None,
    ) -> ChatCompletionResponse:
        payload: dict[str, Any] = {
            "model": model,
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
            chat_completions_url(base_url),
            payload,
            headers=json_headers(headers, api_key_env=api_key_env),
            timeout_s=timeout_s,
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


def chat_completions_url(base_url: str) -> str:
    stripped = base_url.rstrip("/")
    if stripped.endswith("/chat/completions"):
        return stripped
    return f"{stripped}/chat/completions"


def json_headers(
    headers: dict[str, str] | None = None,
    *,
    api_key_env: str | None = None,
    accept: bool = True,
) -> dict[str, str]:
    resolved = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        **(headers or {}),
    }
    if accept:
        resolved.setdefault("Accept", "application/json")
    if api_key_env and "Authorization" not in resolved:
        token = os.environ.get(api_key_env)
        if token:
            resolved["Authorization"] = f"Bearer {token}"
    return resolved


def post_json_body(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout_s: float,
) -> str:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 - benchmark target is user-configured
            request,
            timeout=timeout_s,
        ) as response:
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


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
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
