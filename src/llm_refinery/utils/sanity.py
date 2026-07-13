from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from llm_refinery.core.endpoints import Endpoint
from llm_refinery.core.http_safety import PinnedHttpRoute
from llm_refinery.providers.openai_chat import OpenAICompatibleChatClient

REASONING_TAG_RE = re.compile(r"</?(?:think|thinking)\b", re.IGNORECASE)


def run_api_sanity_check(
    endpoint: Endpoint,
    timeout: int = 180,
    *,
    client: OpenAICompatibleChatClient | None = None,
    trust_env: bool = True,
    ca_bundle: Path | None = None,
    route: PinnedHttpRoute | None = None,
) -> dict[str, Any]:
    """Perform a basic OpenAI-compatible chat-completions sanity check."""
    chat_client = client or OpenAICompatibleChatClient(
        trust_env=trust_env,
        ca_bundle=ca_bundle,
        route=route,
    )
    try:
        response = chat_client.complete(
            endpoint,
            messages=[{"role": "user", "content": "Say hello in exactly five words."}],
            temperature=0,
            max_tokens=2048,
            timeout_s=timeout,
        )
        choices = response.raw.get("choices") or []
        choice = choices[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
            or ""
        )

        if not content.strip():
            detail = (
                "reasoning was returned but visible content was empty"
                if reasoning.strip()
                else "empty content returned"
            )
            raise ValueError(detail)
        if has_reasoning_tags(content):
            raise ValueError("reasoning/thinking tags present in content")
        response_model = response.raw.get("model")
        return {
            "success": True,
            "elapsed_s": round(response.latency_s, 3),
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "finish_reason": choice.get("finish_reason"),
            "content_preview": content[:200],
            "requested_model": endpoint.model,
            "response_model": str(response_model) if response_model is not None else None,
            "model_matches": response_model == endpoint.model
            if response_model is not None
            else None,
        }
    except Exception as exc:  # noqa: BLE001 - sanity failures are returned as result data
        return {"success": False, "error": f"sanity failed: {exc}"}


def has_reasoning_tags(text: str) -> bool:
    return bool(REASONING_TAG_RE.search(text))
