from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


def run_api_sanity_check(
    url: str,
    model_name: str = "local-model",
    timeout: int = 180,
) -> dict[str, Any]:
    """Perform a basic OpenAI-compatible chat-completions sanity check."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Say hello in exactly five words."}],
        "max_tokens": 80,
        "temperature": 0,
        "stream": False,
    }

    start = time.perf_counter()
    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 local URL
            body = response.read().decode(errors="replace")
            elapsed = time.perf_counter() - start

        data = json.loads(body)
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content") or ""
        reasoning = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or message.get("thinking")
            or ""
        )

        if not content.strip():
            raise ValueError("empty content returned")
        if reasoning.strip():
            raise ValueError("reasoning/thinking output present when not expected")

        return {
            "success": True,
            "elapsed_s": round(elapsed, 3),
            "content_len": len(content),
            "reasoning_len": len(reasoning),
            "finish_reason": choice.get("finish_reason"),
            "content_preview": content[:200],
        }
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[-1000:]
        return {"success": False, "error": f"HTTP {exc.code}: {body}"}
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", str(exc))
        return {"success": False, "error": f"URL error: {reason}"}
    except json.JSONDecodeError:
        return {"success": False, "error": "failed to decode JSON response"}
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        return {"success": False, "error": f"sanity failed: {exc}"}
    except Exception as exc:  # noqa: BLE001 - convert unexpected sanity errors into result data
        return {"success": False, "error": f"unexpected error: {exc}"}
