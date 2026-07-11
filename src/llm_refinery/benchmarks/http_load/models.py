from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestResult:
    index: int
    ok: bool
    status_code: int | None
    latency_s: float
    # Backward-compatible first output event, whether reasoning or visible content.
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    completion_chars: int = 0
    server_prompt_eval_duration_s: float | None = None
    server_eval_duration_s: float | None = None
    # ``response_text`` remains the legacy combined reasoning + visible text field.
    response_text: str = ""
    check_passed: bool | None = None
    error: str | None = None
    visible_ttft_s: float | None = None
    reasoning_ttft_s: float | None = None
    tpot_s: float | None = None
    # Interarrival times between non-empty streaming events. Providers may batch multiple tokens.
    itl_s: tuple[float, ...] = ()
    visible_completion_chars: int = 0
    reasoning_completion_chars: int = 0
    # None identifies old artifacts whose visible/reasoning channels were not separated.
    visible_response_text: str | None = None
    reasoning_response_text: str = ""

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "ok": self.ok,
            "status_code": self.status_code,
            "latency_s": self.latency_s,
            "ttft_s": self.ttft_s,
            "visible_ttft_s": self.visible_ttft_s,
            "reasoning_ttft_s": self.reasoning_ttft_s,
            "tpot_s": self.tpot_s,
            "itl_s": list(self.itl_s),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_chars": self.completion_chars,
            "visible_completion_chars": self.visible_completion_chars,
            "reasoning_completion_chars": self.reasoning_completion_chars,
            "server_prompt_eval_duration_s": self.server_prompt_eval_duration_s,
            "server_eval_duration_s": self.server_eval_duration_s,
            "response_text": self.response_text,
            "visible_response_text": self.visible_response_text,
            "reasoning_response_text": self.reasoning_response_text,
            "check_passed": self.check_passed,
            "error": self.error,
        }
