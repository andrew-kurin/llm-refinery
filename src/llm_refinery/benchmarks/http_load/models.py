from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RequestResult:
    index: int
    ok: bool
    status_code: int | None
    latency_s: float
    ttft_s: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    completion_chars: int = 0
    server_prompt_eval_duration_s: float | None = None
    server_eval_duration_s: float | None = None
    response_text: str = ""
    check_passed: bool | None = None
    error: str | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "ok": self.ok,
            "status_code": self.status_code,
            "latency_s": self.latency_s,
            "ttft_s": self.ttft_s,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "completion_chars": self.completion_chars,
            "server_prompt_eval_duration_s": self.server_prompt_eval_duration_s,
            "server_eval_duration_s": self.server_eval_duration_s,
            "response_text": self.response_text,
            "check_passed": self.check_passed,
            "error": self.error,
        }
