from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Self

from llm_refinery.core.config import ConfigError, reject_unknown_keys
from llm_refinery.core.endpoints import Endpoint
from llm_refinery.core.runs import stable_hash


class _Unset:
    pass


UNSET = _Unset()
LimitOverride = int | None | _Unset


class AgentTask(Protocol):
    def safe_json(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class AgentEvalRequestConfig:
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 600.0
    retries: int = 1
    seed: int | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any] | None) -> AgentEvalRequestConfig:
        raw = raw or {}
        reject_unknown_keys(
            raw,
            {"temperature", "max_tokens", "timeout_s", "retries", "seed", "extra_body"},
            context="agent-eval request",
        )
        max_tokens = int(raw.get("max_tokens", 1024))
        timeout_s = float(raw.get("timeout_s", 600.0))
        retries = int(raw.get("retries", 1))
        if max_tokens <= 0:
            raise ConfigError("agent-eval request.max_tokens must be positive")
        if timeout_s <= 0:
            raise ConfigError("agent-eval request.timeout_s must be positive")
        if retries < 0:
            raise ConfigError("agent-eval request.retries cannot be negative")
        return cls(
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            retries=retries,
            seed=int(raw["seed"]) if raw.get("seed") is not None else None,
            extra_body=dict(raw.get("extra_body") or {}),
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout_s": self.timeout_s,
            "retries": self.retries,
            "seed": self.seed,
            "extra_body": self.extra_body,
        }


@dataclass(frozen=True)
class AgentEvalRequest:
    task: AgentTask
    prompt_variant: str
    response_type: str
    system: str
    prompt: str
    config: AgentEvalRequestConfig

    @property
    def task_key(self) -> str:
        task_id = getattr(self.task, "task_id", None)
        if task_id is not None:
            return str(task_id)
        return stable_hash(self.task.safe_json())

    @property
    def key(self) -> str:
        return stable_hash(
            {
                "task": self.task.safe_json(),
                "prompt_variant": self.prompt_variant,
                "response_type": self.response_type,
                "system": self.system,
                "prompt": self.prompt,
                "request": self.config.safe_json(),
            }
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "task": self.task.safe_json(),
            "prompt_variant": self.prompt_variant,
            "response_type": self.response_type,
            "system": self.system,
            "prompt_preview": self.prompt[:500],
            "prompt_chars": len(self.prompt),
            "prompt_hash": stable_hash(self.prompt),
            "request": self.config.safe_json(),
        }


@dataclass(frozen=True)
class AgentEvalResult:
    request: AgentEvalRequest
    ok: bool
    latency_s: float
    response_text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    workflow_step_count: int | None = None
    workflow_step_abs_error: int | None = None
    code_syntax_ok: bool | None = None
    code_model_function_present: bool | None = None
    code_contract_ok: bool | None = None
    code_reference_import_recall: float | None = None
    code_reference_call_recall: float | None = None
    error: str | None = None

    def as_jsonable(self) -> dict[str, Any]:
        return {
            "request": self.request.safe_json(),
            "ok": self.ok,
            "latency_s": self.latency_s,
            "response_text": self.response_text,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "workflow_step_count": self.workflow_step_count,
            "workflow_step_abs_error": self.workflow_step_abs_error,
            "code_syntax_ok": self.code_syntax_ok,
            "code_model_function_present": self.code_model_function_present,
            "code_contract_ok": self.code_contract_ok,
            "code_reference_import_recall": self.code_reference_import_recall,
            "code_reference_call_recall": self.code_reference_call_recall,
            "error": self.error,
        }


class ChatClient(Protocol):
    def complete(self, target: Endpoint, request: AgentEvalRequest) -> AgentEvalResult: ...


class AgentBenchmarkSpec(Protocol):
    @property
    def kind(self) -> str: ...

    def with_overrides(
        self, *, limit: LimitOverride = UNSET, task_ids: tuple[int, ...] | None = None
    ) -> Self: ...

    def safe_json(self) -> dict[str, Any]: ...

    def load_tasks(self) -> list[AgentTask]: ...

    def expand_requests(
        self, tasks: list[AgentTask], request_config: AgentEvalRequestConfig
    ) -> list[AgentEvalRequest]: ...

    def score_result(self, result: AgentEvalResult) -> AgentEvalResult: ...

    def summarize_results(
        self, results: list[AgentEvalResult], wall_duration_s: float
    ) -> dict[str, float]: ...
