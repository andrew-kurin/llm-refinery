from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from llm_refinery.benchmarks.agent import load_agent_benchmark_spec
from llm_refinery.benchmarks.agent.base import (
    UNSET,
    AgentBenchmarkSpec,
    AgentEvalRequest,
    AgentEvalRequestConfig,
    AgentEvalResult,
    ChatClient,
    LimitOverride,
)
from llm_refinery.config import ConfigError, stable_hash
from llm_refinery.core.runs import make_run_id, prepare_artifact_dir, record_benchmark_run
from llm_refinery.providers.openai import OpenAICompatibleChatClient
from llm_refinery.storage import ResultStore, utc_now


@dataclass(frozen=True)
class AgentEvalTarget:
    name: str
    provider: str
    base_url: str
    model: str
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AgentEvalTarget:
        name = str(raw.get("name") or "").strip()
        if not name:
            raise ConfigError("each agent-eval target requires a non-empty 'name'")

        provider = str(raw.get("provider") or "openai").strip().lower()
        if provider != "openai":
            raise ConfigError(f"target {name!r} provider must be 'openai', got {provider!r}")

        base_url = str(raw.get("base_url") or "").strip().rstrip("/")
        if not base_url:
            raise ConfigError(f"target {name!r} requires 'base_url'")

        model = str(raw.get("model") or "").strip()
        if not model:
            raise ConfigError(f"target {name!r} requires 'model'")

        return cls(
            name=name,
            provider=provider,
            base_url=base_url,
            model=model,
            api_key_env=str(raw["api_key_env"]) if raw.get("api_key_env") else None,
            headers={str(k): str(v) for k, v in dict(raw.get("headers") or {}).items()},
        )

    def safe_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "header_names": sorted(self.headers),
        }


@dataclass(frozen=True)
class AgentEvalConfig:
    name: str
    database: Path
    benchmark: AgentBenchmarkSpec
    targets: list[AgentEvalTarget]
    request: AgentEvalRequestConfig = field(default_factory=AgentEvalRequestConfig)
    source_path: Path | None = None

    @classmethod
    def from_mapping(cls, raw: dict[str, Any], source_path: Path | None = None) -> AgentEvalConfig:
        name = str(raw.get("name") or (source_path.stem if source_path else "agent-eval"))
        targets_raw = raw.get("targets") or []
        if not targets_raw:
            raise ConfigError("agent-eval config requires at least one target in 'targets'")

        benchmark_raw = dict(raw.get("benchmark") or {})
        benchmark = load_agent_benchmark_spec(benchmark_raw, source_path=source_path)

        return cls(
            name=name,
            database=Path(str(raw.get("database") or "results/llm_refinery.duckdb")),
            benchmark=benchmark,
            targets=[AgentEvalTarget.from_mapping(dict(item)) for item in targets_raw],
            request=AgentEvalRequestConfig.from_mapping(raw.get("request")),
            source_path=source_path,
        )


def load_agent_eval_config(path: str | Path) -> AgentEvalConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")
    return AgentEvalConfig.from_mapping(raw, source_path=config_path)


class OpenAIChatClient:
    def __init__(self, client: OpenAICompatibleChatClient | None = None) -> None:
        self.client = client or OpenAICompatibleChatClient()

    def complete(self, target: AgentEvalTarget, request: AgentEvalRequest) -> AgentEvalResult:
        started = time.perf_counter()
        last_error: str | None = None
        for attempt in range(request.config.retries + 1):
            try:
                return self._complete_once(target, request, started_at=started)
            except Exception as exc:  # noqa: BLE001 - store benchmark request failures
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= request.config.retries:
                    break
        return AgentEvalResult(
            request=request,
            ok=False,
            latency_s=time.perf_counter() - started,
            error=last_error,
        )

    def _complete_once(
        self, target: AgentEvalTarget, request: AgentEvalRequest, *, started_at: float
    ) -> AgentEvalResult:
        response = self.client.complete(
            base_url=target.base_url,
            model=target.model,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            temperature=request.config.temperature,
            max_tokens=request.config.max_tokens,
            timeout_s=request.config.timeout_s,
            seed=request.config.seed,
            extra_body=request.config.extra_body,
            headers=target.headers,
            api_key_env=target.api_key_env,
        )
        content = response.content
        return AgentEvalResult(
            request=request,
            ok=bool(content.strip()),
            latency_s=time.perf_counter() - started_at,
            response_text=content,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            error=None if content.strip() else "empty response content",
        )


def run_agent_eval(
    config: AgentEvalConfig,
    *,
    target_names: tuple[str, ...] = (),
    limit: LimitOverride = UNSET,
    task_ids: tuple[int, ...] = (),
    dry_run: bool = False,
    client: ChatClient | None = None,
) -> int:
    benchmark = config.benchmark.with_overrides(
        limit=limit,
        task_ids=task_ids or None,
    )
    tasks = benchmark.load_tasks()
    targets = _selected_targets(config.targets, target_names)
    requests = benchmark.expand_requests(tasks, config.request)

    if dry_run:
        for target in targets:
            print(
                f"agent-eval benchmark={benchmark.kind} target={target.name} "
                f"model={target.model} requests={len(requests)} tasks={len(tasks)}"
            )
        return 0

    chat_client = client or OpenAIChatClient()
    database = config.database
    with ResultStore(database) as store:
        for target in targets:
            _run_target(config, benchmark, target, requests, chat_client, store)
    return 0


def _run_target(
    config: AgentEvalConfig,
    benchmark: AgentBenchmarkSpec,
    target: AgentEvalTarget,
    requests: list[AgentEvalRequest],
    client: ChatClient,
    store: ResultStore,
) -> None:
    key = stable_hash(
        {
            "suite": config.name,
            "benchmark": benchmark.safe_json(),
            "target": target.safe_json(),
            "request_count": len(requests),
            "request": config.request.safe_json(),
        }
    )
    run_id = make_run_id(key)
    artifact_dir = prepare_artifact_dir(store.database, run_id)
    responses_path = artifact_dir / f"{benchmark.kind}-responses.jsonl"
    errors_path = artifact_dir / "errors.txt"

    print(
        f"agent-eval benchmark={benchmark.kind} target={target.name} "
        f"requests={len(requests)} model={target.model}"
    )
    started = utc_now()
    monotonic_start = time.perf_counter()
    results: list[AgentEvalResult] = []
    for index, request in enumerate(requests, start=1):
        print(
            f"[{index}/{len(requests)}] task={request.task_key} "
            f"type={request.response_type} variant={request.prompt_variant}",
            flush=True,
        )
        result = benchmark.score_result(client.complete(target, request))
        results.append(result)
    ended = utc_now()
    duration_s = time.perf_counter() - monotonic_start

    metrics = benchmark.summarize_results(results, duration_s)
    status = "ok" if metrics.get("error_count", 0.0) == 0 and results else "failed"
    error = _first_error(results) if status != "ok" else None

    responses_path.write_text(
        "\n".join(json.dumps(result.as_jsonable(), sort_keys=True) for result in results) + "\n",
        encoding="utf-8",
    )
    errors_path.write_text(
        "\n".join(result.error or "" for result in results if result.error),
        encoding="utf-8",
    )

    trial_name = f"{config.name}/{target.name}/{benchmark.kind}/{key}"
    record_benchmark_run(
        store,
        run_id=run_id,
        suite=config.name,
        trial_name=trial_name,
        status=status,
        started_at=started,
        ended_at=ended,
        duration_s=duration_s,
        command=(
            f"agent-eval benchmark={benchmark.kind} target={target.name} "
            f"model={target.model} requests={len(requests)}"
        ),
        config_json={
            "benchmark": benchmark.safe_json(),
            "target": target.safe_json(),
            "request": config.request.safe_json(),
            "params": {
                "benchmark": benchmark.kind,
                "target": target.name,
                "model": target.model,
                "prompt_variants": ",".join(
                    sorted({request.prompt_variant for request in requests})
                ),
                "response_types": ",".join(
                    sorted({request.response_type for request in requests})
                ),
                "task_count": len({request.task_key for request in requests}),
                "request_count": len(requests),
            },
            "model": {"name": target.model},
            "prompt_tokens": None,
            "gen_tokens": config.request.max_tokens,
        },
        metrics=metrics,
        stdout_path=responses_path,
        stderr_path=errors_path,
        error=error,
    )
    print(f"stored {status}: {run_id} ({_metric_summary(metrics)})")


def _selected_targets(
    targets: list[AgentEvalTarget], target_names: tuple[str, ...]
) -> list[AgentEvalTarget]:
    wanted = set(target_names)
    selected = [target for target in targets if not wanted or target.name in wanted]
    missing = wanted - {target.name for target in selected}
    if missing:
        raise ConfigError(f"unknown agent-eval target(s): {', '.join(sorted(missing))}")
    return selected


def _first_error(results: list[AgentEvalResult]) -> str | None:
    for result in results:
        if result.error:
            return result.error
    return None


def _metric_summary(metrics: dict[str, float]) -> str:
    keys = [
        "success_rate",
        "latency_p95_s",
        "workflow_step_abs_error_avg",
        "code_syntax_pass_rate",
        "error_count",
    ]
    return ", ".join(f"{key}={metrics[key]:.3f}" for key in keys if key in metrics)
