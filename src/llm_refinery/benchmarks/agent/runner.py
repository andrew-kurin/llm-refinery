from __future__ import annotations

import json
import time
from contextlib import nullcontext

from llm_refinery.application.run_session import RunSession
from llm_refinery.benchmarks.agent.base import (
    UNSET,
    AgentBenchmarkSpec,
    AgentEvalRequest,
    AgentEvalResult,
    ChatClient,
    LimitOverride,
)
from llm_refinery.benchmarks.agent.config import AgentEvalConfig
from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import Endpoint
from llm_refinery.core.runs import CompletedRun, RunSpec
from llm_refinery.providers.openai_chat import OpenAICompatibleChatClient
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.storage.models import SampleRecord


class AgentEvalFailed(RuntimeError):
    pass


class OpenAIChatClient:
    def __init__(self, client: OpenAICompatibleChatClient | None = None) -> None:
        self.client = client or OpenAICompatibleChatClient()

    def complete(self, target: Endpoint, request: AgentEvalRequest) -> AgentEvalResult:
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
        self, target: Endpoint, request: AgentEvalRequest, *, started_at: float
    ) -> AgentEvalResult:
        response = self.client.complete(
            target,
            messages=[
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.prompt},
            ],
            temperature=request.config.temperature,
            max_tokens=request.config.max_tokens,
            timeout_s=request.config.timeout_s,
            seed=request.config.seed,
            extra_body=request.config.extra_body,
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
    keep_going: bool = False,
    client: ChatClient | None = None,
    parent_run_id: str | None = None,
    store: ResultStore | None = None,
) -> list[CompletedRun]:
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
        return []

    chat_client = client or OpenAIChatClient()
    if store is not None and store.database != config.database.resolve():
        raise ValueError(
            f"agent-eval database {config.database.resolve()} does not match shared store"
        )
    outcomes: list[CompletedRun] = []
    failures: list[str] = []
    store_context = nullcontext(store) if store is not None else ResultStore(config.database)
    with store_context as active_store:
        assert active_store is not None
        for target in targets:
            try:
                outcomes.append(
                    _run_target(
                        config,
                        benchmark,
                        target,
                        requests,
                        chat_client,
                        active_store,
                        parent_run_id=parent_run_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - RunSession persisted the failure
                if keep_going:
                    message = f"target={target.name}: {exc}"
                    failures.append(message)
                    print(f"failed: {message}")
                    continue
                raise
    if failures:
        raise AgentEvalFailed(f"{len(failures)} agent-eval target(s) failed")
    return outcomes


def _run_target(
    config: AgentEvalConfig,
    benchmark: AgentBenchmarkSpec,
    target: Endpoint,
    requests: list[AgentEvalRequest],
    client: ChatClient,
    store: ResultStore,
    *,
    parent_run_id: str | None,
) -> CompletedRun:
    command = (
        f"agent-eval benchmark={benchmark.kind} target={target.name} "
        f"model={target.model} requests={len(requests)}"
    )
    config_json = {
        "benchmark": benchmark.safe_json(),
        "target": target.safe_json(),
        "request": config.request.safe_json(),
        "task_keys": sorted({request.task_key for request in requests}),
        "request_hashes": [request.key for request in requests],
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
    }
    spec = RunSpec.create(
        benchmark_kind="agent_eval",
        suite=config.name,
        label=f"{config.name}/{target.name}/{benchmark.kind}",
        command=command,
        config_json=config_json,
        database=store.database,
        parent_run_id=parent_run_id,
    )

    print(command)
    with RunSession(store, spec) as run:
        responses_path = run.artifact(
            "responses", f"{benchmark.kind}-responses.jsonl", "application/x-ndjson"
        )
        errors_path = run.artifact("errors", "errors.txt", "text/plain")
        measurement_path = run.artifact("measurement", "measurement.json", "application/json")
        responses_path.write_text("", encoding="utf-8")
        errors_path.write_text("", encoding="utf-8")

        results: list[AgentEvalResult] = []
        measurement_started = time.perf_counter()
        for index, request in enumerate(requests, start=1):
            print(
                f"[{index}/{len(requests)}] task={request.task_key} "
                f"type={request.response_type} variant={request.prompt_variant}",
                flush=True,
            )
            result = benchmark.score_result(client.complete(target, request))
            results.append(result)
            with responses_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(result.as_jsonable(), sort_keys=True) + "\n")
            if result.error:
                with errors_path.open("a", encoding="utf-8") as handle:
                    handle.write(result.error + "\n")
            store.record_sample(
                SampleRecord(
                    run_id=run.run_id,
                    sample_id=f"{index:06d}-{request.key}",
                    status="ok" if result.ok else "failed",
                    payload_json=_sample_payload(result),
                    metrics=_result_metrics(result),
                    artifact_path=str(responses_path),
                    error=result.error,
                )
            )

        wall_duration_s = time.perf_counter() - measurement_started
        measurement_path.write_text(
            json.dumps({"wall_duration_s": wall_duration_s}, sort_keys=True),
            encoding="utf-8",
        )
        metrics = benchmark.summarize_results(results, wall_duration_s)
        status = "ok" if metrics.get("error_count", 0.0) == 0 and results else "failed"
        error = _first_error(results) if status != "ok" else None
        outcome = run.complete(status=status, metrics=metrics, error=error)

    print(f"stored {status}: {outcome.run_id} ({_metric_summary(metrics)})")
    if status != "ok":
        raise AgentEvalFailed(f"{spec.trial_name} failed: {error}; artifacts: {errors_path}")
    return outcome


def _selected_targets(
    targets: list[Endpoint], target_names: tuple[str, ...]
) -> list[Endpoint]:
    wanted = set(target_names)
    selected = [target for target in targets if not wanted or target.name in wanted]
    missing = wanted - {target.name for target in selected}
    if missing:
        raise ConfigError(f"unknown agent-eval target(s): {', '.join(sorted(missing))}")
    return selected


def _sample_payload(result: AgentEvalResult) -> dict[str, object]:
    payload = result.as_jsonable()
    response_text = str(payload.pop("response_text", ""))
    payload["response_chars"] = len(response_text)
    return payload


def _result_metrics(result: AgentEvalResult) -> dict[str, float]:
    metrics = {"latency_s": result.latency_s}
    optional = {
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "workflow_step_count": result.workflow_step_count,
        "workflow_step_abs_error": result.workflow_step_abs_error,
    }
    metrics.update({key: float(value) for key, value in optional.items() if value is not None})
    if result.code_syntax_ok is not None:
        metrics["code_syntax_ok"] = float(result.code_syntax_ok)
    return metrics


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
