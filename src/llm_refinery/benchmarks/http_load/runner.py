from __future__ import annotations

import json
import time
from contextlib import nullcontext
from pathlib import Path

from llm_refinery.application.run_session import RunSession
from llm_refinery.benchmarks.http_load.config import (
    RECOMMENDED_MEASURED_REQUESTS,
    HttpLoadConfig,
    HttpLoadTrial,
    expand_http_load_trials,
    print_http_load_plan,
)
from llm_refinery.benchmarks.http_load.metrics import summarize_request_results
from llm_refinery.benchmarks.http_load.models import RequestResult
from llm_refinery.benchmarks.http_load.transport import run_requests
from llm_refinery.core.runs import CompletedRun, RunSpec
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.storage.models import SampleRecord


class HttpLoadFailed(RuntimeError):
    pass


def run_http_load(
    config: HttpLoadConfig,
    *,
    target_names: tuple[str, ...] = (),
    scenario_names: tuple[str, ...] = (),
    limit: int | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    database_override: str | Path | None = None,
    parent_run_id: str | None = None,
    store: ResultStore | None = None,
) -> list[CompletedRun]:
    trials = expand_http_load_trials(
        config,
        target_names=target_names,
        scenario_names=scenario_names,
    )
    if limit is not None:
        trials = trials[:limit]

    if dry_run:
        print_http_load_plan(
            config,
            target_names=target_names,
            scenario_names=scenario_names,
            limit=limit,
        )
        return []

    if not trials:
        print("no HTTP load trials to run")
        return []

    database = Path(database_override) if database_override else config.database
    if store is not None and store.database != database.resolve():
        raise ValueError(f"HTTP load database {database.resolve()} does not match shared store")

    outcomes: list[CompletedRun] = []
    failures: list[str] = []
    store_context = nullcontext(store) if store is not None else ResultStore(database)
    with store_context as active_store:
        assert active_store is not None
        for index, trial in enumerate(trials, start=1):
            try:
                outcomes.append(
                    _run_one_http_load(
                        trial,
                        active_store,
                        index=index,
                        total=len(trials),
                        parent_run_id=parent_run_id,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - RunSession has persisted the failure
                if keep_going:
                    message = f"{trial.name}: {exc}"
                    failures.append(message)
                    print(f"failed: {message}")
                    continue
                raise
    if failures:
        raise HttpLoadFailed(f"{len(failures)} HTTP load trial(s) failed")
    return outcomes


def _run_one_http_load(
    trial: HttpLoadTrial,
    store: ResultStore,
    *,
    index: int,
    total: int,
    parent_run_id: str | None,
) -> CompletedRun:
    config_json = {**trial.as_jsonable(), "benchmark": "http_load"}
    label = (
        f"{trial.suite}/{trial.target.name}/{trial.scenario.name}/"
        f"c{trial.concurrency}-n{trial.max_tokens}"
    )
    spec = RunSpec.create(
        benchmark_kind="http_load",
        suite=trial.suite,
        label=label,
        command=trial.command_text,
        config_json=config_json,
        database=store.database,
        parent_run_id=parent_run_id,
    )

    print(f"[{index}/{total}] {spec.trial_name}")
    print(trial.command_text)
    if trial.scenario.requests < RECOMMENDED_MEASURED_REQUESTS:
        print(
            "warning: only "
            f"{trial.scenario.requests} measured requests; tail metrics are exploratory below "
            f"the recommended minimum of {RECOMMENDED_MEASURED_REQUESTS}"
        )

    with RunSession(store, spec) as run:
        responses_path = run.artifact("responses", "responses.jsonl", "application/x-ndjson")
        errors_path = run.artifact("errors", "errors.txt", "text/plain")
        measurement_path = run.artifact("measurement", "measurement.json", "application/json")

        warmup_count = trial.effective_warmup_requests
        warmup_results = run_requests(trial, count=warmup_count)

        measurement_started = time.perf_counter()
        results = run_requests(trial, count=trial.scenario.requests)
        duration_s = time.perf_counter() - measurement_started
        measurement_path.write_text(
            json.dumps(
                {
                    "wall_duration_s": duration_s,
                    "measured_request_count": trial.scenario.requests,
                    "configured_warmup_requests": trial.scenario.warmup_requests,
                    "effective_warmup_requests": warmup_count,
                    "cache_mode": trial.scenario.cache_mode,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        metrics = summarize_request_results(
            results,
            wall_duration_s=duration_s,
            concurrency=trial.concurrency,
            max_tokens=trial.max_tokens,
        )
        warmup_errors = [result for result in warmup_results if not result.ok]
        metrics["configured_warmup_requests"] = float(trial.scenario.warmup_requests)
        metrics["effective_warmup_requests"] = float(warmup_count)
        metrics["warmup_error_count"] = float(len(warmup_errors))
        status = (
            "ok"
            if not warmup_errors and metrics["error_count"] == 0 and metrics["success_count"] > 0
            else "failed"
        )
        error = _first_error([*warmup_results, *results]) if status != "ok" else None

        responses_path.write_text(
            "\n".join(json.dumps(result.as_jsonable(), sort_keys=True) for result in results)
            + "\n",
            encoding="utf-8",
        )
        errors_path.write_text(
            "\n".join(
                [
                    *(
                        f"warmup[{result.index}]: {result.error}"
                        for result in warmup_results
                        if result.error
                    ),
                    *(
                        f"request[{result.index}]: {result.error}"
                        for result in results
                        if result.error
                    ),
                ]
            ),
            encoding="utf-8",
        )
        for result in results:
            store.record_sample(
                SampleRecord(
                    run_id=run.run_id,
                    sample_id=str(result.index),
                    status="ok" if result.ok else "failed",
                    payload_json=_sample_payload(result),
                    metrics=_sample_metrics(result),
                    artifact_path=str(responses_path),
                    error=result.error,
                )
            )

        outcome = run.complete(status=status, metrics=metrics, error=error)

    summary = _http_metric_summary(metrics)
    print(f"stored {status}: {outcome.run_id} ({summary})")
    if status != "ok":
        raise HttpLoadFailed(
            f"{spec.trial_name} had HTTP load errors: {error}; artifacts: {errors_path}"
        )
    return outcome


def _sample_payload(result: RequestResult) -> dict[str, object]:
    payload = result.as_jsonable()
    payload.pop("response_text", None)
    payload.pop("visible_response_text", None)
    payload.pop("reasoning_response_text", None)
    return payload


def _sample_metrics(result: RequestResult) -> dict[str, float]:
    values = {
        "latency_s": result.latency_s,
        "ttft_s": result.ttft_s,
        "visible_ttft_s": result.visible_ttft_s,
        "reasoning_ttft_s": result.reasoning_ttft_s,
        "tpot_s": result.tpot_s,
    }
    return {key: float(value) for key, value in values.items() if value is not None}


def _first_error(results: list[RequestResult]) -> str | None:
    for result in results:
        if result.error:
            return result.error
    return None


def _http_metric_summary(metrics: dict[str, float]) -> str:
    keys = [
        "requests_per_second",
        "observed_latency_p95_s",
        "visible_ttft_p95_s",
        "reasoning_ttft_p95_s",
        "tpot_p95_s",
        "completion_tokens_per_second",
        "check_pass_rate",
        "error_count",
    ]
    return ", ".join(f"{key}={metrics[key]:.3f}" for key in keys if key in metrics)
