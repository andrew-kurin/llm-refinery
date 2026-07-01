from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from llm_refinery.benchmarks.http_load import (
    HttpLoadConfig,
    HttpLoadTrial,
    HttpScenario,
    HttpTarget,
    RequestResult,
    expand_http_load_trials,
    load_http_load_config,
    print_http_load_plan,
    summarize_request_results,
)
from llm_refinery.benchmarks.http_load.transport import run_requests
from llm_refinery.storage import ResultStore, RunRecord, utc_now
from llm_refinery.utils.system import get_system_profile

__all__ = [
    "HttpLoadConfig",
    "HttpLoadTrial",
    "HttpScenario",
    "HttpTarget",
    "RequestResult",
    "expand_http_load_trials",
    "load_http_load_config",
    "print_http_load_plan",
    "run_http_load",
    "summarize_request_results",
]


def run_http_load(
    config: HttpLoadConfig,
    *,
    target_names: tuple[str, ...] = (),
    scenario_names: tuple[str, ...] = (),
    limit: int | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    database_override: str | Path | None = None,
) -> int:
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
        return 0

    if not trials:
        print("no HTTP load trials to run")
        return 0

    database = Path(database_override) if database_override else config.database
    with ResultStore(database) as store:
        for index, trial in enumerate(trials, start=1):
            try:
                _run_one_http_load(config, trial, store, index=index, total=len(trials))
            except Exception as exc:  # noqa: BLE001 - keep-going needs to persist failures
                if keep_going:
                    print(f"failed: {trial.name}: {exc}")
                    continue
                raise
    return 0


def _run_one_http_load(
    config: HttpLoadConfig,
    trial: HttpLoadTrial,
    store: ResultStore,
    *,
    index: int,
    total: int,
) -> None:
    run_id = f"{trial.key}-{uuid.uuid4().hex[:8]}"
    artifact_dir = config.database.parent / "artifacts" / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = artifact_dir / "responses.jsonl"
    stderr_path = artifact_dir / "errors.txt"

    print(f"[{index}/{total}] {trial.name}")
    print(trial.command_text)

    if trial.scenario.warmup_requests:
        run_requests(trial, count=trial.scenario.warmup_requests)

    started = utc_now()
    monotonic_start = time.perf_counter()
    results = run_requests(trial, count=trial.scenario.requests)
    ended = utc_now()
    duration_s = time.perf_counter() - monotonic_start

    metrics = summarize_request_results(
        results,
        wall_duration_s=duration_s,
        concurrency=trial.concurrency,
        max_tokens=trial.max_tokens,
    )
    status = "ok" if metrics["error_count"] == 0 and metrics["success_count"] > 0 else "failed"
    error = _first_error(results) if status != "ok" else None

    stdout_path.write_text(
        "\n".join(json.dumps(result.as_jsonable(), sort_keys=True) for result in results) + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text(
        "\n".join(result.error or "" for result in results if result.error),
        encoding="utf-8",
    )

    store.record_run(
        RunRecord(
            run_id=run_id,
            suite=trial.suite,
            trial_name=trial.name,
            status=status,
            started_at=started,
            ended_at=ended,
            duration_s=duration_s,
            command=trial.command_text,
            cwd=str(Path.cwd()),
            config_json=trial.as_jsonable(),
            metrics=metrics,
            system_json=get_system_profile(),
            stdout_path=str(stdout_path),
            stderr_path=str(stderr_path),
            error=error,
        )
    )

    summary = _http_metric_summary(metrics)
    print(f"stored {status}: {run_id} ({summary})")
    if status != "ok":
        raise RuntimeError(f"{trial.name} had HTTP load errors: {error}; artifacts: {stderr_path}")


def _first_error(results: list[RequestResult]) -> str | None:
    for result in results:
        if result.error:
            return result.error
    return None


def _http_metric_summary(metrics: dict[str, float]) -> str:
    keys = [
        "requests_per_second",
        "latency_p95_s",
        "ttft_p95_s",
        "completion_tokens_per_second",
        "check_pass_rate",
        "error_count",
    ]
    return ", ".join(f"{key}={metrics[key]:.3f}" for key in keys if key in metrics)
