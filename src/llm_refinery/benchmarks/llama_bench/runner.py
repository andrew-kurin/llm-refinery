from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TaskID

from llm_refinery.application.run_session import RunSession
from llm_refinery.benchmarks.llama_bench.command import print_plan
from llm_refinery.benchmarks.llama_bench.command_builder import build_bench_command, shell_join
from llm_refinery.benchmarks.llama_bench.config import LlamaSweepConfig, LlamaTrial, expand_trials
from llm_refinery.benchmarks.llama_bench.parser import parse_llama_bench_metrics
from llm_refinery.benchmarks.llama_bench.process import run_bench_process
from llm_refinery.benchmarks.llama_bench.progress import (
    BenchProgress,
    make_bench_progress,
    trial_description,
    update_rich_progress,
)
from llm_refinery.benchmarks.llama_bench.reporting import metric_summary, tail
from llm_refinery.benchmarks.llama_bench.server import detect_llama_version
from llm_refinery.core.runs import CompletedRun, RunSpec
from llm_refinery.storage.duckdb import ResultStore

PROGRESS_INTERVAL_S = 0.5


class RunFailed(RuntimeError):
    pass


def run_bench(
    config: LlamaSweepConfig,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    database_override: str | Path | None = None,
    show_progress: bool = True,
    progress_interval_s: float = PROGRESS_INTERVAL_S,
    parent_run_id: str | None = None,
) -> list[CompletedRun]:
    trials = expand_trials(config, kind="bench")
    if limit is not None:
        trials = trials[:limit]

    if dry_run:
        print_plan(config, kind="bench", limit=limit)
        return []

    console = Console()
    if not trials:
        console.print("no trials to run")
        return []

    database = Path(database_override) if database_override else config.database
    progress_state = BenchProgress(total=len(trials))
    outcomes: list[CompletedRun] = []
    failures: list[str] = []
    llama_version = detect_llama_version(config.commands["bench"])
    with ResultStore(database) as store:
        if show_progress:
            with make_bench_progress(console) as rich_progress:
                task_id = rich_progress.add_task(
                    "starting",
                    total=len(trials),
                    current_elapsed="-",
                    average="unknown",
                    eta="unknown",
                    suite_elapsed="0s",
                )
                _run_bench_trials(
                    config,
                    trials,
                    store,
                    outcomes,
                    failures,
                    keep_going=keep_going,
                    console=console,
                    progress_state=progress_state,
                    rich_progress=rich_progress,
                    progress_task_id=task_id,
                    progress_interval_s=progress_interval_s,
                    parent_run_id=parent_run_id,
                    llama_version=llama_version,
                )
                update_rich_progress(
                    rich_progress,
                    task_id,
                    progress_state,
                    description="complete",
                    trial_started_monotonic=None,
                )
        else:
            _run_bench_trials(
                config,
                trials,
                store,
                outcomes,
                failures,
                keep_going=keep_going,
                console=console,
                progress_state=progress_state,
                rich_progress=None,
                progress_task_id=None,
                progress_interval_s=progress_interval_s,
                parent_run_id=parent_run_id,
                llama_version=llama_version,
            )
    if failures:
        raise RunFailed(f"{len(failures)} llama-bench trial(s) failed")
    return outcomes


def _run_bench_trials(
    config: LlamaSweepConfig,
    trials: list[LlamaTrial],
    store: ResultStore,
    outcomes: list[CompletedRun],
    failures: list[str],
    *,
    keep_going: bool,
    console: Console,
    progress_state: BenchProgress,
    rich_progress: Progress | None,
    progress_task_id: TaskID | None,
    progress_interval_s: float,
    parent_run_id: str | None,
    llama_version: str | None,
) -> None:
    for index, trial in enumerate(trials, start=1):
        try:
            outcomes.append(
                _run_one_bench(
                    config,
                    trial,
                    store,
                    index=index,
                    total=len(trials),
                    console=console,
                    progress_state=progress_state,
                    rich_progress=rich_progress,
                    progress_task_id=progress_task_id,
                    progress_interval_s=progress_interval_s,
                    parent_run_id=parent_run_id,
                    llama_version=llama_version,
                )
            )
        except Exception as exc:  # noqa: BLE001 - keep-going persists failures via RunSession
            if keep_going:
                message = f"{trial.name}: {exc}"
                failures.append(message)
                console.print(f"failed: {message}", style="red", markup=False)
                continue
            raise


def _run_one_bench(
    config: LlamaSweepConfig,
    trial: LlamaTrial,
    store: ResultStore,
    *,
    index: int,
    total: int,
    console: Console,
    progress_state: BenchProgress,
    rich_progress: Progress | None,
    progress_task_id: TaskID | None,
    progress_interval_s: float,
    parent_run_id: str | None,
    llama_version: str | None,
) -> CompletedRun:
    cmd = build_bench_command(config, trial)
    command_text = shell_join(cmd)
    config_json = {
        **trial.as_jsonable(),
        "benchmark": "llama_bench",
        "params": trial.params,
        "command_argv": cmd,
        "bench": {
            "repetitions": config.bench.repetitions,
            "output": config.bench.output,
            "extra_args": [*trial.model.extra_args, *config.bench.extra_args],
        },
    }
    label = (
        f"{trial.suite}/{trial.model.name}/"
        f"p{trial.prompt_tokens or 0}-g{trial.gen_tokens or 0}"
    )
    spec = RunSpec.create(
        benchmark_kind="llama_bench",
        suite=trial.suite,
        label=label,
        command=command_text,
        config_json=config_json,
        database=store.database,
        parent_run_id=parent_run_id,
    )

    if rich_progress is None or progress_task_id is None:
        console.print(f"[{index}/{total}] {spec.trial_name}", markup=False)
    console.print(command_text, style="dim", markup=False, soft_wrap=True)

    with RunSession(store, spec) as run:
        stdout_path = run.artifact("stdout", "stdout.txt", "text/plain")
        stderr_path = run.artifact("stderr", "stderr.txt", "text/plain")
        trial_started_monotonic = time.perf_counter()
        if rich_progress is not None and progress_task_id is not None:
            update_rich_progress(
                rich_progress,
                progress_task_id,
                progress_state,
                description=trial_description(index, total, spec.trial_name),
                trial_started_monotonic=trial_started_monotonic,
            )

        completed = run_bench_process(
            cmd,
            progress_state=progress_state,
            rich_progress=rich_progress,
            progress_task_id=progress_task_id,
            trial_started_monotonic=trial_started_monotonic,
            progress_interval_s=progress_interval_s,
        )
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        metrics = parse_llama_bench_metrics(completed.stdout)
        status = "ok" if completed.returncode == 0 else "failed"
        error = None if completed.returncode == 0 else f"exit code {completed.returncode}"
        duration_s = run.elapsed_s
        outcome = run.complete(
            status=status,
            metrics=metrics,
            error=error,
            llama_version=llama_version,
        )

    progress_state.record_completion(duration_s)
    if rich_progress is not None and progress_task_id is not None:
        rich_progress.update(progress_task_id, completed=progress_state.completed)
        update_rich_progress(
            rich_progress,
            progress_task_id,
            progress_state,
            description=trial_description(index, total, spec.trial_name),
            trial_started_monotonic=None,
        )

    summary = metric_summary(metrics)
    summary_text = f" ({summary})" if summary else " (no metrics parsed)"
    console.print(f"stored {status}: {outcome.run_id}{summary_text}", markup=False)

    if completed.returncode != 0:
        stderr_tail = tail(completed.stderr or completed.stdout)
        details = f"; stderr tail: {stderr_tail}" if stderr_tail else ""
        raise RunFailed(
            f"{spec.trial_name} failed with exit code {completed.returncode}{details}; "
            f"artifacts: {stdout_path}, {stderr_path}"
        )
    return outcome
