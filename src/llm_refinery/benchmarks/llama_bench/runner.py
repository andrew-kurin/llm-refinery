from __future__ import annotations

import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TaskID

from llm_refinery.bench_parser import parse_llama_bench_metrics
from llm_refinery.benchmarks.llama_bench.command import print_plan
from llm_refinery.benchmarks.llama_bench.process import run_bench_process
from llm_refinery.benchmarks.llama_bench.progress import (
    BenchProgress,
    make_bench_progress,
    trial_description,
    update_rich_progress,
)
from llm_refinery.benchmarks.llama_bench.reporting import metric_summary, tail
from llm_refinery.config import Trial, TuneConfig, expand_trials
from llm_refinery.core.runs import make_run_id, prepare_artifact_dir, record_benchmark_run
from llm_refinery.llama_cmd import build_bench_command, shell_join
from llm_refinery.providers.llama_cpp import detect_llama_version
from llm_refinery.storage import ResultStore, utc_now

PROGRESS_INTERVAL_S = 0.5


class RunFailed(RuntimeError):
    pass


def run_bench(
    config: TuneConfig,
    *,
    limit: int | None = None,
    dry_run: bool = False,
    keep_going: bool = False,
    database_override: str | Path | None = None,
    show_progress: bool = True,
    progress_interval_s: float = PROGRESS_INTERVAL_S,
) -> int:
    trials = expand_trials(config, include_bench_dimensions=True)
    if limit is not None:
        trials = trials[:limit]

    if dry_run:
        print_plan(config, kind="bench", limit=limit)
        return 0

    console = Console()
    if not trials:
        console.print("no trials to run")
        return 0

    database = Path(database_override) if database_override else config.database
    progress_state = BenchProgress(total=len(trials))
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
                    keep_going=keep_going,
                    console=console,
                    progress_state=progress_state,
                    rich_progress=rich_progress,
                    progress_task_id=task_id,
                    progress_interval_s=progress_interval_s,
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
                keep_going=keep_going,
                console=console,
                progress_state=progress_state,
                rich_progress=None,
                progress_task_id=None,
                progress_interval_s=progress_interval_s,
            )
    return 0


def _run_bench_trials(
    config: TuneConfig,
    trials: list[Trial],
    store: ResultStore,
    *,
    keep_going: bool,
    console: Console,
    progress_state: BenchProgress,
    rich_progress: Progress | None,
    progress_task_id: TaskID | None,
    progress_interval_s: float,
) -> None:
    for index, trial in enumerate(trials, start=1):
        try:
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
            )
        except Exception as exc:  # noqa: BLE001 - keep-going needs to persist failures
            if keep_going:
                console.print(f"failed: {trial.name}: {exc}", style="red", markup=False)
                continue
            raise


def _run_one_bench(
    config: TuneConfig,
    trial: Trial,
    store: ResultStore,
    *,
    index: int,
    total: int,
    console: Console,
    progress_state: BenchProgress,
    rich_progress: Progress | None,
    progress_task_id: TaskID | None,
    progress_interval_s: float,
) -> None:
    run_id = make_run_id(trial.key)
    cmd = build_bench_command(config, trial)
    command_text = shell_join(cmd)
    artifact_dir = prepare_artifact_dir(config.database, run_id)
    stdout_path = artifact_dir / "stdout.txt"
    stderr_path = artifact_dir / "stderr.txt"

    if rich_progress is None or progress_task_id is None:
        console.print(f"[{index}/{total}] {trial.name}", markup=False)
    console.print(command_text, style="dim", markup=False, soft_wrap=True)

    started = utc_now()
    monotonic_start = time.perf_counter()
    if rich_progress is not None and progress_task_id is not None:
        update_rich_progress(
            rich_progress,
            progress_task_id,
            progress_state,
            description=trial_description(index, total, trial.name),
            trial_started_monotonic=monotonic_start,
        )

    completed = run_bench_process(
        cmd,
        progress_state=progress_state,
        rich_progress=rich_progress,
        progress_task_id=progress_task_id,
        trial_started_monotonic=monotonic_start,
        progress_interval_s=progress_interval_s,
    )
    ended = utc_now()
    duration_s = time.perf_counter() - monotonic_start

    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")

    metrics = parse_llama_bench_metrics(completed.stdout)
    status = "ok" if completed.returncode == 0 else "failed"
    error = None if completed.returncode == 0 else f"exit code {completed.returncode}"

    record_benchmark_run(
        store,
        run_id=run_id,
        suite=trial.suite,
        trial_name=trial.name,
        status=status,
        started_at=started,
        ended_at=ended,
        duration_s=duration_s,
        command=command_text,
        config_json=trial.as_jsonable(),
        metrics=metrics,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        llama_version=detect_llama_version(config.commands["bench"]),
        error=error,
    )

    progress_state.record_completion(duration_s)
    if rich_progress is not None and progress_task_id is not None:
        rich_progress.update(progress_task_id, completed=progress_state.completed)
        update_rich_progress(
            rich_progress,
            progress_task_id,
            progress_state,
            description=trial_description(index, total, trial.name),
            trial_started_monotonic=None,
        )

    summary = metric_summary(metrics)
    if summary:
        console.print(f"stored {status}: {run_id} ({summary})", markup=False)
    else:
        console.print(f"stored {status}: {run_id} (no metrics parsed)", markup=False)

    if completed.returncode != 0:
        stderr_tail = tail(completed.stderr or completed.stdout)
        details = f"; stderr tail: {stderr_tail}" if stderr_tail else ""
        raise RunFailed(
            f"{trial.name} failed with exit code {completed.returncode}{details}; "
            f"artifacts: {stdout_path}, {stderr_path}"
        )
