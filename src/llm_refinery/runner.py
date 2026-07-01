from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, TaskID

from llm_refinery.assets import ensure_mtp_head
from llm_refinery.bench_parser import parse_llama_bench_metrics
from llm_refinery.benchmarks.llama_bench.progress import (
    BenchProgress,
    format_duration,
)
from llm_refinery.benchmarks.llama_bench.progress import (
    make_bench_progress as _make_bench_progress,
)
from llm_refinery.benchmarks.llama_bench.progress import (
    trial_description as _trial_description,
)
from llm_refinery.benchmarks.llama_bench.progress import (
    update_rich_progress as _update_rich_progress,
)
from llm_refinery.config import Trial, TuneConfig, expand_trials
from llm_refinery.core.runs import make_run_id, prepare_artifact_dir, record_benchmark_run
from llm_refinery.llama_cmd import (
    build_bench_command,
    build_server_command,
    effective_params,
    shell_join,
)
from llm_refinery.storage import ResultStore, utc_now

__all__ = ["BenchProgress", "format_duration", "launch_server", "print_plan", "run_bench"]

PROGRESS_INTERVAL_S = 0.5
class RunFailed(RuntimeError):
    pass


def print_plan(config: TuneConfig, *, kind: str = "bench", limit: int | None = None) -> None:
    trials = expand_trials(config, include_bench_dimensions=(kind == "bench"))
    if limit is not None:
        trials = trials[:limit]

    for index, trial in enumerate(trials):
        if kind == "bench":
            cmd = build_bench_command(config, trial)
        else:
            cmd = build_server_command(config, trial)
        print(f"# [{index}] {trial.name}")
        print(shell_join(cmd))
        print()

    total = len(expand_trials(config, include_bench_dimensions=(kind == "bench")))
    shown = len(trials)
    print(f"planned {shown} of {total} {kind} command(s)")


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
            with _make_bench_progress(console) as rich_progress:
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
                _update_rich_progress(
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


def launch_server(config: TuneConfig, *, index: int = 0, dry_run: bool = False) -> int:
    trials = expand_trials(config, include_bench_dimensions=False)
    if index < 0 or index >= len(trials):
        raise IndexError(f"server trial index {index} outside 0..{len(trials) - 1}")

    trial = trials[index]
    cmd = build_server_command(config, trial)
    print(f"# [{index}] {trial.name}")
    print(shell_join(cmd))
    if dry_run:
        return 0

    prepare_server_assets(config, trial)

    env = os.environ.copy()
    env.update(config.server.env)
    completed = subprocess.run(cmd, env=env, check=False)  # noqa: S603 - command is user config
    return completed.returncode


def prepare_server_assets(config: TuneConfig, trial: Trial) -> None:
    params = effective_params(trial.params, config.server.params, config.server.omit_params)
    mtp_head = params.get("mtp_head")
    if mtp_head is None:
        return
    spec = ensure_mtp_head(mtp_head)
    print(f"MTP head ready: {spec.path}")


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
        _update_rich_progress(
            rich_progress,
            progress_task_id,
            progress_state,
            description=_trial_description(index, total, trial.name),
            trial_started_monotonic=monotonic_start,
        )

    completed = _run_process(
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
        _update_rich_progress(
            rich_progress,
            progress_task_id,
            progress_state,
            description=_trial_description(index, total, trial.name),
            trial_started_monotonic=None,
        )

    metric_summary = _metric_summary(metrics)
    if metric_summary:
        console.print(f"stored {status}: {run_id} ({metric_summary})", markup=False)
    else:
        console.print(f"stored {status}: {run_id} (no metrics parsed)", markup=False)

    if completed.returncode != 0:
        stderr_tail = _tail(completed.stderr or completed.stdout)
        details = f"; stderr tail: {stderr_tail}" if stderr_tail else ""
        raise RunFailed(
            f"{trial.name} failed with exit code {completed.returncode}{details}; "
            f"artifacts: {stdout_path}, {stderr_path}"
        )


def _metric_summary(metrics: dict[str, float], *, limit: int = 4) -> str:
    if not metrics:
        return ""

    preferred = [
        (key, value) for key, value in metrics.items() if key.endswith(".tokens_per_second")
    ]
    remaining = [(key, value) for key, value in metrics.items() if (key, value) not in preferred]
    selected = [*preferred, *remaining][:limit]
    return ", ".join(f"{key}={value:.3f}" for key, value in selected)


def _tail(output: str, *, lines: int = 6) -> str:
    stripped_lines = [line.strip() for line in output.splitlines() if line.strip()]
    return " | ".join(stripped_lines[-lines:])


def _run_process(
    cmd: list[str],
    *,
    progress_state: BenchProgress,
    rich_progress: Progress | None,
    progress_task_id: TaskID | None,
    trial_started_monotonic: float,
    progress_interval_s: float,
) -> subprocess.CompletedProcess[str]:
    if rich_progress is None or progress_task_id is None:
        return subprocess.run(  # noqa: S603 - command is user config
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env=os.environ.copy(),
        )

    process = subprocess.Popen(  # noqa: S603 - command is user config
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=os.environ.copy(),
    )

    poll_interval_s = max(progress_interval_s, 0.1)
    while True:
        try:
            stdout, stderr = process.communicate(timeout=poll_interval_s)
            _update_rich_progress(
                rich_progress,
                progress_task_id,
                progress_state,
                trial_started_monotonic=trial_started_monotonic,
            )
            return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            _update_rich_progress(
                rich_progress,
                progress_task_id,
                progress_state,
                trial_started_monotonic=trial_started_monotonic,
            )
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            raise


def detect_llama_version(command: list[str]) -> str | None:
    candidates: list[list[str]] = []
    if command:
        candidates.append([command[0], "--version"])
        candidates.append(command + ["--version"])

    for candidate in candidates:
        try:
            completed = subprocess.run(  # noqa: S603 - fixed probe derived from user command
                candidate,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        output = (completed.stdout or completed.stderr).strip()
        if completed.returncode == 0 and output:
            return output.splitlines()[0]
    return None
