from __future__ import annotations

import time

from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TaskID, TextColumn

TRIAL_DESCRIPTION_WIDTH = 72


class BenchProgress:
    def __init__(self, total: int, *, started_monotonic: float | None = None):
        self.total = total
        self.completed = 0
        self.completed_durations_s: list[float] = []
        self.started_monotonic = (
            time.perf_counter() if started_monotonic is None else started_monotonic
        )

    @property
    def average_duration_s(self) -> float | None:
        if not self.completed_durations_s:
            return None
        return sum(self.completed_durations_s) / len(self.completed_durations_s)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_monotonic

    def record_completion(self, duration_s: float) -> None:
        self.completed += 1
        self.completed_durations_s.append(duration_s)

    def eta_after_completed_s(self) -> float | None:
        average = self.average_duration_s
        if average is None:
            return None
        return max(self.total - self.completed, 0) * average

    def eta_during_current_s(self, current_elapsed_s: float) -> float | None:
        average = self.average_duration_s
        if average is None:
            return None
        current_remaining_s = max(average - current_elapsed_s, 0.0)
        remaining_after_current = max(self.total - self.completed - 1, 0)
        return current_remaining_s + remaining_after_current * average


def make_bench_progress(console: Console) -> Progress:
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}", style="progress.description", markup=False),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.percentage:>5.1f}%"),
        TextColumn("elapsed {task.fields[suite_elapsed]}"),
        TextColumn("current {task.fields[current_elapsed]}"),
        TextColumn("avg {task.fields[average]}"),
        TextColumn("eta {task.fields[eta]}"),
        console=console,
        transient=False,
        refresh_per_second=4,
        redirect_stdout=False,
        redirect_stderr=False,
    )


def update_rich_progress(
    rich_progress: Progress,
    progress_task_id: TaskID,
    progress_state: BenchProgress,
    *,
    description: str | None = None,
    trial_started_monotonic: float | None,
) -> None:
    current_elapsed_s = (
        time.perf_counter() - trial_started_monotonic
        if trial_started_monotonic is not None
        else None
    )
    if current_elapsed_s is not None:
        eta_s = progress_state.eta_during_current_s(current_elapsed_s)
        current_elapsed_text = format_duration(current_elapsed_s)
    else:
        eta_s = progress_state.eta_after_completed_s()
        current_elapsed_text = "-"

    update_kwargs: dict[str, object] = {
        "completed": progress_state.completed,
        "suite_elapsed": format_duration(progress_state.elapsed_s),
        "current_elapsed": current_elapsed_text,
        "average": _duration_or_unknown(progress_state.average_duration_s),
        "eta": _eta_text(eta_s, progress_state),
    }
    if description is not None:
        update_kwargs["description"] = description
    rich_progress.update(progress_task_id, **update_kwargs)


def trial_description(index: int, total: int, trial_name: str) -> str:
    return f"{index}/{total} {_truncate(trial_name, TRIAL_DESCRIPTION_WIDTH)}"


def format_duration(seconds: float) -> str:
    if 0 < seconds < 1:
        return "<1s"
    seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _duration_or_unknown(seconds: float | None) -> str:
    return format_duration(seconds) if seconds is not None else "unknown"


def _eta_text(seconds: float | None, progress_state: BenchProgress) -> str:
    if progress_state.completed >= progress_state.total:
        return "done"
    return format_duration(seconds) if seconds is not None else "unknown"


def _truncate(value: str, width: int) -> str:
    if len(value) <= width:
        return value
    return f"{value[: width - 1]}…"
