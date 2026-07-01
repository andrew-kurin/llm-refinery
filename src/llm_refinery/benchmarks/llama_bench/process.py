from __future__ import annotations

import os
import subprocess

from rich.progress import Progress, TaskID

from llm_refinery.benchmarks.llama_bench.progress import BenchProgress, update_rich_progress


def run_bench_process(
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
            update_rich_progress(
                rich_progress,
                progress_task_id,
                progress_state,
                trial_started_monotonic=trial_started_monotonic,
            )
            return subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        except subprocess.TimeoutExpired:
            update_rich_progress(
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
