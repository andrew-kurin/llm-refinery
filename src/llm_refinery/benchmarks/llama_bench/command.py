from __future__ import annotations

from llm_refinery.benchmarks.llama_bench.command_builder import (
    build_bench_command,
    build_server_command,
    shell_join,
)
from llm_refinery.benchmarks.llama_bench.config import LlamaSweepConfig, expand_trials


def print_plan(config: LlamaSweepConfig, *, kind: str = "bench", limit: int | None = None) -> None:
    trials = expand_trials(config, kind=kind)
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

    total = len(expand_trials(config, kind=kind))
    shown = len(trials)
    print(f"planned {shown} of {total} {kind} command(s)")
