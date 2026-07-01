from __future__ import annotations

from llm_refinery.config import TuneConfig, expand_trials
from llm_refinery.llama_cmd import build_bench_command, build_server_command, shell_join


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
