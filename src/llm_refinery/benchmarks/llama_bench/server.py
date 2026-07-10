from __future__ import annotations

import os
import subprocess

from llm_refinery.benchmarks.llama_bench.assets import ensure_mtp_head
from llm_refinery.benchmarks.llama_bench.command_builder import build_server_command, shell_join
from llm_refinery.benchmarks.llama_bench.config import LlamaSweepConfig, LlamaTrial, expand_trials


def launch_server(config: LlamaSweepConfig, *, index: int = 0, dry_run: bool = False) -> int:
    trials = expand_trials(config, kind="server")
    if index < 0 or index >= len(trials):
        raise IndexError(f"server trial index {index} outside 0..{len(trials) - 1}")

    trial = trials[index]
    cmd = build_server_command(config, trial)
    print(f"# [{index}] {trial.name}")
    print(shell_join(cmd))
    if dry_run:
        return 0

    prepare_server_assets(trial)

    env = os.environ.copy()
    env.update(config.server.env)
    completed = subprocess.run(cmd, env=env, check=False)  # noqa: S603 - command is user config
    return completed.returncode


def prepare_server_assets(trial: LlamaTrial) -> None:
    mtp_head = trial.params.get("mtp_head")
    if mtp_head is None:
        return
    spec = ensure_mtp_head(mtp_head)
    print(f"MTP head ready: {spec.path}")


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
