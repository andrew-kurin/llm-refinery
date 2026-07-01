from __future__ import annotations

import os
import subprocess

from llm_refinery.assets import ensure_mtp_head
from llm_refinery.config import Trial, TuneConfig, expand_trials
from llm_refinery.llama_cmd import build_server_command, effective_params, shell_join


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
