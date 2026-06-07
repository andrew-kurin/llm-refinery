from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    returncode: int


def run_command(
    cmd: str | list[str],
    check: bool = True,
    capture_output: bool = True,
    env: dict[str, Any] | None = None,
) -> CommandResult:
    """Execute a command and return stdout, stderr, and return code."""
    argv = shlex.split(cmd) if isinstance(cmd, str) else cmd

    try:
        result = subprocess.run(
            argv,
            capture_output=capture_output,
            text=True,
            check=check,
            env=env,
        )
        return CommandResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            returncode=result.returncode,
        )
    except subprocess.CalledProcessError as exc:
        if check:
            raise
        return CommandResult(
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            returncode=exc.returncode,
        )
    except OSError as exc:
        if check:
            raise
        return CommandResult(stdout="", stderr=str(exc), returncode=1)
