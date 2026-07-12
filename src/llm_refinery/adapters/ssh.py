from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable, Sequence
from importlib.resources import files
from typing import Any

from llm_refinery.core.targets import (
    HOST_ACCESS_LOCAL,
    HostAccess,
    HostDiscovery,
)
from llm_refinery.utils.system import get_system_profile

ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
MAX_PROBE_OUTPUT_CHARS = 2_000_000


class OpenSSHClient:
    """Run the fixed, read-only inventory probe through the system OpenSSH client."""

    def __init__(
        self,
        *,
        runner: ProcessRunner = subprocess.run,
        ssh_executable: str = "ssh",
        local_python: str = sys.executable,
    ) -> None:
        self._runner = runner
        self._ssh_executable = ssh_executable
        self._local_python = local_python

    def command(self, access: HostAccess) -> list[str]:
        if access.access == HOST_ACCESS_LOCAL:
            return [self._local_python, "-I", "-"]
        assert access.destination is not None
        connect_timeout = max(1, math.ceil(access.connect_timeout_s))
        return [
            self._ssh_executable,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={connect_timeout}",
            "--",
            access.destination,
            "python3",
            "-I",
            "-",
        ]

    def collect_host_profile(self, access: HostAccess) -> HostDiscovery:
        if access.access == HOST_ACCESS_LOCAL:
            return HostDiscovery(
                transport=access.access,
                destination=None,
                profile=get_system_profile(),
            )
        probe_source = linux_dgx_probe_source()
        argv = self.command(access)
        try:
            result = self._runner(
                argv,
                input=probe_source,
                capture_output=True,
                text=True,
                check=False,
                timeout=access.command_timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"target host inventory timed out after {access.command_timeout_s:g}s"
            ) from exc
        except OSError as exc:
            raise RuntimeError(f"could not execute target host inventory: {exc}") from exc
        if result.returncode != 0:
            detail = _bounded_error(result.stderr or result.stdout)
            raise RuntimeError(
                f"target host inventory failed with exit code {result.returncode}"
                + (f": {detail}" if detail else "")
            )
        if len(result.stdout) > MAX_PROBE_OUTPUT_CHARS:
            raise RuntimeError(
                f"target host inventory exceeded {MAX_PROBE_OUTPUT_CHARS} characters"
            )
        try:
            profile = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("target host inventory returned invalid JSON") from exc
        if not isinstance(profile, dict):
            raise RuntimeError("target host inventory must return a JSON object")
        if profile.get("schema_version") != 1:
            raise RuntimeError(
                "target host inventory returned unsupported schema_version "
                f"{profile.get('schema_version')!r}"
            )
        return HostDiscovery(
            transport=access.access,
            destination=access.destination,
            profile=_sanitize_profile(profile),
        )


def linux_dgx_probe_source() -> str:
    resource = files("llm_refinery.probes").joinpath("linux_dgx_probe.py")
    return resource.read_text(encoding="utf-8")


def _sanitize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Defense in depth against accidental raw IDs or process data from future probes."""
    forbidden = {"machine_id", "machine-id", "command_line", "cmdline", "environment"}

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): sanitize(item)
                for key, item in value.items()
                if str(key).casefold() not in forbidden
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    return sanitize(profile)


def _bounded_error(value: str, *, limit: int = 2000) -> str:
    return " ".join(value.strip().split())[-limit:]


def command_is_read_only(argv: Sequence[str]) -> bool:
    """Expose the adapter invariant for focused tests and policy checks."""
    return list(argv[-3:]) == ["python3", "-I", "-"] or list(argv[-2:]) == ["-I", "-"]


__all__ = ["OpenSSHClient", "command_is_read_only", "linux_dgx_probe_source"]
