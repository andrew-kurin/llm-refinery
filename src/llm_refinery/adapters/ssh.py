from __future__ import annotations

import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
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
PROCESS_TERMINATION_GRACE_S = 0.5


class OpenSSHClient:
    """Run the fixed, read-only inventory probe through the system OpenSSH client."""

    def __init__(
        self,
        *,
        runner: ProcessRunner | None = None,
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
            if self._runner is None:
                result = _run_bounded_process(
                    argv,
                    input_text=probe_source,
                    timeout_s=access.command_timeout_s,
                )
            else:
                # Preserve the injected subprocess.run-compatible seam for focused
                # tests. Production collection uses the streaming implementation.
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
        if _encoded_size(result.stdout) + _encoded_size(result.stderr) > MAX_PROBE_OUTPUT_CHARS:
            raise RuntimeError(
                f"target host inventory exceeded {MAX_PROBE_OUTPUT_CHARS} characters"
            )
        if result.returncode != 0:
            detail = _bounded_error(result.stderr or result.stdout)
            raise RuntimeError(
                f"target host inventory failed with exit code {result.returncode}"
                + (f": {detail}" if detail else "")
            )
        try:
            profile = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("target host inventory returned invalid JSON") from exc
        if not isinstance(profile, dict):
            raise RuntimeError("target host inventory must return a JSON object")
        schema_version = profile.get("schema_version")
        if type(schema_version) is not int or schema_version != 1:
            raise RuntimeError(
                f"target host inventory returned unsupported schema_version {schema_version!r}"
            )
        return HostDiscovery(
            transport=access.access,
            destination=access.destination,
            profile=_sanitize_profile(profile),
        )


def _run_bounded_process(
    argv: Sequence[str],
    *,
    input_text: str,
    timeout_s: float,
) -> subprocess.CompletedProcess[str]:
    """Collect a child process without allowing either output pipe to grow unbounded."""
    process = subprocess.Popen(  # noqa: S603 - argv is fixed by command().
        argv,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    selector = selectors.DefaultSelector()
    input_bytes = input_text.encode("utf-8")
    input_offset = 0
    output = bytearray()
    error = bytearray()
    deadline = time.monotonic() + timeout_s
    completed = False
    try:
        for pipe in (process.stdin, process.stdout, process.stderr):
            os.set_blocking(pipe.fileno(), False)
        selector.register(process.stdin, selectors.EVENT_WRITE, "stdin")
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            events = selector.select(remaining)
            if not events:
                raise subprocess.TimeoutExpired(argv, timeout_s)
            for key, _ in events:
                if key.data == "stdin":
                    try:
                        written = os.write(key.fd, input_bytes[input_offset:])
                    except BrokenPipeError:
                        written = len(input_bytes) - input_offset
                    except BlockingIOError:
                        continue
                    input_offset += written
                    if input_offset >= len(input_bytes):
                        selector.unregister(key.fileobj)
                        process.stdin.close()
                    continue
                remaining_capacity = MAX_PROBE_OUTPUT_CHARS - len(output) - len(error) + 1
                try:
                    chunk = os.read(key.fd, min(64 * 1024, remaining_capacity))
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                destination = output if key.data == "stdout" else error
                destination.extend(chunk)
                if len(output) + len(error) > MAX_PROBE_OUTPUT_CHARS:
                    raise RuntimeError(
                        f"target host inventory exceeded {MAX_PROBE_OUTPUT_CHARS} characters"
                    )

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout_s)
        try:
            returncode = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired(argv, timeout_s) from exc
        completed = True
        return subprocess.CompletedProcess(
            argv,
            returncode,
            stdout=output.decode("utf-8", errors="replace"),
            stderr=error.decode("utf-8", errors="replace"),
        )
    finally:
        selector.close()
        for pipe in (process.stdin, process.stdout, process.stderr):
            pipe.close()
        if not completed:
            _terminate_process_group(process)


def _terminate_process_group(process: subprocess.Popen[bytes]) -> None:
    """Stop SSH and any ProxyCommand/ProxyJump helpers started in its session."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.terminate()
    with suppress(subprocess.TimeoutExpired):
        process.wait(timeout=PROCESS_TERMINATION_GRACE_S)

    # The SSH leader can exit before a helper. Kill any remaining members of
    # the dedicated group even when wait() above has already reaped the leader.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        if process.poll() is None:
            process.kill()
    if process.poll() is None:
        process.wait()


def _encoded_size(value: str) -> int:
    return len(value.encode("utf-8"))


def linux_dgx_probe_source() -> str:
    resource = files("llm_refinery.probes").joinpath("linux_dgx_probe.py")
    return resource.read_text(encoding="utf-8")


def _sanitize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Defense in depth against accidental raw IDs or process data from future probes."""
    forbidden = {
        "machineid",
        "machineidentifier",
        "hardwareuuid",
        "productuuid",
        "commandline",
        "cmdline",
        "environment",
    }

    def forbidden_key(key: Any) -> bool:
        normalized = re.sub(r"[^a-z0-9]", "", str(key).casefold())
        return normalized in forbidden

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): sanitize(item) for key, item in value.items() if not forbidden_key(key)
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
