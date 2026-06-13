from __future__ import annotations

import platform
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

from llm_refinery import __version__

MODEL_PROCESS_MARKERS = ("llama", "ollama", "mlx", "python")
PROCESS_EXCLUDE_MARKERS = ("rg", "pi-bash")
SYSCTL_PROFILE_KEYS = (
    "hw.model",
    "hw.machine",
    "hw.memsize",
    "hw.ncpu",
    "hw.physicalcpu",
    "hw.logicalcpu",
    "hw.perflevel0.physicalcpu",
    "hw.perflevel1.physicalcpu",
    "machdep.cpu.brand_string",
)


def is_port_listening(port: int) -> bool:
    """Return True when a TCP port is currently in LISTEN state."""
    lsof_path = shutil.which("lsof")
    if not lsof_path:
        return False

    try:
        result = subprocess.run(
            [lsof_path, f"-iTCP:{port}", "-sTCP:LISTEN", "-n", "-P"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except OSError:
        return False


def get_system_snapshot() -> str:
    """Return a macOS-centric memory and model-process snapshot."""
    snapshot: list[str] = []
    snapshot.extend(_vm_stat_snapshot())
    snapshot.extend(_swap_snapshot())
    snapshot.extend(_process_snapshot())
    return "\n".join(snapshot)


def get_system_profile() -> dict[str, Any]:
    """Return structured host metadata for historical benchmark comparison."""
    sysctl_values = _sysctl_values(SYSCTL_PROFILE_KEYS)
    memsize = _int_or_none(sysctl_values.get("hw.memsize"))
    profile: dict[str, Any] = {
        "schema_version": 1,
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": socket.gethostname(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "macos": _sw_vers(),
        "hardware": {
            "model": sysctl_values.get("hw.model"),
            "machine": sysctl_values.get("hw.machine"),
            "chip": sysctl_values.get("machdep.cpu.brand_string"),
            "memory_bytes": memsize,
            "memory_gb": round(memsize / 1024**3, 1) if memsize is not None else None,
            "ncpu": _int_or_none(sysctl_values.get("hw.ncpu")),
            "physicalcpu": _int_or_none(sysctl_values.get("hw.physicalcpu")),
            "logicalcpu": _int_or_none(sysctl_values.get("hw.logicalcpu")),
            "perflevel0_physicalcpu": _int_or_none(
                sysctl_values.get("hw.perflevel0.physicalcpu")
            ),
            "perflevel1_physicalcpu": _int_or_none(
                sysctl_values.get("hw.perflevel1.physicalcpu")
            ),
        },
        "project": {
            "llm_refinery_version": __version__,
            "git_head": _git_output("rev-parse", "HEAD"),
            "git_dirty": _git_dirty(),
        },
    }
    return _drop_none(profile)


def _sw_vers() -> dict[str, str]:
    sw_vers_path = shutil.which("sw_vers")
    if not sw_vers_path:
        return {}
    try:
        result = subprocess.run(
            [sw_vers_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}

    values: dict[str, str] = {}
    for line in result.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _sysctl_values(keys: tuple[str, ...]) -> dict[str, str]:
    sysctl_path = shutil.which("sysctl")
    if not sysctl_path:
        return {}

    values: dict[str, str] = {}
    for key in keys:
        try:
            result = subprocess.run(
                [sysctl_path, "-n", key],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0 and result.stdout.strip():
            values[key] = result.stdout.strip()
    return values


def _git_output(*args: str) -> str | None:
    git_path = shutil.which("git")
    if not git_path:
        return None
    try:
        result = subprocess.run(
            [git_path, *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _git_dirty() -> bool | None:
    git_path = shutil.which("git")
    if not git_path:
        return None
    try:
        result = subprocess.run(
            [git_path, "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def _int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(child) for key, child in value.items() if child is not None}
    if isinstance(value, list):
        return [_drop_none(child) for child in value]
    return value


def _vm_stat_snapshot() -> list[str]:
    vm_stat_path = shutil.which("vm_stat")
    if not vm_stat_path:
        return []

    try:
        result = subprocess.run([vm_stat_path], capture_output=True, text=True, check=False)
    except OSError as exc:
        return [f"Error running vm_stat: {exc}", ""]

    if not result.stdout:
        return []

    return ["--- vm_stat ---", *result.stdout.splitlines()[:20], ""]


def _swap_snapshot() -> list[str]:
    sysctl_path = shutil.which("sysctl")
    if not sysctl_path:
        return []

    try:
        result = subprocess.run(
            [sysctl_path, "vm.swapusage"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return [f"Error running sysctl vm.swapusage: {exc}", ""]

    if not result.stdout:
        return []

    return ["--- swap ---", result.stdout.strip(), ""]


def _process_snapshot() -> list[str]:
    ps_path = shutil.which("ps")
    if not ps_path:
        return []

    try:
        result = subprocess.run(
            [ps_path, "-axo", "pid,ppid,comm,%cpu,%mem,rss,args"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return [f"Error running ps: {exc}", ""]

    if not result.stdout:
        return []

    filtered_lines = []
    for line in result.stdout.splitlines():
        if any(marker in line for marker in MODEL_PROCESS_MARKERS) and not any(
            marker in line for marker in PROCESS_EXCLUDE_MARKERS
        ):
            filtered_lines.append(line)

    if not filtered_lines:
        return []

    return ["--- process snapshot ---", *filtered_lines, ""]
