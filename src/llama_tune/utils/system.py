from __future__ import annotations

import shutil
import subprocess

MODEL_PROCESS_MARKERS = ("llama", "ollama", "mlx", "python")
PROCESS_EXCLUDE_MARKERS = ("rg", "pi-bash")


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
