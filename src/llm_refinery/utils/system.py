from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_refinery import __version__

MODEL_PROCESS_MARKERS = (
    "llama",
    "ollama",
    "mlx",
    "python",
    "vllm",
    "trtllm",
    "tensorrt_llm",
    "sglang",
)
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
LINUX_MACHINE_ID_PATHS = (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id"))
DGX_RELEASE_PATHS = (Path("/etc/dgx-release"), Path("/etc/nvidia/dgx-release"))
LINUX_DMI_PATHS = {
    "sys_vendor": Path("/sys/devices/virtual/dmi/id/sys_vendor"),
    "product_name": Path("/sys/devices/virtual/dmi/id/product_name"),
    "product_version": Path("/sys/devices/virtual/dmi/id/product_version"),
    "board_name": Path("/sys/devices/virtual/dmi/id/board_name"),
}


def is_port_listening(port: int) -> bool:
    """Return True when a local TCP port accepts connections.

    ``lsof`` remains the cheapest authoritative check when available. Minimal Linux
    installations, including appliance-style DGX images, may not ship it, so a local
    socket probe is used as a portable fallback.
    """
    lsof_path = shutil.which("lsof")
    if lsof_path:
        try:
            result = subprocess.run(
                [lsof_path, f"-iTCP:{port}", "-sTCP:LISTEN", "-n", "-P"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if result.returncode == 0:
                return True
        except (OSError, subprocess.TimeoutExpired):
            pass

    for host in ("127.0.0.1", "::1"):
        try:
            connection = socket.create_connection((host, port), timeout=0.25)
        except (OSError, ValueError):
            continue
        connection.close()
        return True
    return False


def get_system_snapshot() -> str:
    """Return a concise memory and model-process snapshot for the current host."""
    snapshot: list[str] = []
    if platform.system() == "Linux":
        snapshot.extend(_linux_memory_snapshot())
    else:
        snapshot.extend(_vm_stat_snapshot())
        snapshot.extend(_swap_snapshot())
    snapshot.extend(_process_snapshot())
    return "\n".join(snapshot)


def get_system_profile() -> dict[str, Any]:
    """Return structured, stable host metadata for benchmark comparison."""
    system_name = platform.system()
    hostname = socket.gethostname()
    sysctl_values = _sysctl_values(SYSCTL_PROFILE_KEYS) if system_name == "Darwin" else {}
    linux = _linux_profile() if system_name == "Linux" else {}

    hardware = _hardware_profile(system_name, sysctl_values=sysctl_values, linux=linux)
    profile: dict[str, Any] = {
        "schema_version": 2,
        "captured_at": datetime.now(UTC).isoformat(),
        "hostname": hostname,
        "host_fingerprint": _current_host_fingerprint(
            system_name=system_name,
            hostname=hostname,
            hardware=hardware,
        ),
        "platform": {
            "system": system_name,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_executable": sys.executable,
        },
        "hardware": hardware,
        "project": {
            "llm_refinery_version": __version__,
            "git_head": _git_output("rev-parse", "HEAD"),
            "git_dirty": _git_dirty(),
        },
    }
    if system_name == "Darwin":
        profile["macos"] = _sw_vers()
    if linux:
        profile["linux"] = linux

    nvidia = _nvidia_profile()
    if nvidia:
        profile["nvidia"] = nvidia
    dgx = _dgx_profile(linux)
    if dgx:
        profile["dgx"] = dgx
    return _drop_none(profile)


def host_identity(profile: dict[str, Any]) -> str:
    """Return a stable comparison identity from new or legacy system profiles."""
    explicit = profile.get("host_fingerprint")
    if explicit:
        return str(explicit)

    host = profile.get("host")
    if isinstance(host, dict) and host.get("fingerprint"):
        return str(host["fingerprint"])

    platform_profile = profile.get("platform") or {}
    hardware = profile.get("hardware") or {}
    legacy_identity = {
        "hostname": profile.get("hostname"),
        "system": platform_profile.get("system") if isinstance(platform_profile, dict) else None,
        "machine": platform_profile.get("machine") if isinstance(platform_profile, dict) else None,
        "hardware_model": hardware.get("model") if isinstance(hardware, dict) else None,
        "hardware_machine": hardware.get("machine") if isinstance(hardware, dict) else None,
        "chip": hardware.get("chip") if isinstance(hardware, dict) else None,
        "memory_bytes": hardware.get("memory_bytes") if isinstance(hardware, dict) else None,
        "memory_gb": hardware.get("memory_gb") if isinstance(hardware, dict) else None,
    }
    if not any(value not in (None, "") for value in legacy_identity.values()):
        return "unknown-host"
    return f"legacy-{_stable_digest(legacy_identity)}"


def _hardware_profile(
    system_name: str,
    *,
    sysctl_values: dict[str, str],
    linux: dict[str, Any],
) -> dict[str, Any]:
    if system_name == "Linux":
        proc = linux.get("proc") or {}
        cpu = proc.get("cpuinfo") or {}
        memory = proc.get("meminfo") or {}
        dmi = linux.get("dmi") or {}
        memsize = _kilobytes_to_bytes(memory.get("mem_total_kb"))
        return _drop_none(
            {
                "model": dmi.get("product_name") or linux.get("device_tree_model"),
                "vendor": dmi.get("sys_vendor"),
                "product_version": dmi.get("product_version"),
                "board_name": dmi.get("board_name"),
                "machine": platform.machine(),
                "chip": cpu.get("model_name") or cpu.get("hardware"),
                "cpu_vendor": cpu.get("vendor_id"),
                "memory_bytes": memsize,
                "memory_gb": round(memsize / 1024**3, 1) if memsize is not None else None,
                "ncpu": cpu.get("logical_cpus"),
                "physicalcpu": cpu.get("physical_cores"),
                "logicalcpu": cpu.get("logical_cpus"),
                "physical_packages": cpu.get("physical_packages"),
            }
        )

    memsize = _int_or_none(sysctl_values.get("hw.memsize"))
    return _drop_none(
        {
            "model": sysctl_values.get("hw.model"),
            "machine": sysctl_values.get("hw.machine"),
            "chip": sysctl_values.get("machdep.cpu.brand_string"),
            "memory_bytes": memsize,
            "memory_gb": round(memsize / 1024**3, 1) if memsize is not None else None,
            "ncpu": _int_or_none(sysctl_values.get("hw.ncpu")),
            "physicalcpu": _int_or_none(sysctl_values.get("hw.physicalcpu")),
            "logicalcpu": _int_or_none(sysctl_values.get("hw.logicalcpu")),
            "perflevel0_physicalcpu": _int_or_none(sysctl_values.get("hw.perflevel0.physicalcpu")),
            "perflevel1_physicalcpu": _int_or_none(sysctl_values.get("hw.perflevel1.physicalcpu")),
        }
    )


def _current_host_fingerprint(*, system_name: str, hostname: str, hardware: dict[str, Any]) -> str:
    machine_identifier = _machine_identifier(system_name)
    if machine_identifier:
        identity: dict[str, Any] = {
            "version": 1,
            "system": system_name,
            "machine_identifier": machine_identifier,
        }
    else:
        identity = {
            "version": 1,
            "system": system_name,
            "hostname": hostname,
            "machine": hardware.get("machine"),
            "model": hardware.get("model"),
            "chip": hardware.get("chip"),
            "memory_bytes": hardware.get("memory_bytes"),
        }
    return f"host-{_stable_digest(identity)}"


def _machine_identifier(system_name: str) -> str | None:
    if system_name == "Linux":
        for path in LINUX_MACHINE_ID_PATHS:
            value = _read_text(path, max_chars=256)
            if value:
                return value.strip()
        return None

    if system_name == "Darwin":
        ioreg_path = shutil.which("ioreg")
        if not ioreg_path:
            return None
        output = _command_output([ioreg_path, "-rd1", "-c", "IOPlatformExpertDevice"])
        match = re.search(r'"IOPlatformUUID"\s*=\s*"([^"]+)"', output or "")
        return match.group(1) if match else None
    return None


def _stable_digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _linux_profile() -> dict[str, Any]:
    cpuinfo = _parse_cpuinfo(_read_text(Path("/proc/cpuinfo"), max_chars=2_000_000) or "")
    meminfo = _parse_meminfo(_read_text(Path("/proc/meminfo"), max_chars=100_000) or "")
    os_release = _parse_release_file(_read_text(Path("/etc/os-release"), max_chars=100_000) or "")
    dmi = {
        key: value
        for key, path in LINUX_DMI_PATHS.items()
        if (value := _read_text(path, max_chars=4096))
    }
    device_tree_model = _read_text(Path("/proc/device-tree/model"), max_chars=4096)
    return _drop_none(
        {
            "os_release": os_release,
            "proc": {"cpuinfo": cpuinfo, "meminfo": meminfo},
            "dmi": dmi,
            "device_tree_model": device_tree_model,
        }
    )


def _parse_cpuinfo(text: str) -> dict[str, Any]:
    records: list[dict[str, str]] = []
    for block in text.split("\n\n"):
        record: dict[str, str] = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                record[key.strip().casefold()] = value.strip()
        if record:
            records.append(record)

    first_value: dict[str, str] = {}
    for record in records:
        for key, value in record.items():
            first_value.setdefault(key, value)

    physical_core_ids = {
        (record["physical id"], record["core id"])
        for record in records
        if "physical id" in record and "core id" in record
    }
    physical_packages = {record["physical id"] for record in records if "physical id" in record}
    logical_cpus = sum("processor" in record for record in records)
    processor_name = first_value.get("processor")
    if processor_name and processor_name.isdigit():
        processor_name = None
    return _drop_none(
        {
            "model_name": first_value.get("model name") or processor_name,
            "hardware": first_value.get("hardware"),
            "vendor_id": first_value.get("vendor_id") or first_value.get("cpu implementer"),
            "cpu_architecture": first_value.get("cpu architecture"),
            "cpu_part": first_value.get("cpu part"),
            "cpu_revision": first_value.get("cpu revision"),
            "logical_cpus": logical_cpus or None,
            "physical_cores": len(physical_core_ids) or None,
            "physical_packages": len(physical_packages) or None,
        }
    )


def _parse_meminfo(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        key, separator, value_text = line.partition(":")
        if not separator:
            continue
        parts = value_text.split()
        if not parts:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", key.strip()).casefold()
        if len(parts) > 1 and parts[1].casefold() == "kb":
            normalized += "_kb"
        values[normalized] = value
    return values


def _parse_release_file(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            key, separator, value = stripped.partition(":")
        if separator:
            values[key.strip().casefold()] = value.strip().strip("\"'")
    return values


def _nvidia_profile() -> dict[str, Any]:
    profile: dict[str, Any] = {}
    smi_path = shutil.which("nvidia-smi")
    if smi_path:
        query_fields = ("index", "name", "uuid", "driver_version", "memory.total", "pci.bus_id")
        query = f"--query-gpu={','.join(query_fields)}"
        output = _command_output([smi_path, query, "--format=csv,noheader,nounits"], timeout=5)
        gpus = _parse_nvidia_gpus(output or "", query_fields)
        if gpus:
            profile["gpus"] = gpus
            profile["driver_version"] = gpus[0].get("driver_version")

        summary = _command_output([smi_path], timeout=5)
        driver_match = re.search(r"Driver Version:\s*([0-9.]+)", summary or "")
        if driver_match and not profile.get("driver_version"):
            profile["driver_version"] = driver_match.group(1)
        cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", summary or "")
        if cuda_match:
            # This is the driver's maximum supported CUDA level, not a loaded
            # runtime. Keep the old key as a schema-v2 compatibility alias.
            profile["cuda_driver_supported_version"] = cuda_match.group(1)
            profile["cuda_runtime_version"] = cuda_match.group(1)
        if not gpus:
            listed_gpus = _parse_nvidia_gpu_list(_command_output([smi_path, "-L"], timeout=5) or "")
            if listed_gpus:
                profile["gpus"] = listed_gpus

    driver_text = _read_text(Path("/proc/driver/nvidia/version"), max_chars=16_384)
    if driver_text and not profile.get("driver_version"):
        driver_match = re.search(r"Kernel Module\s+([0-9.]+)", driver_text)
        profile["driver_version"] = driver_match.group(1) if driver_match else driver_text

    nvcc_path = shutil.which("nvcc")
    if nvcc_path:
        nvcc_output = _command_output([nvcc_path, "--version"], timeout=5)
        toolkit_match = re.search(r"\brelease\s+([0-9.]+)", nvcc_output or "")
        if toolkit_match:
            profile["cuda_toolkit_version"] = toolkit_match.group(1)
        elif nvcc_output:
            profile["nvcc_version"] = nvcc_output.splitlines()[-1]
    return _drop_none(profile)


def _parse_nvidia_gpus(text: str, fields: tuple[str, ...]) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    for values in csv.reader(text.splitlines()):
        if len(values) != len(fields):
            continue
        gpu: dict[str, Any] = {
            field.replace(".", "_"): _none_if_na(value)
            for field, value in zip(fields, values, strict=True)
        }
        memory = gpu.get("memory_total")
        if isinstance(memory, str):
            gpu["memory_total_mb"] = _float_or_none(memory)
            del gpu["memory_total"]
        index = gpu.get("index")
        if isinstance(index, str):
            gpu["index"] = _int_or_none(index)
        gpus.append(_drop_none(gpu))
    return gpus


def _parse_nvidia_gpu_list(text: str) -> list[dict[str, Any]]:
    gpus: list[dict[str, Any]] = []
    pattern = re.compile(r"^GPU\s+(\d+):\s+(.+?)\s+\(UUID:\s*([^\)]+)\)$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            gpus.append(
                {"index": int(match.group(1)), "name": match.group(2), "uuid": match.group(3)}
            )
    return gpus


def _dgx_profile(linux: dict[str, Any]) -> dict[str, Any]:
    release_values: dict[str, str] = {}
    release_path: str | None = None
    for path in DGX_RELEASE_PATHS:
        text = _read_text(path, max_chars=100_000)
        if text:
            release_values = _parse_release_file(text)
            release_path = str(path)
            break

    dmi = linux.get("dmi") or {}
    model = dmi.get("product_name") or linux.get("device_tree_model")
    is_dgx = bool(release_values) or (isinstance(model, str) and "dgx" in model.casefold())
    if not is_dgx:
        return {}
    return _drop_none({"model": model, "release_file": release_path, "release": release_values})


def _sw_vers() -> dict[str, str]:
    sw_vers_path = shutil.which("sw_vers")
    if not sw_vers_path:
        return {}
    output = _command_output([sw_vers_path])
    if not output:
        return {}

    values: dict[str, str] = {}
    for line in output.splitlines():
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
        output = _command_output([sysctl_path, "-n", key])
        if output:
            values[key] = output
    return values


def _command_output(command: list[str], *, timeout: int = 2) -> str | None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def _git_output(*args: str) -> str | None:
    git_path = shutil.which("git")
    if not git_path:
        return None
    return _command_output([git_path, *args])


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


def _read_text(path: Path, *, max_chars: int) -> str | None:
    try:
        value = path.read_text(encoding="utf-8", errors="replace")[:max_chars]
    except OSError:
        return None
    value = value.replace("\x00", "").strip()
    return value or None


def _kilobytes_to_bytes(value: object) -> int | None:
    return value * 1024 if isinstance(value, int) else None


def _int_or_none(value: object) -> int | None:
    if not isinstance(value, str | bytes | bytearray | int | float):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: object) -> float | None:
    if not isinstance(value, str | bytes | bytearray | int | float):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _none_if_na(value: str) -> str | None:
    stripped = value.strip()
    return None if stripped.casefold() in {"", "n/a", "[not supported]"} else stripped


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
    output = _command_output([vm_stat_path])
    return ["--- vm_stat ---", *output.splitlines()[:20], ""] if output else []


def _swap_snapshot() -> list[str]:
    sysctl_path = shutil.which("sysctl")
    if not sysctl_path:
        return []
    output = _command_output([sysctl_path, "vm.swapusage"])
    return ["--- swap ---", output, ""] if output else []


def _linux_memory_snapshot() -> list[str]:
    output = _read_text(Path("/proc/meminfo"), max_chars=100_000)
    return ["--- /proc/meminfo ---", *output.splitlines()[:30], ""] if output else []


def _process_snapshot() -> list[str]:
    ps_path = shutil.which("ps")
    if not ps_path:
        return []
    output = _command_output([ps_path, "-axo", "pid,ppid,comm,%cpu,%mem,rss,args"])
    if not output:
        return []

    filtered_lines = []
    for line in output.splitlines():
        normalized = line.casefold()
        if any(marker in normalized for marker in MODEL_PROCESS_MARKERS) and not any(
            marker in normalized for marker in PROCESS_EXCLUDE_MARKERS
        ):
            filtered_lines.append(line)
    return ["--- process snapshot ---", *filtered_lines, ""] if filtered_lines else []
