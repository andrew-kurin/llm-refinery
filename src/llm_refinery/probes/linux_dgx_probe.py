"""A standalone, standard-library-only Linux/DGX inventory probe.

This file is shipped to a target over stdin and executed using ``python3 -I -``.
Keep imports standard-library-only and all operations read-only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from datetime import datetime, timezone

SCHEMA_VERSION = 1
MAX_SMALL_FILE_BYTES = 100_000
MACHINE_ID_PATHS = ("/etc/machine-id", "/var/lib/dbus/machine-id")
HARDWARE_UUID_PATHS = ("/sys/devices/virtual/dmi/id/product_uuid",)
DMI_PATHS = {
    "sys_vendor": "/sys/devices/virtual/dmi/id/sys_vendor",
    "product_name": "/sys/devices/virtual/dmi/id/product_name",
    "product_version": "/sys/devices/virtual/dmi/id/product_version",
    "board_name": "/sys/devices/virtual/dmi/id/board_name",
}
DGX_RELEASE_PATHS = ("/etc/dgx-release", "/etc/nvidia/dgx-release")
OPTIONAL_GPU_FIELDS = {
    "memory.total": "reported_device_memory_mib",
    "memory.used": "reported_device_memory_used_mib",
    "memory.free": "reported_device_memory_free_mib",
    "utilization.gpu": "utilization_gpu_percent",
    "utilization.memory": "utilization_memory_percent",
    "temperature.gpu": "temperature_gpu_c",
    "power.draw": "power_draw_w",
    "pstate": "pstate",
    "clocks.current.sm": "clocks_sm_mhz",
}


def _read_text(path, limit=MAX_SMALL_FILE_BYTES):
    try:
        with open(path, "rb") as handle:
            return handle.read(limit).decode("utf-8", errors="replace").strip("\x00\n ")
    except (OSError, ValueError):
        return None


def _parse_key_values(text):
    values = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            key, separator, value = stripped.partition(":")
        if separator:
            values[key.strip().lower()] = value.strip().strip("\"'")
    return values


def _parse_meminfo(text):
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            continue
        parts = value.split()
        if not parts:
            continue
        try:
            number = int(parts[0])
        except ValueError:
            continue
        normalized = key.strip().lower()
        if len(parts) > 1 and parts[1].lower() == "kb":
            normalized += "_kb"
        values[normalized] = number
    return values


def _cpu_profile(text):
    records = []
    for block in text.split("\n\n"):
        record = {}
        for line in block.splitlines():
            key, separator, value = line.partition(":")
            if separator:
                record[key.strip().lower()] = value.strip()
        if record:
            records.append(record)
    first = {}
    for record in records:
        for key, value in record.items():
            first.setdefault(key, value)
    processor = first.get("model name") or first.get("hardware") or first.get("processor")
    if processor and processor.isdigit():
        processor = None
    return _drop_none(
        {
            "model_name": processor,
            "vendor_id": first.get("vendor_id") or first.get("cpu implementer"),
            "architecture": first.get("cpu architecture"),
            "part": first.get("cpu part"),
            "logical_cpus": sum("processor" in record for record in records) or None,
        }
    )


def _command(argv, timeout=5):
    try:
        result = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _nvidia_profile():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return None
    fields = ("index", "name", "uuid", "driver_version", "pci.bus_id")
    output = _command(
        [
            executable,
            "--query-gpu=" + ",".join(fields),
            "--format=csv,noheader,nounits",
        ],
        timeout=2,
    )
    summary = _command([executable], timeout=2) or ""
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", summary)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", summary)
    driver_supported_cuda = cuda_match.group(1) if cuda_match else None
    profile = _drop_none(
        {
            "available": True,
            "query_supported": output is not None,
            "driver_version": driver_match.group(1) if driver_match else None,
            # nvidia-smi reports the newest CUDA version supported by the
            # installed driver, not the CUDA runtime loaded by this process.
            "cuda_driver_supported_version": driver_supported_cuda,
            # Retain the original field for consumers of probe schema v1.
            "cuda_runtime_version": driver_supported_cuda,
        }
    )
    nvcc = shutil.which("nvcc")
    if nvcc:
        nvcc_output = _command([nvcc, "--version"], timeout=2) or ""
        toolkit_match = re.search(r"\brelease\s+([0-9.]+)", nvcc_output)
        if toolkit_match:
            profile["cuda_toolkit_version"] = toolkit_match.group(1)
    if output is None:
        return profile
    gpus = []
    for row in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(row) != len(fields):
            continue
        gpus.append(
            _drop_none(
                {
                    "index": row[0].strip(),
                    "name": row[1].strip(),
                    "uuid": row[2].strip(),
                    "driver_version": row[3].strip(),
                    "pci_bus_id": row[4].strip(),
                }
            )
        )
    profile["gpus"] = gpus
    _add_optional_gpu_fields(executable, gpus)
    if gpus and not profile.get("driver_version"):
        profile["driver_version"] = gpus[0].get("driver_version")
    return profile


def _add_optional_gpu_fields(executable, gpus):
    by_index = {str(gpu.get("index")): gpu for gpu in gpus}
    for field, output_key in OPTIONAL_GPU_FIELDS.items():
        output = _command(
            [
                executable,
                "--query-gpu=index," + field,
                "--format=csv,noheader,nounits",
            ],
            timeout=1,
        )
        if output is None:
            continue
        for row in csv.reader(output.splitlines(), skipinitialspace=True):
            if len(row) != 2 or row[0].strip() not in by_index:
                continue
            raw_value = row[1].strip()
            if raw_value.lower() in {"", "n/a", "[not supported]", "not supported"}:
                continue
            value = raw_value
            if output_key != "pstate":
                try:
                    value = float(raw_value)
                except ValueError:
                    continue
            by_index[row[0].strip()][output_key] = value


def _usable_hardware_uuid(value):
    if not value:
        return None
    normalized = value.strip().lower()
    compact = normalized.replace("-", "")
    if (
        len(compact) < 16
        or re.fullmatch(r"[0-9a-f]+", compact) is None
        or set(compact) in ({"0"}, {"f"})
        or normalized in {"none", "unknown", "not specified"}
    ):
        return None
    return normalized


def _usable_machine_id(value):
    if not value:
        return None
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{32}", normalized) is None or set(normalized) in (
        {"0"},
        {"f"},
    ):
        return None
    return normalized


def _machine_fingerprint(system_name, hostname):
    hardware_uuid = None
    for path in HARDWARE_UUID_PATHS:
        hardware_uuid = _usable_hardware_uuid(_read_text(path, 1024))
        if hardware_uuid:
            break
    machine_id = None
    for path in MACHINE_ID_PATHS:
        machine_id = _usable_machine_id(_read_text(path, 1024))
        if machine_id:
            break
    if hardware_uuid:
        identity = {"version": 2, "system": system_name, "hardware_uuid": hardware_uuid}
        source = "dmi_product_uuid"
        strength = "hardware"
    elif machine_id:
        # Keep the original machine-id hash stable for already pinned targets.
        identity = {"version": 1, "system": system_name, "machine_id": machine_id}
        source = "machine_id"
        strength = "installation"
    else:
        identity = {
            "version": 1,
            "system": system_name,
            "hostname": hostname,
            "machine": platform.machine(),
        }
        source = "hostname"
        strength = "weak"
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return (
        "host-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
        source,
        strength,
    )


def _dgx_profile(dmi, device_tree_model, releases, nvidia):
    identity_parts = [
        dmi.get("sys_vendor", ""),
        dmi.get("product_name", ""),
        dmi.get("board_name", ""),
        device_tree_model or "",
    ]
    gpu_names = [str(gpu.get("name", "")) for gpu in (nvidia or {}).get("gpus", [])]
    text = " ".join(identity_parts + gpu_names).lower()
    is_spark = any(marker in text for marker in ("dgx spark", "gb10", "grace blackwell"))
    is_dgx = bool(releases) or "dgx" in text or is_spark
    if not is_dgx:
        return None
    return {
        "detected": True,
        "product": "DGX Spark" if is_spark else dmi.get("product_name"),
        "is_spark": is_spark,
        "unified_memory": is_spark,
        "release": releases or None,
    }


def _drop_none(value):
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


def collect():
    errors = []
    system_name = platform.system()
    hostname = socket.gethostname()
    cpu_text = _read_text("/proc/cpuinfo", 2_000_000) or ""
    meminfo = _parse_meminfo(_read_text("/proc/meminfo") or "")
    os_release = _parse_key_values(_read_text("/etc/os-release") or "")
    dmi = {}
    for key, path in DMI_PATHS.items():
        value = _read_text(path, 4096)
        if value:
            dmi[key] = value
    device_tree_model = _read_text("/proc/device-tree/model", 4096)
    releases = {}
    for path in DGX_RELEASE_PATHS:
        value = _read_text(path)
        if value:
            releases[os.path.basename(path)] = _parse_key_values(value) or value
    nvidia = _nvidia_profile()
    mem_total_kb = meminfo.get("memtotal_kb")
    memory_bytes = mem_total_kb * 1024 if mem_total_kb is not None else None
    cpu = _cpu_profile(cpu_text)
    host_fingerprint, fingerprint_source, fingerprint_strength = _machine_fingerprint(
        system_name,
        hostname,
    )
    hardware = _drop_none(
        {
            "model": dmi.get("product_name") or device_tree_model,
            "vendor": dmi.get("sys_vendor"),
            "product_version": dmi.get("product_version"),
            "board_name": dmi.get("board_name"),
            "machine": platform.machine(),
            "chip": cpu.get("model_name"),
            "logicalcpu": cpu.get("logical_cpus"),
            "memory_bytes": memory_bytes,
            "memory_gb": round(memory_bytes / 1024**3, 1) if memory_bytes else None,
        }
    )
    profile = {
        "schema_version": SCHEMA_VERSION,
        # Keep this probe executable by the remote host's Python 3.10 runtime.
        "captured_at": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
        "hostname": hostname,
        "host_fingerprint": host_fingerprint,
        "host_fingerprint_source": fingerprint_source,
        "host_fingerprint_strength": fingerprint_strength,
        "platform": {
            "system": system_name,
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "hardware": hardware,
        "linux": {
            "os_release": os_release,
            "proc": {"cpuinfo": cpu, "meminfo": meminfo},
            "dmi": dmi,
            "device_tree_model": device_tree_model,
        },
        "nvidia": nvidia,
        "capabilities": {
            "nvidia_smi": bool(shutil.which("nvidia-smi")),
            "dgx_release": bool(releases),
        },
        "errors": errors,
    }
    dgx = _dgx_profile(dmi, device_tree_model, releases, nvidia)
    if dgx:
        profile["dgx"] = dgx
    return _drop_none(profile)


def main():
    json.dump(collect(), sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
