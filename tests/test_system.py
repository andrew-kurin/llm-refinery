from __future__ import annotations

import socket

from llm_refinery.utils import system


def test_port_check_uses_socket_when_lsof_is_unavailable(monkeypatch):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    monkeypatch.setattr(system.shutil, "which", lambda _name: None)

    try:
        assert system.is_port_listening(port) is True
    finally:
        listener.close()


def test_host_identity_prefers_fingerprint_and_stably_supports_legacy_profiles():
    assert system.host_identity({"host_fingerprint": "host-explicit"}) == "host-explicit"

    legacy = {
        "hostname": "spark",
        "platform": {"system": "Linux", "machine": "aarch64"},
        "hardware": {"model": "DGX Spark", "memory_bytes": 128 * 1024**3},
    }
    assert system.host_identity(legacy) == system.host_identity(dict(legacy))
    assert system.host_identity(legacy) != system.host_identity(
        {
            "hostname": "mac",
            "platform": {"system": "Darwin", "machine": "arm64"},
            "hardware": {"model": "Mac17,6", "memory_bytes": 128 * 1024**3},
        }
    )
    assert system.host_identity({}) == "unknown-host"


def test_linux_proc_parsers_capture_cpu_topology_and_memory():
    cpu = system._parse_cpuinfo(
        """processor : 0
vendor_id : GenuineIntel
physical id : 0
core id : 0
model name : Example CPU

processor : 1
vendor_id : GenuineIntel
physical id : 0
core id : 1
model name : Example CPU
"""
    )
    memory = system._parse_meminfo(
        """MemTotal:       131072000 kB
MemAvailable:   120000000 kB
SwapTotal:              0 kB
"""
    )

    assert cpu == {
        "model_name": "Example CPU",
        "vendor_id": "GenuineIntel",
        "logical_cpus": 2,
        "physical_cores": 2,
        "physical_packages": 1,
    }
    assert memory["mem_total_kb"] == 131072000
    assert memory["mem_available_kb"] == 120000000
    assert memory["swap_total_kb"] == 0


def test_linux_system_profile_projects_proc_nvidia_and_dgx_metadata(monkeypatch):
    linux = {
        "os_release": {"name": "Ubuntu", "version_id": "24.04"},
        "proc": {
            "cpuinfo": {
                "model_name": "NVIDIA Grace",
                "logical_cpus": 20,
                "physical_cores": 20,
            },
            "meminfo": {"mem_total_kb": 134217728},
        },
        "dmi": {"sys_vendor": "NVIDIA", "product_name": "DGX Spark"},
    }
    monkeypatch.setattr(system.platform, "system", lambda: "Linux")
    monkeypatch.setattr(system.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(system.socket, "gethostname", lambda: "spark")
    monkeypatch.setattr(system, "_linux_profile", lambda: linux)
    monkeypatch.setattr(system, "_machine_identifier", lambda _system_name: "machine-id")
    monkeypatch.setattr(
        system,
        "_nvidia_profile",
        lambda: {
            "driver_version": "580.10",
            "cuda_driver_supported_version": "13.0",
            "cuda_runtime_version": "13.0",
        },
    )
    monkeypatch.setattr(system, "_dgx_profile", lambda _linux: {"model": "DGX Spark"})
    monkeypatch.setattr(system, "_git_output", lambda *_args: None)
    monkeypatch.setattr(system, "_git_dirty", lambda: None)

    profile = system.get_system_profile()

    assert profile["host_fingerprint"].startswith("host-")
    assert profile["linux"]["os_release"]["name"] == "Ubuntu"
    assert profile["hardware"]["model"] == "DGX Spark"
    assert profile["hardware"]["chip"] == "NVIDIA Grace"
    assert profile["hardware"]["memory_gb"] == 128.0
    assert profile["nvidia"]["cuda_runtime_version"] == "13.0"
    assert profile["nvidia"]["cuda_driver_supported_version"] == "13.0"
    assert profile["dgx"]["model"] == "DGX Spark"


def test_nvidia_profile_captures_driver_cuda_and_gpu_details(monkeypatch):
    paths = {"nvidia-smi": "/usr/bin/nvidia-smi", "nvcc": "/usr/local/cuda/bin/nvcc"}
    monkeypatch.setattr(system.shutil, "which", paths.get)

    def fake_command_output(command, *, timeout=2):
        del timeout
        if (
            command[0].endswith("nvidia-smi")
            and len(command) > 1
            and command[1].startswith("--query-gpu")
        ):
            return "0, NVIDIA GB10, GPU-123, 580.10, 122880, 00000000:01:00.0"
        if command[0].endswith("nvidia-smi"):
            return "NVIDIA-SMI 580.10  Driver Version: 580.10  CUDA Version: 13.0"
        if command[0].endswith("nvcc"):
            return "Cuda compilation tools, release 13.0, V13.0.10"
        return None

    monkeypatch.setattr(system, "_command_output", fake_command_output)
    monkeypatch.setattr(system, "_read_text", lambda *_args, **_kwargs: None)

    profile = system._nvidia_profile()

    assert profile["driver_version"] == "580.10"
    assert profile["cuda_runtime_version"] == "13.0"
    assert profile["cuda_driver_supported_version"] == "13.0"
    assert profile["cuda_toolkit_version"] == "13.0"
    assert profile["gpus"][0]["name"] == "NVIDIA GB10"
    assert profile["gpus"][0]["memory_total_mb"] == 122880.0


def test_process_snapshot_recognizes_local_linux_serving_stacks(monkeypatch):
    monkeypatch.setattr(system.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        system,
        "_command_output",
        lambda *_args, **_kwargs: "\n".join(
            [
                "1 0 vllm 10 2 100 vllm serve model",
                "2 0 trtllm 10 2 100 trtllm-serve model",
                "3 0 sglang 10 2 100 python -m sglang.launch_server",
                "4 0 unrelated 0 0 1 sleep 10",
            ]
        ),
    )

    snapshot = "\n".join(system._process_snapshot())

    assert "vllm serve" in snapshot
    assert "trtllm-serve" in snapshot
    assert "sglang.launch_server" in snapshot
    assert "sleep 10" not in snapshot
