from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from llm_refinery.adapters.ssh import MAX_PROBE_OUTPUT_CHARS, OpenSSHClient
from llm_refinery.application.target_discovery import TargetResolver
from llm_refinery.cli import main
from llm_refinery.core.config import ConfigError
from llm_refinery.core.targets import (
    DiscoveryPolicy,
    EndpointSpec,
    HostAccess,
    HostDiscovery,
    ModelDescriptor,
    ServiceDiscovery,
    TargetInspection,
    TargetSpec,
    load_target_spec,
)
from llm_refinery.probes import linux_dgx_probe
from llm_refinery.providers.openai_discovery import OpenAIDiscoveryClient


def _target_mapping(
    *,
    access: str = "ssh",
    destination: str | None = "dgx",
    selection: str = "single",
    model_id: str | None = None,
    endpoint_model: str | None = None,
    api_key_env: str | None = None,
) -> dict[str, Any]:
    endpoint: dict[str, Any] = {
        "protocol": "openai_chat",
        "base_url": "http://spark.local:8000/v1",
    }
    if endpoint_model is not None:
        endpoint["model"] = endpoint_model
    if api_key_env is not None:
        endpoint["api_key_env"] = api_key_env
    host: dict[str, Any] = {"access": access}
    if destination is not None:
        host["destination"] = destination
    model: dict[str, Any] = {"selection": selection, "tokenizer": "org/model-tokenizer"}
    if model_id is not None:
        model["id"] = model_id
    return {
        "name": "spark",
        "host": host,
        "endpoint": endpoint,
        "model": model,
        "discovery": {"service_required": True, "server_info": "optional"},
    }


def _spec(**kwargs: Any) -> TargetSpec:
    return TargetSpec.from_mapping(_target_mapping(**kwargs))


def _host(*, transport: str = "ssh") -> HostDiscovery:
    return HostDiscovery(
        transport=transport,
        destination="dgx" if transport == "ssh" else None,
        profile={
            "schema_version": 1,
            "hostname": "spark-host",
            "host_fingerprint": "host-example",
            "hardware": {"model": "NVIDIA DGX Spark", "memory_gb": 128.0},
        },
    )


def _service(*model_ids: str, health: str = "ok") -> ServiceDiscovery:
    return ServiceDiscovery(
        implementation="vllm",
        base_url="http://spark.local:8000/v1",
        health=health,
        version="0.10.0",
        models=tuple(
            ModelDescriptor(
                id=model_id,
                root=f"org/{model_id}",
                max_model_len=32768,
                owned_by="vllm",
            )
            for model_id in model_ids
        ),
        server_info={"model": model_ids[0]} if model_ids else {},
    )


class _FakeSSH:
    def __init__(self, host: HostDiscovery | None = None) -> None:
        self.host = host or _host()
        self.calls: list[HostAccess] = []

    def collect_host_profile(self, access: HostAccess) -> HostDiscovery:
        self.calls.append(access)
        return self.host


class _FakeService:
    def __init__(
        self,
        service: ServiceDiscovery | None = None,
        error: str | None = None,
    ) -> None:
        self.service = service
        self.error = error
        self.calls: list[tuple[EndpointSpec, DiscoveryPolicy]] = []

    def discover(
        self,
        endpoint: EndpointSpec,
        policy: DiscoveryPolicy,
    ) -> ServiceDiscovery:
        self.calls.append((endpoint, policy))
        if self.error:
            raise RuntimeError(self.error)
        assert self.service is not None
        return self.service


def test_target_spec_loads_wrapped_config_and_keeps_endpoint_unresolved(tmp_path: Path):
    config = tmp_path / "target.yaml"
    config.write_text(
        """
schema_version: 1
target:
  name: dgx-spark
  host:
    access: ssh
    destination: dgx
    connect_timeout_s: 3
    command_timeout_s: 12
  endpoint:
    protocol: openai_chat
    base_url: http://spark.local:8000/v1/
    api_key_env: VLLM_API_KEY
  model:
    selection: single
    tokenizer: nvidia/tokenizer
  discovery:
    service_required: true
    server_info: optional
""",
        encoding="utf-8",
    )

    loaded_path, spec = load_target_spec(config)

    assert loaded_path == config
    assert spec.name == "dgx-spark"
    assert spec.host == HostAccess(
        access="ssh",
        destination="dgx",
        connect_timeout_s=3,
        command_timeout_s=12,
    )
    assert spec.endpoint.name == "dgx-spark"
    assert spec.endpoint.base_url == "http://spark.local:8000/v1"
    assert spec.endpoint.model is None
    assert spec.model.model_id is None
    assert spec.safe_json()["endpoint"]["header_names"] == []


@pytest.mark.parametrize("destination", ["-oProxyCommand=bad", "dgx host", "dgx\nother"])
def test_target_spec_rejects_unsafe_ssh_destinations(destination: str):
    with pytest.raises(ConfigError, match="destination"):
        _spec(destination=destination)


def test_target_spec_validates_explicit_model_and_unknown_fields():
    explicit = _spec(selection="explicit", model_id="served-model")
    assert explicit.model.model_id == "served-model"

    with pytest.raises(ConfigError, match="conflicts"):
        _spec(
            selection="explicit",
            model_id="served-model",
            endpoint_model="different-model",
        )

    raw = _target_mapping()
    raw["launch_command"] = "vllm serve model"
    with pytest.raises(ConfigError, match="unknown field.*launch_command"):
        TargetSpec.from_mapping(raw)


def test_target_spec_normalizes_bare_vllm_server_url_to_openai_v1():
    raw = _target_mapping()
    raw["endpoint"]["base_url"] = "http://spark.local:8000"

    spec = TargetSpec.from_mapping(raw)

    assert spec.endpoint.base_url == "http://spark.local:8000/v1"
    assert spec.endpoint.resolve("served").chat_completions_url.endswith(
        "/v1/chat/completions"
    )


def test_target_spec_rejects_ssh_inventory_with_client_loopback_endpoint():
    raw = _target_mapping()
    raw["endpoint"]["base_url"] = "http://127.0.0.1:8000/v1"

    with pytest.raises(ConfigError, match="loopback"):
        TargetSpec.from_mapping(raw)


def test_ssh_adapter_uses_fixed_argv_probe_stdin_and_never_requests_a_shell():
    captured: dict[str, Any] = {}
    profile = {
        "schema_version": 1,
        "hostname": "spark",
        "machine_id": "must-not-leak",
        "nested": {"cmdline": "must-not-leak", "safe": True},
    }

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(profile), stderr="")

    client = OpenSSHClient(runner=runner, ssh_executable="/usr/bin/ssh")
    access = HostAccess(
        access="ssh",
        destination="dgx",
        connect_timeout_s=2.1,
        command_timeout_s=17,
    )

    discovered = client.collect_host_profile(access)

    assert captured["argv"] == [
        "/usr/bin/ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=3",
        "--",
        "dgx",
        "python3",
        "-I",
        "-",
    ]
    kwargs = captured["kwargs"]
    assert "shell" not in kwargs
    assert kwargs["timeout"] == 17
    assert kwargs["check"] is False
    assert kwargs["text"] is True
    assert "def collect" in kwargs["input"]
    assert "vllm serve" not in kwargs["input"]
    assert "systemctl" not in kwargs["input"]
    assert discovered.profile == {
        "schema_version": 1,
        "hostname": "spark",
        "nested": {"safe": True},
    }


def test_ssh_adapter_reports_timeout_and_rejects_oversized_output():
    def timeout_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    access = HostAccess(access="ssh", destination="dgx", command_timeout_s=4)
    with pytest.raises(RuntimeError, match="timed out after 4s"):
        OpenSSHClient(runner=timeout_runner).collect_host_profile(access)

    def oversized_runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="x" * (MAX_PROBE_OUTPUT_CHARS + 1),
            stderr="",
        )

    with pytest.raises(RuntimeError, match="exceeded"):
        OpenSSHClient(runner=oversized_runner).collect_host_profile(access)


def test_ssh_adapter_local_inventory_does_not_execute_probe(monkeypatch: pytest.MonkeyPatch):
    def runner(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise AssertionError("local inventory must not execute the SSH probe")

    monkeypatch.setattr(
        "llm_refinery.adapters.ssh.get_system_profile",
        lambda: {"schema_version": 2, "hostname": "local-mac"},
    )

    result = OpenSSHClient(runner=runner).collect_host_profile(HostAccess(access="local"))

    assert result.transport == "local"
    assert result.profile["hostname"] == "local-mac"


def test_remote_probe_normalizes_dgx_spark_unified_memory(monkeypatch: pytest.MonkeyPatch):
    files = {
        "/etc/machine-id": "raw-machine-id",
        "/proc/cpuinfo": "processor: 0\nmodel name: NVIDIA Grace\n",
        "/proc/meminfo": "MemTotal: 134217728 kB\nMemAvailable: 120000000 kB\n",
        "/etc/os-release": 'NAME="DGX OS"\nVERSION_ID="24.04"\n',
        "/sys/devices/virtual/dmi/id/sys_vendor": "NVIDIA",
        "/sys/devices/virtual/dmi/id/product_name": "DGX Spark",
        "/proc/device-tree/model": "NVIDIA DGX Spark",
        "/etc/dgx-release": "DGX_NAME=DGX Spark\n",
    }
    monkeypatch.setattr(
        linux_dgx_probe,
        "_read_text",
        lambda path, limit=100_000: files.get(path),
    )
    monkeypatch.setattr(linux_dgx_probe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(linux_dgx_probe.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(linux_dgx_probe.platform, "release", lambda: "6.11")
    monkeypatch.setattr(linux_dgx_probe.platform, "version", lambda: "DGX")
    monkeypatch.setattr(linux_dgx_probe.socket, "gethostname", lambda: "spark")
    monkeypatch.setattr(
        linux_dgx_probe,
        "_nvidia_profile",
        lambda: {
            "driver_version": "580.10",
            "cuda_runtime_version": "13.0",
            "gpus": [{"name": "NVIDIA GB10", "reported_device_memory_mib": 122880}],
        },
    )

    profile = linux_dgx_probe.collect()

    assert profile["hardware"]["memory_gb"] == 128.0
    assert profile["linux"]["proc"]["meminfo"]["memavailable_kb"] == 120000000
    assert profile["dgx"]["is_spark"] is True
    assert profile["dgx"]["unified_memory"] is True
    assert "raw-machine-id" not in json.dumps(profile)


def test_remote_probe_keeps_gpu_identity_when_memory_queries_are_unsupported(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        linux_dgx_probe.shutil,
        "which",
        lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None,
    )

    def fake_command(argv, timeout=5):
        del timeout
        if any(str(arg).startswith("--query-gpu=index,name") for arg in argv):
            return "0,NVIDIA GB10,GPU-1,580.10,00000000:01:00.0"
        if any("memory.total" in str(arg) for arg in argv):
            return None
        if any(str(arg).startswith("--query-gpu=index,") for arg in argv):
            return "0,N/A"
        return "NVIDIA-SMI 580.10 Driver Version: 580.10 CUDA Version: 13.0"

    monkeypatch.setattr(linux_dgx_probe, "_command", fake_command)

    profile = linux_dgx_probe._nvidia_profile()

    assert profile["gpus"] == [
        {
            "index": "0",
            "name": "NVIDIA GB10",
            "uuid": "GPU-1",
            "driver_version": "580.10",
            "pci_bus_id": "00000000:01:00.0",
        }
    ]
    assert profile["cuda_runtime_version"] == "13.0"


def test_resolver_uses_local_profile_and_selects_the_only_served_model():
    ssh = _FakeSSH()
    service = _FakeService(_service("served-model"))
    resolver = TargetResolver(
        ssh_client=ssh,  # type: ignore[arg-type]
        service_client=service,  # type: ignore[arg-type]
        local_system_profile=lambda: {"hostname": "local-mac", "schema_version": 2},
    )

    resolved = resolver.resolve(_spec(access="local", destination=None))

    assert ssh.calls == []
    assert resolved.endpoint.model == "served-model"
    assert resolved.endpoint.base_url == "http://spark.local:8000/v1"
    assert resolved.host.profile["hostname"] == "local-mac"
    assert resolved.selection == "single_discovered"
    assert resolved.topology == {
        "measurement_scope": "local_client_to_network_endpoint"
    }
    assert resolved.tokenizer == "org/model-tokenizer"


def test_resolver_uses_ssh_inventory_and_verifies_explicit_model():
    ssh = _FakeSSH()
    service = _FakeService(_service("first", "chosen"))
    resolver = TargetResolver(
        ssh_client=ssh,  # type: ignore[arg-type]
        service_client=service,  # type: ignore[arg-type]
    )

    resolved = resolver.resolve(
        _spec(selection="explicit", model_id="chosen")
    )

    assert len(ssh.calls) == 1
    assert resolved.endpoint.model == "chosen"
    assert resolved.selection == "explicit_verified"
    assert resolved.topology == {"measurement_scope": "remote_client_to_server"}
    assert resolved.safe_json()["host"]["profile"]["hostname"] == "spark-host"

    inspection = resolver.inspect(_spec(selection="explicit", model_id="chosen"))
    assert [model["id"] for model in inspection.safe_json()["service"]["models"]] == [
        "first",
        "chosen",
    ]


@pytest.mark.parametrize(
    ("models", "message"),
    [
        ((), "no served models"),
        (("alpha", "beta"), "selection is ambiguous"),
    ],
)
def test_resolver_rejects_zero_or_multiple_models_for_single_selection(
    models: tuple[str, ...],
    message: str,
):
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(_service(*models)),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match=message):
        resolver.resolve(_spec())


def test_resolver_rejects_explicit_model_that_is_not_served():
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(_service("available")),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="configured model 'missing'.*available"):
        resolver.resolve(_spec(selection="explicit", model_id="missing"))


def test_openai_discovery_uses_api_auth_and_sanitizes_server_info(
    monkeypatch: pytest.MonkeyPatch,
):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer top-secret-api-key"
        if request.url.path == "/health":
            return httpx.Response(200, text="")
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.10.0"})
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "data": [
                        {
                            "id": "served-model",
                            "root": "org/model",
                            "max_model_len": 65536,
                            "owned_by": "vllm",
                        }
                    ],
                },
            )
        if request.url.path == "/server_info":
            assert request.url.params["config_format"] == "json"
            return httpx.Response(
                200,
                json={
                    "model": "org/model",
                    "dtype": "bfloat16",
                    "tokenizer": "org/tokenizer",
                    "tokenizer_revision": "revision-123",
                    "max_num_batched_tokens": 8192,
                    "api_key": "server-secret",
                    "hf_token": "server-secret",
                    "nested": {
                        "authorization": "server-secret",
                        "password": "server-secret",
                        "safe": True,
                    },
                },
            )
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:num_requests_running 0\n")
        return httpx.Response(404)

    monkeypatch.setenv("VLLM_API_KEY", "top-secret-api-key")
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        discovery_client = OpenAIDiscoveryClient(client=client)
        discovery = discovery_client.discover(
            _spec(api_key_env="VLLM_API_KEY").endpoint,
            DiscoveryPolicy(server_info="optional"),
        )
        metrics = discovery_client.metrics(_spec(api_key_env="VLLM_API_KEY").endpoint)

    assert [request.url.path for request in requests] == [
        "/health",
        "/version",
        "/v1/models",
        "/server_info",
        "/metrics",
    ]
    assert discovery.health == "ok"
    assert discovery.version == "0.10.0"
    assert discovery.models[0].id == "served-model"
    assert discovery.models[0].max_model_len == 65536
    serialized = json.dumps(discovery.server_info)
    assert "server-secret" not in serialized
    assert "api_key" not in serialized
    assert discovery.server_info == {
        "model": "org/model",
        "dtype": "bfloat16",
        "tokenizer": "org/tokenizer",
        "tokenizer_revision": "revision-123",
        "max_num_batched_tokens": 8192,
        "nested": {"safe": True},
    }
    assert metrics == "vllm:num_requests_running 0\n"


def test_openai_discovery_requires_configured_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("must not send request"))
    ) as client, pytest.raises(RuntimeError, match="VLLM_API_KEY"):
        OpenAIDiscoveryClient(client=client).discover(
            _spec(api_key_env="VLLM_API_KEY").endpoint,
            DiscoveryPolicy(),
        )


def test_openai_discovery_preserves_explicit_auth_and_short_circuits_offline_host(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.headers["Authorization"] == "Static credential"
        raise httpx.ConnectError("connection refused", request=request)

    endpoint = EndpointSpec(
        name="spark",
        protocol="openai_chat",
        base_url="http://spark.local:8000/v1",
        api_key_env="MISSING_API_KEY",
        headers={"authorization": "Static credential"},
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        discovery = OpenAIDiscoveryClient(client=client).discover(
            endpoint,
            DiscoveryPolicy(),
        )

    assert len(requests) == 1
    assert requests[0].url.path == "/health"
    assert discovery.health == "unavailable"
    assert discovery.models == ()
    assert discovery.errors == ("health: connection refused",)


def test_resolver_can_return_inventory_when_service_is_unavailable():
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(error="connection refused"),  # type: ignore[arg-type]
    )

    inspection = resolver.inspect(_spec(), allow_service_unavailable=True)

    assert inspection.available is False
    assert inspection.host is not None
    assert inspection.host.profile["hostname"] == "spark-host"
    assert inspection.service is None
    assert inspection.errors == ("service: connection refused",)
    assert inspection.safe_json()["status"] == "unavailable"

    with pytest.raises(RuntimeError, match="connection refused"):
        resolver.resolve(_spec())


def test_service_required_false_makes_default_inspection_host_only():
    raw = _target_mapping()
    raw["discovery"] = {
        "service_required": False,
        "server_info": "off",
        "metrics": False,
    }
    spec = TargetSpec.from_mapping(raw)
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(error="connection refused"),  # type: ignore[arg-type]
    )

    inspection = resolver.inspect(spec)

    assert inspection.available is False
    assert inspection.host is not None
    assert spec.discovery.metrics is False
    with pytest.raises(RuntimeError, match="connection refused"):
        resolver.resolve(spec)


def test_target_inspect_cli_uses_read_only_resolver_and_prints_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = tmp_path / "target.yaml"
    config.write_text(
        """
name: spark
host:
  access: ssh
  destination: dgx
endpoint:
  protocol: openai_chat
  base_url: http://spark.local:8000/v1
model:
  selection: single
""",
        encoding="utf-8",
    )
    calls: list[tuple[TargetSpec, bool]] = []

    class FakeResolver:
        def inspect(
            self,
            spec: TargetSpec,
            *,
            allow_service_unavailable: bool = False,
        ) -> TargetInspection:
            calls.append((spec, allow_service_unavailable))
            return TargetInspection(
                spec=spec,
                host=_host(),
                service=None,
                resolved=None,
                errors=("service: connection refused",),
            )

    monkeypatch.setattr("llm_refinery.commands.targets.TargetResolver", FakeResolver)

    result = CliRunner().invoke(
        main,
        [
            "target",
            "inspect",
            str(config),
            "--ssh-destination",
            "another-spark",
            "--allow-service-unavailable",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["name"] == "spark"
    assert payload["status"] == "unavailable"
    assert payload["host"]["destination"] == "dgx"
    assert calls[0][0].host.destination == "another-spark"
    assert calls[0][1] is True
