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
from llm_refinery.providers import openai_discovery as openai_discovery_module
from llm_refinery.providers.openai_discovery import OpenAIDiscoveryClient


@pytest.fixture(autouse=True)
def _resolve_discovery_test_hosts_to_lan(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )


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
    model: dict[str, Any] = {"selection": selection}
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

    raw = _target_mapping()
    raw["model"]["tokenizer"] = "inert/tokenizer"
    with pytest.raises(ConfigError, match="target.model has unknown field.*tokenizer"):
        TargetSpec.from_mapping(raw)


def test_target_spec_normalizes_bare_vllm_server_url_to_openai_v1():
    raw = _target_mapping()
    raw["endpoint"]["base_url"] = "http://spark.local:8000"

    spec = TargetSpec.from_mapping(raw)

    assert spec.endpoint.base_url == "http://spark.local:8000/v1"
    assert spec.endpoint.resolve("served").chat_completions_url.endswith("/v1/chat/completions")


def test_target_spec_rejects_protocols_discovery_cannot_resolve():
    raw = _target_mapping()
    raw["endpoint"]["protocol"] = "ollama_chat"

    with pytest.raises(ConfigError, match="requires endpoint.protocol 'openai_chat'"):
        TargetSpec.from_mapping(raw)


def test_target_spec_rejects_ssh_inventory_with_client_loopback_endpoint():
    raw = _target_mapping()
    raw["endpoint"]["base_url"] = "http://127.0.0.1:8000/v1"

    with pytest.raises(ConfigError, match="loopback"):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://127.1:8000/v1",
        "http://0.0.0.0:8000/v1",
        "http://[::]:8000/v1",
        "http://localhost.localdomain:8000/v1",
        "http://model.localhost:8000/v1",
    ],
)
def test_target_spec_rejects_all_client_local_urls_for_ssh_targets(base_url: str):
    raw = _target_mapping()
    raw["endpoint"]["base_url"] = base_url

    with pytest.raises(ConfigError, match="loopback or wildcard"):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize(
    "base_url",
    [
        "http://user:secret@spark.local:8000/v1",
        "http://spark.local:8000/v1?tenant=one",
        "http://spark.local:8000/v1#fragment",
    ],
)
def test_target_spec_rejects_url_credentials_queries_and_fragments(base_url: str):
    raw = _target_mapping()
    raw["endpoint"]["base_url"] = base_url

    with pytest.raises(ConfigError, match="user information|query or fragment"):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize("section", ["host", "endpoint", "model", "discovery"])
@pytest.mark.parametrize("value", [False, [], "not-a-mapping"])
def test_target_spec_rejects_falsey_or_malformed_sections(section: str, value: Any):
    raw = _target_mapping()
    raw[section] = value

    with pytest.raises(ConfigError, match=rf"target\.{section} must be a mapping"):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("host", "required"),
        ("discovery", "service_required"),
        ("discovery", "metrics"),
    ],
)
def test_target_spec_rejects_quoted_booleans(section: str, key: str):
    raw = _target_mapping()
    raw[section][key] = "false"

    with pytest.raises(ConfigError, match=rf"target\.{section}\.{key} must be a boolean"):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("host", "access"),
        ("host", "destination"),
        ("model", "selection"),
        ("model", "id"),
        ("endpoint", "api_key_env"),
    ],
)
@pytest.mark.parametrize("value", [False, None, ""])
def test_target_spec_rejects_falsey_present_string_fields(
    section: str,
    key: str,
    value: Any,
):
    raw = _target_mapping()
    raw[section][key] = value

    with pytest.raises(ConfigError, match=rf"target\.{section}\.{key} must be a non-empty string"):
        TargetSpec.from_mapping(raw)


def test_target_spec_rejects_empty_authorization_header():
    raw = _target_mapping()
    raw["endpoint"]["headers"] = {"Authorization": ""}

    with pytest.raises(ConfigError, match="Authorization cannot be empty"):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize("timeout", [float("nan"), float("inf"), "-inf", True])
def test_target_spec_rejects_non_finite_or_boolean_timeouts(timeout: Any):
    raw = _target_mapping()
    raw["host"]["connect_timeout_s"] = timeout

    with pytest.raises(ConfigError, match="positive finite number"):
        TargetSpec.from_mapping(raw)


def test_target_spec_rejects_falsey_non_mapping_headers():
    raw = _target_mapping()
    raw["endpoint"]["headers"] = []

    with pytest.raises(ConfigError, match="target.endpoint.headers must be a mapping"):
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
    assert profile["cuda_driver_supported_version"] == "13.0"


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
    assert resolved.topology == {"measurement_scope": "local_client_to_network_endpoint"}


def test_resolver_uses_ssh_inventory_and_verifies_explicit_model():
    ssh = _FakeSSH()
    service = _FakeService(_service("first", "chosen"))
    resolver = TargetResolver(
        ssh_client=ssh,  # type: ignore[arg-type]
        service_client=service,  # type: ignore[arg-type]
    )

    resolved = resolver.resolve(_spec(selection="explicit", model_id="chosen"))

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


def test_resolver_verifies_pinned_host_fingerprint_independent_of_ssh_alias():
    raw = _target_mapping(destination="another-ssh-alias")
    raw["host"]["expected_fingerprint"] = "host-example"
    spec = TargetSpec.from_mapping(raw)
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(_service("served-model")),  # type: ignore[arg-type]
    )

    inspection = resolver.inspect(spec)

    assert inspection.available is True
    assert inspection.safe_json()["host_identity_binding"] == {
        "expected_fingerprint": "host-example",
        "actual_fingerprint": "host-example",
        "verified": True,
    }


@pytest.mark.parametrize("actual", ["host-other", None])
def test_resolver_fails_closed_on_host_fingerprint_mismatch(actual: str | None):
    raw = _target_mapping()
    raw["host"]["expected_fingerprint"] = "host-example"
    raw["discovery"]["service_required"] = False
    profile = dict(_host().profile)
    if actual is None:
        profile.pop("host_fingerprint")
    else:
        profile["host_fingerprint"] = actual
    resolver = TargetResolver(
        ssh_client=_FakeSSH(HostDiscovery(transport="ssh", destination="dgx", profile=profile)),  # type: ignore[arg-type]
        service_client=_FakeService(error="connection refused"),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="fingerprint does not match"):
        resolver.inspect(
            TargetSpec.from_mapping(raw),
            allow_service_unavailable=True,
        )


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
                    "github_token": "server-secret",
                    "admin-token": "server-secret",
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


def test_openai_discovery_bounds_streams_before_buffering(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_discovery_module, "MAX_RESPONSE_BYTES", 32)
    model_chunks_read = 0

    class ModelStream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal model_chunks_read
            for chunk in (b'{"data":[]}', b"x" * 30, b"must-not-be-read"):
                model_chunks_read += 1
                yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, text="")
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "v"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, stream=ModelStream())
        return httpx.Response(404)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ConfigError, match="model discovery.*response is too large"),
    ):
        OpenAIDiscoveryClient(client=client).discover(
            _spec().endpoint, DiscoveryPolicy(server_info="off")
        )

    assert model_chunks_read == 2


def test_offline_tolerance_does_not_suppress_oversized_health_response(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(openai_discovery_module, "MAX_RESPONSE_BYTES", 32)
    health_body_read = False

    class HealthStream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal health_body_read
            health_body_read = True
            yield b"must-not-be-read"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-length": "100"},
            stream=HealthStream(),
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolver = TargetResolver(
            ssh_client=_FakeSSH(),  # type: ignore[arg-type]
            service_client=OpenAIDiscoveryClient(client=client),
        )
        with pytest.raises(ConfigError, match="health.*response is too large"):
            resolver.inspect(_spec(), allow_service_unavailable=True)

    assert health_body_read is False


def test_openai_discovery_bounds_metrics_while_streaming(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(openai_discovery_module, "MAX_RESPONSE_BYTES", 8)
    chunks_read = 0

    class MetricsStream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal chunks_read
            for chunk in (b"1234", b"56789", b"must-not-be-read"):
                chunks_read += 1
                yield chunk

    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=MetricsStream())
            )
        ) as client,
        pytest.raises(RuntimeError, match="response is too large"),
    ):
        OpenAIDiscoveryClient(client=client).metrics(_spec().endpoint)

    assert chunks_read == 2


def test_openai_discovery_applies_total_stream_deadline(monkeypatch: pytest.MonkeyPatch):
    ticks = iter((0.0, 0.1, 6.0))
    monkeypatch.setattr(openai_discovery_module, "_monotonic", lambda: next(ticks))

    class TricklingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"still arriving"

    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, stream=TricklingStream())
            )
        ) as client,
        pytest.raises(RuntimeError, match="exceeded the total timeout"),
    ):
        OpenAIDiscoveryClient(client=client, timeout_s=5.0).metrics(_spec().endpoint)


def test_openai_discovery_requires_configured_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: pytest.fail("must not send request"))
        ) as client,
        pytest.raises(ConfigError, match="VLLM_API_KEY"),
    ):
        OpenAIDiscoveryClient(client=client).discover(
            _spec(api_key_env="VLLM_API_KEY").endpoint,
            DiscoveryPolicy(),
        )


@pytest.mark.parametrize(
    ("headers", "api_key_env"),
    [
        ({"Authorization": "Bearer super-secret\nInjected: yes"}, None),
        ({}, "VLLM_API_KEY"),
    ],
)
def test_invalid_authorization_never_reaches_transport_or_error_text(
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
    api_key_env: str | None,
):
    if api_key_env is not None:
        monkeypatch.setenv(api_key_env, "super-secret\nInjected: yes")
    endpoint = EndpointSpec(
        name="spark",
        protocol="openai_chat",
        base_url="http://spark.local:8000/v1",
        headers=headers,
        api_key_env=api_key_env,
    )
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: pytest.fail("must not send request"))
        ) as client,
        pytest.raises(ConfigError) as caught,
    ):
        OpenAIDiscoveryClient(client=client).discover(endpoint, DiscoveryPolicy())

    assert "super-secret" not in str(caught.value)
    assert "Injected" not in str(caught.value)


def test_offline_tolerance_does_not_suppress_missing_api_key_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("must not send request"))
    ) as client:
        resolver = TargetResolver(
            ssh_client=_FakeSSH(),  # type: ignore[arg-type]
            service_client=OpenAIDiscoveryClient(client=client),
        )
        with pytest.raises(ConfigError, match="VLLM_API_KEY"):
            resolver.inspect(
                _spec(api_key_env="VLLM_API_KEY"),
                allow_service_unavailable=True,
            )


@pytest.mark.parametrize("unauthorized_path", ["/health", "/v1/models"])
def test_offline_tolerance_does_not_suppress_discovery_authorization_failures(
    unauthorized_path: str,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == unauthorized_path:
            return httpx.Response(401, request=request)
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.10.0"}, request=request)
        return httpx.Response(404, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolver = TargetResolver(
            ssh_client=_FakeSSH(),  # type: ignore[arg-type]
            service_client=OpenAIDiscoveryClient(client=client),
        )
        with pytest.raises(ConfigError, match="authorization failed with HTTP 401"):
            resolver.inspect(_spec(), allow_service_unavailable=True)


def test_optional_server_info_authorization_failure_remains_a_warning():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "0.10.0"}, request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "served-model"}]},
                request=request,
            )
        return httpx.Response(403, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolver = TargetResolver(
            ssh_client=_FakeSSH(),  # type: ignore[arg-type]
            service_client=OpenAIDiscoveryClient(client=client),
        )
        inspection = resolver.inspect(_spec())

    assert inspection.available is True
    assert "server_info: HTTP 403" in inspection.errors


def test_offline_tolerance_does_not_suppress_reachable_wrong_http_service():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolver = TargetResolver(
            ssh_client=_FakeSSH(),  # type: ignore[arg-type]
            service_client=OpenAIDiscoveryClient(client=client),
        )
        with pytest.raises(ConfigError, match="health failed with HTTP 404"):
            resolver.inspect(_spec(), allow_service_unavailable=True)

    assert [request.url.path for request in requests] == ["/health"]


def test_discovery_rejects_cross_origin_redirect_before_following_it(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [
            (2, 1, 6, "", ("192.168.1.41", port))
        ],
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"location": "http://127.0.0.1:8000/health"},
            request=request,
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ConfigError, match="remain on the configured.*origin"),
    ):
        OpenAIDiscoveryClient(client=client).discover(
            _spec().endpoint, DiscoveryPolicy(server_info="off")
        )

    assert len(requests) == 1


def test_discovery_rejects_dns_name_that_resolves_to_client_loopback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("127.0.0.1", port))],
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: pytest.fail("must not send request"))
    ) as client:
        resolver = TargetResolver(
            ssh_client=_FakeSSH(),  # type: ignore[arg-type]
            service_client=OpenAIDiscoveryClient(client=client),
        )
        with pytest.raises(ConfigError, match="client-local"):
            resolver.inspect(_spec(), allow_service_unavailable=True)


def test_discovery_allows_dgx_local_name_that_resolves_to_lan_address(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "v"}, request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "served-model"}]},
                request=request,
            )
        return httpx.Response(404, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        resolver = TargetResolver(
            ssh_client=_FakeSSH(),  # type: ignore[arg-type]
            service_client=OpenAIDiscoveryClient(client=client),
        )
        inspection = resolver.inspect(_spec())

    assert inspection.service is not None
    assert inspection.service.health == "ok"
    assert [model.id for model in inspection.service.models] == ["served-model"]


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


def test_allow_service_unavailable_does_not_suppress_required_host_failure():
    class FailingSSH:
        def collect_host_profile(self, access: HostAccess) -> HostDiscovery:
            raise RuntimeError("permission denied")

    resolver = TargetResolver(
        ssh_client=FailingSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(error="connection refused"),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="host: permission denied"):
        resolver.inspect(_spec(), allow_service_unavailable=True)


@pytest.mark.parametrize(
    ("spec", "service", "message"),
    [
        (_spec(), _service("alpha", "beta"), "selection is ambiguous"),
        (
            _spec(selection="explicit", model_id="missing"),
            _service("available"),
            "configured model 'missing' is not served",
        ),
    ],
)
def test_offline_tolerance_does_not_suppress_healthy_service_model_errors(
    spec: TargetSpec,
    service: ServiceDiscovery,
    message: str,
):
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(service),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match=message):
        resolver.inspect(spec, allow_service_unavailable=True)


def test_service_required_false_does_not_suppress_healthy_service_model_errors():
    raw = _target_mapping()
    raw["discovery"]["service_required"] = False
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(_service("alpha", "beta")),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="selection is ambiguous"):
        resolver.inspect(TargetSpec.from_mapping(raw))


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
