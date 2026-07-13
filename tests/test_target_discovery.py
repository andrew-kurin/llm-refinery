from __future__ import annotations

import json
import os
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from llm_refinery.adapters import ssh as ssh_module
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
    TargetTransport,
    load_target_spec,
)
from llm_refinery.probes import linux_dgx_probe
from llm_refinery.providers import openai_discovery as openai_discovery_module
from llm_refinery.providers.openai_discovery import OpenAIDiscoveryClient
from llm_refinery.utils import system as system_module

_REAL_GETADDRINFO = socket.getaddrinfo


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


class _MockOpenAIDiscoveryClient(OpenAIDiscoveryClient):
    """Test-only client provider that preserves production ownership semantics."""

    def __init__(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
        *,
        timeout_s: float = 5.0,
    ) -> None:
        super().__init__(timeout_s=timeout_s)
        self._handler = handler
        self.created_clients: list[httpx.Client] = []

    def _new_client(
        self,
        transport: TargetTransport,
        *,
        trust_env: bool | None = None,
    ) -> httpx.Client:
        del transport, trust_env
        client = httpx.Client(transport=httpx.MockTransport(self._handler))
        self.created_clients.append(client)
        return client


class _TrackingOpenAIDiscoveryClient(OpenAIDiscoveryClient):
    """Exercise production client construction while retaining ownership evidence."""

    def __init__(self, *, timeout_s: float = 5.0) -> None:
        super().__init__(timeout_s=timeout_s)
        self.created_clients: list[httpx.Client] = []

    def _new_client(
        self,
        transport: TargetTransport,
        *,
        trust_env: bool | None = None,
    ) -> httpx.Client:
        client = super()._new_client(transport, trust_env=trust_env)
        self.created_clients.append(client)
        return client


def _discovery_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    timeout_s: float = 5.0,
) -> _MockOpenAIDiscoveryClient:
    return _MockOpenAIDiscoveryClient(handler, timeout_s=timeout_s)


def _host(*, transport: str = "ssh") -> HostDiscovery:
    return HostDiscovery(
        transport=transport,
        destination="dgx" if transport == "ssh" else None,
        profile={
            "schema_version": 1,
            "hostname": "spark-host",
            "host_fingerprint": "host-example",
            "host_fingerprint_source": "machine_id",
            "host_fingerprint_strength": "installation",
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
        self.calls: list[tuple[EndpointSpec, DiscoveryPolicy, TargetTransport]] = []

    def discover(
        self,
        endpoint: EndpointSpec,
        policy: DiscoveryPolicy,
        transport: TargetTransport,
        *,
        route: Any = None,
    ) -> ServiceDiscovery:
        del route
        self.calls.append((endpoint, policy, transport))
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
    assert spec.transport == TargetTransport()


@pytest.mark.parametrize("schema_version", [1.0, "1", True])
def test_target_spec_rejects_non_integer_schema_version(
    tmp_path: Path,
    schema_version: Any,
):
    config = tmp_path / "target.yaml"
    config.write_text(
        "schema_version: "
        + json.dumps(schema_version)
        + "\ntarget:\n"
        + "  name: dgx\n"
        + "  host:\n    access: local\n"
        + "  endpoint:\n"
        + "    protocol: openai_chat\n"
        + "    base_url: http://127.0.0.1:8000/v1\n"
        + "  model:\n    selection: single\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="schema_version must be 1"):
        load_target_spec(config)


def test_target_transport_resolves_ca_bundle_relative_to_target_file(tmp_path: Path):
    cert_dir = tmp_path / "certs"
    cert_dir.mkdir()
    ca_bundle = cert_dir / "spark.pem"
    ca_bundle.write_text("test CA fixture", encoding="utf-8")
    config = tmp_path / "target.yaml"
    config.write_text(
        """
schema_version: 1
target:
  name: dgx-spark
  host:
    access: ssh
    destination: dgx
  endpoint:
    protocol: openai_chat
    base_url: https://spark.local:8000/v1
  model:
    selection: single
  transport:
    trust_env: false
    ca_bundle: certs/spark.pem
""",
        encoding="utf-8",
    )

    _, spec = load_target_spec(config)

    assert spec.transport == TargetTransport(
        trust_env=False,
        ca_bundle=ca_bundle.resolve(),
    )
    assert spec.safe_json()["transport"] == {
        "trust_env": False,
        "ca_bundle": str(ca_bundle.resolve()),
    }


@pytest.mark.parametrize("value", [None, "", False, 123])
def test_target_transport_rejects_invalid_ca_bundle(value: Any, tmp_path: Path):
    raw = _target_mapping()
    raw["transport"] = {"ca_bundle": value}

    with pytest.raises(ConfigError, match="ca_bundle must be a non-empty path string"):
        TargetSpec.from_mapping(raw, base_dir=tmp_path)


def test_target_transport_rejects_missing_ca_bundle(tmp_path: Path):
    raw = _target_mapping()
    raw["transport"] = {"ca_bundle": "missing.pem"}

    with pytest.raises(ConfigError, match="ca_bundle is not a file"):
        TargetSpec.from_mapping(raw, base_dir=tmp_path)


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

    with pytest.raises(ConfigError, match="loopback or wildcard|wildcard address"):
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


@pytest.mark.parametrize(
    ("base_url", "error"),
    [
        ("http://spark.local:0/v1", "valid hostname and port"),
        ("http://spark.local:/v1", "valid hostname and port"),
        ("http://spark.local:99999/v1", "valid hostname and port"),
        ("http://spark.local:8000/v1\\chat", "backslashes"),
        ("http://spark.local:8000/v1 chat", "without whitespace"),
        ("http://spark.local:8000/v1\nother", "without whitespace"),
    ],
)
def test_target_spec_rejects_ambiguous_endpoint_urls_during_manifest_loading(
    base_url: str,
    error: str,
):
    raw = _target_mapping()
    raw["host"] = {"access": "local"}
    raw["endpoint"]["base_url"] = base_url

    with pytest.raises(ConfigError, match=error):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize("base_url", ["http://0.0.0.0:8000/v1", "http://[::]:8000/v1"])
def test_local_target_spec_rejects_wildcard_endpoint_urls(base_url: str):
    raw = _target_mapping()
    raw["host"] = {"access": "local"}
    raw["endpoint"]["base_url"] = base_url

    with pytest.raises(ConfigError, match="wildcard address"):
        TargetSpec.from_mapping(raw)


@pytest.mark.parametrize("section", ["host", "endpoint", "model", "discovery", "transport"])
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
        ("transport", "trust_env"),
    ],
)
def test_target_spec_rejects_quoted_booleans(section: str, key: str):
    raw = _target_mapping()
    raw.setdefault(section, {})
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


@pytest.mark.parametrize("field", ["connect_timeout_s", "command_timeout_s"])
def test_target_spec_rejects_timeout_above_platform_safe_bound(field: str):
    raw = _target_mapping()
    raw["host"][field] = 1e100

    with pytest.raises(ConfigError, match="must be at most 86400 seconds"):
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
        "hardware-uuid": "must-not-leak",
        "nested": {
            "cmdline": "must-not-leak",
            "Product_UUID": "must-not-leak",
            "machine identifier": "must-not-leak",
            "safe": True,
        },
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
        "ClearAllForwardings=yes",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "RemoteCommand=none",
        "-o",
        "StdinNull=no",
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


def test_ssh_adapter_overrides_alias_remote_command_and_stdin_policy(tmp_path: Path):
    ssh = shutil.which("ssh")
    if ssh is None:
        pytest.skip("OpenSSH client is not installed")
    config = tmp_path / "ssh_config"
    config.write_text(
        """\
Host dgx
    HostName example.invalid
    RemoteCommand false
    StdinNull yes
    PermitLocalCommand yes
    LocalCommand false
""",
        encoding="utf-8",
    )
    command = OpenSSHClient(ssh_executable=ssh).command(
        HostAccess(access="ssh", destination="dgx")
    )

    completed = subprocess.run(
        [command[0], "-G", "-F", str(config), *command[1:]],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert completed.returncode == 0, completed.stderr
    effective = completed.stdout.casefold().splitlines()
    assert "clearallforwardings yes" in effective
    assert "permitlocalcommand no" in effective
    assert "stdinnull no" in effective
    assert "remotecommand false" not in effective


@pytest.mark.parametrize("schema_version", [True, 1.0, "1", None])
def test_ssh_adapter_requires_strict_integer_probe_schema_version(schema_version: Any):
    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        profile = {"schema_version": schema_version, "hostname": "spark"}
        return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(profile), stderr="")

    with pytest.raises(RuntimeError, match="unsupported schema_version"):
        OpenSSHClient(runner=runner).collect_host_profile(
            HostAccess(access="ssh", destination="dgx")
        )


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


def test_ssh_adapter_rejects_excessively_nested_json_without_leaking_recursion_error():
    depth = sys.getrecursionlimit() + 100
    nested = '{"value":' * depth + "null" + "}" * depth
    stdout = '{"schema_version":1,"nested":' + nested + "}"

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    # CPython versions differ on whether the JSON decoder or the subsequent
    # profile sanitizer reaches its recursion guard first. Both paths must
    # expose the same bounded user-facing invariant, never RecursionError.
    with pytest.raises(RuntimeError, match="too deeply nested"):
        OpenSSHClient(runner=runner).collect_host_profile(
            HostAccess(access="ssh", destination="dgx")
        )


def test_ssh_adapter_rejects_profile_that_is_too_deep_to_sanitize(
    monkeypatch: pytest.MonkeyPatch,
):
    profile: dict[str, Any] = {"schema_version": 1}
    cursor = profile
    for _ in range(sys.getrecursionlimit() + 100):
        child: dict[str, Any] = {}
        cursor["nested"] = child
        cursor = child

    monkeypatch.setattr(ssh_module.json, "loads", lambda _value: profile)

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="{}", stderr="")

    with pytest.raises(RuntimeError, match="profile is too deeply nested"):
        OpenSSHClient(runner=runner).collect_host_profile(
            HostAccess(access="ssh", destination="dgx")
        )


def test_ssh_adapter_strips_terminal_sequences_from_failure_details():
    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del kwargs
        stderr = (
            "\x1b[31mpermission denied\x1b[0m\n"
            "\x1b]2;forged terminal title\x07"
            "\x1bPprivate payload\x1b\\"
            "retry\x00\x7f\x85"
        )
        return subprocess.CompletedProcess(argv, 255, stdout="", stderr=stderr)

    with pytest.raises(RuntimeError) as exc_info:
        OpenSSHClient(runner=runner).collect_host_profile(
            HostAccess(access="ssh", destination="dgx")
        )

    message = str(exc_info.value)
    assert message.endswith(": permission denied retry")
    assert "forged terminal title" not in message
    assert "private payload" not in message
    assert all(
        ord(character) >= 0x20 and not 0x7F <= ord(character) <= 0x9F for character in message
    )


def test_ssh_adapter_strips_terminal_sequences_from_execution_errors():
    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        del argv, kwargs
        raise OSError("\x9b31munsafe\x9b0m \x9dforged title\x9csafe")

    with pytest.raises(RuntimeError) as exc_info:
        OpenSSHClient(runner=runner).collect_host_profile(
            HostAccess(access="ssh", destination="dgx")
        )

    assert str(exc_info.value) == "could not execute target host inventory: unsafe safe"


@pytest.mark.parametrize("output_fd", [1, 2])
def test_ssh_adapter_bounds_streamed_stdout_and_stderr(
    tmp_path: Path,
    output_fd: int,
):
    fake_ssh = tmp_path / "fake-ssh"
    fake_ssh.write_text(
        f"""#!{sys.executable}
import os

chunk = b"x" * 65536
remaining = {MAX_PROBE_OUTPUT_CHARS + 1}
while remaining:
    written = os.write({output_fd}, chunk[:remaining])
    remaining -= written
""",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    access = HostAccess(access="ssh", destination="dgx", command_timeout_s=4)

    with pytest.raises(RuntimeError, match="exceeded"):
        OpenSSHClient(ssh_executable=str(fake_ssh)).collect_host_profile(access)


def test_ssh_adapter_starts_and_terminates_a_dedicated_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    fake_ssh = tmp_path / "fake-ssh"
    fake_ssh.write_text(
        f"#!{sys.executable}\nimport time\ntime.sleep(60)\n",
        encoding="utf-8",
    )
    fake_ssh.chmod(0o755)
    real_popen = ssh_module.subprocess.Popen
    real_killpg = ssh_module.os.killpg
    popen_kwargs: dict[str, Any] = {}
    signals: list[int] = []

    def recording_popen(*args: Any, **kwargs: Any):
        popen_kwargs.update(kwargs)
        return real_popen(*args, **kwargs)

    def recording_killpg(process_group: int, sent_signal: int):
        signals.append(sent_signal)
        return real_killpg(process_group, sent_signal)

    monkeypatch.setattr(ssh_module.subprocess, "Popen", recording_popen)
    monkeypatch.setattr(ssh_module.os, "killpg", recording_killpg)

    with pytest.raises(RuntimeError, match="timed out"):
        OpenSSHClient(ssh_executable=str(fake_ssh)).collect_host_profile(
            HostAccess(access="ssh", destination="dgx", command_timeout_s=0.1)
        )

    assert popen_kwargs["start_new_session"] is True
    assert ssh_module.signal.SIGTERM in signals
    assert ssh_module.signal.SIGKILL in signals


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
    machine_id = "0123456789abcdef0123456789abcdef"
    files = {
        "/etc/machine-id": machine_id,
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
    assert profile["host_fingerprint_source"] == "machine_id"
    assert profile["host_fingerprint_strength"] == "installation"
    assert machine_id not in json.dumps(profile)


def test_remote_probe_records_hashed_hardware_identity_without_changing_installation_identity(
    monkeypatch: pytest.MonkeyPatch,
):
    hardware_uuid = "12345678-1234-5678-9abc-def012345678"
    machine_id = "0123456789abcdef0123456789abcdef"
    files = {
        linux_dgx_probe.HARDWARE_UUID_PATHS[0]: hardware_uuid,
        "/etc/machine-id": machine_id,
    }
    monkeypatch.setattr(
        linux_dgx_probe,
        "_read_text",
        lambda path, limit=100_000: files.get(path),
    )
    monkeypatch.setattr(linux_dgx_probe, "_nvidia_profile", lambda: None)

    profile = linux_dgx_probe.collect()

    assert profile["host_fingerprint"].startswith("host-")
    assert profile["host_fingerprint_source"] == "machine_id"
    assert profile["host_fingerprint_strength"] == "installation"
    assert profile["host_hardware_fingerprint"].startswith("host-")
    serialized = json.dumps(profile)
    assert hardware_uuid not in serialized
    assert machine_id not in serialized


@pytest.mark.parametrize(
    "hardware_uuid",
    [
        "1234567890abcdef",
        "0" * 32,
        "f" * 32,
        "12345678-1234-5678-9abc-def01234567z",
    ],
)
def test_remote_probe_rejects_malformed_hardware_uuid(hardware_uuid: str):
    assert linux_dgx_probe._usable_hardware_uuid(hardware_uuid) is None


def test_remote_probe_canonicalizes_hardware_uuid_format():
    canonical = "12345678-1234-5678-9abc-def012345678"

    assert linux_dgx_probe._usable_hardware_uuid(canonical.upper()) == canonical
    assert linux_dgx_probe._usable_hardware_uuid(canonical.replace("-", "")) == canonical


def test_remote_probe_uses_hardware_identity_when_machine_id_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
):
    hardware_uuid = "12345678-1234-5678-9abc-def012345678"
    monkeypatch.setattr(
        linux_dgx_probe,
        "_read_text",
        lambda path, limit=100_000: (
            hardware_uuid if path == linux_dgx_probe.HARDWARE_UUID_PATHS[0] else None
        ),
    )

    fingerprint, source, strength, hardware_fingerprint, aliases = (
        linux_dgx_probe._machine_fingerprint("Linux", "spark", {})
    )

    assert fingerprint == hardware_fingerprint
    assert source == "dmi_product_uuid"
    assert strength == "hardware"
    assert aliases == []


def test_remote_and_local_linux_inventory_share_fingerprint_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    machine_id = "0123456789abcdef0123456789abcdef"
    hardware_uuid = "12345678-1234-5678-9abc-def012345678"
    files = {
        linux_dgx_probe.HARDWARE_UUID_PATHS[0]: hardware_uuid,
        linux_dgx_probe.MACHINE_ID_PATHS[0]: machine_id,
    }
    monkeypatch.setattr(
        linux_dgx_probe,
        "_read_text",
        lambda path, limit=100_000: files.get(path),
    )
    remote_fingerprint, remote_source, remote_strength, remote_hardware, remote_aliases = (
        linux_dgx_probe._machine_fingerprint(
            "Linux",
            "spark",
            {"machine": "aarch64", "model": "DGX Spark"},
        )
    )
    monkeypatch.setattr(system_module, "_machine_identifier", lambda _name: machine_id)
    monkeypatch.setattr(system_module, "_linux_hardware_uuid", lambda: hardware_uuid)
    local_fingerprint, local_source, local_strength, local_hardware, local_aliases = (
        system_module._current_host_identity(
            system_name="Linux",
            hostname="spark",
            hardware={"machine": "aarch64", "model": "DGX Spark"},
        )
    )

    assert (
        remote_fingerprint,
        remote_source,
        remote_strength,
        remote_hardware,
        remote_aliases,
    ) == (
        local_fingerprint,
        local_source,
        local_strength,
        local_hardware,
        local_aliases,
    )


def test_remote_probe_executes_with_python_3_10():
    source = ssh_module.linux_dgx_probe_source()
    python = os.environ.get("LLM_REFINERY_PYTHON_310") or shutil.which("python3.10")
    if python is None:
        pytest.skip("Python 3.10 is not installed; CI supplies it explicitly")

    completed = subprocess.run(
        [python, "-I", "-"],
        input=source,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    profile = json.loads(completed.stdout)
    assert profile["schema_version"] == 1
    assert profile["platform"]["python_version"].startswith("3.10.")


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
        "actual_strength": "installation",
        "verified": True,
    }


def test_resolver_accepts_pre_unification_strong_hardware_pin():
    raw = _target_mapping()
    raw["host"]["expected_fingerprint"] = "host-prior-hardware"
    profile = dict(_host().profile)
    profile.update(
        {
            "host_fingerprint": "host-canonical-installation",
            "host_fingerprint_source": "machine_id",
            "host_fingerprint_strength": "installation",
            "host_hardware_fingerprint": "host-prior-hardware",
            "host_fingerprint_aliases": [
                {
                    "fingerprint": "host-prior-hardware",
                    "source": "dmi_product_uuid",
                    "strength": "hardware",
                }
            ],
        }
    )
    resolver = TargetResolver(
        ssh_client=_FakeSSH(HostDiscovery(transport="ssh", destination="dgx", profile=profile)),  # type: ignore[arg-type]
        service_client=_FakeService(_service("served-model")),  # type: ignore[arg-type]
    )

    inspection = resolver.inspect(TargetSpec.from_mapping(raw))

    assert inspection.available is True
    assert inspection.safe_json()["host_identity_binding"] == {
        "expected_fingerprint": "host-prior-hardware",
        "actual_fingerprint": "host-prior-hardware",
        "actual_strength": "hardware",
        "verified": True,
    }


def test_resolver_rejects_weak_hostname_fingerprint_as_identity_pin():
    raw = _target_mapping()
    raw["host"]["expected_fingerprint"] = "host-example"
    profile = dict(_host().profile)
    profile["host_fingerprint_source"] = "hostname"
    profile["host_fingerprint_strength"] = "weak"
    resolver = TargetResolver(
        ssh_client=_FakeSSH(HostDiscovery(transport="ssh", destination="dgx", profile=profile)),  # type: ignore[arg-type]
        service_client=_FakeService(_service("served-model")),  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="not strong enough"):
        resolver.inspect(TargetSpec.from_mapping(raw))


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
    service = _FakeService(_service("served-model"))
    resolver = TargetResolver(
        ssh_client=_FakeSSH(HostDiscovery(transport="ssh", destination="dgx", profile=profile)),  # type: ignore[arg-type]
        service_client=service,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="fingerprint does not match") as exc_info:
        resolver.inspect(
            TargetSpec.from_mapping(raw),
            allow_service_unavailable=True,
        )

    assert service.calls == []
    inspection = exc_info.value.target_inspection
    assert inspection.host is not None
    assert inspection.host.profile == profile
    assert inspection.service is None
    assert inspection.safe_json()["status"] == "unavailable"


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
                    "max_tokens": 4096,
                    "maxTokens": 2048,
                    "api_key": "server-secret",
                    "hf_token": "server-secret",
                    "github_token": "server-secret",
                    "admin-token": "server-secret",
                    "secret_key": "server-secret",
                    "clientSecret": "server-secret",
                    "aws_secret_access_key": "server-secret",
                    "AWSSecretAccessKey": "server-secret",
                    "XApiKey": "server-secret",
                    "passwd": "server-secret",
                    "passphrase": "server-secret",
                    "nested": {
                        "authorization": "server-secret",
                        "password": "server-secret",
                        "database_password_hash": "server-secret",
                        "safe": True,
                    },
                },
            )
        if request.url.path == "/metrics":
            return httpx.Response(200, text="vllm:num_requests_running 0\n")
        return httpx.Response(404)

    monkeypatch.setenv("VLLM_API_KEY", "top-secret-api-key")
    discovery_client = _discovery_client(handler)
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
        "max_tokens": 4096,
        "maxTokens": 2048,
        "nested": {"safe": True},
    }
    assert metrics == "vllm:num_requests_running 0\n"


def test_openai_discovery_applies_target_transport_to_owned_clients(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    ca_bundle = tmp_path / "private-ca.pem"
    ca_bundle.write_text("fixture", encoding="utf-8")
    ssl_context = object()
    captured: dict[str, Any] = {}

    def create_context(*, cafile: str) -> object:
        captured["cafile"] = cafile
        return ssl_context

    monkeypatch.setattr(
        openai_discovery_module.ssl,
        "create_default_context",
        create_context,
    )

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs

    monkeypatch.setattr(openai_discovery_module.httpx, "Client", FakeClient)

    client = OpenAIDiscoveryClient()._new_client(
        TargetTransport(trust_env=False, ca_bundle=ca_bundle)
    )

    assert isinstance(client, FakeClient)
    assert captured["cafile"] == str(ca_bundle)
    assert captured["client_kwargs"] == {
        "timeout": 5.0,
        "follow_redirects": False,
        "trust_env": False,
        "verify": ssl_context,
    }


def test_openai_discovery_forces_owned_loopback_client_direct(monkeypatch):
    captured: dict[str, Any] = {}

    class FakeClient:
        def __init__(self, **kwargs: Any) -> None:
            captured["client_kwargs"] = kwargs
            self.is_closed = False

        def close(self) -> None:
            self.is_closed = True

    def bounded_response(_client: Any, url: str, **kwargs: Any) -> httpx.Response:
        del kwargs
        request = httpx.Request("GET", url)
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "test"}, request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "served"}]},
                request=request,
            )
        return httpx.Response(404, request=request)

    monkeypatch.setattr(openai_discovery_module.httpx, "Client", FakeClient)
    monkeypatch.setattr(openai_discovery_module, "_get_bounded_response", bounded_response)
    endpoint = EndpointSpec(
        name="local",
        protocol="openai_chat",
        base_url="http://127.0.0.1:8000/v1",
    )

    discovery = OpenAIDiscoveryClient().discover(
        endpoint,
        DiscoveryPolicy(server_info="off"),
    )

    assert discovery.models[0].id == "served"
    assert captured["client_kwargs"]["trust_env"] is False


@pytest.mark.parametrize(
    "model",
    [
        {"id": True},
        {"id": 123},
        {"id": 1.5},
        {"id": ""},
        {"id": "served", "max_model_len": True},
        {"id": "served", "max_model_len": 32768.5},
        {"id": "served", "max_model_len": 0},
        {"id": "served", "max_model_len": -1},
        {"id": "served", "root": 123},
        {"id": "served", "root": ""},
        {"id": "served", "owned_by": False},
        {"id": "served", "owned_by": ""},
    ],
)
def test_openai_discovery_rejects_malformed_model_metadata(model: dict[str, Any]):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "v"}, request=request)
        return httpx.Response(200, json={"data": [model]}, request=request)

    with pytest.raises(ConfigError, match="model discovery returned an invalid response"):
        _discovery_client(handler).discover(
            _spec().endpoint,
            DiscoveryPolicy(server_info="off"),
        )


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

    with pytest.raises(ConfigError, match="model discovery.*response is too large"):
        _discovery_client(handler).discover(_spec().endpoint, DiscoveryPolicy(server_info="off"))

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

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
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

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=MetricsStream(), request=request)

    with pytest.raises(RuntimeError, match="response is too large"):
        _discovery_client(handler).metrics(_spec().endpoint)

    assert chunks_read == 2


def test_openai_discovery_applies_total_stream_deadline(monkeypatch: pytest.MonkeyPatch):
    ticks = iter((0.0, 0.1, 6.0))
    monkeypatch.setattr(openai_discovery_module, "_monotonic", lambda: next(ticks))

    class TricklingStream(httpx.SyncByteStream):
        def __iter__(self):
            yield b"still arriving"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=TricklingStream(), request=request)

    with pytest.raises(RuntimeError, match="exceeded the total timeout"):
        _discovery_client(handler, timeout_s=5.0).metrics(_spec().endpoint)


def test_openai_discovery_waits_for_dequeued_deadline_callback(monkeypatch):
    callback_may_run = threading.Event()
    callback_started = threading.Event()
    release_callback = threading.Event()
    request_finished = threading.Event()

    class RacingTimer:
        def __init__(self, _interval: float, function: Callable[[], None]) -> None:
            self._function = function
            self._thread = threading.Thread(target=self._run, daemon=True)

        def _run(self) -> None:
            callback_may_run.wait(timeout=2)
            self._function()

        def start(self) -> None:
            self._thread.start()

        def cancel(self) -> None:
            # Model Timer.cancel() losing the race after the callback is dequeued.
            pass

        def join(self) -> None:
            self._thread.join(timeout=2)

    class BlockingCloseClient:
        is_closed = False

        def close(self) -> None:
            callback_started.set()
            if release_callback.wait(timeout=2):
                self.is_closed = True

    def bounded_response(*args: Any, **kwargs: Any) -> httpx.Response:
        del args, kwargs
        callback_may_run.set()
        if not callback_started.wait(timeout=1):
            raise AssertionError("deadline callback did not start")
        request = httpx.Request("GET", "http://spark.local:8000/health")
        return httpx.Response(200, request=request)

    monkeypatch.setattr(openai_discovery_module.threading, "Timer", RacingTimer)
    monkeypatch.setattr(
        openai_discovery_module,
        "_get_bounded_response_before_deadline",
        bounded_response,
    )
    client = BlockingCloseClient()
    result: list[httpx.Response] = []
    errors: list[BaseException] = []

    def request() -> None:
        try:
            result.append(
                openai_discovery_module._get_bounded_response(
                    client,  # type: ignore[arg-type]
                    "http://spark.local:8000/health",
                    headers={},
                    timeout_s=5.0,
                )
            )
        except BaseException as exc:  # pragma: no cover - surfaced below
            errors.append(exc)
        finally:
            request_finished.set()

    thread = threading.Thread(target=request)
    thread.start()
    assert callback_started.wait(timeout=1)
    assert not request_finished.wait(timeout=0.05)
    release_callback.set()
    thread.join(timeout=1)

    assert request_finished.is_set()
    assert errors == []
    assert result[0].status_code == 200
    assert client.is_closed is True


def test_openai_discovery_owns_and_closes_test_clients():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/version":
            return httpx.Response(200, json={"version": "test"}, request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "served"}]}, request=request)
        return httpx.Response(404, request=request)

    discovery_client = _discovery_client(handler)
    discovery = discovery_client.discover(
        _spec().endpoint,
        DiscoveryPolicy(server_info="off"),
    )

    assert discovery.models[0].id == "served"
    assert len(discovery_client.created_clients) == 1
    assert all(client.is_closed for client in discovery_client.created_clients)


def test_openai_discovery_replaces_owned_client_after_optional_endpoint_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, request=request)
        if request.url.path == "/v1/models":
            return httpx.Response(
                200,
                json={"data": [{"id": "served"}]},
                request=request,
            )
        return httpx.Response(404, request=request)

    original_request = openai_discovery_module._get_bounded_response

    def bounded_response(client: httpx.Client, url: str, **kwargs: Any) -> httpx.Response:
        if url.endswith("/version"):
            client.close()
            raise httpx.ReadTimeout(
                "version timed out",
                request=httpx.Request("GET", url),
            )
        return original_request(client, url, **kwargs)

    monkeypatch.setattr(openai_discovery_module, "_get_bounded_response", bounded_response)
    discovery_client = _discovery_client(handler)
    discovery = discovery_client.discover(
        _spec().endpoint,
        DiscoveryPolicy(server_info="off"),
    )

    assert discovery.models[0].id == "served"
    assert len(discovery_client.created_clients) == 2
    assert all(client.is_closed for client in discovery_client.created_clients)


def test_openai_discovery_interrupts_trickled_response_headers(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        _REAL_GETADDRINFO,
    )

    class TrickleHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(65536)
            self.request.sendall(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
            for _ in range(100):
                time.sleep(0.02)
                try:
                    self.request.sendall(b"x")
                except OSError:
                    return

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), TrickleHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = EndpointSpec(
        name="local",
        protocol="openai_chat",
        base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
    )
    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="exceeded the total timeout"):
            OpenAIDiscoveryClient(timeout_s=0.08).metrics(endpoint)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert time.perf_counter() - started < 1


def test_openai_discovery_bounds_and_closes_factory_client(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        _REAL_GETADDRINFO,
    )

    class TrickleHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(65536)
            self.request.sendall(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
            for _ in range(100):
                time.sleep(0.02)
                try:
                    self.request.sendall(b"x")
                except OSError:
                    return

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), TrickleHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = EndpointSpec(
        name="local",
        protocol="openai_chat",
        base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
    )
    started = time.perf_counter()
    discovery_client = _TrackingOpenAIDiscoveryClient(timeout_s=0.08)
    try:
        with pytest.raises(RuntimeError, match="exceeded the total timeout"):
            discovery_client.metrics(endpoint)
        elapsed = time.perf_counter() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert elapsed < 1
    assert len(discovery_client.created_clients) == 1
    assert discovery_client.created_clients[0].is_closed is True


def test_openai_discovery_requires_configured_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"must not send request: {request.url}")

    with pytest.raises(ConfigError, match="VLLM_API_KEY"):
        _discovery_client(handler).discover(
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

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"must not send request: {request.url}")

    with pytest.raises(ConfigError) as caught:
        _discovery_client(handler).discover(endpoint, DiscoveryPolicy())

    assert "super-secret" not in str(caught.value)
    assert "Injected" not in str(caught.value)


def test_offline_tolerance_does_not_suppress_missing_api_key_config(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("VLLM_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"must not send request: {request.url}")

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
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

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
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

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
    )
    inspection = resolver.inspect(_spec())

    assert inspection.available is True
    assert "server_info: HTTP 403" in inspection.errors


def test_offline_tolerance_does_not_suppress_reachable_wrong_http_service():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(404, request=request)

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
    )
    with pytest.raises(ConfigError, match="health failed with HTTP 404"):
        resolver.inspect(_spec(), allow_service_unavailable=True)

    assert [request.url.path for request in requests] == ["/health"]


def test_discovery_rejects_cross_origin_redirect_before_following_it(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"location": "http://127.0.0.1:8000/health"},
            request=request,
        )

    with pytest.raises(ConfigError, match="remain on the configured.*origin"):
        _discovery_client(handler).discover(_spec().endpoint, DiscoveryPolicy(server_info="off"))

    assert len(requests) == 1


def test_discovery_rejects_dns_name_that_resolves_to_client_loopback(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("127.0.0.1", port))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"must not send request: {request.url}")

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
    )
    with pytest.raises(ConfigError, match="client-local"):
        resolver.inspect(_spec(), allow_service_unavailable=True)


def test_discovery_requires_ssh_endpoint_hostname_to_resolve(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_resolution(host: str, port: int, **kwargs: Any) -> Any:
        raise OSError("temporary DNS failure")

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        fail_resolution,
    )
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(error="connection refused"),  # type: ignore[arg-type]
    )

    with pytest.raises(ConfigError, match="could not be resolved for safety validation"):
        resolver.inspect(_spec(), allow_service_unavailable=True)


@pytest.mark.parametrize(
    ("service_required", "allow_service_unavailable"),
    [(True, True), (False, False)],
)
def test_local_inspection_tolerates_unresolvable_service_without_retrying_http(
    monkeypatch: pytest.MonkeyPatch,
    service_required: bool,
    allow_service_unavailable: bool,
):
    def fail_resolution(host: str, port: int, **kwargs: Any) -> Any:
        raise socket.gaierror("not found")

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        fail_resolution,
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, request=request)

    raw = _target_mapping(access="local", destination=None)
    raw["discovery"]["service_required"] = service_required
    resolver = TargetResolver(
        service_client=_discovery_client(handler, timeout_s=0.02),
        local_system_profile=lambda: {"hostname": "local-mac", "schema_version": 2},
    )

    inspection = resolver.inspect(
        TargetSpec.from_mapping(raw),
        allow_service_unavailable=allow_service_unavailable,
    )

    assert requests == []
    assert inspection.available is False
    assert inspection.host is not None
    assert inspection.host.profile["hostname"] == "local-mac"
    assert inspection.service is None
    assert inspection.errors == (
        "service: configured endpoint hostname could not be resolved for safety validation",
    )


def test_local_inspection_tolerates_bounded_dns_timeout_without_http_request(
    monkeypatch: pytest.MonkeyPatch,
):
    release = threading.Event()

    def block_resolution(host: str, port: int, **kwargs: Any) -> Any:
        release.wait(timeout=2)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.41", port))]

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        block_resolution,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"must not send request: {request.url}")

    resolver = TargetResolver(
        service_client=_discovery_client(handler, timeout_s=0.02),
        local_system_profile=lambda: {"hostname": "local-mac", "schema_version": 2},
    )
    started = time.perf_counter()
    try:
        inspection = resolver.inspect(
            _spec(access="local", destination=None),
            allow_service_unavailable=True,
        )
    finally:
        release.set()

    assert time.perf_counter() - started < 0.5
    assert inspection.available is False
    assert inspection.service is None
    assert inspection.errors == (
        "service: configured endpoint hostname resolution exceeded its timeout",
    )


@pytest.mark.parametrize("failure", ["unresolvable", "timeout"])
def test_required_local_inspection_fails_closed_on_resolution_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
):
    release = threading.Event()

    def fail_resolution(host: str, port: int, **kwargs: Any) -> Any:
        if failure == "timeout":
            release.wait(timeout=2)
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.168.1.41", port))]
        raise socket.gaierror("not found")

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        fail_resolution,
    )

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"must not send request: {request.url}")

    resolver = TargetResolver(
        service_client=_discovery_client(handler, timeout_s=0.02),
        local_system_profile=lambda: {"hostname": "local-mac", "schema_version": 2},
    )
    try:
        with pytest.raises(
            ConfigError, match="hostname (could not be resolved|resolution exceeded)"
        ):
            resolver.inspect(_spec(access="local", destination=None))
    finally:
        release.set()


def test_local_offline_tolerance_does_not_suppress_unsafe_resolution(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", port))
        ],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail(f"must not send request: {request.url}")

    resolver = TargetResolver(
        service_client=_discovery_client(handler),
        local_system_profile=lambda: {"hostname": "local-mac", "schema_version": 2},
    )

    with pytest.raises(ConfigError, match="client-local"):
        resolver.inspect(
            _spec(access="local", destination=None),
            allow_service_unavailable=True,
        )


def test_discovery_rejects_endpoint_address_assigned_to_benchmark_client(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety._is_client_interface_address",
        lambda address, route_address, port: True,
    )
    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_FakeService(_service("served-model")),  # type: ignore[arg-type]
    )

    with pytest.raises(ConfigError, match="assigned to the benchmark client"):
        resolver.inspect(_spec())


def test_local_route_cache_cannot_bypass_later_ssh_client_identity_check(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety._is_client_interface_address",
        lambda address, route_address, port: True,
    )
    resolver = TargetResolver(
        service_client=_FakeService(_service("served-model")),  # type: ignore[arg-type]
    )

    local_route = resolver._service_route(_spec(access="local", destination=None))

    assert local_route is not None
    with pytest.raises(ConfigError, match="assigned to the benchmark client"):
        resolver._service_route(_spec())


def test_discovery_allows_dgx_local_name_that_resolves_to_lan_address(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "192.168.1.41"
        assert request.headers["host"] == "spark.local:8000"
        assert request.extensions["sni_hostname"] == "spark.local"
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

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
    )
    inspection = resolver.inspect(_spec())

    assert inspection.service is not None
    assert inspection.service.health == "ok"
    assert [model.id for model in inspection.service.models] == ["served-model"]
    assert inspection.safe_json()["route"]["selected_address"] == "192.168.1.41"


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
    discovery = _discovery_client(handler).discover(
        endpoint,
        DiscoveryPolicy(),
    )

    assert len(requests) == 1
    assert requests[0].url.path == "/health"
    assert discovery.health == "unavailable"
    assert discovery.models == ()
    assert discovery.errors == ("health: connection refused",)


def test_offline_tolerance_accepts_connection_reset_read_error():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadError("connection reset", request=request)

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
    )
    inspection = resolver.inspect(_spec(), allow_service_unavailable=True)

    assert inspection.available is False
    assert inspection.errors == ("health: connection reset",)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"), "TLS"),
        (httpx.ProxyError("proxy unavailable"), "proxy connection"),
        (httpx.RemoteProtocolError("invalid response framing"), "HTTP protocol"),
        (httpx.UnsupportedProtocol("unsupported scheme"), "unsupported HTTP protocol"),
    ],
)
def test_offline_tolerance_does_not_suppress_fatal_transport_errors(
    error: httpx.TransportError,
    message: str,
):
    def handler(request: httpx.Request) -> httpx.Response:
        error.request = request
        raise error

    resolver = TargetResolver(
        ssh_client=_FakeSSH(),  # type: ignore[arg-type]
        service_client=_discovery_client(handler),
    )
    with pytest.raises(ConfigError, match=message):
        resolver.inspect(_spec(), allow_service_unavailable=True)


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

    service = _FakeService(_service("served-model"))
    resolver = TargetResolver(
        ssh_client=FailingSSH(),  # type: ignore[arg-type]
        service_client=service,  # type: ignore[arg-type]
    )

    with pytest.raises(RuntimeError, match="host: permission denied") as exc_info:
        resolver.inspect(_spec(), allow_service_unavailable=True)

    assert service.calls == []
    inspection = exc_info.value.target_inspection
    assert inspection.host is None
    assert inspection.service is None
    assert inspection.errors == ("host: permission denied",)


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
