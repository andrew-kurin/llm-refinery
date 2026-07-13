from __future__ import annotations

import json
import sys
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest
from click.testing import CliRunner

from llm_refinery.application.target_discovery import TargetResolver
from llm_refinery.cli import main
from llm_refinery.core.targets import (
    HostDiscovery,
    ModelDescriptor,
    ServiceDiscovery,
    TargetInspection,
    TargetSpec,
)
from llm_refinery.providers.openai_discovery import (
    OpenAIDiscoveryClient,
    _get_bounded_response,
    _json_object,
    _sanitize,
)


def test_target_route_resolution_uses_discovery_timeout_budget(monkeypatch):
    spec = TargetSpec.from_mapping(
        {
            "name": "spark",
            "host": {"access": "ssh", "destination": "dgx"},
            "endpoint": {
                "protocol": "openai_chat",
                "base_url": "http://spark.local:8000/v1",
            },
            "model": {"selection": "single"},
        }
    )
    captured: dict[str, Any] = {}

    def resolve_route(url: str, **kwargs: Any):
        captured.update(url=url, **kwargs)
        return None

    monkeypatch.setattr(
        "llm_refinery.application.target_discovery.resolve_request_route",
        resolve_route,
    )
    resolver = TargetResolver(service_client=OpenAIDiscoveryClient(timeout_s=0.125))

    assert resolver._service_route(spec) is None
    assert captured == {
        "url": spec.endpoint.base_url,
        "require_resolution": True,
        "reject_client_local": True,
        "resolution_timeout_s": 0.125,
    }


def test_local_network_target_route_also_uses_discovery_timeout_budget(monkeypatch):
    spec = TargetSpec.from_mapping(
        {
            "name": "network-service",
            "host": {"access": "local"},
            "endpoint": {
                "protocol": "openai_chat",
                "base_url": "http://service.local:8000/v1",
            },
            "model": {"selection": "single"},
        }
    )
    captured: dict[str, Any] = {}

    def resolve_route(url: str, **kwargs: Any):
        captured.update(url=url, **kwargs)
        return None

    monkeypatch.setattr(
        "llm_refinery.application.target_discovery.resolve_request_route",
        resolve_route,
    )
    resolver = TargetResolver(service_client=OpenAIDiscoveryClient(timeout_s=0.125))

    assert resolver._service_route(spec) is None
    assert captured == {
        "url": spec.endpoint.base_url,
        "require_resolution": True,
        "reject_client_local": False,
        "resolution_timeout_s": 0.125,
    }


@pytest.mark.parametrize(
    "timeout_s",
    [True, "1", 0, -1, float("nan"), float("inf"), threading.TIMEOUT_MAX + 1],
)
def test_discovery_client_rejects_invalid_timeout_values(timeout_s: Any):
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        OpenAIDiscoveryClient(timeout_s=timeout_s)


def test_discovery_requests_identity_encoding_and_rejects_encoded_body_before_reading():
    body_was_read = False

    class EncodedStream(httpx.SyncByteStream):
        def __iter__(self):
            nonlocal body_was_read
            body_was_read = True
            yield b"small compressed representation"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept-encoding"] == "identity"
        return httpx.Response(
            200,
            headers={"content-encoding": "gzip"},
            stream=EncodedStream(),
            request=request,
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ValueError, match="compressed discovery responses"),
    ):
        _get_bounded_response(
            client,
            "http://127.0.0.1:8000/v1/models",
            headers={"Accept-Encoding": "gzip"},
            timeout_s=1,
        )

    assert body_was_read is False


@pytest.mark.parametrize(
    "secret_key",
    ["apiKeyValue", "encryptionKey", "signingKey", "azureStorageKey", "AWSAccessKeyId"],
)
def test_server_info_sanitizer_removes_credential_key_variants(secret_key: str):
    assert _sanitize({"dtype": "bfloat16", secret_key: "must-not-persist"}) == {"dtype": "bfloat16"}


def test_json_parser_translates_recursion_failure_to_bounded_validation_error():
    depth = sys.getrecursionlimit() + 100
    content = ('{"nested":' * depth + "null" + "}" * depth).encode()
    response = httpx.Response(200, content=content)

    with pytest.raises(ValueError, match="too deeply nested"):
        _json_object(response)


def test_server_info_sanitizer_rejects_deep_preparsed_values_without_recursing():
    value: dict[str, Any] = {}
    cursor = value
    for _ in range(100):
        child: dict[str, Any] = {}
        cursor["nested"] = child
        cursor = child

    with pytest.raises(ValueError, match="too deeply nested"):
        _sanitize(value)


def _target_config(path: Path) -> TargetSpec:
    path.write_text(
        """
name: spark
host:
  access: local
endpoint:
  protocol: openai_chat
  base_url: http://127.0.0.1:8000/v1
model:
  selection: single
""",
        encoding="utf-8",
    )
    return TargetSpec.from_mapping(
        {
            "name": "spark",
            "host": {"access": "local"},
            "endpoint": {
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8000/v1",
            },
            "model": {"selection": "single"},
        }
    )


def test_target_human_output_sanitizes_remote_values_but_json_preserves_them(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = tmp_path / "target.yaml"
    spec = _target_config(config)
    unsafe_hostname = "spark\x1b]2;forged terminal title\x07-host"
    unsafe_model = "served\x1b[31m-red"
    unsafe_error = "model\nforged-line\u2028forged-separator\u2029forged-paragraph"
    inspection = TargetInspection(
        spec=spec,
        host=HostDiscovery(
            transport="local",
            destination=None,
            profile={"hostname": unsafe_hostname, "hardware": {"model": "DGX"}},
        ),
        service=ServiceDiscovery(
            implementation="vllm",
            base_url=spec.endpoint.base_url,
            health="ok",
            version="1.0",
            models=(ModelDescriptor(id=unsafe_model),),
        ),
        resolved=None,
        errors=(unsafe_error,),
    )

    class FakeResolver:
        def inspect(self, *_args: Any, **_kwargs: Any) -> TargetInspection:
            return inspection

    monkeypatch.setattr("llm_refinery.commands.targets.TargetResolver", FakeResolver)
    runner = CliRunner()

    human = runner.invoke(main, ["target", "inspect", str(config)])
    assert human.exit_code == 0, human.output
    assert "\x1b" not in human.output
    assert "forged terminal title" not in human.output
    assert "\nforged-line" not in human.output
    assert "\u2028" not in human.output
    assert "\u2029" not in human.output

    machine = runner.invoke(main, ["target", "inspect", str(config), "--json"])
    assert machine.exit_code == 0, machine.output
    payload = json.loads(machine.output)
    assert payload["host"]["profile"]["hostname"] == unsafe_hostname
    assert payload["service"]["models"][0]["id"] == unsafe_model
    assert payload["errors"] == [unsafe_error]


def test_target_human_error_sanitizes_remote_control_sequences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config = tmp_path / "target.yaml"
    _target_config(config)

    class FailingResolver:
        def inspect(self, *_args: Any, **_kwargs: Any) -> TargetInspection:
            raise RuntimeError("failure \x1b]2;forged terminal title\x07 safely reported")

    monkeypatch.setattr("llm_refinery.commands.targets.TargetResolver", FailingResolver)

    result = CliRunner().invoke(main, ["target", "inspect", str(config)])

    assert result.exit_code == 1
    assert "forged terminal title" not in result.output
    assert "failure  safely reported" in result.output
