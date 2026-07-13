import socket

import pytest

from llm_refinery.core.config import ConfigError
from llm_refinery.core.http_safety import (
    environment_proxy_applies,
    pinned_route_trust_env,
    resolve_request_route,
    validate_request_url,
)


def test_resolved_route_pins_safe_address_and_preserves_logical_authority(monkeypatch):
    resolutions = 0

    def resolve(host, port, **kwargs):
        nonlocal resolutions
        resolutions += 1
        return [(2, 1, 6, "", ("192.168.1.41", port))]

    monkeypatch.setattr("llm_refinery.core.http_safety.socket.getaddrinfo", resolve)

    route = resolve_request_route(
        "https://DGX.local:8443/v1/models?details=1",
        require_resolution=True,
    )

    assert route is not None
    assert resolutions == 1
    assert route.request_url("https://dgx.local:8443/v1/models?details=1") == (
        "https://192.168.1.41:8443/v1/models?details=1"
    )
    assert route.request_headers({"Accept": "application/json", "host": "wrong"}) == {
        "Accept": "application/json",
        "Host": "dgx.local:8443",
    }
    assert route.sni_hostname == "dgx.local"


def test_resolved_route_preserves_scoped_ipv6_for_socket_connection(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("fe80::1", port, 0, 7))
        ],
    )
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.if_indextoname",
        lambda index: "en7",
    )

    route = resolve_request_route(
        "http://dgx.local:8000/v1",
        require_resolution=True,
    )

    assert route is not None
    assert route.connect_host == "fe80::1%en7"
    assert route.request_url("http://dgx.local:8000/v1/models") == (
        "http://[fe80::1%en7]:8000/v1/models"
    )


def test_pinned_route_rejects_active_environment_proxy(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128"},
    )
    with pytest.raises(ConfigError, match="cannot use an environment proxy"):
        pinned_route_trust_env("http://dgx.local:8000/v1", trust_env=True, route_is_pinned=True)


def test_pinned_route_disables_proxy_mounts_when_logical_host_is_bypassed(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128", "no": "dgx.local:8000"},
    )

    assert (
        pinned_route_trust_env("http://dgx.local:8000/v1", trust_env=True, route_is_pinned=True)
        is False
    )


def test_pinned_route_no_proxy_port_must_match(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128", "no": "dgx.local:9000"},
    )

    with pytest.raises(ConfigError, match="cannot use an environment proxy"):
        pinned_route_trust_env("http://dgx.local:8000/v1", trust_env=True, route_is_pinned=True)


def test_scheme_qualified_no_proxy_entry_matches_only_its_exact_host(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128", "no": "http://dgx.local:8000"},
    )

    assert (
        pinned_route_trust_env("http://dgx.local:8000/v1", trust_env=True, route_is_pinned=True)
        is False
    )
    with pytest.raises(ConfigError, match="cannot use an environment proxy"):
        pinned_route_trust_env(
            "http://other.dgx.local:8000/v1", trust_env=True, route_is_pinned=True
        )


def test_scheme_qualified_no_proxy_default_port_matches_httpx_normalization(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128", "no": "http://dgx.local:80"},
    )

    assert (
        pinned_route_trust_env("http://dgx.local:8000/v1", trust_env=True, route_is_pinned=True)
        is False
    )


@pytest.mark.parametrize("no_proxy", ["all://dgx.local:80", "HTTP://DGX.LOCAL:80"])
def test_no_proxy_patterns_retain_non_normalized_default_ports(monkeypatch, no_proxy):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128", "no": no_proxy},
    )

    with pytest.raises(ConfigError, match="cannot use an environment proxy"):
        pinned_route_trust_env("http://dgx.local:80/v1", trust_env=True, route_is_pinned=True)


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:8000/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
    ],
)
def test_explicit_loopback_targets_always_disable_environment_proxies(monkeypatch, url):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128"},
    )

    assert pinned_route_trust_env(url, trust_env=True) is False


def test_route_less_remote_target_preserves_environment_proxy_support(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.test:3128"},
    )

    assert pinned_route_trust_env(
        "http://dgx.local:8000/v1",
        trust_env=True,
        route_is_pinned=False,
    )


def test_environment_proxy_applies_respects_no_proxy(monkeypatch):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {
            "http": "http://proxy.test:3128",
            "no": "direct.example",
        },
    )

    assert environment_proxy_applies("http://proxied.example:8000/v1") is True
    assert environment_proxy_applies("http://direct.example:8000/v1") is False


@pytest.mark.parametrize("port", ["", "0"])
def test_request_url_rejects_invalid_explicit_port(port: str):
    with pytest.raises(ConfigError, match="invalid port"):
        validate_request_url(f"http://dgx.example:{port}/v1")


def test_required_hostname_resolution_fails_closed(monkeypatch: pytest.MonkeyPatch):
    def fail_resolution(*args, **kwargs):
        raise socket.gaierror("not found")

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        fail_resolution,
    )

    with pytest.raises(ConfigError, match="could not be resolved for safety validation"):
        validate_request_url(
            "http://dgx-unresolvable.example:8000/v1",
            require_resolution=True,
        )


def test_optional_hostname_resolution_preserves_offline_transport_semantics(
    monkeypatch: pytest.MonkeyPatch,
):
    def fail_resolution(*args, **kwargs):
        raise socket.gaierror("not found")

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        fail_resolution,
    )

    assert validate_request_url("http://offline.example:8000/v1") == (
        "http",
        "offline.example",
        8000,
    )


@pytest.mark.parametrize("host", ["0.0.0.0", "[::]"])
def test_explicit_wildcard_address_is_never_a_request_target(host: str):
    with pytest.raises(ConfigError, match="wildcard address"):
        validate_request_url(f"http://{host}:8000/v1")
    with pytest.raises(ConfigError, match="wildcard address"):
        resolve_request_route(f"http://{host}:8000/v1", require_resolution=True)


@pytest.mark.parametrize("mapped", ["::ffff:127.0.0.1", "::ffff:0.0.0.0"])
def test_ipv4_mapped_ipv6_cannot_bypass_local_address_guard(monkeypatch, mapped: str):
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", (mapped, port, 0, 0))
        ],
    )

    with pytest.raises(ConfigError, match="client-local or wildcard"):
        validate_request_url("http://dgx.local:8000/v1")
    with pytest.raises(ConfigError, match="client-local or wildcard"):
        resolve_request_route("http://dgx.local:8000/v1", require_resolution=True)
