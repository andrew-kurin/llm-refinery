import socket
import threading
import time

import httpx
import pytest

from llm_refinery.core.config import ConfigError
from llm_refinery.core.http_safety import (
    environment_proxy_applies,
    http_origin,
    pinned_route_trust_env,
    resolve_request_route,
    validate_request_url,
)


def test_unicode_and_punycode_hosts_share_httpx_idna2008_origin(monkeypatch):
    resolved_hosts: list[str] = []

    def resolve(host, port, **kwargs):
        resolved_hosts.append(host)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.41", port))]

    monkeypatch.setattr("llm_refinery.core.http_safety.socket.getaddrinfo", resolve)

    unicode_url = "https://faß.de:8443/v1"
    punycode_url = "https://xn--fa-hia.de:8443/v1/models"
    route = resolve_request_route(unicode_url, require_resolution=True)

    assert (
        http_origin(unicode_url)
        == http_origin(punycode_url)
        == (
            "https",
            "xn--fa-hia.de",
            8443,
        )
    )
    assert resolved_hosts == ["xn--fa-hia.de"]
    assert route is not None
    assert route.authority == "xn--fa-hia.de:8443"
    assert route.sni_hostname == "xn--fa-hia.de"
    assert route.request_url(punycode_url) == "https://192.0.2.41:8443/v1/models"


@pytest.mark.parametrize("operation", ["validate", "route"])
def test_hostname_resolution_honors_explicit_wall_clock_timeout(monkeypatch, operation):
    release = threading.Event()

    def blocked_resolution(*args, **kwargs):
        release.wait(timeout=2)
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.41", 8000))]

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        blocked_resolution,
    )
    started = time.perf_counter()
    try:
        with pytest.raises(ConfigError, match="resolution exceeded its timeout"):
            if operation == "validate":
                validate_request_url(
                    "http://slow.example:8000/v1",
                    require_resolution=True,
                    resolution_timeout_s=0.02,
                )
            else:
                resolve_request_route(
                    "http://slow.example:8000/v1",
                    require_resolution=True,
                    resolution_timeout_s=0.02,
                )
    finally:
        release.set()

    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    "timeout_s",
    [True, "1", 0, -1, float("nan"), float("inf"), threading.TIMEOUT_MAX + 1],
)
def test_hostname_resolution_rejects_invalid_timeout_values(timeout_s):
    with pytest.raises(ValueError, match="resolution_timeout_s must be positive"):
        resolve_request_route(
            "http://remote.example:8000/v1",
            require_resolution=True,
            resolution_timeout_s=timeout_s,
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


def test_pinned_route_disables_environment_mounts_when_no_proxy_is_configured(monkeypatch):
    calls = 0

    def no_proxies():
        nonlocal calls
        calls += 1
        return {}

    monkeypatch.setattr("llm_refinery.core.http_safety.getproxies", no_proxies)

    assert (
        pinned_route_trust_env(
            "http://dgx.local:8000/v1",
            trust_env=True,
            route_is_pinned=True,
        )
        is False
    )
    assert calls == 1


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


@pytest.mark.parametrize(
    ("url", "no_proxy"),
    [
        ("http://dgx.local:8000/v1", "dgx.local"),
        ("http://dgx.local.:8000/v1", "dgx.local"),
        ("http://dgx.local:8000/v1", "dgx.local."),
        ("http://dgx.local.:8000/v1", "dgx.local."),
        ("http://dgx.local:8000/v1", "http://dgx.local:8000"),
        ("http://dgx.local.:8000/v1", "http://dgx.local:8000"),
        ("http://dgx.local:8000/v1", "http://dgx.local.:8000"),
        ("http://dgx.local.:8000/v1", "http://dgx.local.:8000"),
    ],
)
def test_environment_proxy_trailing_dot_matches_httpx_mount_selection(url, no_proxy):
    selected_transport: list[str] = []

    def direct_handler(_request: httpx.Request) -> httpx.Response:
        selected_transport.append("direct")
        return httpx.Response(200)

    def proxy_handler(_request: httpx.Request) -> httpx.Response:
        selected_transport.append("proxy")
        return httpx.Response(200)

    direct_transport = httpx.MockTransport(direct_handler)
    proxy_transport = httpx.MockTransport(proxy_handler)
    bypass_pattern = no_proxy if "://" in no_proxy else f"all://*{no_proxy}"
    with httpx.Client(
        transport=direct_transport,
        mounts={"http://": proxy_transport, bypass_pattern: None},
    ) as client:
        client.get(url)

    assert environment_proxy_applies(
        url,
        proxies={
            "http": "http://proxy.test:3128",
            "no": no_proxy,
        },
    ) is (selected_transport == ["proxy"])


@pytest.mark.parametrize(
    ("no_proxy", "bypass_pattern"),
    [
        ("xn--bcher-kva.example", "all://*xn--bcher-kva.example"),
        (
            "http://xn--bcher-kva.example:8000",
            "http://xn--bcher-kva.example:8000",
        ),
    ],
)
def test_environment_proxy_idna_matches_httpx_mount_selection(no_proxy, bypass_pattern):
    url = "http://xn--bcher-kva.example:8000/v1"
    selected_transport: list[str] = []

    direct_transport = httpx.MockTransport(
        lambda _request: selected_transport.append("direct") or httpx.Response(200)
    )
    proxy_transport = httpx.MockTransport(
        lambda _request: selected_transport.append("proxy") or httpx.Response(200)
    )
    with httpx.Client(
        transport=direct_transport,
        mounts={"http://": proxy_transport, bypass_pattern: None},
    ) as client:
        client.get(url)

    assert environment_proxy_applies(
        url,
        proxies={
            "http": "http://proxy.test:3128",
            "no": no_proxy,
        },
    ) is (selected_transport == ["proxy"])


@pytest.mark.parametrize(
    "no_proxy",
    [
        "bücher.example",
        "http://xn--bcher-kva.example:8000,bücher.example",
    ],
)
def test_environment_proxy_treats_invalid_unicode_no_proxy_pattern_as_active(no_proxy):
    assert (
        environment_proxy_applies(
            "http://xn--bcher-kva.example:8000/v1",
            proxies={
                "http": "http://proxy.test:3128",
                "no": no_proxy,
            },
        )
        is True
    )


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
