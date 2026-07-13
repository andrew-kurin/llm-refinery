from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from typing import TypeAlias
from urllib.parse import urlparse, urlsplit, urlunsplit
from urllib.request import getproxies

import httpx

from llm_refinery.core.config import ConfigError

HttpOrigin: TypeAlias = tuple[str, str, int]


@dataclass(frozen=True)
class PinnedHttpRoute:
    """Bind a validated logical HTTP origin to one non-local connection address."""

    origin: HttpOrigin
    connect_host: str
    authority: str
    sni_hostname: str

    def request_url(self, logical_url: str) -> str:
        validate_request_url(
            logical_url,
            expected_origin=self.origin,
            resolve_addresses=False,
        )
        parsed = urlsplit(logical_url)
        # HTTPX/httpcore pass the parsed host directly to socket APIs. Keep an
        # IPv6 scope suffix as ``%interface`` rather than percent-encoding it.
        connect_host = self.connect_host
        if ":" in connect_host:
            connect_host = f"[{connect_host}]"
        explicit_port = parsed.port
        netloc = f"{connect_host}:{explicit_port}" if explicit_port is not None else connect_host
        return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, ""))

    def request_headers(self, headers: dict[str, str]) -> dict[str, str]:
        result = {key: value for key, value in headers.items() if key.casefold() != "host"}
        result["Host"] = self.authority
        return result

    def safe_json(self) -> dict[str, object]:
        scheme, hostname, port = self.origin
        return {
            "logical_origin": {"scheme": scheme, "hostname": hostname, "port": port},
            "selected_address": self.connect_host,
            "authority": self.authority,
        }


def http_origin(url: str) -> HttpOrigin:
    """Return a normalized HTTP origin or fail closed for an invalid URL."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
        raise ConfigError("HTTP request URL must include an HTTP(S) origin")
    if parsed.username is not None or parsed.password is not None:
        raise ConfigError("HTTP request URL cannot include user information")
    if parsed.fragment:
        raise ConfigError("HTTP request URL cannot include a fragment")
    try:
        explicit_port = parsed.port
    except ValueError as exc:
        raise ConfigError("HTTP request URL includes an invalid port") from exc
    if explicit_port == 0 or parsed.netloc.endswith(":"):
        raise ConfigError("HTTP request URL includes an invalid port")
    port = explicit_port if explicit_port is not None else (443 if parsed.scheme == "https" else 80)
    hostname = parsed.hostname.casefold().rstrip(".")
    return parsed.scheme, hostname, port


def validate_request_url(
    url: str,
    *,
    expected_origin: HttpOrigin | None = None,
    resolve_addresses: bool = True,
    require_resolution: bool = False,
) -> HttpOrigin:
    """Reject cross-origin requests and DNS names that resolve back to this client.

    Explicit loopback URLs remain valid for the harness's local-target mode. A
    non-loopback hostname resolving to loopback or an unspecified address is
    rejected, which prevents a remote target name from being attributed to a
    service on the benchmark client.
    """
    if require_resolution and not resolve_addresses:
        raise ValueError("require_resolution needs resolve_addresses=True")
    origin = http_origin(url)
    if expected_origin is not None and origin != expected_origin:
        raise ConfigError("HTTP redirect must remain on the configured endpoint origin")

    _, hostname, port = origin
    explicit_address = _explicit_ip_address(hostname)
    if explicit_address is not None and explicit_address.is_unspecified:
        raise ConfigError("HTTP request URL cannot use a wildcard address")
    if _is_explicit_client_loopback_host(hostname):
        return origin
    if not resolve_addresses:
        return origin
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        if require_resolution:
            raise ConfigError(
                "configured endpoint hostname could not be resolved for safety validation"
            ) from exc
        # Preserve the normal HTTP client's DNS/connection error semantics so an
        # offline hostname can still be handled by offline-tolerant inspection.
        return origin
    for address_info in addresses:
        address_text = str(address_info[4][0]).split("%", 1)[0]
        try:
            address = _normalized_address(ipaddress.ip_address(address_text))
        except ValueError:
            continue
        if address.is_loopback or address.is_unspecified:
            raise ConfigError(
                "configured endpoint hostname resolves to a client-local or wildcard address"
            )
    return origin


def resolve_request_route(
    url: str,
    *,
    require_resolution: bool,
    reject_client_local: bool = False,
) -> PinnedHttpRoute | None:
    """Resolve and validate once, returning a route that avoids a second DNS lookup."""
    origin = http_origin(url)
    _, hostname, port = origin
    explicit_address = _explicit_ip_address(hostname)
    if explicit_address is not None and explicit_address.is_unspecified:
        raise ConfigError("HTTP request URL cannot use a wildcard address")
    if _is_explicit_client_loopback_host(hostname):
        if reject_client_local:
            raise ConfigError(
                "SSH target endpoint must not point to the benchmark client's local host"
            )
        return None
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        if require_resolution:
            raise ConfigError(
                "configured endpoint hostname could not be resolved for safety validation"
            ) from exc
        return None

    safe_addresses: list[tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, str]] = []
    for address_info in addresses:
        route_address = _route_address(address_info)
        address_text = route_address.split("%", 1)[0]
        try:
            route_ip = ipaddress.ip_address(address_text)
        except ValueError:
            continue
        address = _normalized_address(route_ip)
        if address.is_loopback or address.is_unspecified:
            raise ConfigError(
                "configured endpoint hostname resolves to a client-local or wildcard address"
            )
        if reject_client_local and _is_client_interface_address(route_ip, route_address, port):
            raise ConfigError(
                "SSH target endpoint resolves to an address assigned to the benchmark client"
            )
        if all(existing != address for existing, _ in safe_addresses):
            # Keep an IPv6 scope ID for the actual connection while validating
            # the underlying address independently of that interface suffix.
            safe_addresses.append((address, route_address))
    if not safe_addresses:
        if require_resolution:
            raise ConfigError(
                "configured endpoint hostname returned no usable addresses for safety validation"
            )
        return None

    # Prefer IPv4 for LAN appliances with link-local IPv6 plus IPv4 records; the
    # selected address remains fixed for every request made through this route.
    safe_addresses.sort(key=lambda item: isinstance(item[0], ipaddress.IPv6Address))
    parsed = urlsplit(url)
    assert parsed.hostname is not None
    sni_hostname = parsed.hostname.encode("idna").decode("ascii")
    return PinnedHttpRoute(
        origin=origin,
        connect_host=safe_addresses[0][1],
        authority=_authority(sni_hostname, parsed.port),
        sni_hostname=sni_hostname,
    )


def pinned_route_trust_env(
    url: str,
    *,
    trust_env: bool,
    route_is_pinned: bool = True,
) -> bool:
    """Return safe HTTPX ``trust_env`` behavior for a validated target.

    Explicit client-local targets are always direct so credentials cannot leave
    the machine through an environment proxy. HTTPX applies proxy and NO_PROXY
    matching to a rewritten pinned IP rather than the logical hostname, so
    active proxying is also rejected for pinned targets. If the logical origin
    is bypassed, environment mounts are disabled for the client. Route-less
    non-local targets retain normal HTTPX proxy behavior.
    """
    if not trust_env:
        return False
    scheme, hostname, _ = http_origin(url)
    if _is_explicit_client_loopback_host(hostname):
        return False
    if not route_is_pinned:
        return True
    proxies = getproxies()
    if not (proxies.get(scheme) or proxies.get("all")):
        return True
    if environment_proxy_applies(url, proxies=proxies):
        raise ConfigError(
            "IP-pinned target endpoints cannot use an environment proxy; "
            "configure a direct connection with transport.trust_env=false"
        )
    return False


def environment_proxy_applies(
    url: str,
    *,
    proxies: dict[str, str] | None = None,
) -> bool:
    """Return whether HTTPX would proxy the logical origin from the environment."""
    scheme, canonical_hostname, port = http_origin(url)
    if _is_explicit_client_loopback_host(canonical_hostname):
        return False
    environment = getproxies() if proxies is None else proxies
    if not (environment.get(scheme) or environment.get("all")):
        return False
    # Origin comparison intentionally treats a DNS root dot as equivalent, but
    # HTTPX URLPattern matching preserves it and applies HTTPX's own IDNA
    # normalization to the request host. Match against that public URL view so
    # a punycode spelling cannot be mistaken for a NO_PROXY bypass.
    try:
        proxy_hostname = httpx.URL(url).host.casefold()
    except (httpx.InvalidURL, UnicodeError):
        # If HTTPX cannot normalize the logical URL, do not silently convert an
        # environment-proxied target into a direct pinned connection.
        return True
    return not _httpx_no_proxy_bypass(
        scheme=scheme,
        hostname=proxy_hostname,
        port=port,
        no_proxy=str(environment.get("no") or ""),
    )


def _httpx_no_proxy_bypass(
    *,
    scheme: str,
    hostname: str,
    port: int,
    no_proxy: str,
) -> bool:
    """Match HTTPX's NO_PROXY host, domain, scheme, and port behavior.

    ``urllib.request.proxy_bypass`` accepts only a host string on some
    platforms and may apply different macOS SystemConfiguration rules. HTTPX
    converts NO_PROXY entries to URL patterns instead, including an optional
    port. Keep the safety decision aligned with those request-time patterns.
    """
    target_pattern_port = None if port == {"http": 80, "https": 443}.get(scheme) else port
    patterns = [raw_pattern.strip() for raw_pattern in no_proxy.split(",") if raw_pattern.strip()]
    if "*" in patterns:
        return True

    normalized_patterns: list[tuple[str, str, int | None]] = []
    for pattern in patterns:
        normalized = _normalized_httpx_no_proxy_pattern(pattern)
        if normalized is None:
            # Invalid environment patterns cannot establish a trustworthy
            # bypass. Treat the configured proxy as active for pinned routes.
            return False
        normalized_patterns.append(normalized)

    for pattern_scheme, normalized_pattern, pattern_port in normalized_patterns:
        if pattern_scheme not in {"", scheme}:
            continue
        if pattern_port is not None and pattern_port != target_pattern_port:
            continue
        if not normalized_pattern or _no_proxy_host_matches(hostname, normalized_pattern):
            return True
    return False


def _normalized_httpx_no_proxy_pattern(
    pattern: str,
) -> tuple[str, str, int | None] | None:
    """Return the URLPattern fields HTTPX derives from one NO_PROXY entry."""
    if "://" in pattern:
        pattern_url = pattern
    else:
        address_text = pattern.split("/", 1)[0]
        try:
            ipaddress.IPv4Address(address_text)
        except ValueError:
            is_ipv4 = False
        else:
            is_ipv4 = True
        try:
            ipaddress.IPv6Address(address_text)
        except ValueError:
            is_ipv6 = False
        else:
            is_ipv6 = True

        if is_ipv4:
            pattern_url = f"all://{pattern}"
        elif is_ipv6:
            pattern_url = f"all://[{pattern}]"
        elif pattern.casefold() == "localhost":
            pattern_url = f"all://{pattern}"
        else:
            pattern_url = f"all://*{pattern}"

    try:
        parsed = httpx.URL(pattern_url)
    except (httpx.InvalidURL, UnicodeError):
        return None
    pattern_scheme = "" if parsed.scheme == "all" else parsed.scheme
    pattern_hostname = "" if parsed.host == "*" else parsed.host.casefold()
    return pattern_scheme, pattern_hostname, parsed.port


def _no_proxy_host_matches(hostname: str, pattern: str) -> bool:
    candidate_address = _explicit_ip_address(hostname)
    pattern_address = _explicit_ip_address(pattern)
    if pattern_address is not None:
        return candidate_address == pattern_address
    if pattern.startswith("*."):
        return hostname.endswith(pattern[1:])
    if pattern.startswith("*"):
        normalized = pattern[1:]
        return hostname == normalized or hostname.endswith(f".{normalized}")
    return hostname == pattern


def _route_address(address_info: tuple[object, ...]) -> str:
    sockaddr = address_info[4]
    assert isinstance(sockaddr, tuple)
    route_address = str(sockaddr[0])
    if ":" not in route_address or "%" in route_address or len(sockaddr) < 4:
        return route_address
    scope_id = sockaddr[3]
    if not isinstance(scope_id, int) or scope_id <= 0:
        return route_address
    try:
        scope = socket.if_indextoname(scope_id)
    except OSError:
        scope = str(scope_id)
    return f"{route_address}%{scope}"


def _authority(hostname: str, explicit_port: int | None) -> str:
    host = f"[{hostname}]" if ":" in hostname else hostname
    return f"{host}:{explicit_port}" if explicit_port is not None else host


def _is_client_interface_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    route_address: str,
    port: int,
) -> bool:
    family = socket.AF_INET6 if isinstance(address, ipaddress.IPv6Address) else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_DGRAM) as probe:
            probe.connect((route_address, port))
            local_text = str(probe.getsockname()[0]).split("%", 1)[0]
    except OSError:
        return False
    try:
        return _normalized_address(ipaddress.ip_address(local_text)) == _normalized_address(address)
    except ValueError:
        return False


def _normalized_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _explicit_ip_address(
    hostname: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    normalized = hostname.casefold().rstrip(".")
    try:
        return _normalized_address(ipaddress.ip_address(normalized))
    except ValueError:
        try:
            return ipaddress.ip_address(socket.inet_aton(normalized))
        except OSError:
            return None


def _is_explicit_client_loopback_host(hostname: str) -> bool:
    normalized = hostname.casefold().rstrip(".")
    if (
        normalized == "localhost"
        or normalized.startswith("localhost.")
        or normalized.endswith(".localhost")
        or normalized in {"ip6-localhost", "localhost6", "localhost6.localdomain6"}
    ):
        return True
    address = _explicit_ip_address(normalized)
    return address is not None and address.is_loopback


__all__ = [
    "environment_proxy_applies",
    "HttpOrigin",
    "PinnedHttpRoute",
    "http_origin",
    "pinned_route_trust_env",
    "resolve_request_route",
    "validate_request_url",
]
