from __future__ import annotations

import ipaddress
import socket
from typing import TypeAlias
from urllib.parse import urlparse

from llm_refinery.core.config import ConfigError

HttpOrigin: TypeAlias = tuple[str, str, int]


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
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise ConfigError("HTTP request URL includes an invalid port") from exc
    hostname = parsed.hostname.casefold().rstrip(".")
    return parsed.scheme, hostname, port


def validate_request_url(
    url: str,
    *,
    expected_origin: HttpOrigin | None = None,
    resolve_addresses: bool = True,
) -> HttpOrigin:
    """Reject cross-origin requests and DNS names that resolve back to this client.

    Explicit loopback URLs remain valid for the harness's local-target mode. A
    non-loopback hostname resolving to loopback or an unspecified address is
    rejected, which prevents a remote target name from being attributed to a
    service on the benchmark client.
    """
    origin = http_origin(url)
    if expected_origin is not None and origin != expected_origin:
        raise ConfigError("HTTP redirect must remain on the configured endpoint origin")

    _, hostname, port = origin
    if _is_explicit_client_local_host(hostname):
        return origin
    if not resolve_addresses:
        return origin
    try:
        addresses = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
    except OSError:
        # Preserve the normal HTTP client's DNS/connection error semantics so an
        # offline hostname can still be handled by offline-tolerant inspection.
        return origin
    for address_info in addresses:
        address_text = str(address_info[4][0]).split("%", 1)[0]
        try:
            address = ipaddress.ip_address(address_text)
        except ValueError:
            continue
        if address.is_loopback or address.is_unspecified:
            raise ConfigError(
                "configured endpoint hostname resolves to a client-local or wildcard address"
            )
    return origin


def _is_explicit_client_local_host(hostname: str) -> bool:
    normalized = hostname.casefold().rstrip(".")
    if (
        normalized == "localhost"
        or normalized.startswith("localhost.")
        or normalized.endswith(".localhost")
        or normalized in {"ip6-localhost", "localhost6", "localhost6.localdomain6"}
    ):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        try:
            address = ipaddress.ip_address(socket.inet_aton(normalized))
        except OSError:
            return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_unspecified


__all__ = ["HttpOrigin", "http_origin", "validate_request_url"]
