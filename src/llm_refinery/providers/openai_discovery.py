from __future__ import annotations

import os
import re
import ssl
import threading
import time
from typing import Any
from urllib.parse import urljoin

import httpx

from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import OPENAI_CHAT
from llm_refinery.core.http_safety import (
    PinnedHttpRoute,
    http_origin,
    pinned_route_trust_env,
    validate_request_url,
)
from llm_refinery.core.targets import (
    SERVER_INFO_OFF,
    DiscoveryPolicy,
    EndpointSpec,
    ModelDescriptor,
    ServiceDiscovery,
    TargetTransport,
)
from llm_refinery.providers.openai_chat import has_header, json_headers

MAX_RESPONSE_BYTES = 2_000_000
_monotonic = time.monotonic
_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "auth_token",
        "bearer_token",
        "client_secret",
        "client_secret_key",
        "credential",
        "credentials",
        "hf_token",
        "private_key",
    }
)
_SECRET_SUFFIXES = (
    "_api_key",
    "_password",
    "_secret",
    "_access_token",
    "_auth_token",
    "_credential",
    "_credentials",
    "_private_key",
    "_token",
)
_MAX_REDIRECTS = 5


class OpenAIDiscoveryClient:
    """Read-only discovery for a vLLM/OpenAI-compatible serving endpoint."""

    def __init__(
        self,
        *,
        timeout_s: float = 5.0,
    ) -> None:
        self._timeout_s = timeout_s

    def discover(
        self,
        endpoint: EndpointSpec,
        policy: DiscoveryPolicy,
        transport: TargetTransport | None = None,
        *,
        route: PinnedHttpRoute | None = None,
    ) -> ServiceDiscovery:
        if endpoint.protocol != OPENAI_CHAT:
            raise RuntimeError("target model discovery currently requires protocol 'openai_chat'")
        errors: list[str] = []
        headers = _headers(endpoint)
        effective_transport = transport or TargetTransport()
        client_trust_env = pinned_route_trust_env(
            endpoint.base_url,
            trust_env=effective_transport.trust_env,
            route_is_pinned=route is not None,
        )
        client = self._new_client(
            effective_transport,
            trust_env=client_trust_env,
        )

        def active_client() -> httpx.Client:
            nonlocal client
            if client.is_closed:
                client = self._new_client(
                    effective_transport,
                    trust_env=client_trust_env,
                )
            return client

        health = "unavailable"
        version: str | None = None
        models: tuple[ModelDescriptor, ...] = ()
        server_info: dict[str, Any] | None = None
        health_transport_failed = False
        try:
            try:
                _get_bounded_response(
                    active_client(),
                    f"{endpoint.server_root_url}/health",
                    headers=headers,
                    timeout_s=self._timeout_s,
                    route=route,
                )
                health = "ok"
            except (httpx.HTTPError, ValueError) as exc:
                _raise_critical_discovery_error(exc, operation="health")
                errors.append(f"health: {_error_text(exc)}")
                health_transport_failed = isinstance(exc, httpx.TransportError)

            if not health_transport_failed:
                try:
                    response = _get_bounded_response(
                        active_client(),
                        f"{endpoint.server_root_url}/version",
                        headers=headers,
                        timeout_s=self._timeout_s,
                        route=route,
                    )
                    payload = _json_object(response)
                    version_value = payload.get("version") or payload.get("vllm_version")
                    version = str(version_value) if version_value is not None else None
                    if version is None:
                        errors.append("version: response did not include a version")
                except (httpx.HTTPError, ValueError) as exc:
                    _raise_fatal_transport_error(exc, operation="version discovery")
                    errors.append(f"version: {_error_text(exc)}")

                try:
                    response = _get_bounded_response(
                        active_client(),
                        _models_url(endpoint),
                        headers=headers,
                        timeout_s=self._timeout_s,
                        route=route,
                    )
                    models = _models_from_response(response)
                except (httpx.HTTPError, ValueError) as exc:
                    _raise_critical_discovery_error(exc, operation="model discovery")
                    errors.append(f"models: {_error_text(exc)}")

                if policy.server_info != SERVER_INFO_OFF:
                    try:
                        response = _get_bounded_response(
                            active_client(),
                            f"{endpoint.server_root_url}/server_info",
                            params={"config_format": "json"},
                            headers=headers,
                            timeout_s=self._timeout_s,
                            route=route,
                        )
                        server_info = _sanitize(_json_object(response))
                    except (httpx.HTTPError, ValueError) as exc:
                        _raise_fatal_transport_error(exc, operation="server_info discovery")
                        errors.append(f"server_info: {_error_text(exc)}")
        finally:
            client.close()

        return ServiceDiscovery(
            implementation="vllm",
            base_url=endpoint.api_root_url,
            health=health,
            version=version,
            models=models,
            server_info=server_info,
            errors=tuple(errors),
        )

    def metrics(
        self,
        endpoint: EndpointSpec,
        transport: TargetTransport | None = None,
        *,
        route: PinnedHttpRoute | None = None,
    ) -> str:
        """Read the Prometheus snapshot without changing server state."""
        headers = _headers(endpoint)
        effective_transport = transport or TargetTransport()
        client_trust_env = pinned_route_trust_env(
            endpoint.base_url,
            trust_env=effective_transport.trust_env,
            route_is_pinned=route is not None,
        )
        client = self._new_client(
            effective_transport,
            trust_env=client_trust_env,
        )
        try:
            response = _get_bounded_response(
                client,
                f"{endpoint.server_root_url}/metrics",
                headers=headers,
                timeout_s=self._timeout_s,
                route=route,
            )
            return response.text
        except (httpx.HTTPError, ValueError) as exc:
            _raise_fatal_transport_error(exc, operation="metrics discovery")
            raise RuntimeError(f"could not read vLLM metrics: {_error_text(exc)}") from exc
        finally:
            client.close()

    def _new_client(
        self,
        transport: TargetTransport,
        *,
        trust_env: bool | None = None,
    ) -> httpx.Client:
        client_trust_env = transport.trust_env if trust_env is None else trust_env
        verify: bool | ssl.SSLContext = True
        if transport.ca_bundle is not None:
            try:
                verify = ssl.create_default_context(cafile=str(transport.ca_bundle))
            except (OSError, ssl.SSLError) as exc:
                raise ConfigError(
                    f"could not load target transport CA bundle: {transport.ca_bundle}"
                ) from exc
        elif transport.trust_env and not client_trust_env:
            verify = httpx.create_ssl_context(verify=True, trust_env=True)
        return httpx.Client(
            timeout=self._timeout_s,
            follow_redirects=False,
            trust_env=client_trust_env,
            verify=verify,
        )


def _models_url(endpoint: EndpointSpec) -> str:
    api_root = endpoint.api_root_url
    return f"{api_root}/models" if api_root.endswith("/v1") else f"{api_root}/v1/models"


def _headers(endpoint: EndpointSpec) -> dict[str, str]:
    headers = json_headers(
        endpoint.headers,
        api_key_env=endpoint.api_key_env,
        accept=True,
    )
    if (
        endpoint.api_key_env
        and not has_header(headers, "Authorization")
        and not os.environ.get(endpoint.api_key_env)
    ):
        raise ConfigError(
            f"endpoint API key environment variable is not set: {endpoint.api_key_env}"
        )
    return headers


def _get_bounded_response(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str] | None = None,
    timeout_s: float,
    route: PinnedHttpRoute | None = None,
) -> httpx.Response:
    """Stream a small discovery response and stop as soon as its limit is exceeded."""
    deadline = _monotonic() + timeout_s
    expired = threading.Event()

    def expire() -> None:
        expired.set()
        client.close()

    timer = threading.Timer(timeout_s, expire)
    timer.daemon = True
    timer.start()
    try:
        return _get_bounded_response_before_deadline(
            client,
            url,
            headers=headers,
            params=params,
            timeout_s=timeout_s,
            route=route,
            deadline=deadline,
        )
    except httpx.HTTPError as exc:
        if expired.is_set():
            raise httpx.ReadTimeout(
                "discovery response exceeded the total timeout",
                request=httpx.Request("GET", url),
            ) from exc
        raise
    finally:
        timer.cancel()


def _get_bounded_response_before_deadline(
    client: httpx.Client,
    url: str,
    *,
    headers: dict[str, str],
    params: dict[str, str] | None,
    timeout_s: float,
    route: PinnedHttpRoute | None,
    deadline: float,
) -> httpx.Response:
    expected_origin = http_origin(url)
    # HTTPX read timeouts reset after each chunk. A shorter per-read slice plus the
    # monotonic deadline prevents an endless trickle from keeping discovery alive.
    read_timeout_s = min(timeout_s, max(0.1, timeout_s / 4))
    timeout = httpx.Timeout(timeout_s, read=read_timeout_s)
    current_url = url
    current_params = params
    for _redirect_count in range(_MAX_REDIRECTS + 1):
        validate_request_url(
            current_url,
            expected_origin=expected_origin,
            resolve_addresses=False,
        )
        request_url = route.request_url(current_url) if route is not None else current_url
        request_headers = route.request_headers(headers) if route is not None else headers
        extensions = {"sni_hostname": route.sni_hostname} if route is not None else None
        with client.stream(
            "GET",
            request_url,
            headers=request_headers,
            params=current_params,
            timeout=timeout,
            follow_redirects=False,
            extensions=extensions,
        ) as response:
            _check_deadline(deadline, response.request)
            if response.has_redirect_location:
                redirect_url = urljoin(current_url, response.headers["location"])
                validate_request_url(
                    redirect_url,
                    expected_origin=expected_origin,
                    resolve_addresses=False,
                )
                current_url = redirect_url
                current_params = None
                continue
            response.raise_for_status()
            content_length = response.headers.get("content-length")
            if content_length:
                try:
                    declared_size = int(content_length)
                except ValueError:
                    declared_size = None
                if declared_size is not None and declared_size > MAX_RESPONSE_BYTES:
                    raise ValueError("response is too large")
            content = bytearray()
            for chunk in response.iter_bytes():
                _check_deadline(deadline, response.request)
                if len(content) + len(chunk) > MAX_RESPONSE_BYTES:
                    raise ValueError("response is too large")
                content.extend(chunk)
            _check_deadline(deadline, response.request)
            # iter_bytes() has already decoded transfer/content encodings. Do not retain
            # headers that would make the in-memory response decode the bytes a second time.
            bounded_headers = [
                (key, value)
                for key, value in response.headers.multi_items()
                if key.casefold() not in {"content-encoding", "content-length", "transfer-encoding"}
            ]
            return httpx.Response(
                status_code=response.status_code,
                headers=bounded_headers,
                content=bytes(content),
                request=response.request,
            )
    raise ConfigError("vLLM discovery exceeded the maximum same-origin redirects")


def _check_deadline(deadline: float, request: httpx.Request) -> None:
    if _monotonic() > deadline:
        raise httpx.ReadTimeout(
            "discovery response exceeded the total timeout",
            request=request,
        )


def _models_from_response(response: httpx.Response) -> tuple[ModelDescriptor, ...]:
    payload = _json_object(response)
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("response field 'data' must be a list")
    models: list[ModelDescriptor] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"model entry {index} must be an object")
        model_id_value = item.get("id")
        if not isinstance(model_id_value, str) or not model_id_value.strip():
            raise ValueError(f"model entry {index} id must be a non-empty string")
        model_id = model_id_value.strip()
        max_model_len = item.get("max_model_len")
        if max_model_len is not None and (
            isinstance(max_model_len, bool)
            or not isinstance(max_model_len, int)
            or max_model_len <= 0
        ):
            raise ValueError(f"model {model_id!r} max_model_len must be a positive integer")
        root = _optional_model_string(item, "root", model_id=model_id)
        owned_by = _optional_model_string(item, "owned_by", model_id=model_id)
        models.append(
            ModelDescriptor(
                id=model_id,
                root=root,
                max_model_len=max_model_len,
                owned_by=owned_by,
            )
        )
    return tuple(models)


def _json_object(response: httpx.Response) -> dict[str, Any]:
    content_length = response.headers.get("content-length")
    if content_length:
        try:
            parsed_content_length = int(content_length)
        except ValueError:
            parsed_content_length = None
        if parsed_content_length is not None and parsed_content_length > MAX_RESPONSE_BYTES:
            raise ValueError("response is too large")
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("response is too large")
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError("response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    return payload


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize(child)
            for key, child in value.items()
            if not _is_secret_key(str(key))
        }
    if isinstance(value, list):
        return [_sanitize(child) for child in value]
    return value


def _is_secret_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", key.casefold()).strip("_")
    return normalized in _SECRET_KEYS or normalized.endswith(_SECRET_SUFFIXES)


def _error_text(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTP {exc.response.status_code}"
    if isinstance(exc, httpx.LocalProtocolError):
        return "invalid HTTP request or response"
    return str(exc) or type(exc).__name__


def _raise_critical_discovery_error(exc: Exception, *, operation: str) -> None:
    if isinstance(exc, ConfigError):
        raise exc
    _raise_fatal_transport_error(exc, operation=operation)
    if isinstance(exc, httpx.HTTPStatusError) and 300 <= exc.response.status_code < 500:
        if exc.response.status_code in {401, 403}:
            raise ConfigError(
                f"vLLM {operation} authorization failed with HTTP {exc.response.status_code}"
            ) from exc
        raise ConfigError(f"vLLM {operation} failed with HTTP {exc.response.status_code}") from exc
    if isinstance(exc, ValueError) and not isinstance(exc, httpx.HTTPError):
        raise ConfigError(f"vLLM {operation} returned an invalid response: {exc}") from exc


def _raise_fatal_transport_error(exc: Exception, *, operation: str) -> None:
    if not isinstance(exc, httpx.TransportError):
        return
    if isinstance(exc, (httpx.TimeoutException, httpx.NetworkError)):
        if isinstance(exc, httpx.NetworkError) and _looks_like_tls_failure(exc):
            raise ConfigError(f"vLLM {operation} TLS connection failed") from exc
        return
    if isinstance(exc, httpx.ProxyError):
        kind = "proxy connection"
    elif isinstance(exc, httpx.UnsupportedProtocol):
        kind = "unsupported HTTP protocol"
    elif isinstance(exc, httpx.ProtocolError):
        kind = "HTTP protocol"
    else:
        kind = "HTTP transport"
    raise ConfigError(f"vLLM {operation} {kind} failed") from exc


def _looks_like_tls_failure(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, ssl.SSLError):
            return True
        text = str(current).casefold()
        if any(
            marker in text
            for marker in (
                "certificate verify failed",
                "certificate_verify_failed",
                "ssl:",
                "ssl error",
                "tls",
                "wrong version number",
            )
        ):
            return True
        current = current.__cause__ or current.__context__
    return False


def _optional_model_string(
    item: dict[str, Any],
    field_name: str,
    *,
    model_id: str,
) -> str | None:
    value = item.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"model {model_id!r} {field_name} must be a non-empty string")
    return value.strip()


__all__ = ["OpenAIDiscoveryClient"]
