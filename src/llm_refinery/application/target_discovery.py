from __future__ import annotations

from collections.abc import Callable

from llm_refinery.adapters.ssh import OpenSSHClient
from llm_refinery.core.config import ConfigError
from llm_refinery.core.http_safety import (
    HttpOrigin,
    PinnedHttpRoute,
    http_origin,
    resolve_request_route,
)
from llm_refinery.core.targets import (
    HOST_ACCESS_LOCAL,
    HOST_ACCESS_SSH,
    MODEL_SELECTION_EXPLICIT,
    SERVER_INFO_REQUIRED,
    HostDiscovery,
    ModelDescriptor,
    ResolvedTarget,
    TargetInspection,
    TargetSpec,
    host_fingerprint_candidates,
)
from llm_refinery.providers.openai_discovery import OpenAIDiscoveryClient
from llm_refinery.utils.system import get_system_profile


class TargetResolver:
    """Resolve declarative target intent without controlling the remote service."""

    def __init__(
        self,
        *,
        ssh_client: OpenSSHClient | None = None,
        service_client: OpenAIDiscoveryClient | None = None,
        local_system_profile: Callable[[], dict[str, object]] = get_system_profile,
    ) -> None:
        self._ssh_client = ssh_client or OpenSSHClient()
        self._service_client = service_client or OpenAIDiscoveryClient()
        self._local_system_profile = local_system_profile
        self._service_routes: dict[HttpOrigin, PinnedHttpRoute | None] = {}

    def inspect(
        self,
        spec: TargetSpec,
        *,
        allow_service_unavailable: bool = False,
    ) -> TargetInspection:
        errors: list[str] = []
        fatal_errors: list[str] = []
        host: HostDiscovery | None = None
        try:
            host = self._capture_host(spec)
            self._verify_host(spec, host)
        except RuntimeError as exc:
            error = f"host: {exc}"
            errors.append(error)
            if spec.host.required or spec.host.expected_fingerprint is not None:
                fatal_errors.append(error)

        service = None
        route: PinnedHttpRoute | None = None
        if not fatal_errors:
            try:
                route = self._service_route(spec)
                service = self._service_client.discover(
                    spec.endpoint,
                    spec.discovery,
                    spec.transport,
                    route=route,
                )
                errors.extend(service.errors)
            except ConfigError as exc:
                error = f"service: {exc}"
                partial = TargetInspection(
                    spec=spec,
                    host=host,
                    service=None,
                    resolved=None,
                    route=route,
                    errors=tuple(dict.fromkeys([*errors, error])),
                )
                exc.target_inspection = partial  # type: ignore[attr-defined]
                raise
            except RuntimeError as exc:
                errors.append(f"service: {exc}")

        selected: ModelDescriptor | None = None
        selection: str | None = None
        service_available = service is not None and service.health == "ok"
        if service_available and service is not None:
            selected, selection_error = _select_model(spec, service.models)
            if selection_error:
                errors.append(selection_error)
                fatal_errors.append(selection_error)
            elif selected is not None:
                selection = (
                    "explicit_verified"
                    if spec.model.selection == MODEL_SELECTION_EXPLICIT
                    else "single_discovered"
                )
            if spec.discovery.server_info == SERVER_INFO_REQUIRED and service.server_info is None:
                error = "required server_info discovery did not return server information"
                errors.append(error)
                fatal_errors.append(error)
        elif service is not None and not service.errors:
            errors.append(f"service: health is {service.health!r}")

        host_identity_matches = spec.host.expected_fingerprint is None or (
            host is not None
            and spec.host.expected_fingerprint in host_fingerprint_candidates(host.profile)
        )
        host_ready = (host is not None or not spec.host.required) and host_identity_matches
        service_ready = (
            service is not None
            and service_available
            and selected is not None
            and not (
                spec.discovery.server_info == SERVER_INFO_REQUIRED and service.server_info is None
            )
        )
        resolved = None
        if host_ready and service_ready and service is not None and selected is not None:
            effective_host = host or HostDiscovery(
                transport=spec.host.access,
                destination=spec.host.destination,
                profile={"capture_error": "host inventory unavailable"},
            )
            assert selection is not None
            resolved = ResolvedTarget(
                spec_name=spec.name,
                endpoint=spec.endpoint.resolve(selected.id),
                host=effective_host,
                service=service,
                model=selected,
                selection=selection,
                route=route,
                expected_host_fingerprint=spec.host.expected_fingerprint,
            )

        inspection = TargetInspection(
            spec=spec,
            host=host,
            service=service,
            resolved=resolved,
            route=route,
            errors=tuple(dict.fromkeys(errors)),
        )
        service_unavailability_allowed = (
            allow_service_unavailable or not spec.discovery.service_required
        )
        unresolved_is_tolerated = (
            resolved is None
            and not fatal_errors
            and not service_available
            and service_unavailability_allowed
        )
        if resolved is None and not unresolved_is_tolerated:
            detail = "; ".join(inspection.errors) or "target is unavailable"
            resolution_error = RuntimeError(f"could not resolve target {spec.name!r}: {detail}")
            resolution_error.target_inspection = inspection  # type: ignore[attr-defined]
            raise resolution_error
        return inspection

    def snapshot_host(self, spec: TargetSpec) -> HostDiscovery:
        """Capture host state without repeating service/model discovery."""
        host = self._capture_host(spec)
        self._verify_host(spec, host)
        return host

    def _capture_host(self, spec: TargetSpec) -> HostDiscovery:
        if spec.host.access == HOST_ACCESS_LOCAL:
            return HostDiscovery(
                transport=HOST_ACCESS_LOCAL,
                destination=None,
                profile=dict(self._local_system_profile()),
            )
        return self._ssh_client.collect_host_profile(spec.host)

    @staticmethod
    def _verify_host(spec: TargetSpec, host: HostDiscovery) -> None:
        expected_fingerprint = spec.host.expected_fingerprint
        fingerprint_candidates = host_fingerprint_candidates(host.profile)
        if expected_fingerprint is not None and expected_fingerprint not in fingerprint_candidates:
            raise RuntimeError(
                "target host inventory fingerprint does not match "
                f"host.expected_fingerprint ({host.profile.get('host_fingerprint')!r} != "
                f"{expected_fingerprint!r})"
            )
        if (
            expected_fingerprint is not None
            and spec.host.access == HOST_ACCESS_SSH
            and fingerprint_candidates.get(expected_fingerprint) not in {"hardware", "installation"}
        ):
            raise RuntimeError(
                "target host inventory fingerprint is not strong enough for "
                "host.expected_fingerprint verification "
                f"(strength={fingerprint_candidates.get(expected_fingerprint)!r})"
            )

    def metrics(self, spec: TargetSpec) -> str:
        return self._service_client.metrics(
            spec.endpoint,
            spec.transport,
            route=self._service_route(spec),
        )

    def _service_route(self, spec: TargetSpec) -> PinnedHttpRoute | None:
        if spec.host.access != HOST_ACCESS_SSH:
            return None
        origin = http_origin(spec.endpoint.base_url)
        if origin not in self._service_routes:
            self._service_routes[origin] = resolve_request_route(
                spec.endpoint.base_url,
                require_resolution=True,
                reject_client_local=True,
            )
        return self._service_routes[origin]

    def resolve(self, spec: TargetSpec) -> ResolvedTarget:
        inspection = self.inspect(spec)
        if inspection.resolved is None:
            detail = "; ".join(inspection.errors) or "target is unavailable"
            raise RuntimeError(f"could not resolve target {spec.name!r}: {detail}")
        return inspection.resolved


def _select_model(
    spec: TargetSpec,
    models: tuple[ModelDescriptor, ...],
) -> tuple[ModelDescriptor | None, str | None]:
    if spec.model.selection == MODEL_SELECTION_EXPLICIT:
        requested = spec.model.model_id
        assert requested is not None
        selected = next((model for model in models if model.id == requested), None)
        if selected is None:
            available = ", ".join(sorted(model.id for model in models)) or "none"
            return None, (
                f"configured model {requested!r} is not served; available models: {available}"
            )
        return selected, None
    if len(models) == 1:
        return models[0], None
    if not models:
        return None, "model discovery returned no served models"
    available = ", ".join(sorted(model.id for model in models))
    return None, (
        "model selection is ambiguous; configure model.selection='explicit' and model.id "
        f"from: {available}"
    )


__all__ = ["TargetResolver"]
