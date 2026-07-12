from __future__ import annotations

from collections.abc import Callable

from llm_refinery.adapters.ssh import OpenSSHClient
from llm_refinery.core.targets import (
    HOST_ACCESS_LOCAL,
    MODEL_SELECTION_EXPLICIT,
    SERVER_INFO_REQUIRED,
    HostDiscovery,
    ModelDescriptor,
    ResolvedTarget,
    TargetInspection,
    TargetSpec,
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

    def inspect(
        self,
        spec: TargetSpec,
        *,
        allow_service_unavailable: bool = False,
    ) -> TargetInspection:
        errors: list[str] = []
        host: HostDiscovery | None = None
        try:
            host = self.snapshot_host(spec)
        except RuntimeError as exc:
            errors.append(f"host: {exc}")

        service = None
        try:
            service = self._service_client.discover(spec.endpoint, spec.discovery)
            errors.extend(service.errors)
        except RuntimeError as exc:
            errors.append(f"service: {exc}")

        selected: ModelDescriptor | None = None
        selection: str | None = None
        if service is not None:
            selected, selection_error = _select_model(spec, service.models)
            if selection_error:
                errors.append(selection_error)
            elif selected is not None:
                selection = (
                    "explicit_verified"
                    if spec.model.selection == MODEL_SELECTION_EXPLICIT
                    else "single_discovered"
                )

        host_ready = host is not None or not spec.host.required
        service_ready = (
            service is not None
            and service.health == "ok"
            and selected is not None
            and not (
                spec.discovery.server_info == SERVER_INFO_REQUIRED
                and service.server_info is None
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
                tokenizer=spec.model.tokenizer,
            )

        inspection = TargetInspection(
            spec=spec,
            host=host,
            service=service,
            resolved=resolved,
            errors=tuple(dict.fromkeys(errors)),
        )
        if (
            resolved is None
            and not allow_service_unavailable
            and spec.discovery.service_required
        ):
            detail = "; ".join(inspection.errors) or "target is unavailable"
            raise RuntimeError(f"could not resolve target {spec.name!r}: {detail}")
        return inspection

    def snapshot_host(self, spec: TargetSpec) -> HostDiscovery:
        """Capture host state without repeating service/model discovery."""
        if spec.host.access == HOST_ACCESS_LOCAL:
            return HostDiscovery(
                transport=HOST_ACCESS_LOCAL,
                destination=None,
                profile=dict(self._local_system_profile()),
            )
        return self._ssh_client.collect_host_profile(spec.host)

    def metrics(self, spec: TargetSpec) -> str:
        return self._service_client.metrics(spec.endpoint)

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
