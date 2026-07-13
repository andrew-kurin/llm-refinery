from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import click

from llm_refinery.application.target_discovery import TargetResolver
from llm_refinery.core.targets import TargetInspection, load_target_spec
from llm_refinery.utils.terminal import sanitize_terminal_text


@click.group("target", help="Inspect and resolve local or remote inference targets.")
def target_command() -> None:
    pass


@target_command.command("inspect", help="Read host and service metadata without changing it.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--allow-service-unavailable",
    is_flag=True,
    help="Return host inventory successfully when the model service is offline.",
)
@click.option(
    "--ssh-destination",
    help="Override the OpenSSH destination/alias declared by the target.",
)
@click.option("--json", "as_json", is_flag=True, help="Print the complete JSON envelope.")
def target_inspect_command(
    config: Path,
    allow_service_unavailable: bool,
    ssh_destination: str | None,
    as_json: bool,
) -> None:
    _path, spec = load_target_spec(config)
    if ssh_destination is not None:
        if spec.host.access != "ssh":
            raise click.BadParameter(
                "--ssh-destination requires target.host.access: ssh",
                param_hint="--ssh-destination",
            )
        spec = replace(spec, host=replace(spec.host, destination=ssh_destination))
    inspection = TargetResolver().inspect(
        spec,
        allow_service_unavailable=allow_service_unavailable,
    )
    if as_json:
        click.echo(json.dumps(inspection.safe_json(), indent=2, sort_keys=True))
        return
    _print_inspection(inspection)


def _print_inspection(inspection: TargetInspection) -> None:
    status = "available" if inspection.available else "unavailable"
    _human_echo(f"target={inspection.spec.name} status={status}")
    if inspection.host is not None:
        profile = inspection.host.profile
        hardware = profile.get("hardware") or {}
        _human_echo(
            "host="
            f"{profile.get('hostname') or inspection.host.destination or 'unknown'} "
            f"model={hardware.get('model') or hardware.get('chip') or 'unknown'}"
        )
    if inspection.service is not None:
        model_ids = ",".join(model.id for model in inspection.service.models) or "none"
        _human_echo(
            f"service={inspection.service.implementation} "
            f"health={inspection.service.health} "
            f"version={inspection.service.version or 'unknown'} models={model_ids}"
        )
    if inspection.resolved is not None:
        _human_echo(f"selected_model={inspection.resolved.endpoint.model}")
    for error in inspection.errors:
        _human_echo(f"warning: {error}", err=True)


def _human_echo(value: str, *, err: bool = False) -> None:
    click.echo(sanitize_terminal_text(value), err=err)


__all__ = ["target_command", "target_inspect_command"]
