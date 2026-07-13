from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import click

from llm_refinery.commands.common import parse_lm_eval_limit
from llm_refinery.core.targets import MODEL_SELECTION_EXPLICIT
from llm_refinery.workflows.suite import BenchmarkSuiteWorkflow
from llm_refinery.workflows.suite_config import load_suite_config


@click.command("suite", help="Run a recorded quality and optional HTTP-load workflow.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--limit", "limit_text", help="lm-eval limit, or 'all'.")
@click.option("--tasks", help="Comma-separated tasks for lm-eval.")
@click.option(
    "--max-length",
    type=click.IntRange(min=1),
    help="Maximum lm-eval context length.",
)
@click.option("--eos-string", help="EOS string for lm-eval.")
@click.option("--tokenizer", help="Tokenizer id/path for token-aware lm-eval tasks.")
@click.option("--metadata", help="lm-eval metadata JSON.")
@click.option("--gen-kwargs", help="Extra lm-eval generation kwargs.")
@click.option("--package-spec", help="Override the uvx lm-eval package spec.")
@click.option(
    "--with-package",
    "extra_packages",
    multiple=True,
    help="Additional pinned evaluator package; repeat as needed.",
)
@click.option("--offline/--online", default=None, help="Override Hugging Face offline mode.")
@click.option(
    "--include-path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Additional lm-eval task directory.",
)
@click.option(
    "--run-lm-eval/--no-run-lm-eval",
    default=None,
    help="Override whether the quality step runs.",
)
@click.option(
    "--run-http-load/--no-run-http-load",
    default=None,
    help="Override whether the HTTP-load step runs.",
)
@click.option(
    "--require-clean/--no-require-clean",
    default=None,
    help="Override clean-port preflight checks.",
)
@click.option("--base-url", help="Override the suite endpoint URL.")
@click.option("--api-model", help="Override the model sent to the suite endpoint.")
@click.option(
    "--ssh-destination",
    help="Override the OpenSSH destination/alias for a discovery target.",
)
@click.option(
    "--http-load-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Override the HTTP-load configuration path.",
)
@click.option("--target", help="Override the HTTP-load target name.")
def suite_command(
    config: Path,
    limit_text: str | None,
    tasks: str | None,
    max_length: int | None,
    eos_string: str | None,
    tokenizer: str | None,
    metadata: str | None,
    gen_kwargs: str | None,
    package_spec: str | None,
    extra_packages: tuple[str, ...],
    offline: bool | None,
    include_path: Path | None,
    run_lm_eval: bool | None,
    run_http_load: bool | None,
    require_clean: bool | None,
    base_url: str | None,
    api_model: str | None,
    ssh_destination: str | None,
    http_load_config: Path | None,
    target: str | None,
) -> None:
    suite = load_suite_config(config)
    endpoint = suite.endpoint
    target_spec = suite.target
    if endpoint is not None:
        endpoint = replace(
            endpoint,
            base_url=(base_url or endpoint.base_url).rstrip("/"),
            model=api_model or endpoint.model,
        )
    else:
        assert target_spec is not None
        if ssh_destination is not None:
            if target_spec.host.access != "ssh":
                raise click.BadParameter(
                    "--ssh-destination requires target.host.access: ssh",
                    param_hint="--ssh-destination",
                )
            target_spec = replace(
                target_spec,
                host=replace(target_spec.host, destination=ssh_destination),
            )
        endpoint_spec = replace(
            target_spec.endpoint,
            base_url=(base_url or target_spec.endpoint.base_url).rstrip("/"),
        )
        model_selection = target_spec.model
        if api_model is not None:
            endpoint_spec = replace(endpoint_spec, model=api_model)
            model_selection = replace(
                model_selection,
                selection=MODEL_SELECTION_EXPLICIT,
                model_id=api_model,
            )
        target_spec = replace(
            target_spec,
            endpoint=endpoint_spec,
            model=model_selection,
        )
    if endpoint is not None and ssh_destination is not None:
        raise click.BadParameter(
            "--ssh-destination requires a schema_version 2 discovery target",
            param_hint="--ssh-destination",
        )
    quality = replace(
        suite.quality,
        enabled=suite.quality.enabled if run_lm_eval is None else run_lm_eval,
        limit=(parse_lm_eval_limit(limit_text) if limit_text is not None else suite.quality.limit),
        tasks=tasks or suite.quality.tasks,
        max_length=max_length if max_length is not None else suite.quality.max_length,
        eos_string=eos_string or suite.quality.eos_string,
        tokenizer=tokenizer or suite.quality.tokenizer,
        metadata=metadata or suite.quality.metadata,
        gen_kwargs=gen_kwargs if gen_kwargs is not None else suite.quality.gen_kwargs,
        package_spec=package_spec or suite.quality.package_spec,
        extra_packages=extra_packages or suite.quality.extra_packages,
        offline=suite.quality.offline if offline is None else offline,
        include_path=include_path if include_path is not None else suite.quality.include_path,
    )
    http_config_path = http_load_config or suite.http_load.config
    http_load = replace(
        suite.http_load,
        enabled=(suite.http_load.enabled if run_http_load is None else run_http_load),
        config=http_config_path,
        targets=(target,) if target else suite.http_load.targets,
    )
    if http_load_config is not None and run_http_load is None:
        http_load = replace(http_load, enabled=True)
    if http_load.enabled and http_load.config is None:
        raise click.BadParameter("--http-load-config is required when HTTP load is enabled")
    preflight = replace(
        suite.preflight,
        require_clean=(suite.preflight.require_clean if require_clean is None else require_clean),
    )
    effective = replace(
        suite,
        endpoint=endpoint,
        target=target_spec,
        quality=quality,
        http_load=http_load,
        preflight=preflight,
    )
    BenchmarkSuiteWorkflow(effective).execute()
