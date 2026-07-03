from __future__ import annotations

from pathlib import Path

import click

from llm_refinery.commands.common import parse_lm_eval_limit
from llm_refinery.config import load_config
from llm_refinery.workflows.suite import BenchmarkSuiteWorkflow


@click.command("suite", help="Run lm-eval and optional HTTP-load/compare workflow.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--limit",
    "limit_text",
    help="lm-eval limit, or 'all'. Defaults to config eval.limit or 50.",
)
@click.option("--tasks", help="Comma-separated tasks for lm-eval. Defaults to config eval.tasks.")
@click.option(
    "--max-length",
    type=int,
    help="Max length for lm-eval. Defaults to config eval.max_length.",
)
@click.option("--eos-string", help="EOS string for lm-eval. Defaults to config eval.eos_string.")
@click.option(
    "--gen-kwargs",
    help="Extra lm-eval generation kwargs. Defaults to config eval.gen_kwargs.",
)
@click.option(
    "--include-path",
    type=click.Path(file_okay=False, path_type=Path),
    help="Additional lm-eval task directory. Defaults to config eval.include_path.",
)
@click.option("--run-lm-eval/--no-run-lm-eval", default=True, help="Whether to run lm-eval.")
@click.option(
    "--run-http-load/--no-run-http-load",
    default=None,
    help="Whether to run http-load. Defaults to enabled when --http-load-config is set.",
)
@click.option(
    "--require-clean/--no-require-clean",
    default=True,
    help="Fail if other model servers are running.",
)
@click.option(
    "--base-url",
    default="http://127.0.0.1:8080/v1/chat/completions",
    help="API URL for sanity check and quality evals.",
)
@click.option(
    "--api-model",
    help="Model name to send to the OpenAI-compatible API. Defaults to config eval.api_model.",
)
@click.option(
    "--http-load-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Config for http-load. Providing this enables HTTP load unless --no-run-http-load is set.",
)
@click.option("--target", help="Target name for http-load.")
def suite_command(
    config: Path,
    limit_text: str,
    tasks: str | None,
    max_length: int | None,
    eos_string: str | None,
    gen_kwargs: str | None,
    include_path: Path | None,
    run_lm_eval: bool,
    run_http_load: bool | None,
    require_clean: bool,
    base_url: str,
    api_model: str | None,
    http_load_config: Path | None,
    target: str | None,
) -> None:
    tune_config = load_config(config)
    effective_run_http_load = (
        http_load_config is not None if run_http_load is None else run_http_load
    )
    limit = parse_lm_eval_limit(limit_text) if limit_text is not None else tune_config.eval.limit
    workflow = BenchmarkSuiteWorkflow(
        config=tune_config,
        limit=limit,
        tasks=tasks or tune_config.eval.tasks,
        max_length=max_length or tune_config.eval.max_length,
        eos_string=eos_string or tune_config.eval.eos_string,
        gen_kwargs=gen_kwargs if gen_kwargs is not None else tune_config.eval.gen_kwargs,
        include_path=include_path if include_path is not None else tune_config.eval.include_path,
        run_lm_eval=run_lm_eval,
        run_http_load=effective_run_http_load,
        require_clean=require_clean,
        base_url=base_url,
        http_load_config=http_load_config,
        target_name=target,
        api_model=api_model or tune_config.eval.api_model,
    )
    workflow.execute()
