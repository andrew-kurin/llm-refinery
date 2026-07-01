from __future__ import annotations

from pathlib import Path

import click

from llm_refinery.http_load import load_http_load_config, run_http_load


@click.command("http-load", help="Run OpenAI/Ollama-compatible HTTP load evals.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target", "targets", multiple=True, help="Only run the named target. Repeatable.")
@click.option(
    "--scenario",
    "scenarios",
    multiple=True,
    help="Only run the named scenario. Repeatable.",
)
@click.option("--limit", type=int, help="Only run the first N expanded trials.")
@click.option("--dry-run", is_flag=True, help="Print HTTP load trials without running them.")
@click.option("--keep-going", is_flag=True, help="Continue after failed HTTP load trials.")
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Override database path from the config.",
)
def http_load_command(
    config: Path,
    targets: tuple[str, ...],
    scenarios: tuple[str, ...],
    limit: int | None,
    dry_run: bool,
    keep_going: bool,
    db: Path | None,
) -> None:
    load_config = load_http_load_config(config)
    run_http_load(
        load_config,
        target_names=targets,
        scenario_names=scenarios,
        limit=limit,
        dry_run=dry_run,
        keep_going=keep_going,
        database_override=db,
    )
