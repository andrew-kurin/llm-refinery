from __future__ import annotations

from pathlib import Path

import click

from llm_refinery.benchmarks.dabstep.config import load_dabstep_config
from llm_refinery.benchmarks.dabstep.runner import run_dabstep


@click.command("dabstep", help="Run the official DABStep baseline as an external process.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--resume", "resume_run_id", help="Resume an incomplete DABStep run ID.")
@click.option(
    "--dry-run",
    is_flag=True,
    help="Print the official baseline command without running.",
)
def dabstep_command(config: Path, resume_run_id: str | None, dry_run: bool) -> None:
    run_dabstep(
        load_dabstep_config(config),
        dry_run=dry_run,
        resume_run_id=resume_run_id,
    )
