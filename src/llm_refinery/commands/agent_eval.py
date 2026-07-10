from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from llm_refinery.benchmarks.agent.config import load_agent_eval_config
from llm_refinery.benchmarks.agent.runner import run_agent_eval
from llm_refinery.commands.common import parse_lm_eval_limit


@click.command("agent-eval", help="Run agent/data benchmarks such as GeoAnalystBench.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--target", "targets", multiple=True, help="Only run the named target. Repeatable.")
@click.option("--limit", "limit_text", help="Override benchmark limit; use 'all' for all tasks.")
@click.option(
    "--task-id",
    "task_ids",
    multiple=True,
    type=int,
    help="Only run specific task id(s).",
)
@click.option("--dry-run", is_flag=True, help="Print planned benchmark requests without running.")
@click.option("--keep-going", is_flag=True, help="Continue after a target fails.")
def agent_eval_command(
    config: Path,
    targets: tuple[str, ...],
    limit_text: str | None,
    task_ids: tuple[int, ...],
    dry_run: bool,
    keep_going: bool,
) -> None:
    eval_config = load_agent_eval_config(config)
    kwargs: dict[str, Any] = {
        "target_names": targets,
        "task_ids": task_ids,
        "dry_run": dry_run,
        "keep_going": keep_going,
    }
    if limit_text is not None:
        kwargs["limit"] = parse_lm_eval_limit(limit_text)
    run_agent_eval(eval_config, **kwargs)
