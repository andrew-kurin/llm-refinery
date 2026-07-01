from __future__ import annotations

from pathlib import Path

import click

from llm_refinery.storage import ResultStore, utc_now
from llm_refinery.utils.system import get_system_profile


@click.command("backfill-system-metadata", help="Backfill current host metadata into old run rows.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--overwrite", is_flag=True, help="Replace existing system_json values too.")
@click.option("--dry-run", is_flag=True, help="Show how many rows would be updated.")
def backfill_system_metadata_command(database: Path, overwrite: bool, dry_run: bool) -> None:
    if not database.exists():
        raise FileNotFoundError(f"database not found: {database}")

    profile = get_system_profile()
    profile["backfill"] = {
        "assumed_current_hardware": True,
        "backfilled_at": utc_now().isoformat(),
        "source": "llm-refinery backfill-system-metadata",
    }

    with ResultStore(database) as store:
        if dry_run:
            missing_clause = (
                "WHERE system_json IS NULL OR trim(system_json) = '' "
                "OR trim(system_json) = '{}'"
            )
            where_clause = "" if overwrite else missing_clause
            count = store.connection.execute(
                f"SELECT COUNT(*) FROM runs {where_clause}"
            ).fetchone()[0]
        else:
            count = store.backfill_system_json(profile, overwrite=overwrite)

    action = "would backfill" if dry_run else "backfilled"
    overwrite_text = " including existing metadata" if overwrite else " missing metadata only"
    click.echo(f"{action} {count} run(s) in {database}{overwrite_text}")
