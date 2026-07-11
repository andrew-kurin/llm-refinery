from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from llm_refinery.benchmarks.registry import ReparseNotSupported, reparse_run
from llm_refinery.commands.common import table
from llm_refinery.compare import build_compare_rows, build_compare_table_rows
from llm_refinery.storage.duckdb import ResultStore


@click.command("report", help="Show recent runs or top runs by metric.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--metric", help="Metric name to sort by descending.")
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum rows to show.")
@click.option("--metrics", "list_metrics", is_flag=True, help="List known metric names.")
def report_command(database: Path, metric: str | None, limit: int, list_metrics: bool) -> None:
    if not database.exists():
        raise FileNotFoundError(f"database not found: {database}")

    with ResultStore(database) as store:
        if list_metrics:
            metric_names = store.metric_names(limit=limit)
            if not metric_names:
                click.echo("no parsed metrics yet")
                return
            click.echo(table([("metric", "runs"), *metric_names]))
            return

        if metric:
            rows = store.top_by_metric(metric, limit=limit)
            if not rows:
                click.echo(f"no rows for metric {metric!r}")
                return
            metric_table_rows = [("value", "kind", "status", "trial", "run_id")]
            metric_table_rows.extend(
                (
                    f"{row['value']:.3f}",
                    row["benchmark_kind"],
                    row["status"],
                    row["trial_name"],
                    row["run_id"],
                )
                for row in rows
            )
            click.echo(table(metric_table_rows))
            return

        rows = store.recent_runs(limit=limit)
        if not rows:
            click.echo("no runs yet")
            return
        recent_table_rows = [("ended_at", "kind", "status", "duration_s", "trial", "run_id")]
        recent_table_rows.extend(
            (
                str(row["ended_at"]),
                row["benchmark_kind"],
                row["status"],
                f"{row['duration_s']:.1f}",
                row["trial_name"],
                row["run_id"],
            )
            for row in rows
        )
        click.echo(table(recent_table_rows))
        click.echo(
            "\nUse --metrics to list metric names, --metric NAME to rank runs, "
            "or `llm-refinery compare` to compare configs."
        )


@click.command("compare", help="Compare benchmark configs with params and throughput columns.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--metric",
    "metrics",
    multiple=True,
    help="Metric column to show. Repeatable. Defaults to pp_tps and tg_tps.",
)
@click.option(
    "--param",
    "params",
    multiple=True,
    help="Param column to show. Repeatable. Defaults to inferred sweep params.",
)
@click.option(
    "--sort",
    "sort_key",
    help="Column/metric to sort by. Defaults to tg_tps, or first --metric.",
)
@click.option("--ascending", is_flag=True, help="Sort ascending instead of descending.")
@click.option("--prompt-tokens", type=int, help="Only compare runs with this prompt size.")
@click.option("--gen-tokens", type=int, help="Only compare runs with this generation size.")
@click.option("--all-runs", is_flag=True, help="Include duplicate reruns of the same trial.")
@click.option("--include-failed", is_flag=True, help="Include failed runs.")
@click.option(
    "--suite",
    "suites",
    multiple=True,
    help="Only compare runs from this suite. Repeatable.",
)
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum rows to show.")
def compare_command(
    database: Path,
    metrics: tuple[str, ...],
    params: tuple[str, ...],
    sort_key: str | None,
    ascending: bool,
    prompt_tokens: int | None,
    gen_tokens: int | None,
    all_runs: bool,
    include_failed: bool,
    suites: tuple[str, ...],
    limit: int,
) -> None:
    if not database.exists():
        raise FileNotFoundError(f"database not found: {database}")

    with ResultStore(database) as store:
        runs = store.comparison_runs(
            include_failed=include_failed,
            # Host-aware deduplication happens after system_json has been loaded.
            # Selecting one row per trial here would discard otherwise identical
            # benchmark runs collected on another machine.
            latest_per_trial=False,
        )

    runs = _filter_compare_runs(
        runs,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        suites=suites,
    )
    rows = build_compare_rows(
        runs,
        metrics=metrics,
        params=params,
        sort_key=sort_key,
        ascending=ascending,
        limit=limit,
        dedupe_configs=not all_runs,
    )
    table_rows = build_compare_table_rows(rows)
    if not table_rows:
        click.echo("no comparable runs")
        return
    click.echo(table(table_rows))


@click.command("reparse", help="Reparse typed benchmark artifacts and refresh metrics.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--include-failed", is_flag=True, help="Also reparse failed runs.")
@click.option(
    "--force",
    is_flag=True,
    help="Replace existing metrics even when the benchmark parser returns no metrics.",
)
def reparse_command(database: Path, include_failed: bool, force: bool) -> None:
    if not database.exists():
        raise FileNotFoundError(f"database not found: {database}")

    updated = 0
    skipped = 0
    missing = 0
    empty = 0
    errors = 0
    with ResultStore(database) as store:
        for run in store.reparse_candidates(include_failed=include_failed):
            try:
                metrics = reparse_run(run)
            except ReparseNotSupported:
                skipped += 1
                continue
            except FileNotFoundError:
                missing += 1
                continue
            except Exception as exc:  # noqa: BLE001 - preserve metrics and continue other runs
                errors += 1
                click.echo(f"failed to reparse {run['run_id']}: {exc}", err=True)
                continue
            if not metrics and not force:
                empty += 1
                continue
            store.update_run_metrics(run["run_id"], metrics)
            updated += 1

    click.echo(
        f"reparsed {updated} run(s); skipped={skipped}; "
        f"missing_artifacts={missing}; empty_metrics={empty}; errors={errors}"
    )
    if errors:
        raise click.ClickException(f"failed to reparse {errors} run(s)")


def _filter_compare_runs(
    runs: list[dict[str, Any]],
    *,
    prompt_tokens: int | None,
    gen_tokens: int | None,
    suites: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    if prompt_tokens is None and gen_tokens is None and not suites:
        return runs

    wanted_suites = set(suites)
    filtered = []
    for run in runs:
        config = run.get("config_json") or {}
        if wanted_suites and run.get("suite") not in wanted_suites:
            continue
        if prompt_tokens is not None and config.get("prompt_tokens") != prompt_tokens:
            continue
        if gen_tokens is not None and config.get("gen_tokens") != gen_tokens:
            continue
        filtered.append(run)
    return filtered
