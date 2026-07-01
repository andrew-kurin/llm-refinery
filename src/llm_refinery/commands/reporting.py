from __future__ import annotations

from pathlib import Path
from typing import Any

import click

from llm_refinery.bench_parser import parse_llama_bench_metrics
from llm_refinery.commands.common import table
from llm_refinery.compare import build_compare_rows, build_compare_table_rows
from llm_refinery.storage import ResultStore


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
            table_rows = [("value", "status", "trial", "run_id")]
            table_rows.extend(
                (
                    f"{row['value']:.3f}",
                    row["status"],
                    row["trial_name"],
                    row["run_id"],
                )
                for row in rows
            )
            click.echo(table(table_rows))
            return

        rows = store.recent_runs(limit=limit)
        if not rows:
            click.echo("no runs yet")
            return
        table_rows = [("ended_at", "status", "duration_s", "trial", "run_id")]
        table_rows.extend(
            (
                str(row["ended_at"]),
                row["status"],
                f"{row['duration_s']:.1f}",
                row["trial_name"],
                row["run_id"],
            )
            for row in rows
        )
        click.echo(table(table_rows))
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
            latest_per_trial=not all_runs,
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


@click.command("reparse", help="Reparse stored stdout artifacts and refresh parsed metrics.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--include-failed", is_flag=True, help="Also reparse failed runs.")
def reparse_command(database: Path, include_failed: bool) -> None:
    if not database.exists():
        raise FileNotFoundError(f"database not found: {database}")

    updated = 0
    missing = 0
    empty = 0
    with ResultStore(database) as store:
        for run in store.runs_with_artifacts(include_failed=include_failed):
            stdout_path = run.get("stdout_path")
            if not stdout_path or not Path(stdout_path).exists():
                missing += 1
                continue
            metrics = parse_llama_bench_metrics(Path(stdout_path).read_text(encoding="utf-8"))
            if not metrics:
                empty += 1
            store.update_run_metrics(run["run_id"], metrics)
            updated += 1

    click.echo(f"reparsed {updated} run(s); missing_artifacts={missing}; empty_metrics={empty}")


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
