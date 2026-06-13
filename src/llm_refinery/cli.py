from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import click
import duckdb

from llm_refinery import __version__
from llm_refinery.agent_eval import load_agent_eval_config, run_agent_eval
from llm_refinery.bench_parser import parse_llama_bench_metrics
from llm_refinery.compare import build_compare_rows, build_compare_table_rows
from llm_refinery.config import ConfigError, load_config
from llm_refinery.http_load import load_http_load_config, run_http_load
from llm_refinery.lm_eval import (
    TARGET_CHOICES,
    LmEvalConfig,
    LmEvalTarget,
    default_targets,
    run_lm_eval,
)
from llm_refinery.runner import launch_server, print_plan, run_bench
from llm_refinery.storage import ResultStore, utc_now
from llm_refinery.utils.system import get_system_profile
from llm_refinery.workflows.suite import BenchmarkSuiteWorkflow

EXAMPLE_CONFIG = """name: gemma-cache-sweep

database: results/llm_refinery.duckdb

commands:
  bench: ["llama", "bench"]
  server: ["llama", "server"]

models:
  - name: gemma-4-26b-a4b-q4km
    hf: ggml-org/gemma-4-26B-A4B-it-GGUF:Q4_K_M

defaults:
  cache_type_k: q4_0
  cache_type_v: q4_0

sweep:
  cache_type_k: [q4_0, q8_0]
  cache_type_v: [q4_0, q8_0]

bench:
  prompt_tokens: [512, 2048]
  gen_tokens: [128, 512]
  repetitions: 3
  output: json
  params:
    n_gpu_layers: 99
    flash_attn: 1
  omit_params: []
  extra_args: []

server:
  params:
    ctx_size: 16384
    n_gpu_layers: all
    flash_attn: auto
    mlock: true
    parallel: 1
    perf: true
  extra_args: []
  env: {}
"""


class ErrorHandlingGroup(click.Group):
    def invoke(self, ctx: click.Context) -> Any:
        try:
            return super().invoke(ctx)
        except (click.exceptions.Exit, click.Abort, click.ClickException):
            raise
        except (ConfigError, OSError, IndexError, RuntimeError, duckdb.Error) as exc:
            raise click.ClickException(str(exc)) from exc
        except KeyboardInterrupt as exc:
            raise click.Abort() from exc


@click.group(
    cls=ErrorHandlingGroup,
    context_settings={"help_option_names": ["-h", "--help"]},
)
@click.version_option(__version__, "--version", prog_name="llm-refinery")
def main() -> None:
    """Local LLM benchmarking and serving workflow harness."""


@main.command("init", help="Write a starter sweep config.")
@click.argument(
    "path",
    required=False,
    default="sweeps/gemma-cache-sweep.yaml",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--force", is_flag=True, help="Overwrite an existing file.")
def init_command(path: Path, force: bool) -> None:
    target = Path(path)
    if target.exists() and not force:
        raise FileExistsError(f"{target} already exists; pass --force to overwrite")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    click.echo(f"wrote {target}")


@main.command(help="Print expanded commands without running them.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--kind",
    type=click.Choice(["bench", "server"]),
    default="bench",
    show_default=True,
    help="Command type to plan.",
)
@click.option("--limit", type=int, help="Only show the first N planned commands.")
def plan(config: Path, kind: str, limit: int | None) -> None:
    tune_config = load_config(config)
    print_plan(tune_config, kind=kind, limit=limit)


def _parse_lm_eval_limit(value: str) -> int | None:
    if value.lower() == "all":
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise click.BadParameter("must be a positive integer or 'all'") from exc
    if parsed <= 0:
        raise click.BadParameter("must be a positive integer or 'all'")
    return parsed


@main.command("lm-eval", help="Run lm-eval against local OpenAI-compatible endpoints.")
@click.argument(
    "target",
    required=False,
    default="llama_cpp",
    type=click.Choice(TARGET_CHOICES),
)
@click.argument("limit_text", required=False)
@click.option("--tasks", default="ifeval,gsm8k", show_default=True, help="Comma-separated tasks.")
@click.option("--num-concurrent", type=int, default=1, show_default=True)
@click.option("--max-retries", type=int, default=3, show_default=True)
@click.option("--max-length", type=int, default=16384, show_default=True)
@click.option("--eos-string", default="<turn|>", show_default=True)
@click.option("--gen-kwargs", help="Extra lm-eval generation kwargs for API backends.")
@click.option("--log-samples", is_flag=True, help="Pass --log_samples to lm-eval.")
@click.option("--model", help="Override model name for a single target.")
@click.option("--base-url", help="Override chat-completions URL for a single target.")
@click.option(
    "--output-root",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("results/lm_eval"),
    show_default=True,
)
@click.option(
    "--offline/--online",
    default=True,
    show_default=True,
    help="Set HF_DATASETS_OFFLINE.",
)
@click.option("--dry-run", is_flag=True, help="Print uvx commands without running lm-eval.")
def lm_eval_command(
    target: str,
    limit_text: str | None,
    tasks: str,
    num_concurrent: int,
    max_retries: int,
    max_length: int,
    eos_string: str,
    gen_kwargs: str | None,
    log_samples: bool,
    model: str | None,
    base_url: str | None,
    output_root: Path,
    offline: bool,
    dry_run: bool,
) -> None:
    if (model or base_url) and target in {"both", "all"}:
        raise click.BadParameter("--model/--base-url can only override a single target")

    targets = {}
    if model or base_url:
        target_defaults = default_targets()[target]
        targets[target] = LmEvalTarget(
            name=target,
            model=model or target_defaults.model,
            base_url=base_url or target_defaults.base_url,
        )

    run_lm_eval(
        LmEvalConfig(
            target=target,
            limit=_parse_lm_eval_limit(limit_text) if limit_text is not None else 50,
            tasks=tasks,
            num_concurrent=num_concurrent,
            max_retries=max_retries,
            max_length=max_length,
            eos_string=eos_string,
            log_samples=log_samples,
            gen_kwargs=gen_kwargs,
            output_root=output_root,
            offline=offline,
            targets=targets,
        ),
        dry_run=dry_run,
    )


@main.command(help="Run lm-eval and optional HTTP-load/compare workflow.")
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
    "--llama-cpp-base-url",
    default="http://127.0.0.1:8080/v1/chat/completions",
    help="API URL for sanity check.",
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
def suite(
    config: Path,
    limit_text: str,
    tasks: str | None,
    max_length: int | None,
    eos_string: str | None,
    gen_kwargs: str | None,
    run_lm_eval: bool,
    run_http_load: bool | None,
    require_clean: bool,
    llama_cpp_base_url: str,
    api_model: str | None,
    http_load_config: Path | None,
    target: str | None,
) -> None:
    tune_config = load_config(config)
    effective_run_http_load = (
        http_load_config is not None if run_http_load is None else run_http_load
    )
    limit = _parse_lm_eval_limit(limit_text) if limit_text is not None else tune_config.eval.limit
    workflow = BenchmarkSuiteWorkflow(
        config=tune_config,
        limit=limit,
        tasks=tasks or tune_config.eval.tasks,
        max_length=max_length or tune_config.eval.max_length,
        eos_string=eos_string or tune_config.eval.eos_string,
        gen_kwargs=gen_kwargs if gen_kwargs is not None else tune_config.eval.gen_kwargs,
        run_lm_eval=run_lm_eval,
        run_http_load=effective_run_http_load,
        require_clean=require_clean,
        llama_cpp_base_url=llama_cpp_base_url,
        http_load_config=http_load_config,
        target_name=target,
        api_model=api_model or tune_config.eval.api_model,
    )
    workflow.execute()


@main.command(help="Run llama-bench trials and store results.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--limit", type=int, help="Only run the first N expanded trials.")
@click.option("--dry-run", is_flag=True, help="Print commands without running them.")
@click.option("--keep-going", is_flag=True, help="Continue after failed trials.")
@click.option(
    "--progress/--no-progress",
    "show_progress",
    default=True,
    show_default=True,
    help="Show a Rich progress bar while benchmarks run.",
)
@click.option(
    "--progress-interval",
    type=click.FloatRange(min=0.1),
    default=0.5,
    show_default=True,
    help="Seconds between Rich progress field updates.",
)
@click.option(
    "--db",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Override database path from the config.",
)
def bench(
    config: Path,
    limit: int | None,
    dry_run: bool,
    keep_going: bool,
    show_progress: bool,
    progress_interval: float,
    db: Path | None,
) -> None:
    tune_config = load_config(config)
    run_bench(
        tune_config,
        limit=limit,
        dry_run=dry_run,
        keep_going=keep_going,
        database_override=db,
        show_progress=show_progress,
        progress_interval_s=progress_interval,
    )


@main.command(help="Launch llama server for one expanded config.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--index",
    type=int,
    default=0,
    show_default=True,
    help="Expanded server config index.",
)
@click.option("--dry-run", is_flag=True, help="Print the server command without running it.")
@click.pass_context
def server(ctx: click.Context, config: Path, index: int, dry_run: bool) -> None:
    tune_config = load_config(config)
    ctx.exit(launch_server(tune_config, index=index, dry_run=dry_run))


@main.command("agent-eval", help="Run agent/data benchmarks such as GeoAnalystBench.")
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
def agent_eval(
    config: Path,
    targets: tuple[str, ...],
    limit_text: str | None,
    task_ids: tuple[int, ...],
    dry_run: bool,
) -> None:
    eval_config = load_agent_eval_config(config)
    kwargs: dict[str, Any] = {
        "target_names": targets,
        "task_ids": task_ids,
        "dry_run": dry_run,
    }
    if limit_text is not None:
        kwargs["limit"] = _parse_lm_eval_limit(limit_text)
    run_agent_eval(eval_config, **kwargs)


@main.command("http-load", help="Run OpenAI/Ollama-compatible HTTP load evals.")
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
def http_load(
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


@main.command("backfill-system-metadata", help="Backfill current host metadata into old run rows.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--overwrite", is_flag=True, help="Replace existing system_json values too.")
@click.option("--dry-run", is_flag=True, help="Show how many rows would be updated.")
def backfill_system_metadata(database: Path, overwrite: bool, dry_run: bool) -> None:
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


@main.command(help="Show recent runs or top runs by metric.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--metric", help="Metric name to sort by descending.")
@click.option("--limit", type=int, default=20, show_default=True, help="Maximum rows to show.")
@click.option("--metrics", "list_metrics", is_flag=True, help="List known metric names.")
def report(database: Path, metric: str | None, limit: int, list_metrics: bool) -> None:
    if not database.exists():
        raise FileNotFoundError(f"database not found: {database}")

    with ResultStore(database) as store:
        if list_metrics:
            metric_names = store.metric_names(limit=limit)
            if not metric_names:
                click.echo("no parsed metrics yet")
                return
            click.echo(_table([("metric", "runs"), *metric_names]))
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
            click.echo(_table(table_rows))
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
        click.echo(_table(table_rows))
        click.echo(
            "\nUse --metrics to list metric names, --metric NAME to rank runs, "
            "or `llm-refinery compare` to compare configs."
        )


@main.command(help="Compare benchmark configs with params and throughput columns.")
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
def compare(
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
    click.echo(_table(table_rows))


@main.command(help="Reparse stored stdout artifacts and refresh parsed metrics.")
@click.argument(
    "database",
    required=False,
    default="results/llm_refinery.duckdb",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--include-failed", is_flag=True, help="Also reparse failed runs.")
def reparse(database: Path, include_failed: bool) -> None:
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


def _table(rows: list[tuple[object, ...]]) -> str:
    widths = [0] * max(len(row) for row in rows)
    rendered = [[str(cell) for cell in row] for row in rows]
    for row in rendered:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    terminal_width = shutil.get_terminal_size((120, 20)).columns
    output_lines: list[str] = []
    for row_index, row in enumerate(rendered):
        cells = []
        for index, cell in enumerate(row):
            width = widths[index]
            max_width = max(12, min(width, terminal_width // len(widths)))
            if len(cell) > max_width:
                cell = cell[: max_width - 1] + "…"
            cells.append(cell.ljust(max_width))
        output_lines.append("  ".join(cells).rstrip())
        if row_index == 0:
            separator_cells = ["-" * min(width, terminal_width // len(widths)) for width in widths]
            output_lines.append("  ".join(separator_cells))
    return "\n".join(output_lines)


if __name__ == "__main__":
    main()
