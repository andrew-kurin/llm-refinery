from __future__ import annotations

from pathlib import Path

import click

from llm_refinery.config import load_config
from llm_refinery.runner import launch_server, print_plan, run_bench

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


@click.command("init", help="Write a starter sweep config.")
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


@click.command("plan", help="Print expanded commands without running them.")
@click.argument("config", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--kind",
    type=click.Choice(["bench", "server"]),
    default="bench",
    show_default=True,
    help="Command type to plan.",
)
@click.option("--limit", type=int, help="Only show the first N planned commands.")
def plan_command(config: Path, kind: str, limit: int | None) -> None:
    tune_config = load_config(config)
    print_plan(tune_config, kind=kind, limit=limit)


@click.command("bench", help="Run llama-bench trials and store results.")
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
def bench_command(
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


@click.command("server", help="Launch llama server for one expanded config.")
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
def server_command(ctx: click.Context, config: Path, index: int, dry_run: bool) -> None:
    tune_config = load_config(config)
    ctx.exit(launch_server(tune_config, index=index, dry_run=dry_run))
