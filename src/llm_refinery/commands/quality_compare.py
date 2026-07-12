from __future__ import annotations

from pathlib import Path

import click

from llm_refinery.commands.common import table
from llm_refinery.quality_compare import compare_paired_correctness
from llm_refinery.storage.duckdb import ResultStore


@click.command(
    "quality-compare",
    help="Compare two runs on the exact same retained correctness samples.",
)
@click.argument(
    "database",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument("baseline_run_id")
@click.argument("candidate_run_id")
@click.option("--task", help="Restrict the paired comparison to one lm-eval task.")
@click.option(
    "--sample-metric",
    default="correct",
    show_default=True,
    help="Binary metric stored on each sample (for example prompt_level_loose_acc).",
)
def quality_compare_command(
    database: Path,
    baseline_run_id: str,
    candidate_run_id: str,
    task: str | None,
    sample_metric: str,
) -> None:
    with ResultStore(database) as store:
        baseline = store.samples_for_run(baseline_run_id)
        candidate = store.samples_for_run(candidate_run_id)
    comparison = compare_paired_correctness(
        baseline,
        candidate,
        task=task,
        sample_metric=sample_metric,
    )
    rows = [
        ("statistic", "value"),
        ("paired samples", str(comparison.paired_count)),
        ("baseline-only samples", str(comparison.baseline_only_count)),
        ("candidate-only samples", str(comparison.candidate_only_count)),
        ("baseline accuracy", f"{comparison.baseline_accuracy:.6f}"),
        ("candidate accuracy", f"{comparison.candidate_accuracy:.6f}"),
        ("candidate - baseline", f"{comparison.accuracy_delta:+.6f}"),
        (
            "paired delta 95% CI (normal)",
            f"[{comparison.accuracy_delta_ci95_low:+.6f}, "
            f"{comparison.accuracy_delta_ci95_high:+.6f}]",
        ),
        ("candidate wins", str(comparison.candidate_win_count)),
        ("candidate losses", str(comparison.candidate_loss_count)),
        ("ties", str(comparison.tie_count)),
        ("McNemar exact p", f"{comparison.mcnemar_exact_p:.6g}"),
    ]
    click.echo(table(rows))
