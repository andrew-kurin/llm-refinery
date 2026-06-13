from __future__ import annotations

import csv
import json
from pathlib import Path

from click.testing import CliRunner

from llm_refinery.agent_eval import (
    AgentEvalConfig,
    AgentEvalResult,
    AgentEvalTarget,
    GeoAnalystBenchSpec,
    run_agent_eval,
)
from llm_refinery.cli import main
from llm_refinery.storage import ResultStore


def write_geoanalyst_csv(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "Open Source",
                "Task",
                "Instruction",
                "Domain Knowledge",
                "Dataset Description",
                "Human Designed Workflow",
                "Task Length",
                "CodeString",
            ]
        )
        writer.writerow(
            [
                "1",
                "T",
                "Buffer schools",
                "Create buffers around schools.",
                "Buffers expand geometry.",
                "schools.geojson has points.",
                "1. Load dataset\n2. Buffer\n3. Save",
                "3",
                "def model():\n    return None",
            ]
        )
        writer.writerow(
            [
                "2",
                "F",
                "ArcPy task",
                "Use ArcPy.",
                "ArcPy knowledge.",
                "proprietary.gdb",
                "1. Load\n2. Run",
                "2",
                "import arcpy",
            ]
        )


def test_agent_eval_dry_run_plans_geoanalystbench_requests(tmp_path: Path):
    dataset = tmp_path / "GeoAnalystBench.csv"
    write_geoanalyst_csv(dataset)
    config = tmp_path / "geo.yaml"
    config.write_text(
        f"""
name: geo-smoke
database: {tmp_path / 'runs.duckdb'}
benchmark:
  kind: geoanalystbench
  dataset: {dataset}
  limit: 1
  response_types: [workflow]
targets:
  - name: local
    provider: openai
    base_url: http://127.0.0.1:8080/v1
    model: local-model
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["agent-eval", str(config), "--dry-run"])

    assert result.exit_code == 0
    assert "benchmark=geoanalystbench" in result.output
    assert "target=local" in result.output
    assert "requests=1" in result.output
    assert "tasks=1" in result.output


def test_geoanalystbench_agent_eval_records_metrics_and_artifacts(tmp_path: Path):
    dataset = tmp_path / "GeoAnalystBench.csv"
    write_geoanalyst_csv(dataset)
    config = AgentEvalConfig(
        name="geo-smoke",
        database=tmp_path / "runs.duckdb",
        benchmark=GeoAnalystBenchSpec(
            dataset=str(dataset),
            task_ids=(1,),
            limit=None,
            response_types=("workflow", "code"),
        ),
        targets=[
            AgentEvalTarget(
                name="local",
                provider="openai",
                base_url="http://127.0.0.1:8080/v1",
                model="local-model",
            )
        ],
    )

    class FakeClient:
        def complete(self, target, request):
            if request.response_type == "workflow":
                return AgentEvalResult(
                    request=request,
                    ok=True,
                    latency_s=1.0,
                    response_text="1. Load dataset\n2. Buffer\n3. Save",
                    workflow_step_count=3,
                    workflow_step_abs_error=0,
                )
            return AgentEvalResult(
                request=request,
                ok=True,
                latency_s=2.0,
                response_text="def model():\n    return None\n",
                code_syntax_ok=True,
            )

    run_agent_eval(config, client=FakeClient())

    with ResultStore(config.database) as store:
        runs = store.comparison_runs()
    assert len(runs) == 1
    run = runs[0]
    metrics = run["metrics"]
    assert metrics["success_rate"] == 1.0
    assert metrics["workflow_step_abs_error_avg"] == 0.0
    assert metrics["code_syntax_pass_rate"] == 1.0
    assert run["system_json"]["hardware"]

    responses_path = Path(run["stdout_path"])
    responses = [json.loads(line) for line in responses_path.read_text().splitlines()]
    assert len(responses) == 2
    assert responses[0]["request"]["task"]["task_id"] == 1


def test_load_agent_eval_config_rejects_unknown_benchmark(tmp_path: Path):
    config = tmp_path / "bad.yaml"
    config.write_text(
        """
name: bad
benchmark:
  kind: dabstep
targets:
  - name: local
    base_url: http://127.0.0.1:8080/v1
    model: local-model
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["agent-eval", str(config), "--dry-run"])

    assert result.exit_code != 0
    assert "unsupported agent-eval benchmark kind" in result.output
