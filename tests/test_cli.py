import json

from click.testing import CliRunner

from llm_refinery.cli import main
from llm_refinery.storage.duckdb import ResultStore, utc_now
from llm_refinery.storage.models import RunRecord


def test_click_version():
    result = CliRunner().invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "llm-refinery" in result.output
    assert "0.1.0" in result.output


def test_click_plan_command():
    result = CliRunner().invoke(
        main,
        ["plan", "sweeps/gemma-cache-sweep.yaml", "--limit", "1"],
    )

    assert result.exit_code == 0
    assert "llama bench" in result.output
    assert "planned 1 of 36 bench command(s)" in result.output


def test_lm_eval_command_dry_run_uses_python_cli():
    result = CliRunner().invoke(
        main,
        [
            "lm-eval",
            "ollama",
            "50",
            "--model",
            "hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0",
            "--gen-kwargs",
            'reasoning_effort="none"',
            "--max-length",
            "8192",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "==> Running lm-eval target=ollama" in result.output
    assert "uvx --from 'lm-eval[api]==0.4.12'" in result.output
    assert "hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0" in result.output
    assert 'reasoning_effort="none"' in result.output


def test_lm_eval_command_accepts_arbitrary_openai_compatible_target():
    result = CliRunner().invoke(
        main,
        [
            "lm-eval",
            "custom-endpoint",
            "5",
            "--model",
            "custom/model",
            "--base-url",
            "https://example.test/v1",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "target=custom-endpoint" in result.output
    assert "custom/model" in result.output
    assert "https://example.test/v1" in result.output


def test_lm_eval_command_dry_run_supports_include_path_and_suite_db(tmp_path):
    include_path = tmp_path / "tasks"
    include_path.mkdir()
    result = CliRunner().invoke(
        main,
        [
            "lm-eval",
            "llama_cpp",
            "5",
            "--tasks",
            "gpqa_main_fixed_generative",
            "--include-path",
            str(include_path),
            "--suite-name",
            "quality-reasoning",
            "--db",
            str(tmp_path / "runs.duckdb"),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "--include_path" in result.output
    assert str(include_path) in result.output
    assert "gpqa_main_fixed_generative" in result.output


def test_suite_command_uses_dedicated_manifest_and_records_parent_run(tmp_path):
    database = tmp_path / "suite.duckdb"
    manifest = tmp_path / "suite.yaml"
    manifest.write_text(
        f"""
name: smoke-suite
database: {database}
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
quality:
  enabled: false
http_load:
  enabled: false
preflight:
  enabled: false
""",
        encoding="utf-8",
    )

    result = CliRunner().invoke(main, ["suite", str(manifest)])

    assert result.exit_code == 0, result.output
    with ResultStore(database) as store:
        run = store.comparison_runs()[0]
    assert run["benchmark_kind"] == "suite"
    assert run["metrics"]["child_run_count"] == 0.0


def test_suite_command_rejects_zero_max_length(tmp_path):
    manifest = tmp_path / "suite.yaml"
    manifest.write_text("name: unused\n", encoding="utf-8")

    result = CliRunner().invoke(main, ["suite", str(manifest), "--max-length", "0"])

    assert result.exit_code == 2
    assert "0 is not in the range x>=1" in result.output


def test_suite_discovery_overrides_keep_ssh_and_http_planes_separate(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "suite.yaml"
    manifest.write_text(
        f"""
schema_version: 2
name: spark-suite
database: {tmp_path / 'runs.duckdb'}
target:
  name: spark
  host:
    access: ssh
    destination: dgx
  endpoint:
    protocol: openai_chat
    base_url: http://old-spark.local:8000/v1
  model:
    selection: single
quality:
  enabled: false
http_load:
  enabled: false
preflight:
  enabled: false
""",
        encoding="utf-8",
    )
    captured = {}

    class FakeWorkflow:
        def __init__(self, config):
            captured["config"] = config

        def execute(self):
            return None

    monkeypatch.setattr("llm_refinery.commands.suite.BenchmarkSuiteWorkflow", FakeWorkflow)

    result = CliRunner().invoke(
        main,
        [
            "suite",
            str(manifest),
            "--ssh-destination",
            "another-spark",
            "--base-url",
            "http://new-spark.local:9000",
            "--api-model",
            "chosen-model",
            "--target",
            "scenario-target",
        ],
    )

    assert result.exit_code == 0, result.output
    config = captured["config"]
    assert config.target.host.destination == "another-spark"
    assert config.target.endpoint.base_url == "http://new-spark.local:9000/v1"
    assert config.target.endpoint.model == "chosen-model"
    assert config.target.model.selection == "explicit"
    assert config.target.model.model_id == "chosen-model"
    assert config.http_load.targets == ("scenario-target",)


def test_compare_command_shows_params_and_sorts_by_generation_tps(tmp_path):
    database = tmp_path / "runs.duckdb"
    now = utc_now()
    with ResultStore(database) as store:
        store.record_run(
            RunRecord(
                run_id="slow-run",
                benchmark_kind="llama_bench",
                spec_hash="slow-spec",
                suite="suite",
                trial_name="suite/model/slow",
                status="ok",
                started_at=now,
                ended_at=now,
                duration_s=2.0,
                command="llama bench slow",
                cwd=str(tmp_path),
                config_json={
                    "params": {"cache_type_k": "q4_0", "cache_type_v": "q4_0"},
                    "prompt_tokens": 512,
                    "gen_tokens": 128,
                },
                metrics={"pp512.tokens_per_second": 100.0, "tg128.tokens_per_second": 10.0},
                system_json={"hardware": {"model": "Mac-slow", "memory_gb": 32.0}},
            )
        )
        store.record_run(
            RunRecord(
                run_id="fast-run",
                benchmark_kind="llama_bench",
                spec_hash="fast-spec",
                suite="suite",
                trial_name="suite/model/fast",
                status="ok",
                started_at=now,
                ended_at=now,
                duration_s=1.0,
                command="llama bench fast",
                cwd=str(tmp_path),
                config_json={
                    "params": {"cache_type_k": "q8_0", "cache_type_v": "q8_0"},
                    "prompt_tokens": 512,
                    "gen_tokens": 128,
                },
                metrics={"pp512.tokens_per_second": 200.0, "tg128.tokens_per_second": 20.0},
                system_json={"hardware": {"model": "Mac-fast", "memory_gb": 128.0}},
            )
        )

    result = CliRunner().invoke(main, ["compare", str(database), "--limit", "2"])

    assert result.exit_code == 0
    assert "tg_tps" in result.output
    assert "cache_type_k" in result.output
    assert "q8_0" in result.output
    assert result.output.index("fast-run") < result.output.index("slow-run")

    system_result = CliRunner().invoke(
        main,
        [
            "compare",
            str(database),
            "--limit",
            "2",
            "--param",
            "system.hardware.model",
            "--param",
            "system.hardware.memory_gb",
        ],
    )

    assert system_result.exit_code == 0
    assert "Mac-fast" in system_result.output
    assert "128.0" in system_result.output


def test_compare_command_keeps_same_trial_from_distinct_hosts(tmp_path):
    database = tmp_path / "runs.duckdb"
    now = utc_now()
    shared = {
        "benchmark_kind": "llama_bench",
        "spec_hash": "same-spec",
        "suite": "suite",
        "trial_name": "suite/model/same-spec",
        "status": "ok",
        "started_at": now,
        "ended_at": now,
        "duration_s": 1.0,
        "command": "llama-bench",
        "cwd": str(tmp_path),
        "config_json": {"model": "model", "params": {}},
    }
    with ResultStore(database) as store:
        store.record_run(
            RunRecord(
                run_id="mac-run",
                metrics={"tg128.tokens_per_second": 20.0},
                system_json={"hostname": "mac", "host_fingerprint": "host-mac"},
                **shared,
            )
        )
        store.record_run(
            RunRecord(
                run_id="spark-run",
                metrics={"tg128.tokens_per_second": 30.0},
                system_json={"hostname": "spark", "host_fingerprint": "host-spark"},
                **shared,
            )
        )

    result = CliRunner().invoke(
        main,
        ["compare", str(database), "--metric", "tg128.tokens_per_second", "--limit", "10"],
    )

    assert result.exit_code == 0, result.output
    assert "mac-run" in result.output
    assert "spark-run" in result.output
    assert "mac" in result.output
    assert "spark" in result.output


def test_backfill_system_metadata_command(tmp_path, monkeypatch):
    database = tmp_path / "runs.duckdb"
    now = utc_now()
    with ResultStore(database) as store:
        store.record_run(
            RunRecord(
                run_id="old-run",
                benchmark_kind="llama_bench",
                spec_hash="old-spec",
                suite="suite",
                trial_name="suite/model/old",
                status="ok",
                started_at=now,
                ended_at=now,
                duration_s=1.0,
                command="llama bench old",
                cwd=str(tmp_path),
                config_json={"params": {}},
            )
        )
        store.record_run(
            RunRecord(
                run_id="new-run",
                benchmark_kind="llama_bench",
                spec_hash="new-spec",
                suite="suite",
                trial_name="suite/model/new",
                status="ok",
                started_at=now,
                ended_at=now,
                duration_s=1.0,
                command="llama bench new",
                cwd=str(tmp_path),
                config_json={"params": {}},
                system_json={"hardware": {"model": "existing"}},
            )
        )

    monkeypatch.setattr(
        "llm_refinery.commands.system.get_system_profile",
        lambda: {"hardware": {"model": "Mac16,12", "memory_gb": 32.0}},
    )

    result = CliRunner().invoke(main, ["backfill-system-metadata", str(database)])

    assert result.exit_code == 0
    assert "backfilled 1 run(s)" in result.output
    with ResultStore(database) as store:
        rows = store.connection.execute(
            "SELECT run_id, system_json FROM runs ORDER BY run_id"
        ).fetchall()

    by_run_id = {run_id: json.loads(system_json) for run_id, system_json in rows}
    assert by_run_id["old-run"]["hardware"]["model"] == "Mac16,12"
    assert by_run_id["old-run"]["backfill"]["assumed_current_hardware"] is True
    assert by_run_id["new-run"]["hardware"]["model"] == "existing"
