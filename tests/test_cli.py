import json

from click.testing import CliRunner

from llm_refinery.cli import main
from llm_refinery.storage import ResultStore, RunRecord, utc_now


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
    assert "uvx --from 'lm-eval[api]'" in result.output
    assert "hf.co/ggml-org/gemma-4-12B-it-GGUF:Q8_0" in result.output
    assert 'reasoning_effort="none"' in result.output


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


def test_compare_command_shows_params_and_sorts_by_generation_tps(tmp_path):
    database = tmp_path / "runs.duckdb"
    now = utc_now()
    with ResultStore(database) as store:
        store.record_run(
            RunRecord(
                run_id="slow-run",
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


def test_backfill_system_metadata_command(tmp_path, monkeypatch):
    database = tmp_path / "runs.duckdb"
    now = utc_now()
    with ResultStore(database) as store:
        store.record_run(
            RunRecord(
                run_id="old-run",
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
        "llm_refinery.cli.get_system_profile",
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
