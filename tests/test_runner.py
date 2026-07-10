import subprocess
from pathlib import Path

from llm_refinery.benchmarks.llama_bench.config import LlamaSweepConfig
from llm_refinery.benchmarks.llama_bench.progress import BenchProgress, format_duration
from llm_refinery.benchmarks.llama_bench.runner import run_bench
from llm_refinery.benchmarks.llama_bench.server import launch_server
from llm_refinery.storage.duckdb import ResultStore


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(0.25) == "<1s"
    assert format_duration(9.4) == "9s"
    assert format_duration(65) == "1m05s"
    assert format_duration(3661) == "1h01m01s"


def test_launch_server_downloads_mtp_head(tmp_path, monkeypatch):
    source = tmp_path / "source.gguf"
    source.write_bytes(b"draft-head")
    destination = tmp_path / "heads" / "head.gguf"
    config = LlamaSweepConfig.from_mapping(
        {
            "models": [{"name": "m", "hf": "repo/model"}],
            "server": {
                "params": {
                    "mtp_head": {"url": source.as_uri(), "path": str(destination)},
                    "spec_type": "draft-mtp",
                }
            },
        }
    )
    calls: list[list[str]] = []

    def fake_run(cmd, *, env, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("llm_refinery.benchmarks.llama_bench.server.subprocess.run", fake_run)

    assert launch_server(config) == 0
    assert destination.read_bytes() == b"draft-head"
    assert calls
    assert calls[0][calls[0].index("--model-draft") + 1] == str(destination)


def test_bench_database_override_also_controls_artifact_location(
    tmp_path: Path, monkeypatch
):
    configured_database = tmp_path / "configured" / "runs.duckdb"
    override_database = tmp_path / "override" / "runs.duckdb"
    config = LlamaSweepConfig.from_mapping(
        {
            "name": "suite",
            "database": str(configured_database),
            "models": [{"name": "m", "hf": "repo/model"}],
            "bench": {"prompt_tokens": [8], "gen_tokens": [4]},
        }
    )

    def fake_process(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            ["llama", "bench"],
            0,
            '[{"test":"tg4","t/s":12.5}]',
            "",
        )

    monkeypatch.setattr(
        "llm_refinery.benchmarks.llama_bench.runner.run_bench_process",
        fake_process,
    )
    monkeypatch.setattr(
        "llm_refinery.benchmarks.llama_bench.runner.detect_llama_version",
        lambda _command: "test-version",
    )

    outcomes = run_bench(
        config,
        database_override=override_database,
        show_progress=False,
    )

    assert len(outcomes) == 1
    assert not configured_database.exists()
    with ResultStore(override_database) as store:
        run = store.comparison_runs()[0]
    assert Path(run["artifacts"]["stdout"]["path"]).is_relative_to(
        override_database.parent / "artifacts"
    )


def test_bench_progress_estimates_from_completed_trials():
    progress = BenchProgress(total=4, started_monotonic=0.0)

    assert progress.eta_after_completed_s() is None
    assert progress.eta_during_current_s(3.0) is None

    progress.record_completion(10.0)

    assert progress.average_duration_s == 10.0
    assert progress.eta_after_completed_s() == 30.0
    assert progress.eta_during_current_s(4.0) == 26.0
