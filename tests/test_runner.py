import subprocess

from llm_refinery.config import TuneConfig
from llm_refinery.runner import BenchProgress, format_duration, launch_server


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
    config = TuneConfig.from_mapping(
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

    monkeypatch.setattr("llm_refinery.providers.llama_cpp.subprocess.run", fake_run)

    assert launch_server(config) == 0
    assert destination.read_bytes() == b"draft-head"
    assert calls
    assert calls[0][calls[0].index("--model-draft") + 1] == str(destination)


def test_bench_progress_estimates_from_completed_trials():
    progress = BenchProgress(total=4, started_monotonic=0.0)

    assert progress.eta_after_completed_s() is None
    assert progress.eta_during_current_s(3.0) is None

    progress.record_completion(10.0)

    assert progress.average_duration_s == 10.0
    assert progress.eta_after_completed_s() == 30.0
    assert progress.eta_during_current_s(4.0) == 26.0
