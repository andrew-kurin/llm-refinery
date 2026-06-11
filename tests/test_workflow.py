from __future__ import annotations

import subprocess
from pathlib import Path

from llm_refinery.config import ModelSpec, TuneConfig
from llm_refinery.workflows.suite import BenchmarkSuiteWorkflow


def _minimal_tune_config(tmp_path: Path) -> TuneConfig:
    return TuneConfig(
        name="tune-suite",
        database=tmp_path / "tune.duckdb",
        commands={"bench": ["llama", "bench"], "server": ["llama", "server"]},
        models=[ModelSpec(name="model", hf="repo/model")],
    )


def _write_http_load_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "http-load.yaml"
    config_path.write_text(
        """
name: http-suite
database: {database}
targets:
  - name: local
    provider: openai
    base_url: http://127.0.0.1:8080/v1
    model: local-model
scenarios:
  - name: short
    prompt: hello
    max_tokens: [8]
    concurrency: [1]
    requests: 1
""".format(database=tmp_path / "http.duckdb"),
        encoding="utf-8",
    )
    return config_path


def test_suite_http_load_without_target_runs_all_targets(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("llm_refinery.workflows.suite.subprocess.run", fake_run)

    workflow = BenchmarkSuiteWorkflow(
        config=_minimal_tune_config(tmp_path),
        run_lm_eval=False,
        run_http_load=True,
        http_load_config=_write_http_load_config(tmp_path),
    )

    workflow.run_load()

    assert len(calls) == 2
    assert calls[0][:4] == ["uv", "run", "llm-refinery", "http-load"]
    assert "--target" not in calls[0]
    assert calls[1][:4] == ["uv", "run", "llm-refinery", "compare"]
    assert "--suite" in calls[1]
    assert calls[1][calls[1].index("--suite") + 1] == "http-suite"
    assert calls[1][4] == str(tmp_path / "http.duckdb")


def test_suite_http_load_with_target_passes_target(tmp_path, monkeypatch):
    calls: list[list[str]] = []

    def fake_run(cmd, check):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr("llm_refinery.workflows.suite.subprocess.run", fake_run)

    workflow = BenchmarkSuiteWorkflow(
        config=_minimal_tune_config(tmp_path),
        run_lm_eval=False,
        run_http_load=True,
        http_load_config=_write_http_load_config(tmp_path),
        target_name="local",
    )

    workflow.run_load()

    assert "--target" in calls[0]
    assert calls[0][calls[0].index("--target") + 1] == "local"


def test_suite_quality_sets_eval_config(tmp_path, monkeypatch):
    captured = {}

    def fake_run_lm_eval(config, *, dry_run=False):
        captured["config"] = config
        captured["dry_run"] = dry_run

    monkeypatch.setattr("llm_refinery.workflows.suite.run_lm_eval", fake_run_lm_eval)

    workflow = BenchmarkSuiteWorkflow(
        config=_minimal_tune_config(tmp_path),
        limit=None,
        tasks="ifeval",
        max_length=4096,
        eos_string="<|im_end|>",
        gen_kwargs="enable_thinking=False",
        run_lm_eval=True,
        run_http_load=False,
        api_model="repo/model",
    )

    workflow.run_quality()

    config = captured["config"]
    assert captured["dry_run"] is False
    assert config.target == "llama_cpp"
    assert config.limit is None
    assert config.tasks == "ifeval"
    assert config.max_length == 4096
    assert config.eos_string == "<|im_end|>"
    assert config.gen_kwargs == "enable_thinking=False"
    assert config.targets["llama_cpp"].model == "repo/model"
    assert config.targets["llama_cpp"].base_url == "http://127.0.0.1:8080/v1/chat/completions"
