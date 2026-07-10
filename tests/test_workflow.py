from pathlib import Path

from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.workflows.suite import BenchmarkSuiteWorkflow
from llm_refinery.workflows.suite_config import SuiteConfig, load_suite_config


def _write_http_load_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "http-load.yaml"
    config_path.write_text(
        f"""
name: http-suite
database: {tmp_path / 'http.duckdb'}
targets:
  - name: local
    protocol: openai_chat
    base_url: http://127.0.0.1:8080/v1
    model: local-model
scenarios:
  - name: short
    prompt: hello
    max_tokens: [8]
    concurrency: [1]
    requests: 1
""",
        encoding="utf-8",
    )
    return config_path


def test_suite_config_resolves_http_config_relative_to_manifest(tmp_path: Path):
    _write_http_load_config(tmp_path)
    manifest = tmp_path / "suite.yaml"
    manifest.write_text(
        f"""
name: suite
database: {tmp_path / 'runs.duckdb'}
endpoint:
  name: local
  protocol: openai_chat
  base_url: http://127.0.0.1:8080/v1
  model: local-model
quality:
  enabled: false
http_load:
  config: http-load.yaml
  targets: [local]
preflight:
  enabled: false
""",
        encoding="utf-8",
    )

    config = load_suite_config(manifest)

    assert config.http_load.enabled is True
    assert config.http_load.config == tmp_path / "http-load.yaml"
    assert config.http_load.targets == ("local",)


def test_suite_calls_services_directly_and_links_child_runs(tmp_path: Path):
    http_config = _write_http_load_config(tmp_path)
    config = SuiteConfig.from_mapping(
        {
            "name": "suite",
            "database": str(tmp_path / "runs.duckdb"),
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "local-model",
            },
            "quality": {"tasks": "ifeval", "limit": "all"},
            "http_load": {"config": str(http_config), "targets": ["local"]},
            "preflight": {"enabled": False},
        }
    )
    calls = []

    def fake_lm_eval(config, **kwargs):
        calls.append(("quality", config, kwargs))
        return []

    def fake_http_load(config, **kwargs):
        calls.append(("load", config, kwargs))
        return []

    result = BenchmarkSuiteWorkflow(
        config,
        lm_eval_runner=fake_lm_eval,
        http_load_runner=fake_http_load,
        system_snapshot=lambda: "snapshot",
    ).execute()

    assert [call[0] for call in calls] == ["quality", "load"]
    assert calls[0][1].limit is None
    assert calls[0][1].tasks == "ifeval"
    assert calls[0][2]["parent_run_id"] == result.run.run_id
    assert calls[1][2]["parent_run_id"] == result.run.run_id
    assert calls[1][2]["store"] is calls[0][2]["store"]

    with ResultStore(config.database) as store:
        runs = store.comparison_runs()
    assert len(runs) == 1
    assert runs[0]["benchmark_kind"] == "suite"
    assert set(runs[0]["artifacts"]) == {"system_before", "system_after"}
