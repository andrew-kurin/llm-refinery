from pathlib import Path

import pytest

from llm_refinery.core.config import ConfigError
from llm_refinery.core.targets import (
    HostDiscovery,
    ModelDescriptor,
    ResolvedTarget,
    ServiceDiscovery,
    TargetInspection,
    TargetSpec,
)
from llm_refinery.storage.duckdb import ResultStore
from llm_refinery.workflows.suite import BenchmarkSuiteWorkflow
from llm_refinery.workflows.suite_config import SuiteConfig, load_suite_config


def _write_http_load_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "http-load.yaml"
    config_path.write_text(
        f"""
name: http-suite
database: {tmp_path / "http.duckdb"}
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
database: {tmp_path / "runs.duckdb"}
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


def test_local_quality_core_is_pinned_and_release_sized():
    config = load_suite_config(Path("sweeps/local-quality-core-suite.yaml"))

    assert config.quality.limit is None
    assert config.quality.package_spec == "lm-eval[api]==0.4.12"
    assert {"ifeval_pinned", "ifbench"} <= set(config.quality.tasks.split(","))
    assert any(package.startswith("ifbench @ git+") for package in config.quality.extra_packages)
    assert config.quality.offline is False


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
    assert set(runs[0]["artifacts"]) == {"system_before", "system_after", "preflight"}


def test_legacy_suite_preserves_http_manifest_targets(tmp_path: Path):
    http_config = tmp_path / "http-load.yaml"
    http_config.write_text(
        f"""
name: legacy-http
database: {tmp_path / "http.duckdb"}
targets:
  - name: ollama-a
    protocol: ollama_chat
    base_url: http://127.0.0.1:11434
    model: model-a
  - name: ollama-b
    protocol: ollama_chat
    base_url: http://127.0.0.1:11435
    model: model-b
scenarios:
  - name: short
    prompt: hello
    max_tokens: [8]
    concurrency: [1]
    requests: 1
""",
        encoding="utf-8",
    )
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 1,
            "name": "legacy-suite",
            "database": str(tmp_path / "runs.duckdb"),
            "endpoint": {
                "name": "quality-openai",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:11434/v1",
                "model": "quality-model",
            },
            "quality": {"enabled": False},
            "http_load": {
                "config": str(http_config),
                "targets": ["ollama-a", "ollama-b"],
            },
            "preflight": {"enabled": False},
        }
    )
    calls = []

    def fake_http_load(load_config, **kwargs):
        calls.append((load_config, kwargs))
        return []

    BenchmarkSuiteWorkflow(
        config,
        http_load_runner=fake_http_load,
        system_snapshot=lambda: "snapshot",
    ).execute()

    assert [target.protocol for target in calls[0][0].targets] == [
        "ollama_chat",
        "ollama_chat",
    ]
    assert [target.base_url for target in calls[0][0].targets] == [
        "http://127.0.0.1:11434",
        "http://127.0.0.1:11435",
    ]
    assert calls[0][1]["target_names"] == ("ollama-a", "ollama-b")


def test_schema_v2_suite_requires_endpoint_or_target():
    with pytest.raises(ConfigError, match="requires exactly one"):
        SuiteConfig.from_mapping({"schema_version": 2, "name": "missing-target"})


def test_schema_v1_suite_keeps_legacy_default_endpoint():
    config = SuiteConfig.from_mapping({"schema_version": 1, "name": "legacy-default"})

    assert config.endpoint is not None
    assert config.endpoint.base_url == "http://127.0.0.1:8080/v1"


def test_suite_can_require_the_response_model_identity(tmp_path: Path):
    config = SuiteConfig.from_mapping(
        {
            "name": "model-binding",
            "database": str(tmp_path / "runs.duckdb"),
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "request-alias",
            },
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"expected_response_model": "expected-loaded-model"},
        }
    )
    workflow = BenchmarkSuiteWorkflow(
        config,
        port_listener=lambda port: port == 8080,
        sanity_checker=lambda _endpoint: {
            "success": True,
            "response_model": "wrong-loaded-model",
        },
    )

    with pytest.raises(RuntimeError, match="expected-loaded-model"):
        workflow.preflight("snapshot")


def _resolved_dgx_target(*, max_model_len: int = 32768) -> tuple[TargetSpec, TargetInspection]:
    spec = TargetSpec.from_mapping(
        {
            "name": "spark",
            "host": {"access": "ssh", "destination": "dgx"},
            "endpoint": {
                "protocol": "openai_chat",
                "base_url": "http://aitopatom-41de.local:8000/v1",
            },
            "model": {"selection": "single"},
        }
    )
    host = HostDiscovery(
        transport="ssh",
        destination="dgx",
        profile={
            "hostname": "spark",
            "host_fingerprint": "host-spark",
            "hardware": {"model": "DGX Spark"},
        },
    )
    model = ModelDescriptor(
        id="served-model",
        root="org/model",
        max_model_len=max_model_len,
        owned_by="vllm",
    )
    service = ServiceDiscovery(
        implementation="vllm",
        base_url=spec.endpoint.api_root_url,
        health="ok",
        version="0.10.2",
        models=(model,),
    )
    resolved = ResolvedTarget(
        spec_name=spec.name,
        endpoint=spec.endpoint.resolve(model.id),
        host=host,
        service=service,
        model=model,
        selection="single_discovered",
    )
    return spec, TargetInspection(
        spec=spec,
        host=host,
        service=service,
        resolved=resolved,
    )


def _target_config_mapping(spec: TargetSpec) -> dict[str, object]:
    return {
        "name": spec.name,
        "host": {
            "access": spec.host.access,
            "destination": spec.host.destination,
        },
        "endpoint": {
            "name": spec.endpoint.name,
            "protocol": spec.endpoint.protocol,
            "base_url": spec.endpoint.base_url,
        },
        "model": {
            "selection": spec.model.selection,
        },
    }


class _FakeTargetResolver:
    def __init__(self, inspection: TargetInspection):
        self.inspection = inspection
        self.calls = 0
        self.snapshot_calls = 0
        self.metrics_calls = 0

    def inspect(self, spec, *, allow_service_unavailable=False):
        assert spec == self.inspection.spec
        assert allow_service_unavailable is True
        self.calls += 1
        return self.inspection

    def snapshot_host(self, spec):
        assert spec == self.inspection.spec
        self.snapshot_calls += 1
        if self.inspection.host is None:
            raise RuntimeError("target host inventory unavailable")
        return self.inspection.host

    def metrics(self, spec):
        assert spec == self.inspection.spec
        self.metrics_calls += 1
        return 'vllm:num_requests_running{model_name="served-model"} 0\n'


def test_suite_config_loads_reusable_target_relative_to_manifest(tmp_path: Path):
    target_path = tmp_path / "targets" / "spark.yaml"
    target_path.parent.mkdir()
    target_path.write_text(
        """
schema_version: 1
target:
  name: spark
  host:
    access: ssh
    destination: dgx
  endpoint:
    protocol: openai_chat
    base_url: http://aitopatom-41de.local:8000/v1
  model:
    selection: single
""",
        encoding="utf-8",
    )
    manifest = tmp_path / "suite.yaml"
    manifest.write_text(
        f"""
schema_version: 2
name: spark-suite
database: {tmp_path / 'runs.duckdb'}
target: targets/spark.yaml
quality:
  enabled: false
http_load:
  enabled: false
preflight:
  enabled: false
""",
        encoding="utf-8",
    )

    config = load_suite_config(manifest)

    assert config.endpoint is None
    assert config.target is not None
    assert config.target.host.destination == "dgx"
    assert config.target.model.selection == "single"


def test_remote_suite_resolves_once_for_children_and_overlays_http_target(tmp_path: Path):
    spec, inspection = _resolved_dgx_target()
    resolver = _FakeTargetResolver(inspection)
    http_config = _write_http_load_config(tmp_path)
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "spark-suite",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"tasks": "ifeval", "limit": 1},
            "http_load": {"config": str(http_config), "targets": ["local"]},
            "preflight": {"enabled": False},
        }
    )
    calls = []

    def fake_lm_eval(lm_config, **kwargs):
        calls.append(("quality", lm_config, kwargs))
        return []

    def fake_http_load(load_config, **kwargs):
        calls.append(("load", load_config, kwargs))
        return []

    result = BenchmarkSuiteWorkflow(
        config,
        lm_eval_runner=fake_lm_eval,
        http_load_runner=fake_http_load,
        target_resolver=resolver,
        system_snapshot=lambda: "client snapshot",
    ).execute()

    assert [call[0] for call in calls] == ["quality", "load"]
    assert calls[0][1].targets["spark"].model == "served-model"
    assert calls[0][1].tokenizer is None
    assert calls[1][1].targets[0].base_url == "http://aitopatom-41de.local:8000/v1"
    assert calls[1][1].targets[0].model == "served-model"
    assert calls[0][2]["run_context"].to_target_json()["host"]["destination"] == "dgx"
    assert calls[1][2]["run_context"].to_target_json()["model"]["id"] == "served-model"
    assert resolver.calls == 1
    assert resolver.snapshot_calls == 1
    assert resolver.metrics_calls == 2

    with ResultStore(config.database) as store:
        run = store.comparison_runs()[0]
    assert run["run_id"] == result.run.run_id
    assert run["target_json"]["model"]["id"] == "served-model"
    assert set(run["artifacts"]) >= {
        "target_discovery",
        "server_before",
        "server_after",
        "vllm_metrics_before",
        "vllm_metrics_after",
    }


def test_remote_suite_preflight_defaults_to_discovered_model_identity(tmp_path: Path):
    spec, inspection = _resolved_dgx_target()
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "spark-model-binding",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": True, "require_clean": False},
        }
    )
    workflow = BenchmarkSuiteWorkflow(
        config,
        target_resolver=_FakeTargetResolver(inspection),
        sanity_checker=lambda _endpoint: {
            "success": True,
            "response_model": "different-served-model",
        },
        system_snapshot=lambda: "snapshot",
    )

    with pytest.raises(RuntimeError, match="served-model"):
        workflow.execute()


def test_remote_suite_persists_unavailable_discovery_and_starts_no_children(tmp_path: Path):
    spec, available = _resolved_dgx_target()
    unavailable = TargetInspection(
        spec=spec,
        host=available.host,
        service=ServiceDiscovery(
            implementation="vllm",
            base_url=spec.endpoint.api_root_url,
            health="unavailable",
            version=None,
            models=(),
            errors=("connection refused",),
        ),
        resolved=None,
        errors=("connection refused",),
    )
    resolver = _FakeTargetResolver(unavailable)
    snapshot_calls = 0

    def flaky_client_snapshot():
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls > 1:
            raise RuntimeError("local telemetry failed")
        return "client snapshot"

    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "offline-spark",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": False},
        }
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        BenchmarkSuiteWorkflow(
            config,
            target_resolver=resolver,
            system_snapshot=flaky_client_snapshot,
        ).execute()

    with ResultStore(config.database) as store:
        run = store.comparison_runs(include_failed=True)[0]
    assert run["status"] == "failed"
    assert run["target_json"]["status"] == "unavailable"
    assert "connection refused" in run["target_json"]["errors"]
    assert "connection refused" in run["error"]
    assert "local telemetry failed" not in run["error"]


def test_target_limit_validation_only_checks_selected_http_scenarios(tmp_path: Path):
    http_config = tmp_path / "scenarios.yaml"
    http_config.write_text(
        """
name: scenarios
targets:
  - name: stale
    protocol: openai_chat
    base_url: http://127.0.0.1:9999/v1
    model: stale
scenarios:
  - name: selected-short
    prompt: hello
    max_tokens: [8]
    requests: 1
    concurrency: [1]
  - name: unselected-long
    prompt: hello
    max_tokens: [65536]
    requests: 1
    concurrency: [1]
""",
        encoding="utf-8",
    )
    spec, inspection = _resolved_dgx_target(max_model_len=32768)
    assert inspection.resolved is not None
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "selected-scenario",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {
                "enabled": True,
                "config": str(http_config),
                "scenarios": ["selected-short"],
            },
            "preflight": {"enabled": False},
        }
    )

    workflow = BenchmarkSuiteWorkflow(config, target_resolver=_FakeTargetResolver(inspection))

    workflow._validate_resolved_target(inspection.resolved)
