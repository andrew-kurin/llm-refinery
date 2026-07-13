from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from llm_refinery.application.run_session import RunSession
from llm_refinery.core.config import ConfigError
from llm_refinery.core.http_safety import PinnedHttpRoute
from llm_refinery.core.runs import RunSpec
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


@pytest.mark.parametrize("value", [True, 1.9, "8081", None, {}])
def test_suite_config_rejects_non_integer_forbidden_ports(value: object):
    raw = {
        "endpoint": {
            "name": "local",
            "protocol": "openai_chat",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "model",
        },
        "preflight": {"forbidden_ports": [value]},
    }

    with pytest.raises(ConfigError, match="forbidden_ports entries must be"):
        SuiteConfig.from_mapping(raw)


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
            "quality": {
                "tasks": "ifeval",
                "limit": "all",
                "apply_chat_template": False,
            },
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
    assert calls[0][1].apply_chat_template is False
    assert calls[0][2]["parent_run_id"] == result.run.run_id
    assert calls[1][2]["parent_run_id"] == result.run.run_id
    assert calls[1][2]["store"] is calls[0][2]["store"]

    with ResultStore(config.database) as store:
        runs = store.comparison_runs()
    assert len(runs) == 1
    assert runs[0]["benchmark_kind"] == "suite"
    assert set(runs[0]["artifacts"]) == {"system_before", "system_after", "preflight"}


def test_suite_http_comparison_keeps_runs_from_distinct_executors(tmp_path: Path):
    target_json = {
        "name": "spark",
        "host": {
            "profile": {"hostname": "spark", "host_fingerprint": "host-spark"},
        },
        "topology": {"measurement_scope": "remote_lan_end_to_end"},
    }
    shared = {
        "benchmark_kind": "http-load",
        "suite": "http-suite",
        "trial_name": "http-suite/spark/short/c1/t8/r1",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "shared-spec",
        "config_json": {
            "model": "served-model",
            "params": {
                "target": "spark",
                "protocol": "openai_chat",
                "scenario": "short",
                "concurrency": 1,
            },
        },
        "target_json": target_json,
        "metrics": {"observed_latency_p95_s": 1.0},
    }

    class ComparisonStore:
        def comparison_runs(self, *, include_failed=False, latest_per_trial=True):
            assert include_failed is False
            assert latest_per_trial is False
            return [
                {
                    **shared,
                    "run_id": "mac-a-run",
                    "system_json": {
                        "hostname": "mac-a\x1b]2;forged-title\x07\u2028forged-line",
                        "host_fingerprint": "host-mac-a",
                    },
                },
                {
                    **shared,
                    "run_id": "mac-b-run",
                    "system_json": {
                        "hostname": "mac-b",
                        "host_fingerprint": "host-mac-b",
                    },
                },
            ]

    config = SuiteConfig.from_mapping(
        {
            "name": "comparison",
            "database": str(tmp_path / "runs.duckdb"),
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "local-model",
            },
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": False},
        }
    )
    console = Console(record=True, width=220)

    BenchmarkSuiteWorkflow(config, console=console)._print_http_comparison(
        ComparisonStore(),  # type: ignore[arg-type]
        "http-suite",
    )

    rendered = console.export_text()
    assert "mac-a" in rendered
    assert "mac-b" in rendered
    assert "\x1b" not in rendered
    assert "forged-title" not in rendered
    assert "\u2028" not in rendered


def test_suite_sanitizes_preflight_log_and_warning_output(tmp_path: Path):
    unsafe = (
        "value[bold red]markup[/]\x1b]2;forged-title\x07\x1b[31m-red"
        "\u2028forged-line\u2029forged-paragraph"
    )
    config = SuiteConfig.from_mapping(
        {
            "name": "safe-output",
            "database": str(tmp_path / "runs.duckdb"),
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "local-model",
            },
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"require_clean": False},
        }
    )
    stream = StringIO()
    console = Console(file=stream, color_system=None, width=220)
    workflow = BenchmarkSuiteWorkflow(
        config,
        console=console,
        port_listener=lambda port: port == 8080,
        sanity_checker=lambda _endpoint: {
            "success": True,
            "response_model": unsafe,
            "content_preview": unsafe,
        },
    )

    workflow._log(unsafe)
    workflow._warn_validation(unsafe)
    result = workflow.preflight(f"snapshot-before\n{unsafe}\nsnapshot-after")

    output = stream.getvalue()
    assert result["sanity"]["response_model"] == unsafe
    assert "\x1b" not in output
    assert "forged-title" not in output
    assert "\u2028" not in output
    assert "\u2029" not in output
    assert "[bold red]markup[/]" in output
    assert "-red forged-line forged-paragraph" in output
    assert "snapshot-before\n" in output
    assert "snapshot-after" in output


@pytest.mark.parametrize("failed_step", ["quality", "http_load"])
def test_suite_summarizes_persisted_child_before_reraising(
    tmp_path: Path,
    failed_step: str,
):
    http_config = _write_http_load_config(tmp_path)
    config = SuiteConfig.from_mapping(
        {
            "name": "failed-child-suite",
            "database": str(tmp_path / "runs.duckdb"),
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "local-model",
            },
            "quality": {"enabled": failed_step == "quality", "tasks": "ifeval"},
            "http_load": {
                "enabled": failed_step == "http_load",
                "config": str(http_config),
                "targets": ["local"],
            },
            "preflight": {"enabled": False},
        }
    )

    def failing_runner(_config, **kwargs):
        child_spec = RunSpec.create(
            benchmark_kind=failed_step,
            suite=config.name,
            label=f"{config.name}/failed-child",
            command="fake child",
            config_json={},
            database=config.database,
            parent_run_id=kwargs["parent_run_id"],
        )
        with RunSession(kwargs["store"], child_spec) as child:
            child.complete(status="failed", error="persisted child failure")
        raise RuntimeError("runner failed after persistence")

    with pytest.raises(RuntimeError, match="runner failed after persistence"):
        BenchmarkSuiteWorkflow(
            config,
            lm_eval_runner=failing_runner,
            http_load_runner=failing_runner,
            system_snapshot=lambda: "snapshot",
        ).execute()

    with ResultStore(config.database) as store:
        runs = store.comparison_runs(include_failed=True, latest_per_trial=False)
    parent = next(run for run in runs if run["benchmark_kind"] == "suite")
    child = next(run for run in runs if run["parent_run_id"] == parent["run_id"])
    assert parent["status"] == "failed"
    assert parent["metrics"] == {
        "child_run_count": 1.0,
        "failed_child_count": 1.0,
    }
    assert "runner failed after persistence" in parent["error"]
    assert child["status"] == "failed"
    assert child["error"] == "persisted child failure"


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


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("quality", "enabled"),
        ("quality", "offline"),
        ("quality", "trust_env"),
        ("http_load", "enabled"),
        ("preflight", "enabled"),
        ("preflight", "require_clean"),
        ("preflight", "sanity_check"),
    ],
)
def test_suite_config_rejects_non_boolean_flags(section: str, field: str):
    raw = {
        "endpoint": {
            "name": "local",
            "protocol": "openai_chat",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "model",
        },
        section: {field: "false"},
    }

    with pytest.raises(ConfigError, match="must be a boolean"):
        SuiteConfig.from_mapping(raw)


@pytest.mark.parametrize("value", [True, "2", 2.0, 2.9, None])
def test_suite_config_requires_an_integer_schema_version(value):
    with pytest.raises(ConfigError, match="integer 1 or 2"):
        SuiteConfig.from_mapping({"schema_version": value})


def test_suite_quality_resolves_ca_bundle_relative_to_manifest(tmp_path: Path):
    ca_bundle = tmp_path / "private-ca.pem"
    ca_bundle.write_text("test CA", encoding="utf-8")
    config = SuiteConfig.from_mapping(
        {
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "https://model.local/v1",
                "model": "model",
            },
            "quality": {"trust_env": True, "ca_bundle": ca_bundle.name},
        },
        source_path=tmp_path / "suite.yaml",
    )

    assert config.quality.trust_env is True
    assert config.quality.ca_bundle == ca_bundle.resolve()


def test_suite_quality_expands_home_in_ca_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = tmp_path / "home"
    ca_bundle = home / ".config" / "private-ca.pem"
    ca_bundle.parent.mkdir(parents=True)
    ca_bundle.write_text("test CA", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    config = SuiteConfig.from_mapping(
        {
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "https://model.local/v1",
                "model": "model",
            },
            "quality": {"ca_bundle": "~/.config/private-ca.pem"},
        },
        source_path=tmp_path / "manifests" / "suite.yaml",
    )

    assert config.quality.ca_bundle == ca_bundle.resolve()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit", True),
        ("limit", 1.9),
        ("limit", "2"),
        ("max_length", False),
        ("max_length", 8192.9),
        ("max_length", "8192"),
        ("num_fewshot", True),
        ("num_fewshot", 1.5),
        ("num_fewshot", "1"),
    ],
)
def test_suite_quality_rejects_non_integer_numeric_fields(field: str, value: object):
    raw = {
        "endpoint": {
            "name": "local",
            "protocol": "openai_chat",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "model",
        },
        "quality": {field: value},
    }

    with pytest.raises(ConfigError, match=rf"quality\.{field} must be"):
        SuiteConfig.from_mapping(raw)


def test_suite_quality_accepts_zero_fewshot():
    config = SuiteConfig.from_mapping(
        {
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "model",
            },
            "quality": {"num_fewshot": 0},
        }
    )

    assert config.quality.num_fewshot == 0


def test_suite_quality_accepts_tokenizer_with_completions_backend():
    config = SuiteConfig.from_mapping(
        {
            "endpoint": {
                "name": "local",
                "protocol": "openai_chat",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "model",
            },
            "quality": {
                "model_backend": "local-completions",
                "tokenizer": "org/model-tokenizer",
            },
        }
    )

    assert config.quality.model_backend == "local-completions"
    assert config.quality.tokenizer == "org/model-tokenizer"
    assert config.quality.safe_json()["model_backend"] == "local-completions"


def test_suite_quality_requires_chat_template_off_with_remote_vllm_tokenizer():
    endpoint = {
        "name": "local",
        "protocol": "openai_chat",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "model",
    }
    with pytest.raises(ConfigError, match="apply_chat_template must be false"):
        SuiteConfig.from_mapping(
            {
                "endpoint": endpoint,
                "quality": {"model_backend": "local-completions"},
            }
        )

    config = SuiteConfig.from_mapping(
        {
            "endpoint": endpoint,
            "quality": {
                "model_backend": "local-completions",
                "apply_chat_template": False,
            },
        }
    )

    assert config.quality.apply_chat_template is False
    assert config.quality.safe_json()["apply_chat_template"] is False


def test_suite_steps_fail_fast_for_missing_enabled_input_paths(tmp_path: Path):
    endpoint = {
        "name": "local",
        "protocol": "openai_chat",
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "model",
    }

    with pytest.raises(ConfigError, match="include_path is not a directory"):
        SuiteConfig.from_mapping(
            {
                "endpoint": endpoint,
                "quality": {"include_path": str(tmp_path / "missing-tasks")},
            }
        )

    with pytest.raises(ConfigError, match="http_load.config is not a file"):
        SuiteConfig.from_mapping(
            {
                "endpoint": endpoint,
                "quality": {"enabled": False},
                "http_load": {
                    "enabled": True,
                    "config": str(tmp_path / "missing-load.yaml"),
                },
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tasks", []),
        ("tasks", True),
        ("tasks", {}),
        ("tasks", " "),
        ("tasks", None),
        ("eos_string", []),
        ("eos_string", True),
        ("eos_string", {}),
        ("eos_string", " "),
        ("tokenizer", []),
        ("tokenizer", True),
        ("tokenizer", {}),
        ("tokenizer", " "),
        ("gen_kwargs", []),
        ("gen_kwargs", True),
        ("gen_kwargs", {}),
        ("gen_kwargs", " "),
        ("include_path", []),
        ("include_path", True),
        ("include_path", {}),
        ("include_path", " "),
        ("output_root", []),
        ("output_root", True),
        ("output_root", {}),
        ("output_root", " "),
        ("output_root", None),
        ("package_spec", []),
        ("package_spec", True),
        ("package_spec", {}),
        ("package_spec", " "),
        ("package_spec", None),
        ("extra_packages", "package==1"),
        ("extra_packages", {}),
        ("extra_packages", True),
        ("extra_packages", None),
        ("extra_packages", [""]),
        ("extra_packages", [True]),
        ("apply_chat_template", "false"),
        ("apply_chat_template", 0),
    ],
)
def test_suite_quality_rejects_invalid_string_and_list_fields(field: str, value: object):
    with pytest.raises(ConfigError, match=rf"quality\.{field}"):
        SuiteConfig.from_mapping(
            {
                "endpoint": {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "model",
                },
                "quality": {field: value},
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config", []),
        ("config", True),
        ("config", {}),
        ("config", " "),
        ("targets", "local"),
        ("targets", {}),
        ("targets", True),
        ("targets", None),
        ("targets", [""]),
        ("targets", [1]),
        ("scenarios", "short"),
        ("scenarios", {}),
        ("scenarios", True),
        ("scenarios", None),
        ("scenarios", [""]),
        ("scenarios", [1]),
    ],
)
def test_suite_http_load_rejects_invalid_path_and_list_fields(field: str, value: object):
    with pytest.raises(ConfigError, match=rf"http_load\.{field}"):
        SuiteConfig.from_mapping(
            {
                "endpoint": {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "model",
                },
                "http_load": {"enabled": False, field: value},
            }
        )


@pytest.mark.parametrize("value", [False, True, 0, [], {}, " "])
def test_suite_preflight_rejects_invalid_expected_response_model(value: object):
    with pytest.raises(ConfigError, match="preflight.expected_response_model"):
        SuiteConfig.from_mapping(
            {
                "endpoint": {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "model",
                },
                "preflight": {"expected_response_model": value},
            }
        )


@pytest.mark.parametrize(
    ("quality", "message"),
    [
        ({"model_backend": "unknown"}, "quality.model_backend must be one of"),
        ({"model_backend": False}, "quality.model_backend must be a non-empty string"),
        ({"model_backend": 0}, "quality.model_backend must be a non-empty string"),
        ({"model_backend": None}, "quality.model_backend must be a non-empty string"),
        ({"model_backend": ""}, "quality.model_backend must be a non-empty string"),
        (
            {"tokenizer": "org/model-tokenizer"},
            "quality.tokenizer requires model_backend: local-completions",
        ),
    ],
)
def test_suite_quality_rejects_incoherent_model_backend_configuration(
    quality: dict[str, object],
    message: str,
):
    with pytest.raises(ConfigError, match=message):
        SuiteConfig.from_mapping(
            {
                "endpoint": {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "model",
                },
                "quality": quality,
            }
        )


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
            "transport": {"trust_env": False},
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
    route = PinnedHttpRoute(
        origin=("http", "aitopatom-41de.local", 8000),
        connect_host="192.168.1.41",
        authority="aitopatom-41de.local:8000",
        sni_hostname="aitopatom-41de.local",
    )
    resolved = ResolvedTarget(
        spec_name=spec.name,
        endpoint=spec.endpoint.resolve(model.id),
        host=host,
        service=service,
        model=model,
        selection="single_discovered",
        route=route,
    )
    return spec, TargetInspection(
        spec=spec,
        host=host,
        service=service,
        resolved=resolved,
        route=route,
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
        "transport": {
            "trust_env": spec.transport.trust_env,
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
database: {tmp_path / "runs.duckdb"}
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


def test_suite_config_expands_home_in_reusable_target_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    home = tmp_path / "home"
    target_path = home / ".config" / "llm-refinery" / "spark.yaml"
    target_path.parent.mkdir(parents=True)
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
    base_url: http://spark.local:8000/v1
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))

    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "target": "~/.config/llm-refinery/spark.yaml",
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": False},
        },
        source_path=tmp_path / "manifests" / "suite.yaml",
    )

    assert config.target is not None
    assert config.target.host.destination == "dgx"


def test_suite_config_resolves_inline_target_ca_relative_to_manifest(tmp_path: Path):
    ca_bundle = tmp_path / "private-ca.pem"
    ca_bundle.write_text("test CA", encoding="utf-8")
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "target": {
                "name": "spark",
                "host": {"access": "ssh", "destination": "dgx"},
                "endpoint": {
                    "protocol": "openai_chat",
                    "base_url": "https://spark.local:8000/v1",
                },
                "transport": {"ca_bundle": ca_bundle.name},
            },
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": False},
        },
        source_path=tmp_path / "suite.yaml",
    )

    assert config.target is not None
    assert config.target.transport.ca_bundle == ca_bundle.resolve()


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
    assert calls[0][1].model_backend == "local-chat-completions"
    assert calls[0][1].tokenizer is None
    assert calls[0][1].trust_env is False
    assert calls[0][1].pinned_route == inspection.route
    assert calls[1][1].targets[0].base_url == "http://aitopatom-41de.local:8000/v1"
    assert calls[1][1].targets[0].model == "served-model"
    assert calls[1][1].transport.trust_env is False
    assert calls[1][1].transport.pinned_route == inspection.route
    assert calls[0][2]["run_context"].to_target_json()["host"]["destination"] == "dgx"
    assert calls[1][2]["run_context"].to_target_json()["model"]["id"] == "served-model"
    assert calls[1][2]["run_context"].to_target_json()["route"] == {
        "logical_origin": {
            "scheme": "http",
            "hostname": "aitopatom-41de.local",
            "port": 8000,
        },
        "selected_address": "192.168.1.41",
        "authority": "aitopatom-41de.local:8000",
    }
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
    assert (
        "exact context fit cannot be verified"
        in Path(run["artifacts"]["preflight"]["path"]).read_text()
    )


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

    with ResultStore(config.database) as store:
        run = store.comparison_runs(include_failed=True)[0]
    assert "preflight" not in run["artifacts"]
    assert not (config.database.parent / "artifacts" / run["run_id"] / "preflight.json").exists()


def test_remote_suite_preflight_inherits_target_transport(monkeypatch, tmp_path: Path):
    spec, inspection = _resolved_dgx_target()
    captured: dict[str, object] = {}

    def fake_sanity(endpoint, **kwargs):
        captured.update(kwargs)
        return {"success": True, "response_model": endpoint.model}

    monkeypatch.setattr("llm_refinery.workflows.suite.run_api_sanity_check", fake_sanity)
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "spark-transport",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": True, "require_clean": False},
        }
    )

    BenchmarkSuiteWorkflow(
        config,
        target_resolver=_FakeTargetResolver(inspection),
        system_snapshot=lambda: "snapshot",
    ).execute()

    assert captured == {
        "trust_env": False,
        "ca_bundle": None,
        "route": inspection.route,
    }


def test_remote_suite_require_clean_fails_instead_of_silently_skipping(tmp_path: Path):
    spec, inspection = _resolved_dgx_target()
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "spark-clean-check",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {
                "enabled": True,
                "require_clean": True,
                "sanity_check": False,
            },
        }
    )

    with pytest.raises(ConfigError, match="cannot verify ports on a remote endpoint"):
        BenchmarkSuiteWorkflow(
            config,
            target_resolver=_FakeTargetResolver(inspection),
            system_snapshot=lambda: "snapshot",
        ).execute()


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
    assert "preflight" not in run["artifacts"]
    assert resolver.snapshot_calls == 1


def test_remote_suite_does_not_retry_failed_initial_host_inventory(tmp_path: Path):
    spec, _inspection = _resolved_dgx_target()

    class InventoryFailureResolver:
        def __init__(self):
            self.snapshot_calls = 0

        def inspect(self, target_spec, *, allow_service_unavailable=False):
            assert target_spec == spec
            assert allow_service_unavailable is True
            raise RuntimeError("ssh inventory timed out")

        def snapshot_host(self, target_spec):
            assert target_spec == spec
            self.snapshot_calls += 1
            raise AssertionError("failed initial inventory must not be retried")

    resolver = InventoryFailureResolver()
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "inventory-timeout",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": False},
        }
    )

    with pytest.raises(RuntimeError, match="ssh inventory timed out"):
        BenchmarkSuiteWorkflow(
            config,
            target_resolver=resolver,
            system_snapshot=lambda: "snapshot",
        ).execute()

    with ResultStore(config.database) as store:
        run = store.comparison_runs(include_failed=True)[0]
    assert resolver.snapshot_calls == 0
    assert set(run["artifacts"]) == {
        "server_before",
        "system_after",
        "system_before",
        "target_discovery",
    }
    assert "server_after" not in run["artifacts"]


def test_remote_suite_does_not_register_empty_artifacts_before_discovery(tmp_path: Path):
    spec, inspection = _resolved_dgx_target()
    resolver = _FakeTargetResolver(inspection)
    snapshot_calls = 0

    def failing_first_snapshot():
        nonlocal snapshot_calls
        snapshot_calls += 1
        if snapshot_calls == 1:
            raise RuntimeError("local snapshot failed")
        return "after snapshot"

    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "snapshot-failure",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": False},
        }
    )

    with pytest.raises(RuntimeError, match="local snapshot failed"):
        BenchmarkSuiteWorkflow(
            config,
            target_resolver=resolver,
            system_snapshot=failing_first_snapshot,
        ).execute()

    with ResultStore(config.database) as store:
        run = store.comparison_runs(include_failed=True)[0]
    assert resolver.calls == 0
    assert resolver.snapshot_calls == 0
    assert set(run["artifacts"]) == {"system_after"}
    assert Path(run["artifacts"]["system_after"]["path"]).read_text() == "after snapshot"


def test_remote_suite_persists_discovery_exception_before_failing(tmp_path: Path):
    spec, inspection = _resolved_dgx_target()

    class FailingResolver(_FakeTargetResolver):
        def inspect(self, spec, *, allow_service_unavailable=False):
            error = ConfigError("inventory response was malformed")
            error.target_inspection = inspection
            raise error

    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "malformed-spark",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": False},
            "preflight": {"enabled": False},
        }
    )

    with pytest.raises(ConfigError, match="inventory response was malformed"):
        BenchmarkSuiteWorkflow(
            config,
            target_resolver=FailingResolver(inspection),
            system_snapshot=lambda: "snapshot",
        ).execute()

    with ResultStore(config.database) as store:
        run = store.comparison_runs(include_failed=True)[0]
    discovery_path = Path(run["artifacts"]["target_discovery"]["path"])
    server_before_path = Path(run["artifacts"]["server_before"]["path"])
    assert run["target_json"]["failure_stage"] == "target_discovery"
    assert run["target_json"]["requested_target"]["endpoint"]["header_names"] == []
    assert run["target_json"]["host"]["profile"]["hostname"] == "spark"
    assert run["target_json"]["service"]["version"] == "0.10.2"
    assert "inventory response was malformed" in run["target_json"]["errors"][0]
    assert "inventory response was malformed" in discovery_path.read_text()
    assert '"hostname": "spark"' in server_before_path.read_text()
    assert "preflight" not in run["artifacts"]


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
    system: system
    prompt_repeat: 3
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
    assert any(
        "rendered prompt/system of up to 25 characters" in warning
        for warning in workflow._validation_warnings
    )


def test_target_limit_reserves_context_for_the_http_prompt(tmp_path: Path):
    http_config = _write_http_load_config(tmp_path)
    spec, inspection = _resolved_dgx_target(max_model_len=8)
    assert inspection.resolved is not None
    config = SuiteConfig.from_mapping(
        {
            "schema_version": 2,
            "name": "prompt-budget",
            "database": str(tmp_path / "runs.duckdb"),
            "target": _target_config_mapping(spec),
            "quality": {"enabled": False},
            "http_load": {"enabled": True, "config": str(http_config)},
            "preflight": {"enabled": False},
        }
    )

    with pytest.raises(ConfigError, match="leaves no context for the non-empty prompt"):
        BenchmarkSuiteWorkflow(config)._validate_resolved_target(inspection.resolved)
