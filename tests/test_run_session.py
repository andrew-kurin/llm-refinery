from pathlib import Path

import pytest

from llm_refinery.application.run_context import RunContext
from llm_refinery.application.run_session import RunSession
from llm_refinery.core.runs import RunSpec
from llm_refinery.storage.duckdb import ResultStore


def test_run_session_records_identity_metrics_and_typed_artifacts(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="http_load",
        suite="load-suite",
        label="load-suite/local/short",
        command="http-load target=local scenario=short",
        config_json={"params": {"protocol": "openai_chat", "concurrency": 1}},
        database=database,
    )

    with ResultStore(database) as store:
        with RunSession(store, spec, system_profile={"hardware": {"model": "test"}}) as run:
            responses = run.artifact("responses", "responses.jsonl", "application/x-ndjson")
            responses.write_text('{"ok": true}\n', encoding="utf-8")
            completed = run.complete(metrics={"success_count": 1.0})

        rows = store.comparison_runs()
        stored_path = store.connection.execute(
            "SELECT path FROM artifacts WHERE run_id = ?",
            [completed.run_id],
        ).fetchone()

    assert stored_path == (f"artifacts/{completed.run_id}/responses.jsonl",)
    assert completed.status == "ok"
    assert len(rows) == 1
    assert rows[0]["benchmark_kind"] == "http_load"
    assert rows[0]["spec_hash"] == spec.spec_hash
    assert rows[0]["trial_name"].endswith(spec.spec_hash)
    assert rows[0]["metrics"] == {"success_count": 1.0}
    assert rows[0]["target_json"] == {}
    assert rows[0]["artifacts"]["responses"]["media_type"] == "application/x-ndjson"
    assert Path(rows[0]["artifacts"]["responses"]["path"]).read_text() == '{"ok": true}\n'


def test_run_session_refuses_to_resume_with_a_different_spec(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    original = RunSpec.create(
        benchmark_kind="dabstep",
        suite="dabstep",
        label="dabstep/local",
        command="python baseline/run.py --max-steps 10",
        config_json={"max_steps": 10},
        database=database,
    )
    changed = RunSpec.create(
        benchmark_kind="dabstep",
        suite="dabstep",
        label="dabstep/local",
        command="python baseline/run.py --max-steps 20",
        config_json={"max_steps": 20},
        database=database,
    )

    with ResultStore(database) as store:
        with RunSession(store, original, system_profile={}) as run:
            run.complete(status="failed", error="interrupted")
        with (
            pytest.raises(RuntimeError, match="does not match"),
            RunSession(
                store,
                changed,
                resume_run_id=run.run_id,
            ),
        ):
            pass


def test_run_session_separates_executor_and_remote_target_metadata(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="http_load",
        suite="load-suite",
        label="load-suite/dgx",
        command="http-load target=dgx",
        config_json={},
        database=database,
    )
    context = RunContext(
        executor_system_json={"hostname": "mac", "host_fingerprint": "mac-id"},
        target_json={
            "host": {
                "transport": "ssh",
                "destination": "dgx",
                "profile": {"hostname": "spark", "host_fingerprint": "spark-id"},
            },
            "topology": {"measurement_scope": "remote_lan_end_to_end"},
        },
    )

    with ResultStore(database) as store:
        with RunSession(store, spec, run_context=context) as run:
            run.complete()
        stored = store.comparison_runs()[0]

    assert stored["system_json"]["hostname"] == "mac"
    assert stored["target_json"]["host"]["profile"]["hostname"] == "spark"
    assert run.run_context.to_target_json() == stored["target_json"]


def test_run_session_can_bind_target_after_enter_and_propagate_it_on_resume(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="suite",
        suite="suite",
        label="suite/dgx",
        command="suite",
        config_json={},
        database=database,
    )
    target_json = {
        "host": {"destination": "dgx"},
        "service": {"health": "unavailable"},
    }

    with ResultStore(database) as store:
        with RunSession(store, spec, system_profile={"hostname": "mac"}) as run:
            run.set_target_json(target_json)
            running_target = store.connection.execute(
                "SELECT target_json FROM runs WHERE run_id = ?", [run.run_id]
            ).fetchone()
            assert running_target is not None and "unavailable" in running_target[0]
            run.complete(status="failed", error="service unavailable")

        with RunSession(
            store,
            spec,
            resume_run_id=run.run_id,
            system_profile={"hostname": "mac"},
        ) as resumed:
            assert resumed.target_json == target_json
            assert resumed.run_context.to_executor_system_json() == {"hostname": "mac"}
            resumed.complete()
            with pytest.raises(RuntimeError, match="already completed"):
                resumed.set_target_json({"host": {"destination": "another-host"}})

        stored = store.comparison_runs()[0]

    assert stored["target_json"] == target_json


def test_resume_captures_and_validates_current_executor_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="dabstep",
        suite="suite",
        label="suite/local",
        command="dabstep",
        config_json={},
        database=database,
    )

    with ResultStore(database) as store:
        with RunSession(
            store,
            spec,
            system_profile={"hostname": "mac-one", "host_fingerprint": "mac-one"},
        ) as run:
            run.complete(status="failed", error="retry")

        monkeypatch.setattr(
            "llm_refinery.application.run_session.get_system_profile",
            lambda: {"hostname": "mac-two", "host_fingerprint": "mac-two"},
        )
        with pytest.raises(RuntimeError, match="resume executor host"):
            RunSession(store, spec, resume_run_id=run.run_id)


def test_resume_preserves_stored_executor_observation_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="dabstep",
        suite="suite",
        label="suite/local",
        command="dabstep",
        config_json={},
        database=database,
    )
    stored_profile = {
        "hostname": "mac",
        "host_fingerprint": "mac-one",
        "captured_at": "initial",
    }

    with ResultStore(database) as store:
        with RunSession(store, spec, system_profile=stored_profile) as run:
            run.complete(status="failed", error="retry")

        monkeypatch.setattr(
            "llm_refinery.application.run_session.get_system_profile",
            lambda: {
                "hostname": "renamed-mac",
                "host_fingerprint": "mac-one",
                "captured_at": "resume",
            },
        )
        with RunSession(store, spec, resume_run_id=run.run_id) as resumed:
            assert resumed.system_profile == stored_profile
            resumed.complete()

        stored = store.comparison_runs()[0]

    assert stored["system_json"] == stored_profile


def test_resume_fails_closed_when_executor_capture_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="dabstep",
        suite="suite",
        label="suite/local",
        command="dabstep",
        config_json={},
        database=database,
    )

    with ResultStore(database) as store:
        with RunSession(
            store,
            spec,
            system_profile={"hostname": "mac", "host_fingerprint": "mac-one"},
        ) as run:
            run.complete(status="failed", error="retry")

        def fail_capture():
            raise OSError("inventory unavailable")

        monkeypatch.setattr("llm_refinery.application.run_session.get_system_profile", fail_capture)
        with pytest.raises(RuntimeError, match="cannot verify resume executor host"):
            RunSession(store, spec, resume_run_id=run.run_id)


def test_legacy_resume_requires_explicit_recovery_and_rebinds_executor(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="dabstep",
        suite="suite",
        label="suite/local",
        command="dabstep",
        config_json={},
        database=database,
    )
    recovered_profile = {
        "hostname": "mac",
        "host_fingerprint": "mac-one",
        "captured_at": "recovery",
    }

    with ResultStore(database) as store:
        with RunSession(
            store,
            spec,
            system_profile={"capture_error": "legacy inventory failure"},
        ) as run:
            run.complete(status="failed", error="retry")

        with pytest.raises(RuntimeError, match="stored run has no host identity"):
            RunSession(store, spec, resume_run_id=run.run_id, system_profile=recovered_profile)

        with RunSession(
            store,
            spec,
            resume_run_id=run.run_id,
            system_profile=recovered_profile,
            allow_unverified_executor=True,
        ) as resumed:
            assert resumed.system_profile["host_fingerprint"] == "mac-one"
            recovery = resumed.system_profile["_llm_refinery_provenance"][
                "unverified_executor_recovery"
            ]
            assert recovery["status"] == "unverified"
            assert recovery["reason"] == "stored_executor_identity_unknown"
            assert recovery["recorded_at"]
            assert recovery["original_system_json"] == {"capture_error": "legacy inventory failure"}
            resumed.complete(status="failed", error="retry again")

        stored = store.comparison_runs(include_failed=True)[0]
        assert stored["system_json"]["_llm_refinery_provenance"]["unverified_executor_recovery"][
            "original_system_json"
        ] == {"capture_error": "legacy inventory failure"}

        with pytest.raises(RuntimeError, match="resume executor host"):
            RunSession(
                store,
                spec,
                resume_run_id=run.run_id,
                system_profile={
                    "hostname": "other-mac",
                    "host_fingerprint": "mac-two",
                },
                allow_unverified_executor=True,
            )


def test_unverified_resume_cannot_bind_an_unknown_current_executor(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="dabstep",
        suite="suite",
        label="suite/local",
        command="dabstep",
        config_json={},
        database=database,
    )

    with ResultStore(database) as store:
        with RunSession(store, spec, system_profile={}) as run:
            run.complete(status="failed", error="retry")

        with pytest.raises(RuntimeError, match="current system profile also has no"):
            RunSession(
                store,
                spec,
                resume_run_id=run.run_id,
                system_profile={},
                allow_unverified_executor=True,
            )


def test_run_session_rejects_ambiguous_target_inputs(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="suite",
        suite="suite",
        label="suite/dgx",
        command="suite",
        config_json={},
        database=database,
    )

    with ResultStore(database) as store, pytest.raises(ValueError, match="either run_context"):
        RunSession(store, spec, run_context=RunContext(), target_json={})


def test_run_context_target_identity_excludes_volatile_observations():
    base = {
        "schema_version": 1,
        "name": "spark",
        "host": {
            "destination": "dgx",
            "profile": {
                "hostname": "spark",
                "host_fingerprint": "host-spark",
                "captured_at": "first",
            },
        },
        "service": {
            "implementation": "vllm",
            "base_url": "http://spark.local:8000/v1",
            "version": "0.10.0",
            "health": "ok",
            "server_info": {"dtype": "bfloat16"},
        },
        "route": {
            "logical_origin": {
                "scheme": "http",
                "hostname": "spark.local",
                "port": 8000,
            },
            "selected_address": "192.168.1.41",
            "authority": "spark.local:8000",
        },
        "model": {"id": "served", "root": "org/model"},
        "topology": {"measurement_scope": "remote_lan_end_to_end"},
        "errors": [],
    }
    changed_observation = {
        **base,
        "host": {
            **base["host"],
            "destination": "another-ssh-alias",
            "profile": {**base["host"]["profile"], "captured_at": "later"},
        },
        "service": {**base["service"], "health": "busy"},
        "errors": ["temporary"],
    }

    assert (
        RunContext(target_json=base).target_identity_json()
        == RunContext(target_json=changed_observation).target_identity_json()
    )

    changed_route = {
        **base,
        "route": {**base["route"], "selected_address": "192.168.1.42"},
    }
    assert (
        RunContext(target_json=base).target_identity_json()
        != RunContext(target_json=changed_route).target_identity_json()
    )


def test_resume_rejects_changed_executor_or_target_provenance(tmp_path: Path):
    database = tmp_path / "runs.duckdb"
    spec = RunSpec.create(
        benchmark_kind="dabstep",
        suite="suite",
        label="suite/dgx",
        command="dabstep",
        config_json={},
        database=database,
    )
    original = RunContext(
        executor_system_json={"hostname": "mac-one", "host_fingerprint": "mac-one"},
        target_json={
            "host": {
                "destination": "dgx",
                "profile": {"host_fingerprint": "spark-one"},
            },
            "route": {
                "logical_origin": {
                    "scheme": "http",
                    "hostname": "spark.local",
                    "port": 8000,
                },
                "selected_address": "192.168.1.41",
                "authority": "spark.local:8000",
            },
            "model": {"id": "model-one"},
        },
    )

    with ResultStore(database) as store:
        with RunSession(store, spec, run_context=original) as run:
            run.complete(status="failed", error="retry")

        with pytest.raises(RuntimeError, match="executor host"):
            RunSession(
                store,
                spec,
                resume_run_id=run.run_id,
                run_context=RunContext(
                    executor_system_json={
                        "hostname": "mac-two",
                        "host_fingerprint": "mac-two",
                    },
                    target_json=original.target_json,
                ),
            )

        with pytest.raises(RuntimeError, match="target identity"):
            RunSession(
                store,
                spec,
                resume_run_id=run.run_id,
                run_context=RunContext(
                    executor_system_json=original.executor_system_json,
                    target_json={
                        "host": {
                            "destination": "dgx",
                            "profile": {"host_fingerprint": "spark-one"},
                        },
                        "model": {"id": "model-two"},
                    },
                ),
            )

        changed_route = original.to_target_json()
        changed_route["route"]["selected_address"] = "192.168.1.42"
        with pytest.raises(RuntimeError, match="target identity"):
            RunSession(
                store,
                spec,
                resume_run_id=run.run_id,
                run_context=RunContext(
                    executor_system_json=original.executor_system_json,
                    target_json=changed_route,
                ),
            )
