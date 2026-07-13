from llm_refinery.compare import build_compare_rows, build_compare_table_rows


def test_build_compare_rows_supports_lm_eval_root_model_and_params():
    rows = build_compare_rows(
        [
            {
                "run_id": "quality-12b",
                "trial_name": "quality/12b",
                "status": "ok",
                "duration_s": 10.0,
                "config_json": {"benchmark": "lm-eval", "model": "gemma-12b", "target": "ollama"},
                "metrics": {"gpqa_main_fixed_generative.flexible-extract.exact_match": 0.529},
                "system_json": {},
            }
        ],
        metrics=("gpqa_main_fixed_generative.flexible-extract.exact_match",),
        params=("target",),
    )

    assert rows[0]["model"] == "gemma-12b"
    assert rows[0]["target"] == "ollama"


def test_compare_deduplicates_reruns_by_spec_hash_not_display_columns():
    shared = {
        "trial_name": "suite/model",
        "status": "ok",
        "duration_s": 1.0,
        "config_json": {"model": "same-model", "params": {}},
        "metrics": {"score": 1.0},
    }
    rows = build_compare_rows(
        [
            {**shared, "run_id": "config-a", "spec_hash": "spec-a"},
            {**shared, "run_id": "config-b", "spec_hash": "spec-b"},
            {**shared, "run_id": "rerun-a", "spec_hash": "spec-a"},
        ],
        metrics=("score",),
        limit=10,
    )

    assert {row["run_id"] for row in rows} == {"config-a", "config-b"}


def test_compare_keeps_same_spec_from_distinct_hosts_and_deduplicates_host_reruns():
    shared = {
        "trial_name": "suite/model",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "shared-spec",
        "config_json": {"model": "same-model", "params": {}},
    }
    rows = build_compare_rows(
        [
            {
                **shared,
                "run_id": "mac-latest",
                "metrics": {"score": 3.0},
                "system_json": {"hostname": "mac", "host_fingerprint": "host-mac"},
            },
            {
                **shared,
                "run_id": "spark-latest",
                "metrics": {"score": 2.0},
                "system_json": {"hostname": "spark", "host_fingerprint": "host-spark"},
            },
            {
                **shared,
                "run_id": "mac-older",
                "metrics": {"score": 1.0},
                "system_json": {"hostname": "mac", "host_fingerprint": "host-mac"},
            },
        ],
        metrics=("score",),
        limit=10,
    )

    assert [row["run_id"] for row in rows] == ["mac-latest", "spark-latest"]
    assert [row["host"] for row in rows] == ["mac", "spark"]
    assert "host" in build_compare_table_rows(rows)[0]


def test_compare_derives_distinct_host_identities_for_legacy_profiles():
    shared = {
        "trial_name": "suite/model",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "shared-spec",
        "config_json": {"model": "same-model", "params": {}},
        "metrics": {"score": 1.0},
    }
    rows = build_compare_rows(
        [
            {
                **shared,
                "run_id": "mac",
                "system_json": {"hardware": {"model": "Mac17,6", "memory_gb": 128.0}},
            },
            {
                **shared,
                "run_id": "spark",
                "system_json": {"hardware": {"model": "DGX Spark", "memory_gb": 128.0}},
            },
        ],
        metrics=("score",),
        limit=10,
    )

    assert {row["run_id"] for row in rows} == {"mac", "spark"}


def test_compare_separates_executor_target_and_measurement_topology():
    shared = {
        "trial_name": "suite/model",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "shared-spec",
        "config_json": {"model": "served-model", "params": {}},
        "system_json": {"hostname": "mac", "host_fingerprint": "host-mac"},
        "target_json": {
            "host": {
                "destination": "dgx",
                "profile": {"hostname": "spark", "host_fingerprint": "host-spark"},
            },
            "topology": {"measurement_scope": "remote_lan_end_to_end"},
        },
    }
    rows = build_compare_rows(
        [
            {**shared, "run_id": "remote-latest", "metrics": {"score": 3.0}},
            {**shared, "run_id": "remote-older", "metrics": {"score": 2.0}},
            {
                **shared,
                "run_id": "loopback",
                "metrics": {"score": 1.0},
                "target_json": {
                    **shared["target_json"],
                    "topology": {"measurement_scope": "local_loopback"},
                },
            },
        ],
        metrics=("score",),
        limit=10,
    )

    assert [row["run_id"] for row in rows] == ["remote-latest", "loopback"]
    assert rows[0]["host"] == "spark"
    assert rows[0]["executor_host"] == "mac"
    assert rows[0]["target_host"] == "spark"
    assert rows[0]["topology"] == "remote_lan_end_to_end"
    header = build_compare_table_rows(rows)[0]
    assert all(column in header for column in ("host", "executor_host", "target_host", "topology"))


def test_compare_keeps_same_remote_run_from_distinct_executors():
    target = {
        "host": {"profile": {"hostname": "spark", "host_fingerprint": "host-spark"}},
        "topology": {"measurement_scope": "remote_lan_end_to_end"},
    }
    shared = {
        "trial_name": "suite/model",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "shared-spec",
        "config_json": {"model": "same-model", "params": {}},
        "metrics": {"score": 1.0},
        "target_json": target,
    }
    rows = build_compare_rows(
        [
            {
                **shared,
                "run_id": "mac-a",
                "system_json": {"hostname": "mac-a", "host_fingerprint": "mac-a"},
            },
            {
                **shared,
                "run_id": "mac-b",
                "system_json": {"hostname": "mac-b", "host_fingerprint": "mac-b"},
            },
        ],
        metrics=("score",),
        limit=10,
    )

    assert {row["run_id"] for row in rows} == {"mac-a", "mac-b"}


def test_compare_keeps_parent_runs_with_different_discovered_models():
    shared = {
        "trial_name": "suite",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "declarative-suite-spec",
        "config_json": {"params": {}},
        "system_json": {"host_fingerprint": "mac"},
        "metrics": {"score": 1.0},
    }

    def target(model_id: str):
        return {
            "name": "spark",
            "host": {"profile": {"host_fingerprint": "spark"}},
            "service": {
                "implementation": "vllm",
                "base_url": "http://spark.local:8000/v1",
                "version": "0.10.0",
            },
            "model": {"id": model_id, "requested_id": model_id},
            "topology": {"measurement_scope": "remote_client_to_server"},
        }

    rows = build_compare_rows(
        [
            {**shared, "run_id": "model-a", "target_json": target("model-a")},
            {**shared, "run_id": "model-b", "target_json": target("model-b")},
        ],
        metrics=("score",),
        limit=10,
    )

    assert {row["run_id"] for row in rows} == {"model-a", "model-b"}


def test_compare_keeps_same_target_routed_to_distinct_addresses():
    shared = {
        "trial_name": "suite",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "declarative-suite-spec",
        "config_json": {"params": {}},
        "system_json": {"host_fingerprint": "mac"},
        "metrics": {"score": 1.0},
    }

    def target(selected_address: str):
        return {
            "schema_version": 1,
            "name": "spark",
            "host": {"profile": {"hostname": "spark"}},
            "service": {
                "implementation": "vllm",
                "base_url": "http://spark.local:8000/v1",
                "version": "0.10.0",
            },
            "route": {
                "logical_origin": {
                    "scheme": "http",
                    "hostname": "spark.local",
                    "port": 8000,
                },
                "selected_address": selected_address,
                "authority": "spark.local:8000",
            },
            "model": {"id": "served"},
        }

    rows = build_compare_rows(
        [
            {**shared, "run_id": "route-a", "target_json": target("192.168.1.41")},
            {**shared, "run_id": "route-b", "target_json": target("192.168.1.42")},
        ],
        metrics=("score",),
        limit=10,
    )

    assert {row["run_id"] for row in rows} == {"route-a", "route-b"}


def test_compare_keeps_historical_inventory_fingerprints_distinct():
    shared = {
        "trial_name": "suite",
        "status": "ok",
        "duration_s": 1.0,
        "spec_hash": "declarative-suite-spec",
        "config_json": {"params": {}},
        "system_json": {"host_fingerprint": "mac"},
        "metrics": {"score": 1.0},
    }

    def target(host_fingerprint: str):
        return {
            "schema_version": 1,
            "name": "spark",
            "host": {"inventory": {"host_fingerprint": host_fingerprint}},
            "service": {"base_url": "http://spark.local:8000/v1"},
            "model": {"id": "served"},
        }

    rows = build_compare_rows(
        [
            {**shared, "run_id": "host-a", "target_json": target("host-a")},
            {**shared, "run_id": "host-b", "target_json": target("host-b")},
        ],
        metrics=("score",),
        limit=10,
    )

    assert {row["run_id"] for row in rows} == {"host-a", "host-b"}


def test_compare_supports_target_and_executor_dotted_params():
    rows = build_compare_rows(
        [
            {
                "run_id": "remote",
                "trial_name": "suite/model",
                "status": "ok",
                "duration_s": 1.0,
                "config_json": {"params": {}},
                "metrics": {"score": 1.0},
                "system_json": {"hardware": {"model": "Mac"}},
                "target_json": {"service": {"version": "0.10.2"}},
            }
        ],
        metrics=("score",),
        params=("executor.hardware.model", "target.service.version"),
    )

    assert rows[0]["executor.hardware.model"] == "Mac"
    assert rows[0]["target.service.version"] == "0.10.2"


def test_compare_sort_keeps_missing_metrics_last_when_ascending():
    runs = [
        {
            "run_id": "missing",
            "trial_name": "suite/missing",
            "status": "ok",
            "duration_s": 1.0,
            "config_json": {"params": {"scenario": "bench"}},
            "metrics": {},
        },
        {
            "run_id": "slow",
            "trial_name": "suite/slow",
            "status": "ok",
            "duration_s": 1.0,
            "config_json": {"params": {"scenario": "http"}},
            "metrics": {"latency_p95_s": 5.0},
        },
        {
            "run_id": "fast",
            "trial_name": "suite/fast",
            "status": "ok",
            "duration_s": 1.0,
            "config_json": {"params": {"scenario": "http"}},
            "metrics": {"latency_p95_s": 2.0},
        },
    ]

    rows = build_compare_rows(
        runs,
        metrics=("latency_p95_s",),
        sort_key="latency_p95_s",
        ascending=True,
        limit=3,
        dedupe_configs=False,
    )

    assert [row["run_id"] for row in rows] == ["fast", "slow", "missing"]
