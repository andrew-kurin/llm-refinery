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
