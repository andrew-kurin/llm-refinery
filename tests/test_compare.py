from llama_tune.compare import build_compare_rows


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
