import pytest

from llm_refinery.benchmarks.http_load.config import (
    HttpLoadConfig,
    HttpScenario,
    expand_http_load_trials,
)
from llm_refinery.benchmarks.http_load.metrics import summarize_request_results
from llm_refinery.benchmarks.http_load.models import RequestResult
from llm_refinery.benchmarks.http_load.runner import HttpLoadFailed, run_http_load
from llm_refinery.benchmarks.http_load.transport import (
    messages_for_scenario,
    read_ollama_stream,
    read_openai_stream,
    with_check_result,
)
from llm_refinery.core.config import ConfigError
from llm_refinery.storage.duckdb import ResultStore


def test_expand_http_load_trials_crosses_targets_scenarios_concurrency_and_tokens():
    config = HttpLoadConfig.from_mapping(
        {
            "name": "suite",
            "targets": [
                {
                    "name": "llama",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local",
                },
                {
                    "name": "ollama",
                    "protocol": "ollama_chat",
                    "base_url": "http://127.0.0.1:11434",
                    "model": "gemma",
                },
            ],
            "scenarios": [
                {
                    "name": "chat",
                    "prompt": "hello",
                    "max_tokens": [32, 64],
                    "concurrency": [1, 2],
                    "requests": 3,
                }
            ],
        }
    )

    trials = expand_http_load_trials(config)

    assert len(trials) == 8
    assert {trial.target.name for trial in trials} == {"llama", "ollama"}
    assert {trial.concurrency for trial in trials} == {1, 2}
    assert {trial.max_tokens for trial in trials} == {32, 64}
    assert all("params" in trial.as_jsonable() for trial in trials)


def test_http_scenario_supports_prompt_pools_and_explicit_unique_cache_mode():
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local",
                }
            ],
            "scenarios": [
                {
                    "name": "unique",
                    "prompts": ["alpha", "beta"],
                    "cache_mode": "unique",
                    "requests": 2,
                    "concurrency": 2,
                }
            ],
        }
    )

    scenario = config.scenarios[0]
    first = messages_for_scenario(scenario, index=0, request_nonce="run-a")[-1]["content"]
    second = messages_for_scenario(scenario, index=1, request_nonce="run-a")[-1]["content"]
    repeated_run = messages_for_scenario(scenario, index=0, request_nonce="run-b")[-1]["content"]

    assert scenario.prompts == ("alpha", "beta")
    assert first.startswith("[llm-refinery cache-bust run-a:0]")
    assert first.endswith("alpha")
    assert second.endswith("beta")
    assert len({first, second, repeated_run}) == 3
    assert scenario.safe_json()["cache_mode"] == "unique"
    assert scenario.safe_json()["prompt_pool_size"] == 2
    assert expand_http_load_trials(config)[0].effective_warmup_requests == 2


def test_http_scenario_rejects_concurrency_greater_than_measured_requests():
    with pytest.raises(ConfigError, match="concurrency cannot exceed requests"):
        HttpLoadConfig.from_mapping(
            {
                "targets": [
                    {
                        "name": "local",
                        "protocol": "openai_chat",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "model": "local",
                    }
                ],
                "scenarios": [
                    {
                        "name": "invalid",
                        "prompt": "hello",
                        "requests": 1,
                        "concurrency": 2,
                    }
                ],
            }
        )


def test_http_load_runner_records_samples_and_typed_artifacts(tmp_path, monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "name": "suite",
            "database": str(tmp_path / "runs.duckdb"),
            "targets": [
                {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local",
                }
            ],
            "scenarios": [
                {
                    "name": "chat",
                    "prompt": "hello",
                    "max_tokens": 8,
                    "concurrency": 1,
                    "requests": 1,
                }
            ],
        }
    )
    fake_result = RequestResult(
        index=0,
        ok=True,
        status_code=200,
        latency_s=1.0,
        completion_tokens=8,
        completion_chars=16,
        response_text="non-empty",
        visible_response_text="non-empty",
    )
    calls = []

    def fake_run_requests(_trial, *, count, index_offset=0):
        calls.append((count, index_offset))
        return [
            RequestResult(
                **{
                    **fake_result.__dict__,
                    "index": request_index,
                }
            )
            for request_index in range(index_offset, index_offset + count)
        ]

    monkeypatch.setattr(
        "llm_refinery.benchmarks.http_load.runner.run_requests",
        fake_run_requests,
    )

    outcomes = run_http_load(config)

    assert len(outcomes) == 1
    with ResultStore(config.database) as store:
        run = store.comparison_runs()[0]
        samples = store.samples_for_run(run["run_id"])
    assert set(run["artifacts"]) == {"errors", "measurement", "responses"}
    assert calls == [(1, 0), (1, 0)]
    assert run["metrics"]["effective_warmup_requests"] == 1
    assert run["metrics"]["measured_request_count_recommendation_met"] == 0
    assert len(samples) == 1
    assert samples[0]["metrics"] == {"latency_s": 1.0}


def test_summarize_request_results_calculates_latency_and_throughput_metrics():
    results = [
        RequestResult(
            index=0,
            ok=True,
            status_code=200,
            latency_s=1.0,
            ttft_s=0.2,
            visible_ttft_s=0.3,
            reasoning_ttft_s=0.2,
            tpot_s=0.04,
            itl_s=(0.03, 0.05),
            prompt_tokens=10,
            completion_tokens=20,
            completion_chars=80,
            server_eval_duration_s=0.5,
        ),
        RequestResult(
            index=1,
            ok=True,
            status_code=200,
            latency_s=3.0,
            ttft_s=0.4,
            visible_ttft_s=0.6,
            reasoning_ttft_s=0.4,
            tpot_s=0.08,
            itl_s=(0.07,),
            prompt_tokens=10,
            completion_tokens=20,
            completion_chars=80,
            server_eval_duration_s=1.0,
        ),
        RequestResult(
            index=2,
            ok=False,
            status_code=500,
            latency_s=0.5,
            error="boom",
        ),
    ]

    metrics = summarize_request_results(
        results,
        wall_duration_s=4.0,
        concurrency=2,
        max_tokens=64,
    )

    assert metrics["request_count"] == 3
    assert metrics["success_count"] == 2
    assert metrics["error_count"] == 1
    assert metrics["requests_per_second"] == 0.5
    assert metrics["latency_p50_s"] == 2.0
    assert metrics["observed_latency_p95_s"] == 2.8
    assert metrics["failed_latency_p95_s"] == 0.5
    assert metrics["ttft_p95_s"] == 0.39
    assert metrics["visible_ttft_p95_s"] == 0.585
    assert metrics["reasoning_ttft_p95_s"] == 0.39
    assert metrics["tpot_p95_s"] == 0.078
    assert metrics["itl_observation_count"] == 3
    assert metrics["completion_tokens_total"] == 40
    assert metrics["completion_tokens_per_second"] == 10
    assert metrics["server_eval_tps_avg"] == 30
    assert metrics["measured_request_count_recommended_min"] == 100
    assert metrics["measured_request_count_recommendation_met"] == 0


def test_read_openai_stream_extracts_content_text_and_reasoning_content():
    response = iter(
        [
            b'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}\n',
            b'data: {"choices":[{"delta":{"thinking":"more "}}]}\n',
            b'data: {"choices":[{"text":"hello "}]}\n',
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n',
            b'data: {"usage":{"prompt_tokens":3,"completion_tokens":4},"choices":[]}\n',
            b"data: [DONE]\n",
        ]
    )

    result = read_openai_stream(0, response, 0.0, 200)

    assert result.response_text == "think more hello world"
    assert result.reasoning_response_text == "think more "
    assert result.visible_response_text == "hello world"
    assert result.completion_chars == len("think more hello world")
    assert result.reasoning_completion_chars == len("think more ")
    assert result.visible_completion_chars == len("hello world")
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
    assert result.ttft_s is not None
    assert result.reasoning_ttft_s is not None
    assert result.visible_ttft_s is not None
    assert result.reasoning_ttft_s <= result.visible_ttft_s
    assert result.tpot_s is not None
    assert len(result.itl_s) == 3


def test_read_ollama_stream_extracts_thinking_content():
    response = iter(
        [
            b'{"message":{"content":"","thinking":"think "},"done":false}\n',
            b'{"message":{"content":"answer"},"done":false}\n',
            b'{"done":true,"prompt_eval_count":3,"eval_count":4,"eval_duration":1000000000}\n',
        ]
    )

    result = read_ollama_stream(0, response, 0.0, 200)

    assert result.response_text == "think answer"
    assert result.reasoning_response_text == "think "
    assert result.visible_response_text == "answer"
    assert result.completion_chars == len("think answer")
    assert result.completion_tokens == 4
    assert result.server_eval_duration_s == 1.0
    assert result.ttft_s is not None
    assert result.reasoning_ttft_s is not None
    assert result.visible_ttft_s is not None
    assert result.tpot_s is not None


def test_correctness_checks_and_empty_visible_responses_are_request_failures():
    scenario = HttpScenario(
        name="checked",
        prompt="prompt",
        expected_contains=["required"],
    )

    missing = with_check_result(
        RequestResult(
            index=0,
            ok=True,
            status_code=200,
            latency_s=1.0,
            response_text="wrong",
            visible_response_text="wrong",
        ),
        scenario,
    )
    reasoning_only = with_check_result(
        RequestResult(
            index=1,
            ok=True,
            status_code=200,
            latency_s=1.0,
            response_text="required in hidden reasoning",
            visible_response_text="",
            reasoning_response_text="required in hidden reasoning",
        ),
        scenario,
    )

    assert missing.ok is False
    assert missing.check_passed is False
    assert "correctness check failed" in (missing.error or "")
    assert reasoning_only.ok is False
    assert reasoning_only.check_passed is False
    assert reasoning_only.error == "empty visible response"


def test_failed_correctness_result_fails_the_http_trial(tmp_path, monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "name": "suite",
            "database": str(tmp_path / "runs.duckdb"),
            "targets": [
                {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local",
                }
            ],
            "scenarios": [
                {
                    "name": "chat",
                    "prompt": "hello",
                    "requests": 1,
                    "expected_contains": "required",
                }
            ],
        }
    )
    good = RequestResult(
        index=-1,
        ok=True,
        status_code=200,
        latency_s=0.1,
        response_text="required",
        visible_response_text="required",
        check_passed=True,
    )
    failed = RequestResult(
        index=0,
        ok=False,
        status_code=200,
        latency_s=0.2,
        response_text="wrong",
        visible_response_text="wrong",
        check_passed=False,
        error="correctness check failed; missing: required",
    )

    calls = 0

    def fake_run_requests(_trial, *, count, index_offset=0):
        nonlocal calls
        calls += 1
        return [good] if calls == 1 else [failed]

    monkeypatch.setattr(
        "llm_refinery.benchmarks.http_load.runner.run_requests",
        fake_run_requests,
    )

    with pytest.raises(HttpLoadFailed, match="correctness check failed"):
        run_http_load(config)

    with ResultStore(config.database) as store:
        run = store.comparison_runs(include_failed=True)[0]
    assert run["status"] == "failed"
    assert run["metrics"]["error_count"] == 1
    assert run["metrics"]["check_pass_rate"] == 0
    assert run["metrics"]["observed_latency_p95_s"] == 0.2
