from llm_refinery.http_load import (
    HttpLoadConfig,
    RequestResult,
    _read_ollama_stream,
    _read_openai_stream,
    expand_http_load_trials,
    summarize_request_results,
)


def test_expand_http_load_trials_crosses_targets_scenarios_concurrency_and_tokens():
    config = HttpLoadConfig.from_mapping(
        {
            "name": "suite",
            "targets": [
                {
                    "name": "llama",
                    "provider": "openai",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local",
                },
                {
                    "name": "ollama",
                    "provider": "ollama",
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


def test_summarize_request_results_calculates_latency_and_throughput_metrics():
    results = [
        RequestResult(
            index=0,
            ok=True,
            status_code=200,
            latency_s=1.0,
            ttft_s=0.2,
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
    assert metrics["ttft_p95_s"] == 0.39
    assert metrics["completion_tokens_total"] == 40
    assert metrics["completion_tokens_per_second"] == 10
    assert metrics["server_eval_tps_avg"] == 30


def test_read_openai_stream_extracts_content_text_and_reasoning_content():
    response = iter(
        [
            b'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}\n',
            b'data: {"choices":[{"delta":{"thinking":"more "}}]}\n',
            b'data: {"choices":[{"text":"hello "}]}\n',
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n',
            b'data: {"usage":{"prompt_tokens":3,"completion_tokens":4},"choices":[]}\n',
            b'data: [DONE]\n',
        ]
    )

    result = _read_openai_stream(0, response, 0.0, 200)

    assert result.response_text == "think more hello world"
    assert result.completion_chars == len("think more hello world")
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
    assert result.ttft_s is not None


def test_read_ollama_stream_extracts_thinking_content():
    response = iter(
        [
            b'{"message":{"content":"","thinking":"think "},"done":false}\n',
            b'{"message":{"content":"answer"},"done":false}\n',
            b'{"done":true,"prompt_eval_count":3,"eval_count":4,"eval_duration":1000000000}\n',
        ]
    )

    result = _read_ollama_stream(0, response, 0.0, 200)

    assert result.response_text == "think answer"
    assert result.completion_chars == len("think answer")
    assert result.completion_tokens == 4
    assert result.server_eval_duration_s == 1.0
    assert result.ttft_s is not None
