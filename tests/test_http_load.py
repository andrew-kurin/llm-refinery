import json
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from llm_refinery.application.run_context import RunContext
from llm_refinery.benchmarks.http_load import transport as http_transport
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
    pooled_http_client,
    read_ollama_stream,
    read_openai_stream,
    run_requests,
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
    assert all(trial.transport.trust_env is True for trial in trials)


def test_http_transport_config_supports_direct_mode_and_relative_ca_bundle(tmp_path):
    ca_bundle = tmp_path / "private-ca.pem"
    ca_bundle.write_text("test certificate bundle", encoding="utf-8")
    config = HttpLoadConfig.from_mapping(
        {
            "transport": {"trust_env": False, "ca_bundle": ca_bundle.name},
            "targets": [
                {
                    "name": "remote",
                    "protocol": "openai_chat",
                    "base_url": "https://remote.test/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello"}],
        },
        source_path=tmp_path / "http-load.yaml",
    )

    trial = expand_http_load_trials(config)[0]

    assert trial.transport.trust_env is False
    assert trial.transport.ca_bundle == ca_bundle
    assert trial.as_jsonable()["transport"] == {
        "trust_env": False,
        "ca_bundle": str(ca_bundle),
    }


def test_http_transport_config_rejects_string_boolean():
    with pytest.raises(ConfigError, match="transport.trust_env must be a boolean"):
        HttpLoadConfig.from_mapping(
            {
                "transport": {"trust_env": "false"},
                "targets": [
                    {
                        "name": "local",
                        "protocol": "openai_chat",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "model": "local",
                    }
                ],
                "scenarios": [{"name": "chat", "prompt": "hello"}],
            }
        )


@pytest.mark.parametrize("value", [None, False, 0, ""])
def test_http_transport_config_rejects_invalid_ca_bundle(value):
    with pytest.raises(ConfigError, match="ca_bundle must be a non-empty path string"):
        HttpLoadConfig.from_mapping(
            {
                "transport": {"ca_bundle": value},
                "targets": [
                    {
                        "name": "local",
                        "protocol": "openai_chat",
                        "base_url": "http://127.0.0.1:8080/v1",
                        "model": "local",
                    }
                ],
                "scenarios": [{"name": "chat", "prompt": "hello"}],
            }
        )


def test_http_transport_config_expands_home_in_ca_bundle(tmp_path, monkeypatch):
    home = tmp_path / "home"
    ca_bundle = home / ".config" / "private-ca.pem"
    ca_bundle.parent.mkdir(parents=True)
    ca_bundle.write_text("test certificate bundle", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))

    config = HttpLoadConfig.from_mapping(
        {
            "transport": {"ca_bundle": "~/.config/private-ca.pem"},
            "targets": [
                {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8080/v1",
                    "model": "local",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello"}],
        },
        source_path=tmp_path / "manifests" / "http-load.yaml",
    )

    assert config.transport.ca_bundle == ca_bundle.resolve()


def test_http_client_applies_transport_proxy_and_ca_settings(tmp_path, monkeypatch):
    ca_bundle = tmp_path / "private-ca.pem"
    ca_bundle.write_text("test certificate bundle", encoding="utf-8")
    config = HttpLoadConfig.from_mapping(
        {
            "transport": {"trust_env": False, "ca_bundle": str(ca_bundle)},
            "targets": [
                {
                    "name": "remote",
                    "protocol": "openai_chat",
                    "base_url": "https://remote.test/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello"}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    ssl_context = object()
    captured = {}

    monkeypatch.setattr(
        http_transport.ssl,
        "create_default_context",
        lambda *, cafile: ssl_context if cafile == str(ca_bundle) else None,
    )
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(http_transport.httpx, "Client", FakeClient)

    http_transport._new_http_client(trial)

    assert captured["trust_env"] is False
    assert captured["verify"] is ssl_context


def test_http_client_forces_explicit_loopback_target_direct(monkeypatch):
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
            "scenarios": [{"name": "chat", "prompt": "hello"}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    captured = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(http_transport.httpx, "Client", FakeClient)

    http_transport._new_http_client(trial)

    assert captured["trust_env"] is False


def test_http_client_resolves_remote_origin_once_before_measured_requests(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "dgx",
                    "protocol": "openai_chat",
                    "base_url": "http://aitopatom-41de.local:8000/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [
                {
                    "name": "chat",
                    "prompt": "hello",
                    "stream": False,
                    "requests": 2,
                    "concurrency": 2,
                }
            ],
        }
    )
    trial = expand_http_load_trials(config)[0]
    resolutions = 0

    def resolve(host, port, **kwargs):
        nonlocal resolutions
        resolutions += 1
        return [(2, 1, 6, "", ("192.168.1.41", port))]

    monkeypatch.setattr("llm_refinery.core.http_safety.socket.getaddrinfo", resolve)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "192.168.1.41"
        assert request.headers["host"] == "aitopatom-41de.local:8000"
        assert request.extensions["sni_hostname"] == "aitopatom-41de.local"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"completion_tokens": 1},
            },
            request=request,
        )

    with pooled_http_client(trial) as client:
        for pooled_client in client._clients:  # type: ignore[attr-defined]
            pooled_client._transport = httpx.MockTransport(handler)  # type: ignore[attr-defined]
        assert all(result.ok for result in run_requests(trial, count=2, client=client))

    assert resolutions == 1


def test_http_client_uses_active_proxy_without_ip_pinning(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "remote",
                    "protocol": "openai_chat",
                    "base_url": "http://model.example:8000/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello"}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.getproxies",
        lambda: {"http": "http://proxy.example:3128"},
    )

    def unexpected_resolution(*args, **kwargs):
        raise AssertionError("proxy-routed origins must not be resolved or IP-pinned locally")

    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        unexpected_resolution,
    )

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(http_transport.httpx, "Client", FakeClient)

    client = http_transport._new_http_client(trial)

    assert captured["trust_env"] is True
    assert client._llm_refinery_routes == {  # type: ignore[attr-defined]
        ("http", "model.example", 8000): None
    }


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

    clients: list[httpx.Client] = []

    def fake_run_requests(_trial, *, count, index_offset=0, client=None):
        calls.append((count, index_offset))
        clients.append(client)
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

    context = RunContext(
        target_json={
            "host": {"profile": {"host_fingerprint": "spark"}},
            "service": {"implementation": "vllm", "version": "0.10.0"},
            "model": {"id": "local"},
            "topology": {"measurement_scope": "remote_client_to_server"},
        }
    )
    outcomes = run_http_load(config, run_context=context)

    assert len(outcomes) == 1
    with ResultStore(config.database) as store:
        run = store.comparison_runs()[0]
        samples = store.samples_for_run(run["run_id"])
    assert set(run["artifacts"]) == {"errors", "measurement", "responses"}
    assert calls == [(1, 0), (1, 0)]
    assert clients[0] is clients[1]
    assert clients[0].is_closed
    assert run["metrics"]["effective_warmup_requests"] == 1
    assert run["metrics"]["measured_request_count_recommendation_met"] == 0
    assert run["config_json"]["execution_target"]["model"]["id"] == "local"
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
            b'data: {"choices":[{"delta":{"reasoning":"current "}}]}\n',
            b'data: {"choices":[{"delta":{"reasoning_content":"think "}}]}\n',
            b'data: {"choices":[{"delta":{"thinking":"more "}}]}\n',
            b'data: {"choices":[{"text":"hello "}]}\n',
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n',
            b'data: {"usage":{"prompt_tokens":3,"completion_tokens":4},"choices":[]}\n',
            b"data: [DONE]\n",
        ]
    )

    result = read_openai_stream(0, response, 0.0, 200)

    assert result.response_text == "current think more hello world"
    assert result.reasoning_response_text == "current think more "
    assert result.visible_response_text == "hello world"
    assert result.completion_chars == len("current think more hello world")
    assert result.reasoning_completion_chars == len("current think more ")
    assert result.visible_completion_chars == len("hello world")
    assert result.prompt_tokens == 3
    assert result.completion_tokens == 4
    assert result.ttft_s is not None
    assert result.reasoning_ttft_s is not None
    assert result.visible_ttft_s is not None
    assert result.reasoning_ttft_s <= result.visible_ttft_s
    assert result.tpot_s is not None
    assert len(result.itl_s) == 4


def test_run_requests_shares_and_closes_one_connection_pool(monkeypatch):
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
                    "name": "pooled",
                    "prompt": "hello",
                    "requests": 4,
                    "concurrency": 2,
                }
            ],
        }
    )
    trial = expand_http_load_trials(config)[0]
    clients: list[httpx.Client] = []

    def fake_execute(_trial, index, *, request_nonce=None, client=None):
        clients.append(client)
        return RequestResult(
            index=index,
            ok=True,
            status_code=200,
            latency_s=0.1,
            response_text="ok",
            visible_response_text="ok",
        )

    monkeypatch.setattr(
        "llm_refinery.benchmarks.http_load.transport.execute_http_request",
        fake_execute,
    )

    results = run_requests(trial, count=4)

    assert [result.index for result in results] == [0, 1, 2, 3]
    assert len({id(client) for client in clients}) == trial.concurrency
    assert all(client.is_closed for client in clients)


def test_connection_pool_replaces_timed_out_clients_without_retaining_them():
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
                    "name": "pooled",
                    "prompt": "hello",
                    "requests": 2,
                    "concurrency": 2,
                }
            ],
        }
    )
    trial = expand_http_load_trials(config)[0]

    with pooled_http_client(trial) as pool:
        for _ in range(5):
            with pool.lease() as client:
                client.close()
            assert len(pool._clients) == trial.concurrency  # type: ignore[attr-defined]
            assert all(not client.is_closed for client in pool._clients)  # type: ignore[attr-defined]


@pytest.mark.parametrize("timeout_s", [float("inf"), float("-inf"), float("nan"), 1e100])
def test_http_scenario_rejects_unsupported_timeout(timeout_s):
    with pytest.raises(ConfigError, match="timeout_s must be positive and no greater"):
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
                        "name": "invalid-timeout",
                        "prompt": "hello",
                        "timeout_s": timeout_s,
                    }
                ],
            }
        )


def test_connection_pool_reuses_http11_connections_across_batches():
    connections: set[tuple[str, int]] = set()
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            nonlocal request_count
            request_count += 1
            connections.add(self.client_address)
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            body = json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *args):
            del args

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        config = HttpLoadConfig.from_mapping(
            {
                "targets": [
                    {
                        "name": "local",
                        "protocol": "openai_chat",
                        "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                        "model": "local",
                    }
                ],
                "scenarios": [
                    {
                        "name": "pooled",
                        "prompt": "hello",
                        "requests": 4,
                        "concurrency": 2,
                        "stream": False,
                    }
                ],
            }
        )
        trial = expand_http_load_trials(config)[0]

        with pooled_http_client(trial) as client:
            assert all(result.ok for result in run_requests(trial, count=2, client=client))
            assert all(result.ok for result in run_requests(trial, count=4, client=client))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert request_count == 6
    assert 1 <= len(connections) <= trial.concurrency


def test_http_client_pool_keeps_deadline_cancellation_request_local():
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": "http://127.0.0.1:8000/v1",
                    "model": "local",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello", "requests": 2, "concurrency": 2}],
        }
    )
    trial = expand_http_load_trials(config)[0]

    with pooled_http_client(trial) as pool:
        with pool.lease() as first, pool.lease() as second:
            assert first is not second
            first.close()
            assert second.is_closed is False

        assert len([client for client in pool._clients if not client.is_closed]) == 2


def test_deadline_watchdog_cancel_waits_for_dequeued_callback():
    watchdog = http_transport._DeadlineWatchdog()
    callback_started = threading.Event()
    release_callback = threading.Event()
    cancellation_finished = threading.Event()

    def callback() -> None:
        callback_started.set()
        assert release_callback.wait(timeout=2)

    token = watchdog.schedule(time.perf_counter() - 1, callback)
    assert callback_started.wait(timeout=1)

    def cancel() -> None:
        watchdog.cancel(token)
        cancellation_finished.set()

    thread = threading.Thread(target=cancel)
    thread.start()
    assert not cancellation_finished.wait(timeout=0.05)
    release_callback.set()
    thread.join(timeout=1)

    assert cancellation_finished.is_set()


def test_http_load_rejects_dns_name_resolving_to_client_loopback(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "remote-alias",
                    "protocol": "openai_chat",
                    "base_url": "http://local-alias.example:8000/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello"}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("127.0.0.1", port))],
    )

    with pytest.raises(ConfigError, match="client-local"):
        http_transport._new_http_client(trial)


def test_http_load_rejects_cross_origin_redirect_before_following(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "dgx",
                    "protocol": "openai_chat",
                    "base_url": "http://aitopatom-41de.local:8000/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello", "stream": False, "requests": 1}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            307,
            headers={"location": "http://127.0.0.1:9000/v1/chat/completions"},
            request=request,
        )

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ConfigError, match="remain on the configured.*origin"),
    ):
        http_transport.execute_http_request(trial, 0, client=client)

    assert len(requests) == 1


def test_http_load_redacts_credentials_echoed_by_error_response(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "authenticated",
                    "protocol": "openai_chat",
                    "base_url": "http://remote.test:8000/v1",
                    "model": "served-model",
                    "headers": {"Authorization": "Bearer top-secret-token"},
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello", "stream": False, "requests": 1}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            text="gateway echoed Bearer top-secret-token",
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = http_transport.execute_http_request(trial, 0, client=client)

    assert result.ok is False
    assert "top-secret-token" not in (result.error or "")
    assert "[REDACTED]" in (result.error or "")


def test_http_load_bounds_nonstream_response_bodies(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "remote",
                    "protocol": "openai_chat",
                    "base_url": "http://remote.test:8000/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello", "stream": False, "requests": 1}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    monkeypatch.setattr(http_transport, "_MAX_SUCCESS_RESPONSE_BYTES", 10)
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * 11, request=request)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_requests(trial, count=1, client=client)[0]

    assert result.ok is False
    assert "response exceeded 10 bytes" in (result.error or "")


def test_http_load_bounds_unterminated_stream_without_line_buffering(monkeypatch):
    response = httpx.Response(
        200,
        content=b"x" * 11,
        request=httpx.Request("POST", "http://remote.test/v1/chat/completions"),
    )

    with pytest.raises(ValueError, match="response exceeded 10 bytes"):
        list(
            http_transport._iter_bounded_lines(
                response,
                deadline=time.perf_counter() + 1,
                max_bytes=10,
            )
        )


def test_http_load_worker_fallback_redacts_server_echoed_token(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "authenticated",
                    "protocol": "ollama_chat",
                    "base_url": "http://remote.test:8000",
                    "model": "served-model",
                    "api_key_env": "HTTP_LOAD_SECRET",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello", "requests": 1}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    monkeypatch.setenv("HTTP_LOAD_SECRET", "top-secret-token")
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'{"error":"top-secret-token"}\n',
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = run_requests(trial, count=1, client=client)[0]

    assert result.ok is False
    assert "top-secret-token" not in (result.error or "")
    assert "[REDACTED]" in (result.error or "")


def test_http_load_missing_api_key_fails_before_workers_or_network(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "authenticated",
                    "protocol": "openai_chat",
                    "base_url": "http://remote.test:8000/v1",
                    "model": "served-model",
                    "api_key_env": "MISSING_HTTP_LOAD_KEY",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello", "requests": 4}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    monkeypatch.delenv("MISSING_HTTP_LOAD_KEY", raising=False)
    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={}, request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(ConfigError, match="MISSING_HTTP_LOAD_KEY"),
    ):
        run_requests(trial, count=4, client=client)

    assert request_count == 0


def test_http_load_absolute_deadline_interrupts_blocking_stream(monkeypatch):
    class BlockingStream(httpx.SyncByteStream):
        def __init__(self) -> None:
            self.closed = threading.Event()

        def __iter__(self):
            yield b'data: {"choices": []}\n\n'
            self.closed.wait(timeout=2)

        def close(self) -> None:
            self.closed.set()

    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "remote",
                    "protocol": "openai_chat",
                    "base_url": "http://remote.test:8000/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [
                {
                    "name": "chat",
                    "prompt": "hello",
                    "stream": True,
                    "requests": 1,
                    "timeout_s": 0.05,
                }
            ],
        }
    )
    trial = expand_http_load_trials(config)[0]
    monkeypatch.setattr(
        "llm_refinery.core.http_safety.socket.getaddrinfo",
        lambda host, port, **kwargs: [(2, 1, 6, "", ("192.168.1.41", port))],
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=BlockingStream(), request=request)

    started = time.perf_counter()
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = http_transport.execute_http_request(trial, 0, client=client)

    assert time.perf_counter() - started < 1
    assert result.ok is False
    assert "total timeout" in (result.error or "")


def test_http_load_absolute_deadline_interrupts_trickled_response_headers():
    class TrickleHandler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.recv(65536)
            self.request.sendall(b"HTTP/1.1 200 OK\r\nX-Trickle: ")
            for _ in range(100):
                time.sleep(0.02)
                try:
                    self.request.sendall(b"x")
                except OSError:
                    return

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), TrickleHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "local",
                    "protocol": "openai_chat",
                    "base_url": f"http://127.0.0.1:{server.server_address[1]}/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [
                {
                    "name": "chat",
                    "prompt": "hello",
                    "stream": False,
                    "requests": 1,
                    "timeout_s": 0.08,
                }
            ],
        }
    )
    trial = expand_http_load_trials(config)[0]
    started = time.perf_counter()
    try:
        with httpx.Client() as client:
            result = http_transport.execute_http_request(trial, 0, client=client)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert time.perf_counter() - started < 1
    assert result.ok is False
    assert "total timeout" in (result.error or "")


def test_http_load_stream_iteration_enforces_total_deadline(monkeypatch):
    response = httpx.Response(
        200,
        content=b'data: {"choices": []}\n\ndata: [DONE]\n\n',
        request=httpx.Request("POST", "http://remote.test/v1/chat/completions"),
    )
    observed_times = iter((1.0, 1.0, 3.0))
    monkeypatch.setattr(
        http_transport.time,
        "perf_counter",
        lambda: next(observed_times),
    )

    lines = http_transport._iter_bounded_lines(
        response,
        deadline=2.0,
        max_bytes=1_000,
    )

    assert next(lines).startswith("data:")
    with pytest.raises(TimeoutError, match="total timeout"):
        next(lines)


def test_http_load_follows_same_origin_method_preserving_redirect(monkeypatch):
    config = HttpLoadConfig.from_mapping(
        {
            "targets": [
                {
                    "name": "dgx",
                    "protocol": "openai_chat",
                    "base_url": "http://aitopatom-41de.local:8000/v1",
                    "model": "served-model",
                }
            ],
            "scenarios": [{"name": "chat", "prompt": "hello", "stream": False, "requests": 1}],
        }
    )
    trial = expand_http_load_trials(config)[0]
    resolutions = 0

    def resolve(host, port, **kwargs):
        nonlocal resolutions
        resolutions += 1
        return [(2, 1, 6, "", ("192.168.1.41", port))]

    monkeypatch.setattr("llm_refinery.core.http_safety.socket.getaddrinfo", resolve)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(
                307,
                headers={"location": "/v1/redirected-chat"},
                request=request,
            )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"completion_tokens": 1},
            },
            request=request,
        )

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        result = http_transport.execute_http_request(trial, 0, client=client)

    assert result.ok is True
    assert [request.url.path for request in requests] == [
        "/v1/chat/completions",
        "/v1/redirected-chat",
    ]
    assert all(request.url.host == "192.168.1.41" for request in requests)
    assert all(request.headers["host"] == "aitopatom-41de.local:8000" for request in requests)
    assert all(request.extensions["sni_hostname"] == "aitopatom-41de.local" for request in requests)
    assert resolutions == 1


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

    def fake_run_requests(_trial, *, count, index_offset=0, client=None):
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
