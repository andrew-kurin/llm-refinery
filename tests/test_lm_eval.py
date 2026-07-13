import http.client
import json
import os
import ssl
import subprocess
import sys
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import pytest

from llm_refinery.benchmarks.lm_eval import runner as lm_eval_runner
from llm_refinery.benchmarks.lm_eval.command import build_lm_eval_command, lm_eval_api_key
from llm_refinery.benchmarks.lm_eval.config import LmEvalConfig
from llm_refinery.benchmarks.lm_eval.parser import (
    latest_lm_eval_result,
    lm_eval_sample_files,
    parse_lm_eval_metrics,
    parse_lm_eval_samples,
    summarize_lm_eval_samples,
)
from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint
from llm_refinery.core.http_safety import PinnedHttpRoute
from llm_refinery.storage.duckdb import ResultStore


def test_lm_eval_command_supports_include_path_for_fixed_tasks(tmp_path):
    config = LmEvalConfig(tasks="gpqa_main_generative_n_shot", include_path=tmp_path)
    cmd = build_lm_eval_command(
        config,
        Endpoint(
            name="llama_cpp",
            protocol=OPENAI_CHAT,
            model="local-model",
            base_url="http://localhost/v1/chat/completions",
        ),
    )

    assert "--include_path" in cmd
    assert str(tmp_path) in cmd


def test_lm_eval_command_supports_long_context_metadata(tmp_path):
    config = LmEvalConfig(
        tasks="ruler",
        metadata='{"max_seq_lengths":[4096,8192]}',
        extra_packages=("scorer==1.2.3",),
        output_root=tmp_path,
    )
    cmd = build_lm_eval_command(
        config,
        Endpoint(
            name="local",
            protocol=OPENAI_CHAT,
            model="local-model",
            base_url="http://localhost/v1/chat/completions",
        ),
    )

    model_args = cmd[cmd.index("--model_args") + 1]
    assert "tokenizer=" not in model_args
    assert "eos_string=" not in model_args
    assert cmd[cmd.index("--metadata") + 1] == '{"max_seq_lengths":[4096,8192]}'
    scorer_index = cmd.index("scorer==1.2.3")
    assert cmd[scorer_index - 1] == "--with"


def test_lm_eval_command_rounds_subsecond_timeout_up_for_pinned_adapter():
    config = LmEvalConfig(request_timeout_s=0.05)
    cmd = build_lm_eval_command(
        config,
        Endpoint(
            name="local",
            protocol=OPENAI_CHAT,
            model="local-model",
            base_url="http://localhost/v1/chat/completions",
        ),
    )

    model_args = cmd[cmd.index("--model_args") + 1]
    assert "timeout=1" in model_args.split(",")


def test_lm_eval_chat_backend_rejects_ignored_tokenizer():
    with pytest.raises(ConfigError, match="ignores client-side tokenization"):
        LmEvalConfig(tokenizer="org/model-tokenizer")


def test_lm_eval_rejects_unknown_model_backend():
    with pytest.raises(ConfigError, match="model_backend must be one of"):
        LmEvalConfig(model_backend="unknown")


def test_lm_eval_rejects_blank_tokenizer():
    with pytest.raises(ConfigError, match="tokenizer must be a non-empty string"):
        LmEvalConfig(model_backend="local-completions", tokenizer="  ")


def test_lm_eval_command_uses_explicit_run_output_path(tmp_path):
    output_path = tmp_path / "target" / "run-id"
    command = build_lm_eval_command(
        LmEvalConfig(output_root=tmp_path),
        Endpoint(
            name="local",
            protocol=OPENAI_CHAT,
            model="local-model",
            base_url="http://127.0.0.1:8000/v1",
        ),
        output_path=output_path,
    )

    assert command[command.index("--output_path") + 1] == str(output_path)


def test_lm_eval_relay_preserves_online_network_environment_and_bypasses_loopback():
    config = LmEvalConfig(num_concurrent=2, trust_env=True)
    command = build_lm_eval_command(
        config,
        Endpoint(
            name="remote",
            protocol=OPENAI_CHAT,
            model="served-model",
            base_url="https://model.example/v1",
        ),
    )
    child_environment = lm_eval_runner._lm_eval_environment(
        {
            "HTTPS_PROXY": "http://proxy.example:8080",
            "SSL_CERT_FILE": "/etc/ssl/executor.pem",
            "NO_PROXY": "internal.example",
            "no_proxy": "service.local,localhost",
        }
    )

    assert config.num_concurrent == 2
    assert config.trust_env is True
    assert "env" not in command
    assert child_environment["HTTPS_PROXY"] == "http://proxy.example:8080"
    assert child_environment["SSL_CERT_FILE"] == "/etc/ssl/executor.pem"
    assert child_environment["NO_PROXY"] == ("internal.example,service.local,localhost,127.0.0.1")
    assert child_environment["no_proxy"] == child_environment["NO_PROXY"]


def test_lm_eval_no_trust_env_isolates_child_without_applying_target_ca(tmp_path):
    ca_bundle = tmp_path / "dgx-ca.pem"
    ca_bundle.write_text("test", encoding="utf-8")
    command = build_lm_eval_command(
        LmEvalConfig(trust_env=False, ca_bundle=ca_bundle),
        Endpoint(
            name="remote",
            protocol=OPENAI_CHAT,
            model="served-model",
            base_url="https://model.example/v1",
        ),
    )
    env_index = command.index("env")
    lm_eval_index = command.index("lm_eval")
    wrapper = command[env_index:lm_eval_index]
    pairs = [wrapper[index : index + 2] for index in range(len(wrapper))]

    assert ["-u", "HTTP_PROXY"] in pairs
    assert ["-u", "SSL_CERT_FILE"] in pairs
    assert str(ca_bundle) not in command


def test_lm_eval_relay_preserves_host_and_refuses_upstream_redirects(tmp_path, monkeypatch):
    requests: list[tuple[str, str]] = []
    redirect = False

    class Upstream(BaseHTTPRequestHandler):
        def do_POST(self):
            requests.append((self.path, self.headers["Host"]))
            self.rfile.read(int(self.headers["Content-Length"]))
            if redirect:
                self.send_response(307)
                self.send_header("Location", "http://127.0.0.1:9/stolen")
                self.end_headers()
                return
            body = b'{"choices":[]}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Retry-After", "2")
            self.send_header("retry-after-ms", "50")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    logical_base = f"http://spark.local:{upstream.server_port}/v1"
    route = PinnedHttpRoute(
        origin=("http", "spark.local", upstream.server_port),
        connect_host="127.0.0.1",
        authority=f"spark.local:{upstream.server_port}",
        sni_hostname="spark.local",
    )
    target = Endpoint(
        name="spark",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url=logical_base,
    )
    config = LmEvalConfig(
        target="spark",
        output_root=tmp_path,
        targets={"spark": target},
        pinned_route=route,
        trust_env=True,
    )
    monkeypatch.setattr("llm_refinery.core.http_safety.getproxies", lambda: {})
    try:
        with lm_eval_runner._lm_eval_target(target, config, dry_run=False) as relay_target:
            response = httpx.post(relay_target.chat_completions_url, json={"model": "x"})
            assert response.status_code == 200
            assert response.json() == {"choices": []}
            assert response.headers["retry-after"] == "2"
            assert response.headers["retry-after-ms"] == "50"

            redirect = True
            response = httpx.post(relay_target.chat_completions_url, json={"model": "x"})
            assert response.status_code == 502
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert requests == [
        ("/v1/chat/completions", f"spark.local:{upstream.server_port}"),
        ("/v1/chat/completions", f"spark.local:{upstream.server_port}"),
    ]


def test_lm_eval_completions_relay_forwards_remote_tokenizer_contract(tmp_path):
    requests: list[tuple[str, str, str, str | None, dict[str, object] | None]] = []

    class Upstream(BaseHTTPRequestHandler):
        def do_GET(self):
            self._respond()

        def do_POST(self):
            self._respond()

        def _respond(self):
            body = None
            status = 200
            if self.command == "POST":
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(
                (
                    self.command,
                    self.path,
                    self.headers["Host"],
                    self.headers.get("Authorization"),
                    body,
                )
            )
            if self.path == "/tenant/tokenizer_info":
                response = {"eos_token": "</s>", "bos_token": "<s>"}
            elif self.path == "/tenant/tokenize":
                response = {"tokens": [1, 2, 3]}
            elif self.path == "/tenant/detokenize":
                response = {"prompt": "decoded"}
            elif self.path == "/tenant/v1/completions":
                if self.headers.get("Authorization") != "Bearer target-secret":
                    status = 401
                    response = {"error": "unauthorized"}
                else:
                    response = {"choices": []}
            else:
                self.send_error(404)
                return
            encoded = json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format, *args):  # noqa: A002
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    logical_base = f"http://spark.local:{upstream.server_port}/tenant/v1/chat/completions"
    target = Endpoint(
        name="spark",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url=logical_base,
    )
    config = LmEvalConfig(
        target="spark",
        model_backend="local-completions",
        output_root=tmp_path,
        targets={"spark": target},
        pinned_route=PinnedHttpRoute(
            origin=("http", "spark.local", upstream.server_port),
            connect_host="127.0.0.1",
            authority=f"spark.local:{upstream.server_port}",
            sni_hostname="spark.local",
        ),
    )
    try:
        with lm_eval_runner._lm_eval_target(
            target,
            config,
            dry_run=False,
            api_key="target-secret",
        ) as relay_target:
            tokenizer_base = relay_target.completions_url.replace("/v1/completions", "").rstrip("/")
            client = httpx.Client(trust_env=False)
            try:
                assert client.get(f"{tokenizer_base}/tokenizer_info").json() == {
                    "eos_token": "</s>",
                    "bos_token": "<s>",
                }
                assert client.post(
                    f"{tokenizer_base}/tokenize",
                    json={"prompt": "test", "add_special_tokens": False},
                ).json() == {"tokens": [1, 2, 3]}
                assert client.post(
                    f"{tokenizer_base}/detokenize",
                    json={"tokens": [1, 2, 3]},
                ).json() == {"prompt": "decoded"}
                assert (
                    client.post(
                        relay_target.completions_url,
                        json={"model": "served-model", "prompt": "test"},
                    ).status_code
                    == 401
                )
                assert client.get(relay_target.completions_url).status_code == 404
                assert client.post(f"{tokenizer_base}/tokenizer_info", json={}).status_code == 404
            finally:
                client.close()
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    expected_host = f"spark.local:{upstream.server_port}"
    assert requests == [
        (
            "GET",
            "/tenant/tokenizer_info",
            expected_host,
            "Bearer target-secret",
            None,
        ),
        (
            "POST",
            "/tenant/tokenize",
            expected_host,
            "Bearer target-secret",
            {"prompt": "test", "add_special_tokens": False},
        ),
        (
            "POST",
            "/tenant/detokenize",
            expected_host,
            "Bearer target-secret",
            {"tokens": [1, 2, 3]},
        ),
        (
            "POST",
            "/tenant/v1/completions",
            expected_host,
            None,
            {"model": "served-model", "prompt": "test"},
        ),
    ]


def test_lm_eval_relay_enforces_absolute_upstream_deadline(tmp_path, monkeypatch):
    class TricklingUpstream(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{")
            self.wfile.flush()
            threading.Event().wait(0.5)
            with suppress(BrokenPipeError):
                self.wfile.write(b"}")

        def log_message(self, format, *args):  # noqa: A002
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), TricklingUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    logical_base = f"http://spark.local:{upstream.server_port}/v1"
    route = PinnedHttpRoute(
        origin=("http", "spark.local", upstream.server_port),
        connect_host="127.0.0.1",
        authority=f"spark.local:{upstream.server_port}",
        sni_hostname="spark.local",
    )
    target = Endpoint(
        name="spark",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url=logical_base,
    )
    config = LmEvalConfig(
        target="spark",
        output_root=tmp_path,
        targets={"spark": target},
        pinned_route=route,
        request_timeout_s=0.05,
    )
    monkeypatch.setattr("llm_refinery.core.http_safety.getproxies", lambda: {})
    try:
        with lm_eval_runner._lm_eval_target(target, config, dry_run=False) as relay_target:
            started = time.monotonic()
            response = httpx.post(relay_target.chat_completions_url, json={"model": "x"})
            elapsed = time.monotonic() - started
        assert response.status_code == 502
        assert elapsed < 0.3
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_lm_eval_route_less_relay_enforces_deadline_for_stalled_endpoint(tmp_path):
    class StalledUpstream(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            threading.Event().wait(0.5)
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with suppress(BrokenPipeError):
                self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), StalledUpstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    target = Endpoint(
        name="local",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url=f"http://127.0.0.1:{upstream.server_port}/v1",
    )
    config = LmEvalConfig(
        target="local",
        output_root=tmp_path,
        targets={"local": target},
        request_timeout_s=0.05,
    )
    try:
        with lm_eval_runner._lm_eval_target(target, config, dry_run=False) as relay_target:
            assert relay_target.base_url != target.base_url
            started = time.monotonic()
            response = httpx.post(relay_target.chat_completions_url, json={"model": "x"})
            elapsed = time.monotonic() - started
        assert response.status_code == 502
        assert elapsed < 0.3
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)


def test_lm_eval_relay_teardown_cancels_and_joins_active_requests(tmp_path, monkeypatch):
    upstream_started = threading.Event()
    upstream_closed = threading.Event()
    downstream_statuses: list[int] = []

    class BlockingResponse:
        status_code = 200
        headers: dict[str, str] = {}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def iter_bytes(self):
            upstream_started.set()
            assert upstream_closed.wait(timeout=2)
            raise httpx.ReadError("closed during relay teardown")

    class BlockingClient:
        def __init__(self, **kwargs):
            del kwargs

        def stream(self, *args, **kwargs):
            del args, kwargs
            return BlockingResponse()

        def close(self):
            upstream_closed.set()

    logical_base = "http://spark.local:8000/v1"
    target = Endpoint(
        name="spark",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url=logical_base,
    )
    config = LmEvalConfig(
        target="spark",
        output_root=tmp_path,
        targets={"spark": target},
        pinned_route=PinnedHttpRoute(
            origin=("http", "spark.local", 8000),
            connect_host="192.168.1.41",
            authority="spark.local:8000",
            sni_hostname="spark.local",
        ),
    )
    monkeypatch.setattr("llm_refinery.core.http_safety.getproxies", lambda: {})
    monkeypatch.setattr(lm_eval_runner.httpx, "Client", BlockingClient)

    def request_relay(port: int) -> None:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        try:
            connection.request(
                "POST",
                "/v1/chat/completions",
                body=b"{}",
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            downstream_statuses.append(response.status)
            response.read()
        finally:
            connection.close()

    with lm_eval_runner._lm_eval_target(target, config, dry_run=False) as relay_target:
        parsed_relay = urlsplit(relay_target.base_url)
        assert parsed_relay.port is not None
        request_thread = threading.Thread(target=request_relay, args=(parsed_relay.port,))
        request_thread.start()
        assert upstream_started.wait(timeout=1)

    request_thread.join(timeout=1)

    assert upstream_closed.is_set()
    assert not request_thread.is_alive()
    assert downstream_statuses == [502]


@pytest.mark.parametrize(
    "field,value",
    [
        ("request_timeout_s", float("inf")),
        ("request_timeout_s", True),
        ("process_timeout_s", float("nan")),
        ("process_timeout_s", 0),
    ],
)
def test_lm_eval_rejects_unbounded_timeout_configuration(field, value):
    with pytest.raises(ConfigError, match=field):
        LmEvalConfig(**{field: value})


def test_lm_eval_process_has_absolute_timeout(tmp_path, monkeypatch):
    database = tmp_path / "runs.duckdb"

    def timeout_process(command, *, env, timeout_s):
        assert timeout_s == 0.05
        return (
            subprocess.CompletedProcess(
                command,
                124,
                stdout="partial output",
                stderr="partial error",
            ),
            True,
        )

    monkeypatch.setattr(lm_eval_runner, "_run_lm_eval_process", timeout_process)
    config = LmEvalConfig(
        target="local",
        limit=1,
        process_timeout_s=0.05,
        output_root=tmp_path / "output",
        database=database,
        targets={
            "local": Endpoint(
                name="local",
                protocol=OPENAI_CHAT,
                model="local-model",
                base_url="http://127.0.0.1:8080/v1",
            )
        },
    )

    with pytest.raises(lm_eval_runner.LmEvalFailed, match="timed out after 0.05s"):
        lm_eval_runner.run_lm_eval(config)

    with ResultStore(database) as store:
        run = store.comparison_runs(include_failed=True)[0]
    assert run["status"] == "failed"
    assert run["error"] == "lm-eval process timed out after 0.05s"
    assert "partial output" in Path(run["artifacts"]["stdout"]["path"]).read_text()
    assert "partial error" in Path(run["artifacts"]["stderr"]["path"]).read_text()


def test_lm_eval_timeout_terminates_descendant_processes(tmp_path):
    ready = tmp_path / "child-ready"
    survived = tmp_path / "child-survived"
    child_source = (
        "import signal,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"Path({str(ready)!r}).write_text('ready'); "
        "time.sleep(1); "
        f"Path({str(survived)!r}).write_text('survived'); "
        "time.sleep(60)"
    )
    parent_source = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_source!r}]); "
        "time.sleep(60)"
    )

    completed, timed_out = lm_eval_runner._run_lm_eval_process(
        [sys.executable, "-c", parent_source],
        env=os.environ.copy(),
        timeout_s=0.3,
    )

    assert timed_out is True
    assert completed.returncode == 124
    assert ready.is_file()
    threading.Event().wait(1)
    assert not survived.exists()


def test_lm_eval_completions_backend_uses_completions_url_and_tokenizer():
    config = LmEvalConfig(
        model_backend="local-completions",
        tokenizer="org/model-tokenizer",
    )
    target = Endpoint(
        name="remote",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url="http://remote.test/v1/chat/completions",
    )

    command = build_lm_eval_command(config, target)
    model_args = command[command.index("--model_args") + 1]

    assert "base_url=http://remote.test/v1/completions" in model_args
    assert "tokenizer=org/model-tokenizer" in model_args
    assert "tokenizer_backend=huggingface" in model_args


@pytest.mark.parametrize(
    "headers, message",
    [
        ({"X-Tenant": "tenant-a"}, "unsupported header name.*X-Tenant"),
        ({"Authorization": "Basic secret"}, "only a Bearer Authorization"),
    ],
)
def test_lm_eval_rejects_headers_it_cannot_pass_safely(headers, message):
    target = Endpoint(
        name="remote",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url="http://remote.test/v1",
        headers=headers,
    )

    with pytest.raises(ConfigError, match=message):
        build_lm_eval_command(LmEvalConfig(), target)


def test_lm_eval_bearer_header_is_resolved_only_into_environment():
    target = Endpoint(
        name="remote",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url="http://remote.test/v1",
        headers={"authorization": "Bearer top-secret-token"},
    )

    command = build_lm_eval_command(LmEvalConfig(), target)

    assert "top-secret-token" not in " ".join(command)
    assert lm_eval_api_key(target, environ={}) == "top-secret-token"


def test_lm_eval_api_key_env_must_exist_and_overrides_ambient_openai_key():
    target = Endpoint(
        name="remote",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url="http://remote.test/v1",
        api_key_env="VLLM_API_KEY",
    )

    assert (
        lm_eval_api_key(
            target,
            environ={"VLLM_API_KEY": "target-key", "OPENAI_API_KEY": "unrelated-key"},
        )
        == "target-key"
    )
    with pytest.raises(ConfigError, match="VLLM_API_KEY"):
        lm_eval_api_key(target, environ={"OPENAI_API_KEY": "unrelated-key"})


def test_lm_eval_rejects_invalid_env_api_key_without_disclosing_it():
    target = Endpoint(
        name="remote",
        protocol=OPENAI_CHAT,
        model="served-model",
        base_url="http://remote.test/v1",
        api_key_env="VLLM_API_KEY",
    )

    with pytest.raises(ConfigError) as caught:
        lm_eval_api_key(
            target,
            environ={"VLLM_API_KEY": "top-secret-token\nInjected: yes"},
        )

    assert "top-secret-token" not in str(caught.value)
    assert "Injected" not in str(caught.value)


def test_release_quality_tasks_pin_dataset_revisions():
    task_root = Path("evals/lm_eval_tasks")
    expected_revisions = {
        task_root / "ifeval_pinned" / "ifeval_pinned.yaml": (
            "966cd89545d6b6acfd7638bc708b98261ca58e84"
        ),
        task_root / "gpqa_fixed" / "_gpqa_fixed_generative_yaml": (
            "633f5ee89ab8ad4522a9f850766b73f62147ffdd"
        ),
        task_root / "mmlu_pro_pinned" / "_default_template_yaml": (
            "b189ec765aa7ed75c8acfea42df31fdae71f97be"
        ),
        task_root / "musr_generative" / "_musr_generative_template_yaml": (
            "7c365b439a222150f317764d4f16ae6c96d7d94a"
        ),
    }

    for task_path, revision in expected_revisions.items():
        assert f"revision: {revision}" in task_path.read_text(encoding="utf-8")


def test_latest_lm_eval_result_ignores_stale_result_files(tmp_path):
    old_result = tmp_path / "results_older.json"
    old_result.write_text("{}", encoding="utf-8")
    os.utime(old_result, (100.0, 100.0))

    assert latest_lm_eval_result(tmp_path, newer_than=200.0) is None

    new_result = tmp_path / "results_newer.json"
    new_result.write_text("{}", encoding="utf-8")
    os.utime(new_result, (300.0, 300.0))

    assert latest_lm_eval_result(tmp_path, newer_than=200.0) == new_result


def test_parse_lm_eval_metrics_normalizes_filters_and_stderr(tmp_path):
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": 0.8143,
                        "exact_match_stderr,strict-match": 0.0107,
                        "exact_match,flexible-extract": 0.834,
                        "alias": "gsm8k",
                    },
                    "ifeval": {
                        "prompt_strict_acc": 0.8854,
                        "prompt_strict_acc_stderr": 0.012,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert parse_lm_eval_metrics(result_path) == {
        "gsm8k.strict-match.exact_match": 0.8143,
        "gsm8k.strict-match.exact_match_stderr": 0.0107,
        "gsm8k.flexible-extract.exact_match": 0.834,
        "ifeval.prompt_strict_acc": 0.8854,
        "ifeval.prompt_strict_acc_stderr": 0.012,
        "gsm8k.strict-match.exact_match_ci95_low": 0.793328,
        "gsm8k.strict-match.exact_match_ci95_high": 0.835272,
        "ifeval.prompt_strict_acc_ci95_low": 0.86188,
        "ifeval.prompt_strict_acc_ci95_high": 0.90892,
    }


def test_parse_lm_eval_samples_retains_item_evidence_and_wilson_intervals(tmp_path):
    result_path = tmp_path / "results_2026-07-10T12-00-00.json"
    result_path.write_text("{}", encoding="utf-8")
    sample_path = tmp_path / "samples_ifeval_2026-07-10T12-00-00.jsonl"
    sample_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "doc_id": 7,
                        "filter": "none",
                        "prompt_hash": "prompt-7",
                        "target_hash": "target-7",
                        "filtered_resps": ["answer"],
                        "prompt_level_strict_acc": 1,
                        "prompt_level_loose_acc": 1,
                    }
                ),
                json.dumps(
                    {
                        "doc_id": 8,
                        "filter": "none",
                        "prompt_hash": "prompt-8",
                        "target_hash": "target-8",
                        "filtered_resps": ["wrong"],
                        "prompt_level_strict_acc": 0,
                        "prompt_level_loose_acc": 1,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    assert lm_eval_sample_files(result_path) == [sample_path]
    samples = parse_lm_eval_samples(sample_path, result_path=result_path)
    assert [sample.sample_id for sample in samples] == [
        "ifeval:none:7",
        "ifeval:none:8",
    ]
    assert [sample.metrics["correct"] for sample in samples] == [1.0, 0.0]
    assert samples[0].payload["response_hash"] != samples[1].payload["response_hash"]
    summary = summarize_lm_eval_samples(samples)
    assert summary["samples.recorded_count"] == 2
    assert summary["samples.ifeval.correct_rate"] == 0.5
    assert summary["samples.ifeval.correct_rate_ci95_low"] < 0.5
    assert summary["samples.ifeval.correct_rate_ci95_high"] > 0.5


def test_ifbench_normalized_correctness_uses_its_primary_loose_metric(tmp_path):
    result_path = tmp_path / "results_stamp.json"
    result_path.write_text("{}", encoding="utf-8")
    sample_path = tmp_path / "samples_ifbench_stamp.jsonl"
    sample_path.write_text(
        json.dumps(
            {
                "doc_id": 1,
                "filter": "none",
                "prompt_level_loose_acc": 1,
                "prompt_level_strict_acc": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = parse_lm_eval_samples(sample_path, result_path=result_path)[0]
    assert sample.metrics["correct"] == 1.0


def test_lm_eval_runner_isolates_output_by_run_without_changing_spec_hash(
    monkeypatch,
    tmp_path,
):
    output_paths: list[Path] = []

    def fake_process(command, *, env, timeout_s):
        del env, timeout_s
        output_path = Path(command[command.index("--output_path") + 1])
        output_paths.append(output_path)
        output_path.mkdir(parents=True)
        (output_path / "results_stamp.json").write_text(
            json.dumps({"results": {"ifeval": {"prompt_strict_acc": 1.0}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr=""), False

    monkeypatch.setattr(lm_eval_runner, "_run_lm_eval_process", fake_process)
    config = LmEvalConfig(
        target="local",
        tasks="ifeval",
        limit=1,
        output_root=tmp_path / "lm-eval-output",
        database=tmp_path / "runs.duckdb",
        targets={
            "local": Endpoint(
                name="local",
                protocol=OPENAI_CHAT,
                model="local-model",
                base_url="http://127.0.0.1:8080/v1",
            )
        },
    )

    first = lm_eval_runner.run_lm_eval(config)[0]
    second = lm_eval_runner.run_lm_eval(config)[0]

    assert first.spec_hash == second.spec_hash
    assert first.run_id != second.run_id
    assert output_paths == [
        config.output_root / "local" / first.run_id,
        config.output_root / "local" / second.run_id,
    ]
    assert output_paths[0] != output_paths[1]


def test_lm_eval_completions_full_chat_url_keeps_stable_logical_command(
    monkeypatch,
    tmp_path,
):
    executed_commands: list[list[str]] = []

    def fake_process(command, *, env, timeout_s):
        del env, timeout_s
        executed_commands.append(command)
        output_path = Path(command[command.index("--output_path") + 1])
        output_path.mkdir(parents=True)
        (output_path / "results_stamp.json").write_text(
            json.dumps({"results": {"task": {"score": 1.0}}}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr=""), False

    monkeypatch.setattr(lm_eval_runner, "_run_lm_eval_process", fake_process)
    logical_url = "http://127.0.0.1:8080/v1/chat/completions"
    config = LmEvalConfig(
        target="local",
        model_backend="local-completions",
        tasks="task",
        limit=1,
        output_root=tmp_path / "lm-eval-output",
        database=tmp_path / "runs.duckdb",
        targets={
            "local": Endpoint(
                name="local",
                protocol=OPENAI_CHAT,
                model="local-model",
                base_url=logical_url,
            )
        },
    )

    first = lm_eval_runner.run_lm_eval(config)[0]
    second = lm_eval_runner.run_lm_eval(config)[0]

    assert first.spec_hash == second.spec_hash
    assert first.run_id != second.run_id
    with ResultStore(config.database) as store:
        commands = [run["command"] for run in store.comparison_runs(latest_per_trial=False)]
    assert len(commands) == 2
    assert all("127.0.0.1:8080/v1/completions" in command for command in commands)
    assert all("<run-id>" in command for command in commands)
    assert all(command == commands[0] for command in commands)
    assert all(
        "127.0.0.1:8080/v1/completions" not in " ".join(command) for command in executed_commands
    )


def test_lm_eval_runner_persists_logged_samples(monkeypatch, tmp_path):
    output_root = tmp_path / "lm-eval-output"
    database = tmp_path / "runs.duckdb"
    child_env: dict[str, str] = {}
    executed_command: list[str] = []
    executed_output_paths: list[Path] = []

    def fake_process(command, *, env, timeout_s):
        executed_command.extend(command)
        child_env.update(env)
        output_path = Path(command[command.index("--output_path") + 1])
        executed_output_paths.append(output_path)
        output_dir = output_path / "model"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = "2026-07-10T12-00-00"
        (output_dir / f"results_{stamp}.json").write_text(
            json.dumps({"results": {"ifeval": {"prompt_strict_acc": 1.0}}}),
            encoding="utf-8",
        )
        (output_dir / f"samples_ifeval_{stamp}.jsonl").write_text(
            json.dumps(
                {
                    "doc_id": 42,
                    "filter": "none",
                    "filtered_resps": ["valid"],
                    "prompt_level_strict_acc": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return (
            subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout="ok top-secret-token Bearer top-secret-token",
                stderr="request failed with top-secret-token",
            ),
            False,
        )

    monkeypatch.setattr(lm_eval_runner, "_run_lm_eval_process", fake_process)
    monkeypatch.setattr(
        lm_eval_runner.httpx,
        "create_ssl_context",
        lambda **_kwargs: ssl.create_default_context(),
    )
    monkeypatch.setenv("TEST_LM_EVAL_API_KEY", "top-secret-token")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setenv("SSL_CERT_DIR", "/ambient/certificates")
    ca_bundle = tmp_path / "private-ca.pem"
    ca_bundle.write_text("test CA", encoding="utf-8")
    config = LmEvalConfig(
        target="local",
        tasks="ifeval",
        limit=1,
        log_samples=True,
        output_root=output_root,
        ca_bundle=ca_bundle,
        database=database,
        targets={
            "local": Endpoint(
                name="local",
                protocol=OPENAI_CHAT,
                model="local-model",
                base_url="http://127.0.0.1:8080/v1",
                api_key_env="TEST_LM_EVAL_API_KEY",
            )
        },
    )

    outcome = lm_eval_runner.run_lm_eval(config)[0]

    assert outcome.status == "ok"
    assert outcome.metrics["samples.recorded_count"] == 1
    assert executed_output_paths == [output_root / "local" / outcome.run_id]
    with ResultStore(database) as store:
        samples = store.samples_for_run(outcome.run_id)
        run = store.comparison_runs()[0]
    assert samples[0]["sample_id"] == "ifeval:none:42"
    assert samples[0]["metrics"]["correct"] == 1.0
    assert samples[0]["artifact_path"].endswith("samples_ifeval_2026-07-10T12-00-00.jsonl")
    stdout = Path(run["artifacts"]["stdout"]["path"]).read_text(encoding="utf-8")
    stderr = Path(run["artifacts"]["stderr"]["path"]).read_text(encoding="utf-8")
    assert str(output_root / "local" / "<run-id>") in run["command"]
    assert outcome.run_id not in run["command"]
    assert "top-secret-token" not in stdout + stderr
    assert "[REDACTED]" in stdout + stderr
    assert child_env["HTTP_PROXY"] == "http://proxy.invalid:3128"
    assert child_env["SSL_CERT_DIR"] == "/ambient/certificates"
    env_index = executed_command.index("env")
    lm_eval_index = executed_command.index("lm_eval")
    wrapper = executed_command[env_index:lm_eval_index]
    assert ["-u", "HTTP_PROXY"] in [wrapper[index : index + 2] for index in range(len(wrapper))]
    assert ["-u", "SSL_CERT_DIR"] in [wrapper[index : index + 2] for index in range(len(wrapper))]
    assert str(ca_bundle) not in wrapper
