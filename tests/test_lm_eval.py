import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

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


def test_lm_eval_chat_backend_rejects_ignored_tokenizer():
    with pytest.raises(ConfigError, match="ignores client-side tokenization"):
        LmEvalConfig(tokenizer="org/model-tokenizer")


def test_lm_eval_rejects_proxy_environment_with_async_api_client():
    with pytest.raises(ConfigError, match="num_concurrent=1"):
        LmEvalConfig(num_concurrent=2, trust_env=True)


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
    command = build_lm_eval_command(config, target)
    env_index = command.index("env")
    lm_eval_index = command.index("lm_eval")
    wrapper = command[env_index:lm_eval_index]
    assert ["-u", "HTTP_PROXY"] in [wrapper[index : index + 2] for index in range(len(wrapper))]
    try:
        with lm_eval_runner._lm_eval_target(target, config, dry_run=False) as relay_target:
            response = httpx.post(relay_target.chat_completions_url, json={"model": "x"})
            assert response.status_code == 200
            assert response.json() == {"choices": []}

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


def test_lm_eval_runner_persists_logged_samples(monkeypatch, tmp_path):
    output_root = tmp_path / "lm-eval-output"
    database = tmp_path / "runs.duckdb"
    real_run = subprocess.run
    child_env: dict[str, str] = {}
    executed_command: list[str] = []

    def fake_run(*args, **kwargs):
        command = args[0] if args else kwargs.get("args")
        if not isinstance(command, list) or not command or command[0] != "uvx":
            return real_run(*args, **kwargs)
        executed_command.extend(command)
        child_env.update(kwargs["env"])
        output_dir = output_root / "local" / "model"
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
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="ok top-secret-token Bearer top-secret-token",
            stderr="request failed with top-secret-token",
        )

    monkeypatch.setattr(lm_eval_runner.subprocess, "run", fake_run)
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
    with ResultStore(database) as store:
        samples = store.samples_for_run(outcome.run_id)
        run = store.comparison_runs()[0]
    assert samples[0]["sample_id"] == "ifeval:none:42"
    assert samples[0]["metrics"]["correct"] == 1.0
    assert samples[0]["artifact_path"].endswith("samples_ifeval_2026-07-10T12-00-00.jsonl")
    stdout = Path(run["artifacts"]["stdout"]["path"]).read_text(encoding="utf-8")
    stderr = Path(run["artifacts"]["stderr"]["path"]).read_text(encoding="utf-8")
    assert "top-secret-token" not in stdout + stderr
    assert "[REDACTED]" in stdout + stderr
    assert child_env["HTTP_PROXY"] == "http://proxy.invalid:3128"
    assert child_env["SSL_CERT_DIR"] == "/ambient/certificates"
    env_index = executed_command.index("env")
    lm_eval_index = executed_command.index("lm_eval")
    wrapper = executed_command[env_index:lm_eval_index]
    assert ["-u", "HTTP_PROXY"] in [wrapper[index : index + 2] for index in range(len(wrapper))]
    assert ["-u", "SSL_CERT_DIR"] in [wrapper[index : index + 2] for index in range(len(wrapper))]
    assert f"SSL_CERT_FILE={ca_bundle}" in wrapper
    assert f"REQUESTS_CA_BUNDLE={ca_bundle}" in wrapper
    assert f"CURL_CA_BUNDLE={ca_bundle}" in wrapper
