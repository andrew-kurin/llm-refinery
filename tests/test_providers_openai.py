import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_refinery.core.config import ConfigError
from llm_refinery.core.http_safety import PinnedHttpRoute
from llm_refinery.providers.openai_chat import (
    DEFAULT_USER_AGENT,
    json_headers,
    openai_choice_text,
    post_json_body,
)


def test_json_headers_adds_bearer_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    assert json_headers({"X-Test": "1"}, api_key_env="OPENAI_API_KEY") == {
        "Accept": "application/json",
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
        "X-Test": "1",
    }


def test_json_headers_honors_case_insensitive_explicit_headers(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "environment-secret")

    headers = json_headers(
        {
            "authorization": "Static credential",
            "content-type": "application/custom+json",
            "accept": "application/problem+json",
        },
        api_key_env="OPENAI_API_KEY",
    )

    assert headers == {
        "authorization": "Static credential",
        "content-type": "application/custom+json",
        "accept": "application/problem+json",
        "User-Agent": DEFAULT_USER_AGENT,
    }


def test_json_headers_requires_a_configured_api_key_environment_variable(monkeypatch):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="environment variable is not set: MISSING_API_KEY"):
        json_headers(api_key_env="MISSING_API_KEY")


def test_json_headers_explicit_authorization_does_not_require_an_unused_env_var(
    monkeypatch,
):
    monkeypatch.delenv("MISSING_API_KEY", raising=False)

    headers = json_headers(
        {"Authorization": "Static credential"},
        api_key_env="MISSING_API_KEY",
    )

    assert headers["Authorization"] == "Static credential"


def test_post_json_body_does_not_follow_a_credentialed_redirect():
    redirected_headers: list[dict[str, str]] = []

    class RedirectDestination(BaseHTTPRequestHandler):
        def do_GET(self):
            redirected_headers.append(dict(self.headers))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"{}")

        do_POST = do_GET

        def log_message(self, format, *args):  # noqa: A002
            return

    destination = ThreadingHTTPServer(("127.0.0.1", 0), RedirectDestination)
    destination_thread = threading.Thread(target=destination.serve_forever, daemon=True)
    destination_thread.start()

    class RedirectSource(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(302)
            self.send_header(
                "Location",
                f"http://127.0.0.1:{destination.server_port}/stolen",
            )
            self.end_headers()

        def log_message(self, format, *args):  # noqa: A002
            return

    source = ThreadingHTTPServer(("127.0.0.1", 0), RedirectSource)
    source_thread = threading.Thread(target=source.serve_forever, daemon=True)
    source_thread.start()
    try:
        with pytest.raises(RuntimeError, match="HTTP 302"):
            post_json_body(
                f"http://127.0.0.1:{source.server_port}/v1/chat/completions",
                {"model": "test"},
                headers={
                    "Authorization": "Bearer must-not-leak",
                    "Content-Type": "application/json",
                },
                timeout_s=2,
            )
    finally:
        source.shutdown()
        destination.shutdown()
        source.server_close()
        destination.server_close()
        source_thread.join(timeout=2)
        destination_thread.join(timeout=2)

    assert redirected_headers == []


def test_post_json_body_does_not_persist_untrusted_error_body():
    class ErrorServer(BaseHTTPRequestHandler):
        def do_POST(self):
            body = b"gateway echoed Bearer must-not-persist"
            self.send_response(401)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), ErrorServer)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(RuntimeError, match="HTTP 401") as caught:
            post_json_body(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
                {"model": "test"},
                headers={"Authorization": "Bearer must-not-persist"},
                timeout_s=2,
            )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert "must-not-persist" not in str(caught.value)


def test_post_json_body_uses_pinned_address_with_logical_host():
    observed_hosts: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            observed_hosts.append(self.headers["Host"])
            self.rfile.read(int(self.headers["Content-Length"]))
            body = b"{}"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # noqa: A002
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logical_url = f"http://spark.local:{server.server_port}/v1/chat/completions"
    route = PinnedHttpRoute(
        origin=("http", "spark.local", server.server_port),
        connect_host="127.0.0.1",
        authority=f"spark.local:{server.server_port}",
        sni_hostname="spark.local",
    )
    try:
        body = post_json_body(
            logical_url,
            {"model": "test"},
            headers={"Content-Type": "application/json"},
            timeout_s=2,
            trust_env=False,
            route=route,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert body == "{}"
    assert observed_hosts == [f"spark.local:{server.server_port}"]


def test_openai_choice_text_collects_content_and_reasoning_fields():
    assert (
        openai_choice_text(
            {
                "delta": {"reasoning_content": "think ", "content": "answer "},
                "text": "tail",
            }
        )
        == "answer think tail"
    )
