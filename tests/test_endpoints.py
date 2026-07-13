import pytest

from llm_refinery.core.config import ConfigError
from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint
from llm_refinery.core.runs import stable_hash


def test_endpoint_uses_protocol_and_rejects_legacy_provider_field():
    endpoint = Endpoint.from_mapping(
        {
            "name": "remote",
            "protocol": "openai_chat",
            "base_url": "https://example.test/v1/",
            "model": "model-id",
            "api_key_env": "API_KEY",
            "headers": {"X-Secret": "secret"},
        },
        allowed_protocols=frozenset({OPENAI_CHAT}),
    )

    assert endpoint.base_url == "https://example.test/v1"
    assert endpoint.chat_completions_url == "https://example.test/v1/chat/completions"
    assert endpoint.completions_url == "https://example.test/v1/completions"
    assert endpoint.safe_json() == {
        "name": "remote",
        "protocol": "openai_chat",
        "base_url": "https://example.test/v1",
        "model": "model-id",
        "api_key_env": "API_KEY",
        "header_names": ["X-Secret"],
        "headers_hash": stable_hash({"X-Secret": "secret"}),
    }

    with pytest.raises(ConfigError, match="unknown field.*provider"):
        Endpoint.from_mapping(
            {
                "name": "legacy",
                "provider": "openai",
                "base_url": "https://example.test/v1",
                "model": "model-id",
            }
        )


@pytest.mark.parametrize(
    ("base_url", "error"),
    [
        ("https://user:password@example.test/v1", "user information"),
        ("https://example.test/v1?api_key=secret", "query or fragment"),
        ("https://example.test/v1?", "query or fragment"),
        ("https://example.test/v1#credentials", "query or fragment"),
        ("https://example.test:/v1", "valid hostname and port"),
        ("https://example.test:0/v1", "valid hostname and port"),
        ("https://example.test:99999/v1", "valid hostname and port"),
        ("http://0.0.0.0:8000/v1", "wildcard address"),
        ("http://[::]:8000/v1", "wildcard address"),
        ("https://example.test/v1\\chat", "backslashes"),
        ("https://example.test/v1 chat", "without whitespace"),
    ],
)
def test_endpoint_rejects_credential_bearing_or_ambiguous_base_urls(
    base_url: str,
    error: str,
):
    with pytest.raises(ConfigError, match=error):
        Endpoint(
            name="remote",
            protocol=OPENAI_CHAT,
            base_url=base_url,
            model="served-model",
        )


def test_endpoint_constructs_api_urls_from_the_validated_path_only():
    endpoint = Endpoint(
        name="remote",
        protocol=OPENAI_CHAT,
        base_url="https://example.test:8443/prefix/v1/chat/completions/",
        model="served-model",
    )

    assert endpoint.base_url == "https://example.test:8443/prefix/v1/chat/completions"
    assert endpoint.chat_completions_url == "https://example.test:8443/prefix/v1/chat/completions"
    assert endpoint.completions_url == "https://example.test:8443/prefix/v1/completions"
