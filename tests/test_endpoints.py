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
