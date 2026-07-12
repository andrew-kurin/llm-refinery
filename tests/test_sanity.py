from llm_refinery.core.endpoints import OPENAI_CHAT, Endpoint
from llm_refinery.providers.openai_chat import ChatCompletionResponse
from llm_refinery.utils.sanity import has_reasoning_tags, run_api_sanity_check


def test_has_reasoning_tags_detects_think_tags():
    assert has_reasoning_tags("<think>\n\n</think>\n\nHello")
    assert has_reasoning_tags("<thinking>hidden</thinking>visible")
    assert not has_reasoning_tags("Hello without hidden reasoning markers")


class _FakeClient:
    def __init__(self, message):
        self.message = message

    def complete(self, *_args, **_kwargs):
        return ChatCompletionResponse(
            content=str(self.message.get("content") or ""),
            usage={},
            raw={
                "model": "loaded-model",
                "choices": [{"message": self.message, "finish_reason": "stop"}],
            },
            latency_s=0.25,
        )


def _endpoint():
    return Endpoint(
        name="local",
        protocol=OPENAI_CHAT,
        model="requested-alias",
        base_url="http://127.0.0.1:8080/v1",
    )


def test_sanity_rejects_reasoning_only_response():
    result = run_api_sanity_check(
        _endpoint(),
        client=_FakeClient({"content": "", "reasoning_content": "still working"}),
    )

    assert result["success"] is False
    assert "visible content was empty" in result["error"]


def test_sanity_surfaces_endpoint_model_binding():
    result = run_api_sanity_check(
        _endpoint(),
        client=_FakeClient({"content": "Hello from the loaded model."}),
    )

    assert result["success"] is True
    assert result["requested_model"] == "requested-alias"
    assert result["response_model"] == "loaded-model"
    assert result["model_matches"] is False
