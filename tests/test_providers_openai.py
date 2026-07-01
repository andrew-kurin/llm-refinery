from llm_refinery.providers.openai import chat_completions_url, json_headers, openai_choice_text


def test_chat_completions_url_accepts_base_or_endpoint():
    assert chat_completions_url("http://localhost:8080/v1") == (
        "http://localhost:8080/v1/chat/completions"
    )
    assert chat_completions_url("http://localhost:8080/v1/chat/completions") == (
        "http://localhost:8080/v1/chat/completions"
    )


def test_json_headers_adds_bearer_token(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "secret")

    assert json_headers({"X-Test": "1"}, api_key_env="OPENAI_API_KEY") == {
        "Accept": "application/json",
        "Authorization": "Bearer secret",
        "Content-Type": "application/json",
        "X-Test": "1",
    }


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
