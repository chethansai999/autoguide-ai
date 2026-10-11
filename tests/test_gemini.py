"""Gemini provider, tested offline with an httpx MockTransport (no network)."""

from __future__ import annotations

import json

import httpx
import pytest

from core.config import Settings
from core.llm import GeminiAPIError, GeminiClient, build_llm, is_connection_error

BASE = "http://127.0.0.1/v1beta"  # an IP literal: the DNS probe never leaves the machine


def settings(**overrides) -> Settings:
    values = dict(llm_provider="gemini", gemini_api_key="AQ.test-key", gemini_api_base=BASE,
                  gemini_model="gemini-test-flash", openai_api_key="", anthropic_api_key="")
    values.update(overrides)
    return Settings(**values)


def test_request_uses_api_key_header_and_native_format():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [
                {"text": "thinking...", "thought": True},
                {"text": "The 2019 F-150 SuperCrew 4x4 has a 5-star overall rating [1]."},
            ]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 120, "candidatesTokenCount": 18, "thoughtsTokenCount": 30},
        })

    client = GeminiClient(settings(), transport=httpx.MockTransport(handler))
    result = client.complete("system rules", [
        {"role": "user", "content": "Is the 2019 F-150 safe?"},
        {"role": "assistant", "content": "Which version?"},
        {"role": "user", "content": "SuperCrew 4x4"},
    ])
    assert seen["url"] == f"{BASE}/models/gemini-test-flash:generateContent"
    assert seen["key"] == "AQ.test-key" and seen["auth"] is None
    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "system rules"
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert result.text == "The 2019 F-150 SuperCrew 4x4 has a 5-star overall rating [1]."  # thought part dropped
    assert result.provider == "gemini" and result.prompt_tokens == 120 and result.completion_tokens == 48


def test_json_mode_sets_mime_type():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": "{}"}]}}]})

    GeminiClient(settings(), transport=httpx.MockTransport(handler)).complete("s", [{"role": "user", "content": "x"}], json_mode=True)
    assert seen["body"]["generationConfig"]["responseMimeType"] == "application/json"


@pytest.mark.parametrize("status,payload", [
    (400, {"error": {"message": "API key not valid. Please pass a valid API key."}}),
    (429, {"error": {"message": "Quota exceeded"}}),
    (404, {"error": {"message": "models/gemini-test-flash is not found"}}),
])
def test_provider_errors_are_raised_not_masked(status, payload):
    client = GeminiClient(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(status, json=payload)))
    with pytest.raises(GeminiAPIError) as caught:
        client.complete("s", [{"role": "user", "content": "x"}])
    assert caught.value.status_code == status
    assert payload["error"]["message"][:20] in str(caught.value)
    assert not is_connection_error(caught.value)  # a bad key must never become a silent fallback


def test_empty_answer_is_an_error():
    response = {"candidates": [{"content": {"parts": []}, "finishReason": "MAX_TOKENS"}]}
    client = GeminiClient(settings(), transport=httpx.MockTransport(lambda r: httpx.Response(200, json=response)))
    with pytest.raises(GeminiAPIError, match="MAX_TOKENS"):
        client.complete("s", [{"role": "user", "content": "x"}])


def test_provider_selection():
    assert isinstance(build_llm(settings()), GeminiClient)
    assert settings(llm_provider="auto").resolved_llm_provider() == "gemini"
    assert settings(llm_provider="auto", openai_api_key="sk-real").resolved_llm_provider() == "openai"
