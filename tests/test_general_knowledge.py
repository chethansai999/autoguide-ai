"""General-knowledge fallback for concept questions when the article index is
unavailable.

When no chunks are loaded, a concept question (e.g. "How does automatic
emergency braking work?") cannot be answered from evidence. If a real LLM
provider is configured, the assistant now answers from the model's general
automotive knowledge with an explicit disclaimer, instead of refusing.
With the mock provider the honest refusal stays, since mock text is a
placeholder, not an answer.
"""

from __future__ import annotations

import pytest

from agent.orchestrator import ChatService
from core.llm import LLMResult
from core.schemas import ChatRequest, TurnStatus
from rag.generation import GroundedGenerator
from rag.prompts import GENERAL_KNOWLEDGE_DISCLAIMER

QUESTION = "How does automatic emergency braking work?"


class FakeLLM:
    """Offline stand-in for a real provider: returns canned text, records the
    system prompt so tests can verify the guardrails were sent."""

    provider = "fake"
    model = "fake-model"

    def __init__(self, text: str):
        self.text = text
        self.system_prompts: list[str] = []

    def complete(self, system, messages, **kwargs):
        self.system_prompts.append(system)
        return LLMResult(text=self.text, provider=self.provider, model=self.model)


def _no_index_service(monkeypatch, tmp_path, provider: str, fake_text: str = "canned answer"):
    """A ChatService with no chunks loaded (empty DATA_DIR) and a fake LLM."""
    from core import config
    from rag import retrieval

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_PROVIDER", provider)
    config.get_settings.cache_clear()
    retrieval._retrievers.clear()
    settings = config.get_settings()
    assert settings.resolved_llm_provider() == provider
    fake = FakeLLM(fake_text)
    generator = GroundedGenerator(llm=fake)
    try:
        return ChatService(settings=settings, generator=generator), fake
    finally:
        config.get_settings.cache_clear()
        retrieval._retrievers.clear()


def test_mock_provider_still_refuses_without_index(monkeypatch, tmp_path):
    service, _ = _no_index_service(monkeypatch, tmp_path, "mock")
    r = service.handle(ChatRequest(message=QUESTION))
    assert r.status == TurnStatus.REFUSED
    assert r.refusal_reason == "index_unavailable"
    assert "article library" in r.answer


def test_real_provider_answers_from_general_knowledge(monkeypatch, tmp_path):
    service, fake = _no_index_service(monkeypatch, tmp_path, "gemini", "AEB uses radar and cameras.")
    r = service.handle(ChatRequest(message=QUESTION))
    assert r.status == TurnStatus.ANSWERED
    assert r.refusal_reason is None
    assert r.answer == "AEB uses radar and cameras."
    assert r.disclaimer == GENERAL_KNOWLEDGE_DISCLAIMER
    assert "general_knowledge_no_sources" in r.trace.guard_flags
    assert r.citations == []
    # the guardrails went to the model: no invented ratings/recalls, no citations
    prompt = fake.system_prompts[0]
    assert "Do NOT invent NHTSA star ratings" in prompt
    assert "Do not use numbered citations" in prompt


def test_prediction_still_refused_with_real_provider(monkeypatch, tmp_path):
    service, _ = _no_index_service(monkeypatch, tmp_path, "gemini")
    r = service.handle(ChatRequest(message="When will my airbag fail?"))
    assert r.status == TurnStatus.REFUSED
    assert r.refusal_reason == "unsupported_prediction"


def test_off_topic_still_refused_with_real_provider(monkeypatch, tmp_path):
    service, _ = _no_index_service(monkeypatch, tmp_path, "gemini")
    r = service.handle(ChatRequest(message="How much does it cost to replace an airbag?"))
    assert r.status == TurnStatus.REFUSED
    assert r.refusal_reason == "out_of_scope"


def test_generate_general_knowledge_unit():
    fake = FakeLLM("plain answer")
    gen = GroundedGenerator(llm=fake)
    out = gen.generate_general_knowledge(QUESTION)
    assert out.text == "plain answer"
    assert out.valid_citations and not out.cited_refs and not out.invalid_refs
    assert out.flags == ["general_knowledge_no_sources"]
