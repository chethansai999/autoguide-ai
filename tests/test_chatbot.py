"""rag/chatbot.py facade keeps its original contract."""
from core.llm import MockLLM
from rag.chatbot import AutoGuideChatbot
from rag.generation import GroundedGenerator


class FakeChecker:
    def __init__(self, sufficient=True):
        self.sufficient = sufficient

    def check_evidence(self, question, vehicle_filter=None):
        if not self.sufficient:
            return {"sufficient": False, "reason": "no_evidence", "evidence": []}
        return {"sufficient": True, "reason": "ok", "evidence": [
            {"id": "c1", "title": "Airbags", "url": "https://www.nhtsa.gov/airbags", "text": "Airbags inflate in a crash."}]}


def test_answer_with_sources():
    bot = AutoGuideChatbot(checker=FakeChecker(), generator=GroundedGenerator(llm=MockLLM()))
    result = bot.answer("How do airbags work?")
    assert result["status"] == "success" and result["sources"][0]["url"].endswith("/airbags")


def test_refusal_passthrough():
    bot = AutoGuideChatbot(checker=FakeChecker(sufficient=False), generator=GroundedGenerator(llm=MockLLM()))
    assert bot.answer("?")["status"] == "refused"
