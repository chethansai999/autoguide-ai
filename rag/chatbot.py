"""Backward-compatible facade over the RAG answer path.

The product path is agent/orchestrator.py (adds NLU, clarification, vehicle
data, sessions). This class keeps the original interface for scripts and
tests that call RAG directly:

    AutoGuideChatbot().answer(question, vehicle_filter=None)
        -> {"answer", "sources", "status", ["reason"]}

Prompts and refusal messages now live in rag/prompts.py; generation and
citation validation in rag/generation.py; evidence in rag/evidence.py.
"""

from __future__ import annotations

from core.schemas import Evidence
from rag.evidence import EvidenceChecker
from rag.generation import GroundedGenerator
from rag.prompts import REFUSAL_MESSAGES, SYSTEM_PROMPT  # noqa: F401  (re-exported)


class AutoGuideChatbot:

    def __init__(self, checker=None, generator=None):
        self.checker = checker or EvidenceChecker()
        self.generator = generator or GroundedGenerator()

    def answer(self, question, vehicle_filter=None):
        """Generate a grounded answer using NHTSA evidence."""

        if not question or not question.strip():
            return {"answer": "Please enter a vehicle safety question.", "sources": [], "status": "invalid_question"}

        result = self.checker.check_evidence(question=question, vehicle_filter=vehicle_filter)
        if not result["sufficient"]:
            reason = result["reason"]
            return {
                "answer": REFUSAL_MESSAGES.get(reason, REFUSAL_MESSAGES["no_evidence"]),
                "sources": [],
                "status": "refused",
                "reason": reason,
            }

        evidence = [
            Evidence(
                ref=i, evidence_id=str(r["id"]), kind="chunk",
                title=r.get("title", ""), url=r.get("url", ""), text=r.get("text", ""),
                attribution=r.get("attribution") or r.get("chunk", {}).get("attribution", {}),
            )
            for i, r in enumerate(result["evidence"], start=1)
        ]
        try:
            generated = self.generator.generate(question, evidence)
        except Exception as error:
            print(f"LLM request failed: {type(error).__name__}")
            return {"answer": REFUSAL_MESSAGES["error"], "sources": [], "status": "error"}

        if not generated.valid_citations:
            return {"answer": REFUSAL_MESSAGES["citation_validation_failed"], "sources": [],
                    "status": "citation_validation_failed"}

        sources = [
            {"reference": e.ref, "title": e.title, "url": e.url, "attribution": e.attribution}
            for e in evidence if e.ref in generated.cited_refs
        ]
        return {"answer": generated.text, "sources": sources, "status": "success"}


def main():
    chatbot = AutoGuideChatbot()
    question = "How does automatic emergency braking work?"
    result = chatbot.answer(question)
    print("\nQuestion:", question)
    print("\nStatus:", result["status"])
    print("\nAnswer:\n", result["answer"])
    for source in result["sources"]:
        print(f"[{source['reference']}] {source['title']}\n{source['url']}")


if __name__ == "__main__":
    main()
