"""Behaviour when the LLM provider can't be reached (flaky Wi-Fi, DNS blips).

Network failures fall back to an extractive, clearly labelled answer built
from the same evidence. Errors the provider actually returned (bad key,
quota) are never hidden. Evaluation turns the fallback off.
"""

from __future__ import annotations

import socket

import pytest

from core.llm import LLMResult, is_connection_error
from core.schemas import ChatRequest, Evidence, TurnStatus
from rag.generation import FALLBACK_FLAG, GroundedGenerator
from rag.prompts import LLM_UNREACHABLE_NOTE


class APIConnectionError(Exception):
    """Same class name as openai.APIConnectionError / anthropic.APIConnectionError."""


class AuthenticationError(Exception):
    """Shaped like openai.AuthenticationError: the provider answered with 401."""

    status_code = 401


class RaisingLLM:
    provider, model = "openai", "gpt-test"

    def __init__(self, error: Exception):
        self.error = error

    def complete(self, *args, **kwargs) -> LLMResult:
        raise self.error


def dns_failure() -> Exception:
    try:
        try:
            raise socket.gaierror(8, "nodename nor servname provided, or not known")
        except socket.gaierror as inner:
            raise APIConnectionError("Connection error.") from inner
    except APIConnectionError as outer:
        return outer


EVIDENCE = [Evidence(
    ref=1, evidence_id="vehicle:test", kind="vehicle_record", title="2019 Ford F-150 SuperCrew 4x4",
    url="https://www.nhtsa.gov/ratings",
    text="Vehicle: 2019 FORD F-150\nOverall rating: 4 stars\nFrontal crash rating: 4 stars",
)]


def test_connection_errors_are_recognised():
    assert is_connection_error(dns_failure())
    assert is_connection_error(socket.gaierror(8, "nodename nor servname provided"))
    assert is_connection_error(ConnectionResetError())
    assert not is_connection_error(AuthenticationError("Incorrect API key provided"))
    assert not is_connection_error(ValueError("bad request"))


def test_provider_error_with_earlier_network_context_is_not_masked():
    """A 401 raised after a retried network error must stay a 401."""
    try:
        try:
            raise APIConnectionError("first attempt")
        except APIConnectionError:
            raise AuthenticationError("Incorrect API key provided")
    except AuthenticationError as error:
        assert not is_connection_error(error)


def test_unreachable_llm_falls_back_to_labelled_extractive_answer():
    answer = GroundedGenerator(llm=RaisingLLM(dns_failure()), allow_fallback=True).generate(
        "Is the 2019 Ford F-150 SuperCrew 4x4 safe?", EVIDENCE
    )
    assert answer.text.startswith(LLM_UNREACHABLE_NOTE)
    assert "Mock answer" not in answer.text
    assert "overall rating 4 stars" in answer.text and "frontal crash rating 4 stars" in answer.text
    assert answer.valid_citations and answer.cited_refs == {1}
    assert FALLBACK_FLAG in answer.flags
    trace = answer.llm.as_trace()
    assert trace["provider"] == "extractive-fallback" and trace["cost_usd"] == 0.0
    assert trace["fallback_reason"].startswith("APIConnectionError")


def test_extractive_answer_picks_what_was_asked():
    from rag.generation import extractive_answer

    record = Evidence(
        ref=2, evidence_id="vehicle:x", kind="vehicle_record", title="t", url="u",
        text=("Vehicle: 2016 JEEP WRANGLER\nVariant: WRANGLER · 4-door · 4WD\n"
              "Source: NHTSA 5-Star Safety Ratings data\nOverall rating: not rated\n"
              "Frontal crash rating: not rated\nRollover rating: 3 stars\n"
              "Blind spot detection: optional\n"
              "Note: NHTSA lists 2 versions of this vehicle and these values are the same for all of them."),
    )
    rollover = extractive_answer("2016 jeep wrangler rollover rating", [record])
    assert "rollover rating 3 stars" in rollover and "frontal" not in rollover
    assert "same for all 2 versions" in rollover and rollover.endswith("[2].")
    feature = extractive_answer("Does it have blind spot detection?", [record])
    assert "blind spot detection optional" in feature and "rollover" not in feature


def test_unreachable_host_fails_fast_and_is_remembered(monkeypatch):
    import time as _time

    from core import netcheck

    netcheck.reset()
    calls = []

    def slow_dns(*args, **kwargs):
        calls.append(1)
        _time.sleep(2)
        raise socket.gaierror(8, "nodename nor servname provided")

    monkeypatch.setattr(netcheck.socket, "getaddrinfo", slow_dns)
    try:
        start = _time.perf_counter()
        with pytest.raises(netcheck.HostUnreachable):
            netcheck.ensure_reachable("api.example.invalid", timeout=0.2, cooldown=30)
        assert _time.perf_counter() - start < 1.5  # did not wait for the slow lookup
        with pytest.raises(netcheck.HostUnreachable):  # cool-down: no second lookup
            netcheck.ensure_reachable("api.example.invalid", timeout=0.2, cooldown=30)
        assert len(calls) == 1
        assert is_connection_error(netcheck.HostUnreachable("down"))
    finally:
        netcheck.reset()


def test_provider_errors_are_not_hidden():
    generator = GroundedGenerator(llm=RaisingLLM(AuthenticationError("Incorrect API key")), allow_fallback=True)
    with pytest.raises(AuthenticationError):
        generator.generate("Is the 2019 Ford F-150 safe?", EVIDENCE)


def test_fallback_can_be_turned_off():
    generator = GroundedGenerator(llm=RaisingLLM(dns_failure()), allow_fallback=False)
    with pytest.raises(APIConnectionError):
        generator.generate("Is the 2019 Ford F-150 safe?", EVIDENCE)


def test_orchestrator_answers_from_ratings_when_llm_unreachable(service, monkeypatch):
    monkeypatch.setattr(service.generator, "_llm", RaisingLLM(dns_failure()))
    monkeypatch.setattr(service.generator, "allow_fallback", True)
    r = service.handle(ChatRequest(message="What is the overall rating of the 2018 Subaru Outback?"))
    assert r.status == TurnStatus.ANSWERED, r.answer
    assert r.answer.startswith(LLM_UNREACHABLE_NOTE)
    assert r.citations
    assert FALLBACK_FLAG in r.trace.guard_flags
    assert "error" not in r.trace.llm and r.trace.llm["fallback_reason"]


def test_orchestrator_reports_provider_error(service, monkeypatch):
    monkeypatch.setattr(service.generator, "_llm", RaisingLLM(AuthenticationError("Incorrect API key")))
    r = service.handle(ChatRequest(message="What is the overall rating of the 2018 Subaru Outback?"))
    assert r.status == TurnStatus.ERROR
    assert "AuthenticationError" in r.trace.llm["error"]


def test_extractive_answer_says_when_versions_differ():
    from rag.generation import extractive_answer

    record = Evidence(
        ref=1, evidence_id="vehicle:y", kind="vehicle_record", title="t", url="u",
        text=("Vehicle: 2022 TOYOTA COROLLA\nVariant: COROLLA · 4-door · FWD\nOverall rating: 5 stars\n"
              "Note: NHTSA lists 5 versions of this vehicle with different values; this is one of them."),
    )
    answer = extractive_answer("Is it safe?", [record])
    assert "one of 5 versions, which differ" in answer and "same for all" not in answer
