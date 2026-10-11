from fastapi.testclient import TestClient
import pytest


def test_api_endpoints(loaded):
    from api.main import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        ready = client.get("/ready").json()
        assert ready["vehicles_loaded"] == 39 and ready["llm_provider"] == "mock"
        assert ready["ready"] is False and ready["external_services"] == "not_checked"
        assert ready["checks"]["vector_index"] is True
        chat = client.post("/v1/chat", json={"message": "What does blind spot detection do?"}).json()
        assert chat["status"] == "answered" and chat["citations"]
        msgs = client.get(f"/v1/sessions/{chat['session_id']}/messages").json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        fb = client.post("/v1/feedback", json={"message_id": chat["message_id"], "rating": 1})
        assert fb.status_code == 200
        res = client.get("/v1/vehicles/resolve", params={"make": "ford", "model": "f150", "year": 2019}).json()
        assert res["status"] == "ambiguous"


def test_ui_client_contract(loaded):
    from api.client import answer_question

    out = answer_question("How does automatic emergency braking work?")
    assert {"answer", "sources", "trace", "refused", "demo", "status", "session_id"} <= set(out)
    assert out["sources"][0]["ref"] == "[1]" and out["demo"] is True


def test_readiness_reports_missing_and_incompatible_artifacts(loaded, tmp_path, monkeypatch):
    from api import main
    from agent.orchestrator import ChatService
    from core.config import get_settings

    settings = get_settings().model_copy(update={
        "data_dir": tmp_path, "chunk_version": "not-built", "nlu_backend": "hf",
        "nlu_model_dir": tmp_path / "missing-models",
    })
    service = ChatService(settings=settings)
    monkeypatch.setattr(main, "_service", service)
    with TestClient(main.app) as client:
        response = client.get("/ready")
        assert response.status_code == 503
        report = response.json()
        assert report["vehicles_loaded"] == 39
        codes = {b["code"] for b in report["blockers"]}
        assert {"active_index", "chunks", "vector_index", "nlu_artifacts"} <= codes
        assert service._nlu is None, "readiness must not load an NLU model"
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("problem", ["missing_database", "missing_collection", "wrong_ids", "malformed_chunks"])
def test_readiness_detects_stale_vector_artifacts(loaded, tmp_path, monkeypatch, problem):
    from api import readiness
    from agent.orchestrator import ChatService
    from core.config import get_settings

    settings = get_settings().model_copy()
    service = ChatService(settings=settings)
    if problem == "missing_database":
        settings.chroma_db_path = tmp_path / "missing-chroma"
    elif problem == "missing_collection":
        def missing(*_args):
            raise ValueError("collection missing")
        monkeypatch.setattr(readiness, "_vector_ids", missing)
    elif problem == "wrong_ids":
        monkeypatch.setattr(readiness, "_vector_ids", lambda *_args: {"stale-id"})
    else:
        settings.data_dir = tmp_path
        path = settings.chunks_dir() / "chunks.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text('{"id": "broken", "text": 7}\n')
    report = readiness.inspect_readiness(service)
    assert report["checks"]["active_index"] is True
    assert report["checks"]["vector_index"] is False
    if problem == "missing_database":
        assert not settings.chroma_db_path.exists(), "probe must not create an empty database"


def test_readiness_accepts_consistent_local_artifacts(tmp_path, monkeypatch):
    """Simulated provider configuration, without network or semantic models."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from api import main, readiness
    from core.config import Settings

    settings = Settings(_env_file=None, data_dir=tmp_path, llm_provider="openai",
                        openai_api_key="offline-test-placeholder", embedding_backend="sentence-transformers")
    path = settings.chunks_dir() / "chunks.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"id": "fixture-id", "text": "Fixture evidence."}\n')
    registry = SimpleNamespace(collection=settings.collection_name(), strategy=settings.chunk_strategy,
                               version=settings.chunk_version, embedding_model=settings.embedding_model, chunk_count=1)
    factory = MagicMock()
    db = factory.return_value.__enter__.return_value
    db.scalar.return_value = 1
    db.scalars.return_value.all.return_value = [registry]
    service = SimpleNamespace(settings=settings, session_factory=factory)
    monkeypatch.setattr(main, "_service", service)
    monkeypatch.setattr(readiness, "_vector_ids", lambda *_args: {"fixture-id"})
    with TestClient(main.app) as client:
        response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True
    assert response.json()["external_services"] == "not_checked"
    assert "offline-test-placeholder" not in response.text


def test_feedback_rejects_unknown_and_user_messages(loaded):
    from api.main import app

    with TestClient(app) as client:
        out = client.post("/v1/chat", json={"message": "hello"}).json()
        user = client.get(f"/v1/sessions/{out['session_id']}/messages").json()[0]
        assert client.post("/v1/feedback", json={"message_id": "missing", "rating": 1}).status_code == 404
        assert client.post("/v1/feedback", json={"message_id": user["id"], "rating": 1}).status_code == 422


def test_http_client_feedback_and_degraded_readiness(loaded, monkeypatch):
    import httpx
    from api import client
    from core.config import get_settings

    settings = get_settings().model_copy(update={"autoguide_api_url": "http://backend.test/"})
    monkeypatch.setattr(client, "get_settings", lambda: settings)
    calls = []

    def post(url, **kwargs):
        calls.append((url, kwargs["json"]))
        return httpx.Response(200, json={"ok": True}, request=httpx.Request("POST", url))

    def get(url, **_kwargs):
        return httpx.Response(503, json={"ready": False}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(httpx, "get", get)
    client.submit_feedback("assistant-id", -1)
    assert calls == [("http://backend.test/v1/feedback", {"message_id": "assistant-id", "rating": -1, "comment": None})]
    assert client.readiness() == {"ready": False}
    with pytest.raises(ValueError):
        client.submit_feedback("assistant-id", 0)


def test_chat_recalls_share_transaction_and_reuse_cache(loaded, monkeypatch):
    """A successful API response must not deadlock against the active chat write."""
    import httpx
    from api import main
    from agent.orchestrator import ChatService
    from core.config import get_settings
    from db.models import RecallCache
    from structured.recalls import RecallsTool

    settings = get_settings().model_copy(update={"enable_live_recalls_api": True})
    service = ChatService(settings=settings)
    requests = []

    def reply(request):
        requests.append(request)
        return httpx.Response(200, json={"results": [{
            "NHTSACampaignNumber": "FIXTURE-RECALL", "Component": "AIR BAGS",
            "Summary": "Fixture airbag issue.", "Remedy": "Fixture dealer remedy.",
        }]})

    with httpx.Client(transport=httpx.MockTransport(reply)) as http:
        service.recalls = RecallsTool(service.session_factory, settings, http=http)
        monkeypatch.setattr(main, "_service", service)
        with TestClient(main.app) as client:
            first = client.post("/v1/chat", json={"message": "Are there recalls for the 2022 Toyota Camry?"}).json()
            assert first["status"] == "answered", first
            assert first["citations"]
            assert first["trace"]["evidence_decision"]["recalls"] == "found"
            second = client.post("/v1/chat", json={"message": "Are there recalls for the 2022 Toyota Camry?"}).json()
            assert second["status"] == "answered"
        assert len(requests) == 1, "the committed cache must serve the next turn"
        with service.session_factory() as db:
            assert db.get(RecallCache, "TOYOTA|CAMRY|2022") is not None
