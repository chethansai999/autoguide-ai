"""The Streamlit UI renders and completes a clarification round trip."""
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_ui_chat_and_clarification(loaded):
    app = AppTest.from_file("../app.py", default_timeout=60)
    app.run()
    assert not app.exception
    app.chat_input[0].set_value("Is the 2019 Ford F-150 safe?").run()
    assert not app.exception
    last = app.session_state.messages[-1]
    assert last["status"] == "needs_clarification" and last["clarification"]["options"]
    option_buttons = [b for b in app.button if b.label in last["clarification"]["options"]]
    assert option_buttons, "clarification options must render as buttons"
    option_buttons[0].click().run()
    assert not app.exception
    assert app.session_state.messages[-1]["status"] == "answered"
    assert app.session_state.messages[-1]["sources"]
    message_id = app.session_state.messages[-1]["message_id"]
    next(b for b in app.button if b.key == f"feedback_up_{message_id}").click().run()
    assert not app.exception
    assert app.session_state.messages[-1]["feedback_rating"] == 1
    from db.models import Feedback
    from db.session import get_sessionmaker
    from sqlalchemy import select

    with get_sessionmaker()() as db:
        ratings = db.scalars(select(Feedback).where(Feedback.message_id == message_id)).all()
    assert len(ratings) == 1 and ratings[0].rating == 1
    app.run()
    assert not any(b.key == f"feedback_up_{message_id}" for b in app.button)


def test_ui_feedback_failure_can_be_retried(loaded, monkeypatch):
    from api import client

    def fail(*_args, **_kwargs):
        raise ConnectionError("offline")

    monkeypatch.setattr(client, "submit_feedback", fail)
    app = AppTest.from_file("../app.py", default_timeout=60).run()
    app.chat_input[0].set_value("hello").run()
    message_id = app.session_state.messages[-1]["message_id"]
    next(b for b in app.button if b.key == f"feedback_down_{message_id}").click().run()
    assert not app.exception
    assert not app.session_state.messages[-1].get("feedback_rating")
    assert any("Feedback could not be saved" in e.value for e in app.error)


def test_ui_uses_remote_backend_status(loaded, monkeypatch):
    from api import client

    monkeypatch.setattr(client, "readiness", lambda: {
        "llm_provider": "mock", "llm_model": "remote-test-model", "vehicles_loaded": 7,
        "checks": {"vector_index": False}, "ready": False,
    })
    app = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not app.exception
    captions = [c.value for c in app.caption]
    assert "Configured generation model: remote-test-model" in captions
    assert "Vehicle records: 7" in captions
    assert any("Article library unavailable" in w.value for w in app.warning)


def test_ui_displays_approved_web_sources_and_wikipedia_credit(loaded, monkeypatch):
    from api import client

    monkeypatch.setattr(client, "answer_question", lambda *_a, **_k: {
        "answer": "Fixture history [3].", "session_id": "fixture-web-ui", "status": "answered",
        "sources": [{"ref": "[3]", "title": "Toyota Camry", "url": "https://en.wikipedia.org/wiki/Toyota_Camry"},
                    {"ref": "[4]", "title": "Fuel guide", "url": "https://www.fueleconomy.gov/feg/atv.shtml"},
                    {"ref": "[5]", "title": "Rejected", "url": "https://en.wikipedia.org.evil.test/car"}],
    })
    app = AppTest.from_file("../app.py", default_timeout=60).run()
    assert any("Website passages: Wikipedia 0" in caption.value for caption in app.caption)
    app.chat_input[0].set_value("Toyota Camry history").run()
    assert not app.exception
    links = [element.proto.label for element in app.get("link_button")]
    assert "[3] Toyota Camry" in links and "[4] Fuel guide" in links
    assert not any("Rejected" in label for label in links)
    assert any("Wikipedia contributors" in caption.value and "CC BY-SA 4.0" in caption.value for caption in app.caption)
