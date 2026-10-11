"""Intensive UI audit: every view, every button, every flow must work.

Boots app.py with Streamlit's AppTest runner, walks all five views, and
clicks every interactive control, asserting that no uncaught exception
escapes and that the expected state changes actually happen.

Runs against the mock LLM so the audit stays fast and deterministic.
Real-generation E2E is covered separately (ollama/local), see test_e2e note.
"""

import os
from pathlib import Path

import pytest

os.environ["LLM_PROVIDER"] = "mock"
os.environ["EMBEDDING_BACKEND"] = "hashing"
os.environ["RERANKER_BACKEND"] = "none"
os.environ["NLU_BACKEND"] = "rules"
os.environ["AUTOGUIDE_API_URL"] = ""

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "app.py")


@pytest.fixture(scope="module", autouse=True)
def _real_database():
    """Point this module's app boots at the real dev database.

    Other test modules run against conftest's temporary fixture database
    (39-row sample). Their session fixture leaves DATABASE_URL pointing at
    it and a cached backend behind, so this audit would otherwise assert
    against the wrong data. Swap in the real DB for this module only and
    restore everything afterwards, so later modules are unaffected.
    """
    import streamlit as st

    from api import client as ui_client
    from core import config
    from db import session as db_session

    saved_env = dict(os.environ)
    os.environ["DATABASE_URL"] = f"sqlite:///{ROOT / 'data' / 'autoguide.db'}"
    os.environ["SAFERCAR_CSV"] = str(ROOT / "data" / "structured" / "Safercar_data.csv")
    config.get_settings.cache_clear()
    db_session.get_engine.cache_clear()
    ui_client._service.cache_clear()
    st.cache_data.clear()
    yield
    os.environ.clear()
    os.environ.update(saved_env)
    config.get_settings.cache_clear()
    db_session.get_engine.cache_clear()
    ui_client._service.cache_clear()
    st.cache_data.clear()


def boot():
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, f"boot raised: {at.exception}"
    return at


def go_view(at, view):
    at.button(key=f"nav_{view}").click().run()
    assert not at.exception, f"nav to {view} raised: {at.exception}"


def test_all_five_views_render_and_navigate():
    at = boot()
    for view in ("chat", "lookup", "recalls", "features", "resources"):
        go_view(at, view)
        assert at.session_state["view"] == view


def test_sidebar_system_details_show_vehicle_count():
    at = boot()
    captions = " ".join(c.value for c in at.caption)
    assert "Vehicle records: 17,313" in captions


def test_topic_buttons_all_work():
    at = boot()
    for i in range(4):
        at.button(key=f"topic{i}").click().run()
        assert not at.exception, f"topic{i} raised: {at.exception}"
        # the click's rerun processes the queued question end to end
        msgs = at.session_state["messages"]
        assert any(m["role"] == "assistant" for m in msgs), f"topic{i} produced no answer"
        assert msgs[-1]["role"] == "assistant"


def test_suggestion_buttons_all_work():
    at = boot()
    # INDEX_READY is False in this repo state -> 6 vehicle suggestions
    for i in range(6):
        at2 = boot()
        at2.button(key=f"suggest{i}").click().run()
        assert not at2.exception
        at2.run()
        assert not at2.exception
        roles = [m["role"] for m in at2.session_state["messages"]]
        assert "assistant" in roles, f"suggest{i} produced no answer"


def test_chat_input_flow_and_feedback_buttons():
    at = boot()
    at.chat_input[0].set_value("What is the overall NHTSA star rating of the 2020 Toyota Camry?").run()
    assert not at.exception
    msgs = at.session_state["messages"]
    assert msgs and msgs[-1]["role"] == "assistant"
    assert msgs[-1].get("message_id"), "assistant message has no message_id"
    mid = msgs[-1]["message_id"]
    at.button(key=f"feedback_up_{mid}").click().run()
    assert not at.exception
    assert msgs[-1].get("feedback_rating") == 1


def test_new_conversation_and_export_chat():
    at = boot()
    at.chat_input[0].set_value("Is the 2020 Honda CR-V safe?").run()
    assert at.session_state["messages"]
    assert at.download_button[0].label == "↓ Export Chat"
    new_conv = next(b for b in at.button if "New Conversation" in b.label)
    new_conv.click().run()
    assert not at.exception
    assert at.session_state["messages"] == []
    assert at.session_state["session_id"] is None


def _pick(options, *names, default=0):
    """Pick an option case-insensitively; fall back to options[default]."""
    lowered = [str(o).lower() for o in options]
    for name in names:
        if name.lower() in lowered:
            return options[lowered.index(name.lower())]
    return options[default]


def test_lookup_cascade_variant_and_cross_links():
    at = boot()
    go_view(at, "lookup")
    # cascade: select non-default make/model/year (CSV stores them uppercase)
    makes = at.selectbox(key="lookup_make_sel").options
    assert len(makes) > 10
    target_make = _pick(makes, "TOYOTA", "FORD", default=1)
    assert target_make != makes[0], "fallback selection did not leave the default"
    at.selectbox(key="lookup_make_sel").set_value(target_make).run()
    assert not at.exception
    models = at.selectbox(key="lookup_family_sel").options
    assert models, "no models after make selection"
    target_model = _pick(models, "CAMRY", "F-150")
    assert target_model != models[0] or len(models) == 1
    at.selectbox(key="lookup_family_sel").set_value(target_model).run()
    years = at.selectbox(key="lookup_year_sel").options
    assert years, "no years after model selection"
    target_year = 2020 if 2020 in years else years[0]
    at.selectbox(key="lookup_year_sel").set_value(target_year).run()
    assert not at.exception
    # metrics render
    labels = [m.label for m in at.metric]
    assert "Overall rating" in labels
    # variant picker when multiple variants
    if len(at.selectbox) > 3:
        opts = at.selectbox(key="lookup_variant_sel").options
        if len(opts) > 1:
            at.selectbox(key="lookup_variant_sel").set_value(opts[-1]).run()
            assert not at.exception
    # dataframe of features renders with rows
    assert len(at.dataframe) >= 1
    # cross-links: ask assistant -> chat view; the question is answered in the same cycle
    at.button(key="lookup_ask").click().run()
    assert not at.exception
    assert at.session_state["view"] == "chat"
    msgs = at.session_state["messages"]
    assert msgs and msgs[-1]["role"] == "assistant"
    # check recalls -> recalls view with cascading selection preselected
    at2 = boot()
    go_view(at2, "lookup")
    at2.selectbox(key="lookup_make_sel").set_value(target_make).run()
    at2.selectbox(key="lookup_family_sel").set_value(target_model).run()
    at2.selectbox(key="lookup_year_sel").set_value(target_year).run()
    at2.button(key="lookup_recalls").click().run()
    assert not at2.exception
    assert at2.session_state["view"] == "recalls"


def test_recalls_flow_runs_without_exception():
    at = boot()
    go_view(at, "recalls")
    makes = at.selectbox(key="recall_make_sel").options
    make = _pick(makes, "FORD", "TOYOTA")
    at.selectbox(key="recall_make_sel").set_value(make).run()
    models = at.selectbox(key="recall_family_sel").options
    model = _pick(models, "F-150", "CAMRY")
    at.selectbox(key="recall_family_sel").set_value(model).run()
    years = at.selectbox(key="recall_year_sel").options
    year = 2020 if 2020 in years else years[0]
    at.selectbox(key="recall_year_sel").set_value(year).run()
    at.button(key="recall_submit").click().run()
    assert not at.exception, f"recall check raised: {at.exception}"
    # result or honest error must be stored/rendered
    assert at.session_state.get("recall_result") is not None


def test_features_all_18_buttons_work():
    at = boot()
    go_view(at, "features")
    keys = [b.key for b in at.button if b.key and b.key.startswith("feature_")]
    assert len(keys) == 18, f"expected 18 feature buttons, found {len(keys)}: {keys}"
    for key in keys:
        at2 = boot()
        go_view(at2, "features")
        at2.button(key=key).click().run()
        assert not at2.exception, f"{key} raised: {at2.exception}"
        assert at2.session_state["view"] == "chat"
        # the queued question is processed in the same rerun cycle
        msgs = at2.session_state["messages"]
        assert msgs and msgs[-1]["role"] == "assistant", f"{key} left no answer"


def test_resources_view_renders_sections():
    # AppTest does not expose st.link_button elements; assert the resource
    # sections render here and validate the link URLs statically below.
    at = boot()
    go_view(at, "resources")
    md = " ".join(m.value for m in at.markdown)
    for section in ("Official links", "Check a recall by VIN",
                    "How NHTSA 5-Star Ratings work", "About this demo"):
        assert section in md, f"missing resources section: {section}"


def test_all_link_button_urls_are_safe():
    import re
    from urllib.parse import urlparse

    src = (Path(__file__).resolve().parent.parent / "app.py").read_text()
    calls = re.findall(r"st\.link_button\((?:f?[\"']([^\"']+)[\"']\s*,\s*)?([\"'])(https?://[^\"']+)\2", src)
    assert calls, "no link_button calls found in app.py"
    allowed = ("nhtsa.gov", "en.wikipedia.org", "www.fueleconomy.gov", "creativecommons.org")
    for label, _q, url in calls:
        u = urlparse(url)
        assert u.scheme == "https", f"non-https link '{label}': {url}"
        host = u.hostname or ""
        assert any(host == d or host.endswith("." + d) for d in allowed), \
            f"unexpected link host '{label}': {url}"
